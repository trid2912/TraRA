import os
import json
import numpy as np
from PIL import Image, ImageDraw
from math import ceil
import cv2
from skimage.feature import hog
from sklearn.preprocessing import normalize

TARGET_SIZE = (64, 64)

def make_fixed_crop(crop: Image.Image):
    canvas = Image.new("RGB", TARGET_SIZE, (0,0,0))
    crop.thumbnail(TARGET_SIZE, Image.LANCZOS)
    offset = ((TARGET_SIZE[0] - crop.width)//2, (TARGET_SIZE[1] - crop.height)//2)
    canvas.paste(crop, offset)
    return canvas

def compute_rootsift(crop: Image.Image):
    fixed = make_fixed_crop(crop)
    gray = np.array(fixed.convert("L"), dtype=np.uint8)
    sift = cv2.SIFT_create()
    step = 8
    kp = [cv2.KeyPoint(x, y, step) for y in range(step//2, gray.shape[0], step)
                                  for x in range(step//2, gray.shape[1], step)]
    if not kp: return np.zeros(128, dtype=np.float32)
    _, desc = sift.compute(gray, kp)
    if desc is None or len(desc) == 0: return np.zeros(128, dtype=np.float32)
    desc = desc.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-8)
    desc = np.sqrt(desc)
    return desc.mean(axis=0)

def compute_hog(crop: Image.Image):
    fixed = make_fixed_crop(crop)
    gray = np.array(fixed.convert("L"))
    return hog(gray, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(3,3), block_norm='L2-Hys',
               transform_sqrt=True, feature_vector=True)

def compute_area(masked_crop: Image.Image):
    arr = np.array(masked_crop)
    foreground = np.any(arr != 0, axis=-1)
    count = foreground.sum()  # raw number of non-zero pixels
    return np.array([count], dtype=np.float32)

def clean_and_output_json(json_dir, frames_dir, output_json_dir):
    os.makedirs(output_json_dir, exist_ok=True)

    for json_file in sorted(os.listdir(json_dir)):
        if not json_file.endswith(".json"): continue
        video_name = json_file.rsplit(".", 1)[0]
        json_path = os.path.join(json_dir, json_file)
        print(f"Processing {video_name}...")

        with open(json_path) as f:
            data = json.load(f)

        # Group by original tracker ID
        raw_tracks = {}
        for fid_str, anns in data.items():
            fid = int(fid_str)
            for ann in anns:
                old_id = ann.get("ID", f"temp_{fid}")
                raw_tracks.setdefault(old_id, []).append((fid, ann))

        # New global ID counter
        global_new_id = 0
        new_id_mapping = {}  # (video, old_id, subcluster) → new global ID
        cleaned_data = {}    # final output: frame_id → list of anns with new ID

        for old_id, instances in raw_tracks.items():
            instances.sort(key=lambda x: x[0])
            crops, frame_ids = [], []

            # Extract exact masked crops
            for fid, ann in instances:
                img_path = os.path.join(frames_dir, video_name, f"{fid}.jpg")
                if not os.path.exists(img_path): continue
                img = Image.open(img_path).convert("RGB")
                poly = [tuple(map(int, p)) for p in ann["segmentation"][0]]
                xs, ys = zip(*poly)
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

                crop = img.crop((x0,y0,x1,y1))
                mask = Image.new("L", crop.size, 0)
                ImageDraw.Draw(mask).polygon([(x-x0,y-y0) for x,y in poly], fill=255)
                masked = Image.composite(crop, Image.new("RGB", crop.size), mask)

                crops.append(masked)
                frame_ids.append(fid)

            if len(crops) < 5: 
                # short track → keep original ID or assign new
                new_id = global_new_id
                global_new_id += 1
                for fid, ann in instances:
                    fid_str = str(fid)
                    cleaned_data.setdefault(fid_str, []).append({**ann, "ID": new_id})
                continue

            # Extract features
            sift = np.stack([compute_rootsift(c) for c in crops])
            hog  = np.stack([compute_hog(c) for c in crops])
            area = np.stack([compute_area(c) for c in crops])

            sift = normalize(sift, norm='l2')
            hog  = normalize(hog, norm='l2')
            area = normalize(area, norm='l2')

            X = np.hstack([sift, hog, area])
            X = normalize(X, norm='l2')

            # Adaptive threshold
            consec_dists = np.linalg.norm(X[1:] - X[:-1], axis=1)
            threshold = max(np.mean(consec_dists), 0.48)
            # threshold = max(np.mean(consec_dists) , 0.8)
            # Online clustering: assign to most similar existing cluster
            active_clusters = []

            for i in range(len(X)):
                min_dist = np.inf
                best_cluster_idx = -1

                for ci, cluster in enumerate(active_clusters):
                    dist = np.linalg.norm(X[i] - cluster['last_feat'])
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_idx = ci

                if best_cluster_idx != -1 and min_dist < threshold:
                    active_clusters[best_cluster_idx]['indices'].append(i)
                    active_clusters[best_cluster_idx]['last_feat'] = X[i]
                else:
                    active_clusters.append({
                        'indices': [i],
                        'last_feat': X[i]
                    })

            # Assign new global IDs to each sub-cluster
            # ---------------------------------------------
            # Merge clusters smaller than 3 into nearest valid cluster
            # ---------------------------------------------

            # Separate clusters
            valid_clusters = [c for c in active_clusters if len(c['indices']) >= 3]
            small_clusters = [c for c in active_clusters if len(c['indices']) < 3]

            # If no valid cluster exists, merge everything into one
            if len(valid_clusters) == 0:
                merged = {'indices': []}
                for c in active_clusters:
                    merged['indices'].extend(c['indices'])
                valid_clusters = [merged]
                small_clusters = []

            # Compute mean feature for each cluster
            cluster_means = []
            for c in valid_clusters:
                feats = X[c['indices']]
                cluster_means.append(feats.mean(axis=0))

            # Merge each small cluster into nearest valid cluster
            for sc in small_clusters:
                sc_mean = X[sc['indices']].mean(axis=0)

                # find nearest valid cluster
                dists = [np.linalg.norm(sc_mean - vc_mean) for vc_mean in cluster_means]
                best_idx = int(np.argmin(dists))

                # merge into that cluster
                valid_clusters[best_idx]['indices'].extend(sc['indices'])

            # ---------------------------------------------
            # Assign new IDs from the merged cluster set
            # ---------------------------------------------

            for cluster in valid_clusters:
                new_id = global_new_id
                global_new_id += 1

                for idx in cluster['indices']:
                    fid = frame_ids[idx]
                    orig_fid, orig_ann = instances[idx]
                    fid_str = str(fid)
                    cleaned_data.setdefault(fid_str, []).append({**orig_ann, "ID": new_id})

        # Save cleaned JSON
        output_path = os.path.join(output_json_dir, f"{video_name}.json")
        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)

        print(f"Saved {output_path} — {global_new_id} real identities")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, help="Input JSONs with raw tracks")
    parser.add_argument("--frames_dir", required=True, help="Frames folder")
    parser.add_argument("--output_json_dir", required=True, help="Where to save cleaned JSONs")
    args = parser.parse_args()

    clean_and_output_json(
        json_dir=args.json_dir,
        frames_dir=args.frames_dir,
        output_json_dir=args.output_json_dir
    )