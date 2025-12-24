import os
import cv2
import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from math import ceil, sqrt
from collections import Counter
import random

def extract_bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(max(0, min(xs))), int(max(0, min(ys))), int(max(xs)), int(max(ys))  # x1,y1,x2,y2


def decode_and_crop_bbox(frame_path, bbox):
    if not os.path.exists(frame_path):
        return None
    img = cv2.imread(frame_path)
    if img is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - 1)
    y1 = max(0, y1 - 1)
    x2 = min(img.shape[1] - 1, x2 + 1)
    y2 = min(img.shape[0] - 1, y2 + 1)
    if x2 < x1 or y2 < y1:
        return None
    return img[y1:y2 + 1, x1:x2 + 1].copy()


def arrange_grid_dynamic(crops, pad=5, bg_color=(255, 255, 255), line_color=(0, 0, 0)):
    """Arrange variable-sized crops into a roughly square grid."""
    if not crops:
        return None
    n = len(crops)
    grid_size = int(ceil(sqrt(n)))
    rows, cols = ceil(n / grid_size), grid_size

    max_h = max(c.shape[0] for c in crops)
    max_w = max(c.shape[1] for c in crops)

    padded_crops = []
    for c in crops:
        h, w = c.shape[:2]
        padded = cv2.copyMakeBorder(
            c, 0, max_h - h, 0, max_w - w,
            cv2.BORDER_CONSTANT, value=bg_color
        )
        padded_crops.append(padded)

    grid_h = rows * max_h + (rows - 1) * pad
    grid_w = cols * max_w + (cols - 1) * pad
    grid_img = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            y = r * (max_h + pad)
            x = c * (max_w + pad)
            if idx < n:
                grid_img[y:y+max_h, x:x+max_w] = padded_crops[idx]

    if pad > 0:
        for c in range(1, cols):
            x = c * max_w + (c - 1) * pad
            grid_img[:, x:x+pad] = line_color
        for r in range(1, rows):
            y = r * max_h + (r - 1) * pad
            grid_img[y:y+pad, :] = line_color

    return grid_img


# -------------------
# Process a single XML video
# -------------------

def process_video_xml(xml_path, frames_root, out_frames_dir, entries, id_start, args):
    """
    Parse xml_path, collect all word boxes (from different trajectories),
    group them into fixed-size batches, build one grid per batch,
    and use the most common word as gpt label.
    """
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return id_start, 0

    root = tree.getroot()

    # Collect all word boxes across frames
    samples = []  # [(frame_jpg, bbox, word), ...]
    for frame in root.findall("frame"):
        frame_id = frame.get("ID")
        if frame_id is None:
            continue
        try:
            frame_idx = int(frame_id)
            frame_jpg = f"{frame_idx}.jpg"
        except:
            frame_jpg = f"{frame_id}.jpg"

        for obj in frame.findall("object"):
            trans = (obj.get("Transcription") or "").strip()
            if not trans or trans == "##DONT#CARE##":
                continue
            points = []
            for p in obj.findall("Point"):
                try:
                    x = int(float(p.get("x")))
                    y = int(float(p.get("y")))
                    points.append((x, y))
                except:
                    continue
            if not points:
                continue
            bbox = extract_bbox_from_points(points)
            samples.append((frame_jpg, bbox, trans))

    if not samples:
        return id_start, 0

    random.seed(0)
    random.shuffle(samples)

    Path(out_frames_dir).mkdir(parents=True, exist_ok=True)

    batch_size = getattr(args, "batch_size", 9)
    count = 0
    current_id = id_start

    # group into batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        crops = []
        words = []
        for frame_jpg, bbox, word in batch:
            frame_path = os.path.join(frames_root, frame_jpg)
            crop = decode_and_crop_bbox(frame_path, bbox)
            if crop is not None:
                crops.append(crop)
                words.append(word)

        if len(crops) < 2:
            continue  # skip too small groups

        grid = arrange_grid_dynamic(crops)
        if grid is None:
            continue

        # Determine most common word
        word_counts = Counter(words)
        common_word, _ = word_counts.most_common(1)[0]

        out_name = f"{current_id}.png"
        out_path = os.path.join(out_frames_dir, out_name)

        try:
            success = cv2.imwrite(out_path, grid)
            if not success:
                ext_img = cv2.imencode('.png', grid)[1]
                with open(out_path, 'wb') as f:
                    f.write(ext_img.tobytes())
        except Exception as e:
            print(f"Failed to save {out_path}: {e}")
            continue

        # JSON entry
        prompt = (
            f"<image>\nYou see {len(crops)} cropped text regions from different words in the same video.\n"
            f"What is the most common word among them?"
        )

        entry = {
            "id": current_id,
            "image": out_path,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": common_word}
            ]
        }
        entries.append(entry)

        count += 1
        current_id += 1

    return current_id, count

def main(args):
    entries = []
    current_id = 1

    xml_files = sorted(Path(args.xml_root).glob("Video_*.xml"))

    for xml_file in xml_files:
        video_name = xml_file.stem

        # extract numeric part
        parts = video_name.split("_")
        if len(parts) < 2:
            continue
        try:
            video_num = int(parts[1])
        except:
            continue

        if video_num < 2 or video_num > 37:
            continue

        frames_root = os.path.join(args.frames_root, video_name.replace("_GT",""))
        if not os.path.isdir(frames_root):
            print(f"Warning: missing frames for {video_name}, skipping.")
            continue

        out_frames_dir = os.path.join(args.output_root, video_name)
        current_id, saved_count = process_video_xml(str(xml_file), frames_root, out_frames_dir, entries, current_id, args)
        print(f"Processed {video_name}... saved {saved_count} mixed-word grids.")

    # Save combined JSON
    json_dir = os.path.join(args.output_root, "json")
    Path(json_dir).mkdir(parents=True, exist_ok=True)
    out_json_path = os.path.join(json_dir, "all_videos.json")

    try:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON with {len(entries)} entries to {out_json_path}")
    except Exception as e:
        print(f"Failed to write JSON file {out_json_path}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", required=True, help="Root folder containing per-video frame folders (Video_XXXX/000001.jpg)")
    p.add_argument("--xml_root", required=True, help="Root folder containing per-video XML files named like Video_123_4_5.xml")
    p.add_argument("--output_root", required=True, help="Output folder; will create per-video subfolders and output_root/json/all_videos.json")
    p.add_argument("--batch_size", type=int, default=9, help="Number of word crops per grid image")
    p.add_argument("--padding", type=int, default=6)
    args = p.parse_args()
    main(args)
