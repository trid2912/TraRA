import os
import cv2
import json
import numpy as np
# from PIL import Image
from PIL import Image, ImageDraw

from math import ceil, sqrt
from pycocotools import mask as maskUtils
from ovis.model.modeling_ovis import Ovis
import torch
import re
import hashlib
from typing import List, Tuple


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_from_segmentation(seg: List[List[List[float]]], w: int, h: int) -> Tuple[int, int, int, int]:
    """Compute bounding box from polygon segmentation."""
    if not seg or not seg[0]:
        return 0, 0, w, h
    xs, ys = zip(*seg[0])
    l, t, r, b = min(xs), min(ys), max(xs), max(ys)
    l = int(clamp(l, 0, w))
    t = int(clamp(t, 0, h))
    r = int(clamp(r, 0, w))
    b = int(clamp(b, 0, h))
    if r <= l:
        r = min(w, l + 1)
    if b <= t:
        b = min(h, t + 1)
    return l, t, r, b

def arrange_square_grid(crops, pad=5, bg_color=(255, 255, 255), line_color=(0, 0, 0)):
    """Arrange crops into a nearly square grid with black margins between them."""
    if not crops:
        return None

    n = len(crops)
    grid_size = int(ceil(np.sqrt(n)))
    rows, cols = ceil(n / grid_size), grid_size

    max_h = max(c.height for c in crops)
    max_w = max(c.width for c in crops)

    # Pad crops to same size
    padded_crops = []
    for c in crops:
        padded = Image.new("RGB", (max_w, max_h), bg_color)
        padded.paste(c, (0, 0))
        padded_crops.append(padded)

    # Create grid canvas
    grid_h = rows * max_h + (rows - 1) * pad
    grid_w = cols * max_w + (cols - 1) * pad
    grid_img = Image.new("RGB", (grid_w, grid_h), bg_color)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= n:
                break
            y = r * (max_h + pad)
            x = c * (max_w + pad)
            grid_img.paste(padded_crops[idx], (x, y))

    if pad > 0:
        grid_cv = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2BGR)
        for c in range(1, cols):
            x = c * max_w + (c - 1) * pad
            grid_cv[:, x:x + pad] = line_color
        for r in range(1, rows):
            y = r * max_h + (r - 1) * pad
            grid_cv[y:y + pad, :] = line_color
        grid_img = Image.fromarray(cv2.cvtColor(grid_cv, cv2.COLOR_BGR2RGB))

    return grid_img


def ovis_chat_images(model, prompt, images, max_new_tokens=256, do_sample=False) -> str:
    with torch.no_grad():
        response, _, _ = model.chat(
            prompt=prompt,
            images=images if images else None,
            videos=None,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )
    return response.strip() if isinstance(response, str) else str(response)


def clean_pred(pred: str) -> str:
    """Clean ASCII predictions; keep others unchanged."""
    if all(ord(c) < 128 for c in pred):
        cleaned = re.sub(r'[^A-Za-z0-9]', '', pred)
        return cleaned, True
    else:
        return pred, False


def process_jsons(model, json_dir, frames_dir, out_dir, ovis_ckpt="AIDC-AI/Ovis2.5-9B", ovis_dtype="bfloat16"):
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "grids")
    os.makedirs(viz_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "pred_log.jsonl")
    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        with open(os.path.join(json_dir, json_file), "r", encoding="utf-8") as f:
            video_data = json.load(f)

        # Group annotations by trajectory ID
        traj = {}
        for frame_id, anns in video_data.items():
            for idx, ann in enumerate(anns):
                tid = ann.get("ID", f"__NOID__{frame_id}_{idx}")
                traj.setdefault(tid, []).append((frame_id, idx, ann))

        # Process each trajectory
        for tid, instances in traj.items():
            crops = []
            for frame_id, idx, ann in instances:
                frame_path = os.path.join(frames_dir, json_file.split(".")[0], f"{frame_id}.jpg")
                if not os.path.exists(frame_path):
                    print(f"Warning: Frame {frame_path} not found, skipping.")
                    continue
                try:
                    img = Image.open(frame_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Failed to load {frame_path}: {e}")
                    continue

                W, H = img.size
                x0, y0, x1, y1 = bbox_from_segmentation(ann["segmentation"], W, H)
                if x1 <= x0 or y1 <= y0:
                    continue
                # crop = img.crop((x0, y0, x1, y1))
                # Create mask for the full image
                mask = Image.new("L", (W, H), 0)
                poly = ann["segmentation"][0]
                poly_flat = [tuple(map(int, p)) for p in poly]
                ImageDraw.Draw(mask).polygon(poly_flat, outline=255, fill=255)

                # Apply mask
                masked_img = Image.fromarray(np.array(img) * (np.array(mask)[..., None] // 255))

                # Now crop normally
                crop = masked_img.crop((x0, y0, x1, y1))
                crops.append(crop)

            if not crops:
                continue

            # Build grid
            grid = arrange_square_grid(crops, pad=5)
            if grid is None:
                continue

            prompt = (
                f"<image>\nYou see {len(crops)} cropped text regions from a video. "
                f"Identify the most common word among all regions."
            )
            pred = ovis_chat_images(model, prompt, grid, do_sample=False, max_new_tokens=128)
            pred, update = clean_pred(pred)
            # breakpoint()
            # Save prediction to log file
            with open(log_path, "a", encoding="utf-8") as lf:
                log_entry = {
                    "video": json_file,
                    "trajectory_id": tid,
                    "prediction": pred,
                    "num_crops": len(crops)
                }
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Save grid image safely
            # safe_tid = re.sub(r'[^A-Za-z0-9_-]', '_', tid)
            safe_tid = re.sub(r'[^A-Za-z0-9_-]', '_', str(tid))

            safe_name = f"{json_file.split('.')[0]}_{safe_tid}.jpg"
            grid_path = os.path.join(viz_dir, safe_name)
            grid.save(grid_path)
            for frame_id_k, anns_k in video_data.items():
                for ann_k in anns_k:
                    # Use string equality for IDs, but also compare numeric forms if possible
                    ann_id = ann_k.get("ID")
                    if ann_id is None:
                        continue
                    if str(ann_id) == str(tid):
                        # breakpoint()
                        ann_k["transcription"] = pred
                        # updated_count += 1
        # Save updated JSON
        out_path = os.path.join(out_dir, json_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(video_data, f, indent=2, ensure_ascii=False)

        print(f"Processed {json_file} → {out_path}")

    print(f"✅ All results saved. Predictions logged at: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, help="Path to input json annotations")
    parser.add_argument("--frames_dir", required=True, help="Path to extracted frames")
    parser.add_argument("--out_dir", required=True, help="Where to save updated jsons")
    parser.add_argument("--ovis_ckpt", default="AIDC-AI/Ovis2.5-9B")
    parser.add_argument("--ovis_dtype", default="bfloat16")
    args = parser.parse_args()

    print("Loading Ovis model...")
    model = (
        Ovis.from_pretrained(
            args.ovis_ckpt,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda:0",
        ).eval()
    )
    print("Model loaded.")
    process_jsons(model, args.json_dir, args.frames_dir, args.out_dir, args.ovis_ckpt, args.ovis_dtype)
