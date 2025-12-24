# TraRA: Trajectory-level Recognition Aggregation for Video Text Spotting

📄 **Official implementation of _TraRA: Trajectory-level Recognition Aggregation for Video Text Spotting_**  

---

## 🔍 Overview

**TraRA** is a plug-and-play framework for **Video Text Spotting (VTS)** that enhances recognition accuracy by aggregating information at the **trajectory level**, rather than relying on noisy frame-wise predictions.

By combining **Temporal Clustering** and **Vision–Language Aggregation**, TraRA significantly improves robustness against motion blur, occlusion, and incomplete text observations.

---

## ✨ Key Contributions

- **Trajectory-level recognition** instead of frame-level voting  
- **Temporal Clustering (TC)** for stable text trajectories  
- **Vision–Language Aggregation (VLA)** using a LoRA-adapted VLM  
- Compatible with existing VTS systems such as **GoMatching++** and **TransDETR**

---

## 🧠 Method Overview

### Temporal Clustering (TC)
Groups frame-level detections into consistent trajectories using:
- HOG + SIFT + geometric features
- Adaptive distance thresholds
- Time-aware clustering

### Vision–Language Aggregation (VLA)
A LoRA-adapted Vision–Language Model aggregates all cropped text instances in a trajectory and predicts a single robust word.

---


## ⚙️ Installation

TraRA is designed to work **on top of existing VTS frameworks**.

### Step 1: Install Base Frameworks

Follow the official instructions of:

- **GoMatching++**  
  https://github.com/Hxyz-123/GoMatching

- **TransDETR**  
  https://github.com/weijiawu/TransDETR

- **Ovis (Vision–Language Model)**  
  https://github.com/AIDC-AI/Ovis

> TraRA does not modify these frameworks and uses their outputs directly.

---

### Step 2: Clone TraRA

```bash
git clone https://github.com/trid2912/TraRA.git
cd TraRA
```

## 🚀 Usage

This repository provides a three-stage pipeline for trajectory refinement and text aggregation in video-based text spotting.

---

## 1️⃣ Temporal Clustering

This stage refines noisy tracking results by splitting inconsistent trajectories using visual appearance cues.

### Description
- Extracts masked crops for each trajectory
- Computes RootSIFT, HOG, and area-based features
- Applies adaptive clustering to split ID switches
- Outputs cleaned, re-identified trajectories

```bash
python cluster_json.py \
  --json_dir path/to/raw_jsons \
  --frames_dir path/to/video_frames \
  --output_json_dir path/to/clustered_jsons
```

## 2️⃣ Vision–Language Aggregation

This step groups frame-level detections into trajectories and predicts a unified transcription using a vision–language model.

### Description
- Loads per-frame detection results (JSON format)
- Groups detections by trajectory ID
- Crops text regions and arranges them into a grid
- Uses a VLM (e.g., Ovis-2.5) to infer the recognition for each trajectory

```bash
python full_word_vlm.py \
  --json_dir path/to/vts_outputs \
  --frames_dir path/to/video_frames \
  --out_dir path/to/output_dir
```
## 3️⃣ Dataset Construction from XML Annotations

This stage converts word-level XML annotations into grid-based training samples for vision–language models.

### Description
- Parses XML word annotations
- Crops and groups text regions into fixed-size grids
- Assigns the dominant word label per grid
- Produces images and a unified JSON annotation file

```bash
python build_dataset_from_xml.py \
  --frames_root path/to/frames \
  --xml_root path/to/xml_annotations \
  --output_root path/to/output_dataset \
  --batch_size number_of_crops
```

## 🧠 Notes

- Frame images should follow a consistent naming format (e.g., `000001.jpg`).
- Designed for integration with **Ovis-style VLMs**.
- Recommended GPU memory: ≥ 24 GB.

