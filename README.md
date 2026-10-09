
# Surveillance Camera Vehicle Detection with YOLO

An optimized vehicle detection system designed for surveillance camera streams (CCTV). This pipeline addresses specific challenges in security camera footage, including high-angle oblique views, small/distant targets, heavy occlusion due to high traffic density, and extreme lighting variations between day and night.

[![Video Demo]](video_demo.mp4)
---

## 1. Key Features

* **Surveillance View Optimization:** Trained on high-angle camera datasets to reduce geometric/perspective distortion errors.
* **Multi-Class Vehicle Detection:** Supports detailed classification of vehicle types (e.g., `car`, `motorcycle`, `bus`, `truck`, `bicycle`).
* **Streamlined Pipeline:** Directly integrated with the Ultralytics framework, supporting batch inference on video streams/RTSP feeds and model export to ONNX/TensorRT for edge deployment.

---

## 2. Project Structure

```text
Surveillance-Camera-Vehicle-Detection/
├── Training/
│   ├── train_yolo.ipynb    # Training pipeline execution notebook/script
│   └── kaggle/             # Directory containing datasets, models, and training logs
├── .gitignore
└── README.md

```

* **Kaggle Notebook:** [Top View Camera Vehicle Detection](https://www.kaggle.com/code/trngqucnam/top-view-camera-vehicle-detection)

---

## 3. Dataset & Augmentation Setup

### Dataset Structure

The dataset is formatted according to standard YOLO annotations:

* **Train Set:** 3,525 images (70%)
* **Validation Set:** 1,007 images (20%)
* **Test Set:** 503 images (10%)
* **Source:** [Surveillance Cameras Dataset on Kaggle](https://www.kaggle.com/datasets/miladbayat/surveillance-cameras)

### Data Augmentation Strategy

To simulate real-world outdoor conditions, the following data augmentations were applied:

* `mosaic: 0.8`: Combines 4 random images to help the model learn context and detect small, background vehicles.
* `scale: 0.5`: Applies object scaling (zoom-in / zoom-out).
* `fliplr: 0.5`: Flips images horizontally while preserving camera perspective rules.
* `close_mosaic: 10`: Disables Mosaic during the final 10 epochs to stabilize loss on original ground-truth distributions.
* `degrees: 3.0`: Applies small rotations to handle slightly tilted camera angles.

---

## 4. Training & Benchmark Results

### Experimental Setup

* **Hardware:** Kaggle Notebook (2x NVIDIA Tesla T4 16GB VRAM)
* **Framework:** `ultralytics`

### Core Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| Model Architecture | `YOLO26S` |
| Input Resolution (`imgsz`) | `640` |
| Epochs | `160` |
| Batch Size | `64` |
| Optimizer | `auto` |

### Evaluation Results (Validation Set)

| Model | Image Size | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- | --- | --- |
| `best.pt` | `640` | `0.9532` | `0.9559` | `0.9814` | `0.8537` |

---

## 5. Quickstart & Inference

### 1. Environment Setup

```bash
git clone https://github.com/Trqc-Nam/Surveillance-Camera-Vehicle-Detection.git
cd Surveillance-Camera-Vehicle-Detection

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install ultralytics

```

### 2. Model Training & Testing

Run `Training/train_yolo.ipynb` to execute the training and evaluation pipeline.

---

5. **Xuất mô hình ra thiết bị biên:** *Model export for edge deployment*.
6. **Môi trường thực nghiệm:** *Experimental Setup / Benchmark Setup*.
7. **Phân phối loss gốc:** *Ground-truth loss distributions*.
