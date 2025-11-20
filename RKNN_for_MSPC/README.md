# YOLOv11-M (MANet) Deployment on RK3576

This repository contains the complete workflow for deploying **YOLOv11-M (MANet)** on the **RK3576** embedded platform, including ONNX export, RKNN conversion, mixed quantization, inference scripts, FPS benchmarking, and real-device testing.  
This README explains the purpose and origin of each file and provides instructions to reproduce the deployment process.

---

## 📁 Repository Contents

### 1️⃣ Inference & Performance Benchmarking Scripts

| File | Description |
|------|-------------|
| `batch_infer_fps.py` | Batch inference script (input size 640), used to measure FPS. |
| `batch_infer_fps_s320.py` | Batch inference for **320×320 MANet model**, the final deployment configuration. |
| `convert_fast.py` | Fast RKNN conversion script for YOLOv11-M (640 input). |
| `convert_fast_s320.py` | RKNN conversion + hybrid quantization for **320 MANet model (no DFL)**. |

---

## 2️⃣ Dataset Files for Quantization

| File | Description |
|------|-------------|
| `dataset_320.txt` | Index file for RKNN Toolkit2 hybrid quantization (320 resolution). |
| `dataset320_img.zip` | Small calibration dataset (≈100 sampled images) used during hybrid quantization. |

---

## 3️⃣ Model Files

### Original ONNX Models
| File | Description |
|------|-------------|
| `YOLOv11-M.onnx` | Original YOLOv11-M baseline (640×640). |
| `YOLOv11-M-modflv2.onnx` | MANet variant with DFL removed. |
| `YOLOv11-M-s320.onnx` | Final **320 input MANet ONNX model** used for RKNN conversion. |

### RKNN Models (Ready for RK3576 Deployment)
| File | Description |
|------|-------------|
| `YOLOv11-M.rknn` | RKNN model (640 input). |
| `YOLOv11-M-nodflv2_rk3576.rknn` | MANet + no DFL model for RK3576. |
| `YOLOv11-M-s320_rk3576.rknn` | **Final deployment model: MANet + no DFL + 320 input → 50.76 FPS** |

---

## 4️⃣ Real Device Test: FPS = 50.76

The repository includes a real photo:

**`real_test.jpg`**

This image was captured on the RK3576 board, showing:

- Real-time detection output  
- MANet (no DFL) model  
- Input resolution: **320×320**  
- **Measured FPS: 50.76**  

![Real Test](real_test.jpg)

---

## 🔧 How to Reproduce Deployment on RK3576

### Step 1 — Prepare Python Environment
```bash
conda create -n rknn_env python=3.12
pip install rknn-toolkit2==2.3.2
```

---

### Step 2 — Convert ONNX → RKNN
Run:
```bash
python3 convert_fast_s320.py
```

This script will:

- Load the MANet 320 ONNX  
- Apply RKNN hybrid quantization using `dataset_320.txt`  
- Export `YOLOv11-M-s320_rk3576.rknn`  

---

### Step 3 — Run Inference & Benchmark FPS
```bash
python3 batch_infer_fps_s320.py
```

Expected output:
```
FPS: 50.76
```

---

## 📝 Project Summary

- Successfully deployed **YOLOv11-M (MANet)** on RK3576  
- Removed DFL to fully support RKNN inference  
- Optimized input size to **320×320** for best performance  
- Final real-device performance: **50.76 FPS**  
- Complete workflow included ONNX export → RKNN conversion → quantization → on-device evaluation  

---

