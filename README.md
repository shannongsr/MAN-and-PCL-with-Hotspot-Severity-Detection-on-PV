# 🌞 Severity-Aware Thermal Hotspot Detection (YOLOv11-MSPC)

This repository contains the official implementation of our ongoing research project on **“Mixed Aggregation Networks and Progressive Cooling Training for severity-aware thermal hotspot detection on photovoltaic panels”**.
The work is currently under development and the manuscript is in preparation.

This repository provides reproducible scripts for **training**, **noise robustness evaluation**, and **post-processing optimization** of the proposed YOLOv11-M detector under fixed-pattern noise (FPN).  
It includes implementations of **progressive curriculum training**, **progressive+cooldown training**, **simulated annealing (SA) post-processing tuners**, **GPU power–aware benchmarking**, and **embedded deployment on RK3576** using the YOLOv11‑M (MANet) architecture.

---

## 📦 Repository Structure

```
.
.
├── MSPC/
│   ├── eval_fps_fp32.py              # FP32 evaluation with GPU power sampling (NVML)
│   ├── eval_nms.py                   # Unified evaluator with NMS / Soft-NMS / WBF options
│   ├── ir_noise.py                   # Fixed-pattern noise (FPN) dataset augmentation tool
│   ├── nonpregre_train.py            # Baseline YOLO training script (no progressive regime)
│   ├── PCL_train.py                  # Progressive + Cooldown Training (PCL = progressive + cooldown)
│   ├── progressive_train.py          # Progressive Curriculum Learning (progressive-only)
│   ├── sa_nms_tuner.py               # Simulated Annealing tuner for hard NMS parameters
│   ├── sa_softnms_linear_tuner.py    # Simulated Annealing tuner for Soft-NMS (Linear)
│   └── sa_wbf_tuner.py               # Simulated Annealing tuner for WBF fusion
│
├── RKNN_for_MSPC/                # NEW: RK3576 deployment folder
│   ├── README.md                 # Full deployment instructions
│   ├── YOLOv11-M-s320_rk3576.rknn
│   ├── convert_fast_s320.py
│   ├── batch_infer_fps_s320.py
│   ├── dataset_320.txt
│   ├── dataset320_img.zip
│   └── real_test.jpg
│
└── README.md
```

---

## 🆕 RKNN Deployment (RK3576)

A new module **`RKNN_for_MSPC/`** has been added to provide:

- **YOLOv11‑M MANet (no DFL) RKNN models**
- **320×320 optimized model achieving 50.76 FPS on RK3576**
- **ONNX→RKNN conversion scripts**
- **Hybrid quantization dataset + calibration files**
- **Batch inference FPS benchmarking scripts**
- **Real device demonstration photo**

Real‑device performance (from `real_test.jpg` in the folder):

**➡️ YOLOv11‑M MANet (no DFL), 320×320 input, 50.76 FPS on RK3576**

![RKNN Test](RKNN_for_MSPC/real_test.jpg)

Full deployment details (conversion, quantization, inference) are documented in the sub‑README:

👉 **https://github.com/shannongsr/MAN-and-PCL-with-Hotspot-Severity-Detection-on-PV/tree/main/RKNN_for_MSPC**

---

## ⚙️ Environment

| Dependency | Version (recommended) |
|-------------|----------------------|
| Python      | ≥ 3.10               |
| PyTorch     | ≥ 2.1                |
| Ultralytics | 8.3+ (YOLOv8/11)     |
| NumPy       | ≥ 1.24               |
| OpenCV      | ≥ 4.8                |
| PyYAML      | ≥ 6.0                |
| tqdm        | ≥ 4.65               |
| pynvml      | ≥ 11.5 (for power eval) |

```bash
pip install torch ultralytics opencv-python numpy pyyaml tqdm pynvml
```

---

## 🚀 Key Modules and Usage

### 1️⃣ FPN Noise Augmentation — `MSPC/ir_noise.py`
Adds controllable **Fixed Pattern Noise (FPN)** to images to simulate sensor artifacts during UAV thermal inspection.

```bash
python MSPC/ir_noise.py
```
- Configurable noise strength levels: `fpn_s3`, `fpn_s5`, `fpn_s7`, etc.  
- Each level defines `(σ_bu, σ_br, σ_bc)` for pixel/row/column variance.  
- Outputs augmented images and an updated dataset YAML when configured.

---

### 2️⃣ Training Scripts

#### 🔹 Baseline (no progressive regime)
```bash
python MSPC/nonpregre_train.py
```

#### 🔹 **Progressive Curriculum Learning (PCL-only stage)** — *progressive-only*
Introduces **progressive noise intensity** as epochs increase.
```bash
python MSPC/progressive_train.py
```

#### 🔹 **Progressive + Cooldown Training (PCL = progressive → cooldown)** ✅
Two-phase robust training:
1. Stage 1 — Progressive FPN noise augmentation.  
2. Stage 2 — Clean fine-tuning (cooldown phase).  
3. Optional — Re-BN recalibration on clean data.

```bash
python MSPC/PCL_train.py
```

---

### 3️⃣ Post-processing Evaluation — `MSPC/eval_nms.py`
Unified interface to compare **NMS**, **Soft-NMS**, and **WBF**.

```bash
# Hard NMS
python MSPC/eval_nms.py --postproc none

# Soft-NMS (Gaussian)
python MSPC/eval_nms.py --postproc softnms --snms-method gaussian --snms-sigma 0.5

# Soft-NMS (Linear)
python MSPC/eval_nms.py --postproc softnms --snms-method linear --snms-Nt 0.5

# Weighted Boxes Fusion (WBF)
python MSPC/eval_nms.py --postproc wbf --wbf-iou 0.55
```

---

### 4️⃣ Simulated Annealing Tuners (Auto HPO for Post-processing)

Automatically optimizes post-processing parameters (e.g., `conf`, `IoU`, `max_det`) to maximize **mAP50–95**.

| Script | Tuned Method | Parameters |
|---------|---------------|------------|
| `MSPC/sa_nms_tuner.py` | Hard NMS | `(conf, iou, max_det)` |
| `MSPC/sa_softnms_linear_tuner.py` | Soft-NMS (Linear) | `(conf, Nt, max_det)` |
| `MSPC/sa_wbf_tuner.py` | WBF | `(conf, wbf_iou, max_det)` |

Example:

```bash
python MSPC/sa_wbf_tuner.py   --weights runs/detect/YOLO11-M/weights/best.pt   --data dataset/test.yaml   --imgsz 640 --batch 4   --metric map5095 --iters 60
```
Each tuner outputs:
- A detailed CSV log of all iterations.
- A YAML file with the best parameter set.

---

### 5️⃣ FP32 Evaluation + Power Profiling — `MSPC/eval_fps_fp32.py`
Performs full **FP32 validation** with real-time GPU **power sampling** using NVIDIA NVML.

```bash
python MSPC/eval_fps_fp32.py   --weights runs/detect/YOLO11-M/weights/best.pt   --data dataset/data.yaml   --imgsz 640 --device cuda   --split test --batch 1   --repeats 3
```
Outputs:
- Average FPS, GFLOPs, latency (ms/img)  
- Power metrics: average W, energy (J/img), FPS/W  

---

## 🧠 Experimental Workflow Summary

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `MSPC/ir_noise.py` | Generate synthetic FPN noise datasets |
| 2 | `MSPC/progressive_train.py` or `MSPC/PCL_train.py` | Train for robustness (progressive-only / progressive→cooldown) |
| 3 | `MSPC/eval_nms.py` | Compare post-processing methods |
| 4 | `MSPC/sa_*.py` | Auto-tune post-processing hyperparameters |
| 5 | `MSPC/eval_fps_fp32.py` | Benchmark energy–accuracy trade-offs |

---

## 📊 Example Results (waiting for perfection)
| Model | mAP50 | mAP50–95 | GFLOPs | Power (W) | FPS/W |
|--------|------:|---------:|-------:|----------:|------:|
| YOLOv11 | 79.7 | 49.0 | 6.3 | 32.2 | 1.59 |
| **YOLOv11-M (ours)** | **81.1** | **50.3** | **8.3** | **32.4** | **1.72** |


---

## 📩 Contact
For questions or collaborations:  
📧 **shirong.guo@monash.edu**  
📧 **shannongsr@yeah.net**  
🏫 **Monash University**
