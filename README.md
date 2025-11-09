# 🌞 Severity-Aware Thermal Hotspot Detection (YOLOv11-M)

Official implementation of the paper:  
**“Mixed Aggregation Networks and Progressive Cooling Training for Thermal Hotspot Severity Detection on Photovoltaic Panels”**

This repository provides reproducible scripts for **training**, **noise robustness evaluation**, and **post-processing optimization** of the proposed YOLOv11-M detector under fixed-pattern noise (FPN).  
It includes implementations of **progressive curriculum training**, **simulated annealing (SA) post-processing tuners**, and **GPU power–aware benchmarking**.

---

## 📦 Repository Structure

```
.MSPC/
├── eval_fps_fp32.py              # FP32 evaluation with GPU power sampling (NVML)
├── eval_nms.py                   # Unified evaluator with NMS / Soft-NMS / WBF options
├── ir_noise.py                   # Fixed-pattern noise (FPN) dataset augmentation tool
├── nonpregre_train.py            # Baseline YOLO training script (no progressive regime)
├── PCL_train.py                  # Progressive Curriculum Learning (PCL) training
├── progressive_train.py          # Progressive FPN + cooldown fine-tuning pipeline
├── sa_nms_tuner.py               # Simulated Annealing tuner for hard NMS parameters
├── sa_softnms_linear_tuner.py    # Simulated Annealing tuner for Soft-NMS (Linear)
├── sa_wbf_tuner.py               # Simulated Annealing tuner for WBF fusion
└── README.md                     # (this file)
```

---

## ⚙️ Environment

| Dependency | Version (recommended) |
|-------------|----------------------|
| Python      | ≥ 3.10               |
| PyTorch     | ≥ 2.1                |
| Ultralytics | 8.3+ (YOLOv8/11)    |
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

### 1️⃣ FPN Noise Augmentation — `ir_noise.py`
Adds controllable **Fixed Pattern Noise (FPN)** to images to simulate sensor artifacts during UAV thermal inspection.

```bash
python ir_noise.py
```

- Configurable noise strength levels: `fpn_s3`, `fpn_s5`, `fpn_s7`, etc.  
- Each level defines `(σ_bu, σ_br, σ_bc)` for pixel/row/column variance.  
- Outputs augmented images and updated dataset YAML.

---

### 2️⃣ Training Scripts

#### 🔹 Baseline (no progressive regime)
```bash
python nonpregre_train.py
```

#### 🔹 Progressive Curriculum Learning
Introduces **progressive noise intensity** as epochs increase.

```bash
python PCL_train.py
```

#### 🔹 Progressive + Cooldown Training
Two-phase robust training:
1. Stage 1 — Progressive FPN noise augmentation.  
2. Stage 2 — Clean fine-tuning (cooldown phase).  
3. Optional — Re-BN recalibration on clean data.

```bash
python progressive_train.py
```

---

### 3️⃣ Post-processing Evaluation — `eval_nms.py`
Unified interface to compare **NMS**, **Soft-NMS**, and **WBF**.

```bash
# Hard NMS
python eval_nms.py --postproc none

# Soft-NMS (Gaussian)
python eval_nms.py --postproc softnms --snms-method gaussian --snms-sigma 0.5

# Soft-NMS (Linear)
python eval_nms.py --postproc softnms --snms-method linear --snms-Nt 0.5

# Weighted Boxes Fusion (WBF)
python eval_nms.py --postproc wbf --wbf-iou 0.55
```

---

### 4️⃣ Simulated Annealing Tuners

Automatically optimizes post-processing parameters (e.g., `conf`, `IoU`, `max_det`) to maximize **mAP50–95**.

| Script | Tuned Method | Parameters |
|---------|---------------|------------|
| `sa_nms_tuner.py` | Hard NMS | `(conf, iou, max_det)` |
| `sa_softnms_linear_tuner.py` | Soft-NMS (Linear) | `(conf, Nt, max_det)` |
| `sa_wbf_tuner.py` | WBF | `(conf, wbf_iou, max_det)` |

Example:

```bash
python sa_wbf_tuner.py   --weights runs/detect/YOLO11-M/weights/best.pt   --data dataset/test.yaml   --imgsz 640 --batch 4   --metric map5095 --iters 60
```

Each tuner outputs:
- A detailed CSV log of all iterations.
- A YAML file with the best parameter set.

---

### 5️⃣ FP32 Evaluation + Power Profiling — `eval_fps_fp32.py`
Performs full **FP32 validation** with real-time GPU **power sampling** using NVIDIA NVML.

```bash
python eval_fps_fp32.py   --weights runs/detect/YOLO11-M/weights/best.pt   --data dataset/data.yaml   --imgsz 640 --device cuda   --split test --batch 1   --repeats 3
```

Outputs:
- Average FPS, GFLOPs, latency (ms/img)  
- Power metrics: average W, energy (J/img), FPS/W  

---

## 🧠 Experimental Workflow Summary

| Step | Script | Purpose |
|------|---------|----------|
| 1 | `ir_noise.py` | Generate synthetic FPN noise datasets |
| 2 | `PCL_train.py` / `progressive_train.py` | Train detector with progressive robustness |
| 3 | `eval_nms.py` | Compare post-processing methods |
| 4 | `sa_*.py` | Auto-tune post-processing hyperparameters |
| 5 | `eval_fps_fp32.py` | Benchmark energy–accuracy trade-offs |

---

## 📊 Example Results
| Model | mAP50 | mAP50–95 | GFLOPs | Power (W) | FPS/W |
|--------|--------|-----------|---------|-------------|--------|
| YOLOv11 | 75.7 | 49.0 | 6.3 | 32.2 | 1.59 |
| **YOLOv11-M (ours)** | **81.1** | **50.3** | **8.3** | **32.4** | **1.72** |

---


## 📩 Contact
For questions or collaborations:  
**Email:** shirong.guo@monash.edu 
**Institution:** Monash University
