# YOLOv11-M (MANet) on RK3576 – Deployment Files Overview

本仓库包含完整的 **YOLOv11‑M (MANet)** 在 **RK3576 开发板** 上的部署工作流程，包括模型转换、量化、推理脚本、FPS 测试脚本及测试图像示例。本 README 对每个文件的用途、来源以及如何在 RK3576 上复现这些结果进行说明。

---

## 📁 文件结构与用途说明

### 🔧 1. 推理与性能测试脚本
| 文件名 | 作用说明 |
|-------|----------|
| `batch_infer_fps.py` | 使用 **默认输入尺度（640）** 的批量推理脚本，用于统计 FPS。|
| `batch_infer_fps_s320.py` | 使用 **320 输入尺度** 的 MANet 版本批量推理脚本（对应最终部署配置）。|
| `convert_fast.py` | 将 YOLOv11‑M ONNX 模型转换为 RKNN 的快速转换脚本（640 输入）。|
| `convert_fast_s320.py` | 将 YOLOv11‑M MANet 模型转换为 RKNN（320 输入），并进行混合量化。|


### 📄 2. 数据集相关文件
| 文件名 | 说明 |
|-------|------|
| `dataset_320.txt` | RKNN Toolkit2 混合量化所需的数据集索引文件（320 尺寸）。|
| `dataset320_img.zip` | 实际用于量化的小数据集 zip 包。|

---

## 🧩 3. 模型文件说明

### 💠 原始 ONNX 模型
| 文件名 | 用途 |
|-------|------|
| `YOLOv11-M.onnx` | 官方 YOLOv11‑M 基线模型（640 输入）。|
| `YOLOv11-M-modflv2.onnx` | MANet 结构版本（未使用 DFL）。|
| `YOLOv11-M-s320.onnx` | 最终用于 RK3576 部署的 **320 输入、移除 DFL 的 MANet 模型**。|

### 🚀 RKNN 部署模型（可直接在 RK3576 上运行）
| 文件名 | 说明 |
|-------|------|
| `YOLOv11-M.rknn` | 640 输入的 RKNN 模型。|
| `YOLOv11-M-nodflv2_rk3576.rknn` | MANet + 移除 DFL 的 RKNN 模型。|
| `YOLOv11-M-s320_rk3576.rknn` | **最终部署版本：MANet + 移除 DFL + 输入 320 → FPS = 50.76** |

---

## 📷 4. 实测结果（real_test.jpg）

下图（仓库中的 `real_test.jpg`）来自 RK3576 实机拍摄，展示了：

- 开发板屏幕输出
- 部署的 YOLOv11‑M MANet (no DFL) 模型
- 输入分辨率：**320 × 320**
- **实测 FPS = 50.76**

![real test](real_test.jpg)

---

## 🔨 如何复现 RK3576 上的部署流程

### 1️⃣ 准备 Python 环境
```bash
conda create -n rknn_env python=3.12
pip install rknn-toolkit2==2.3.2
```

### 2️⃣ 量化并生成 RKNN 模型
```bash
python3 convert_fast_s320.py
```

此脚本将：
- 加载 MANet 结构 ONNX
- 使用 dataset_320.txt 进行混合量化
- 导出 `YOLOv11-M-s320_rk3576.rknn`

---

### 3️⃣ 运行推理并测试 FPS
```bash
python3 batch_infer_fps_s320.py
```

输出示例：
```
FPS: 50.76
```

---

## ⭐ 项目总结（可用于论文 / 报告）
- 成功将 YOLOv11‑M (MANet) 结构移植到 RK3576  
- 移除 DFL（距离分布回归层）以增强 RKNN 兼容性  
- 输入尺度降至 320×320，框头保持默认结构  
- **最终推理速度：50.76 FPS（实机）**  
- 全流程包含 ONNX 导出 → RKNN 转换 → 混合量化 → 实机部署  

---

