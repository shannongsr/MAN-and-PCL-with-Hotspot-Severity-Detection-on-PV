import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import yaml
import os
import random

# =========================
# 路径设置（按需修改）
# =========================
val_img_dir = Path("/content/drive/MyDrive/yolov13-solar/dataset/test/images")
val_lbl_dir = Path("/content/drive/MyDrive/yolov13-solar/dataset/test/labels")
output_base_dir = Path("/content/drive/MyDrive/yolov13-solar/dataset/fpn_augmented")

# =========================
# 数据集与类别（按需修改）
# =========================
NAMES = ['Serious hot spot', 'Slight hot spot', 'dirt']  # 你的类别名称
IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")

# =========================
# 固定图样噪声（FPN）等级设置
# 说明：
#   σ_bu: 空间独立像素项（white/uncorrelated）
#   σ_br: 行常数项
#   σ_bc: 列常数项
# 论文实验示例常用 σ_bc = σ_bu = 5（可作为中档基准）；这里提供多档可调
# =========================
FPN_LEVELS = {
    # tag: (sigma_bu, sigma_br, sigma_bc)
    "fpn_s3":  (3.0, 3.0, 3.0),   
    "fpn_s5":  (5.0, 5.0, 5.0),    # 中度（论文常用数量级）
    "fpn_s7":  (7.0, 7.0, 7.0),    # 偏重
    "fpn_s10":  (10.0, 10.0, 10.0),    # 偏重
    "fpn_s12": (12.0, 12.0, 12.0), # 重度
}

# =========================
# 可选：叠加时的整体强度系数（保持1即可）
# =========================
GLOBAL_SCALE = 1.0

# =========================
# 随机种子（为了可复现）
# =========================
SEED = 2025
np.random.seed(SEED)
random.seed(SEED)

def ensure_uint8(arr):
    """裁剪并转换为 uint8"""
    return np.clip(arr, 0, 255).astype(np.uint8)

def to_float_gray_or_rgb(img):
    """
    返回 (img_float, channels, H, W, is_color)
    - 若为单通道，保持灰度
    - 若为三通道，保持 BGR，但以 float32 处理
    """
    if img is None:
        raise ValueError("Failed to read image.")
    if img.ndim == 2:
        h, w = img.shape
        return img.astype(np.float32), 1, h, w, False
    elif img.ndim == 3:
        h, w, c = img.shape
        if c == 1:
            return img[..., 0].astype(np.float32), 1, h, w, False
        else:
            return img.astype(np.float32), c, h, w, True
    else:
        raise ValueError("Unsupported image shape: {}".format(img.shape))

def make_fpn_template(h, w, sigma_bu=5.0, sigma_br=5.0, sigma_bc=5.0, dtype=np.float32):
    """
    生成论文模型的 FPN 模板：
    b(i,j) = b_w(i,j) + b_r(j) + b_c(i)
    - b_w ~ N(0, σ_bu), 独立像素噪声
    - b_r 行常数 ~ N(0, σ_br)
    - b_c 列常数 ~ N(0, σ_bc)
    返回形状 (H, W) 的 float32 噪声矩阵
    """
    # 像素独立项
    bw = np.random.normal(loc=0.0, scale=sigma_bu, size=(h, w)).astype(dtype)

    # 行常数项（对每一行采样一个值，广播到该行）
    br_line = np.random.normal(loc=0.0, scale=sigma_br, size=(h, 1)).astype(dtype)
    br = np.repeat(br_line, w, axis=1)

    # 列常数项（对每一列采样一个值，广播到该列）
    bc_col = np.random.normal(loc=0.0, scale=sigma_bc, size=(1, w)).astype(dtype)
    bc = np.repeat(bc_col, h, axis=0)

    b = bw + br + bc
    return b

def add_fpn_noise(img_bgr_or_gray, sigma_bu, sigma_br, sigma_bc, global_scale=1.0):
    """
    将 FPN 噪声叠加到图像：
    - 对灰度：直接叠加
    - 对彩色：生成一张 FPN（同一传感器假设），对每个通道统一叠加
    """
    img, c, h, w, is_color = to_float_gray_or_rgb(img_bgr_or_gray)

    # 生成一次 FPN 模板（单传感器假设）
    b = make_fpn_template(h, w, sigma_bu=sigma_bu, sigma_br=sigma_br, sigma_bc=sigma_bc)

    if is_color and c >= 3:
        # 对每个通道叠加同一张 FPN（BGR）
        noisy = img.copy()
        for ch in range(c):
            noisy[..., ch] = img[..., ch] + global_scale * b
    else:
        # 单通道
        noisy = img + global_scale * b

    return ensure_uint8(noisy)

def write_yaml(yaml_path: Path, dataset_root: Path, val_rel, test_rel, names):
    data = {
        'path': str(dataset_root),
        'train': '',              # 这里留空（你主要用于val/test评估）
        'val': val_rel,
        'test': test_rel,
        'nc': len(names),
        'names': names
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

def main():
    print("🔄 Generating Fixed-Pattern-Noise (FPN) datasets ...")

    # 创建各噪声等级的数据子集
    for tag, (s_bu, s_br, s_bc) in FPN_LEVELS.items():
        img_out_dir = output_base_dir / tag / "images"
        lbl_out_dir = output_base_dir / tag / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        # 遍历图片
        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(sorted(val_img_dir.glob(f"*{ext}")))

        for img_path in tqdm(img_paths, desc=f"{tag}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            noisy = add_fpn_noise(
                img,
                sigma_bu=s_bu,
                sigma_br=s_br,
                sigma_bc=s_bc,
                global_scale=GLOBAL_SCALE
            )
            cv2.imwrite(str(img_out_dir / img_path.name), noisy)

            # 拷贝标签
            lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out_dir / lbl_path.name)

        # 为该子集写单独的 yaml
        yaml_path = output_base_dir / tag / f"{tag}.yaml"
        write_yaml(
            yaml_path=yaml_path,
            dataset_root=output_base_dir,
            val_rel=f"{tag}/images",
            test_rel=f"{tag}/images",
            names=NAMES
        )

    # 生成“总”yaml，包含所有子集（便于一次性评估多强度FPN）
    all_yaml_path = output_base_dir / "fpn_all.yaml"
    write_yaml(
        yaml_path=all_yaml_path,
        dataset_root=output_base_dir,
        val_rel=[f"{tag}/images" for tag in FPN_LEVELS.keys()],
        test_rel=[f"{tag}/images" for tag in FPN_LEVELS.keys()],
        names=NAMES
    )

    print("✅ FPN 噪声数据集与配置文件生成完成！")
    print(f"📂 根目录：{output_base_dir}")
    print(f"📄 汇总配置：{all_yaml_path}")

if __name__ == "__main__":
    os.makedirs(output_base_dir, exist_ok=True)
    main()
