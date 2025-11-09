import numpy as np
import cv2
import torch
from ultralytics import YOLO
# ✅ use the task-specific trainer
from ultralytics.models.yolo.detect.train import DetectionTrainer

# ================================
# 1) FPN 生成与叠加
#    b(i,j) = b_w(i,j) + b_r(j) + b_c(i)
# ================================
def make_fpn_template(h, w, sigma_bu=5.0, sigma_br=5.0, sigma_bc=5.0, dtype=np.float32):
    # 像素独立项（white/uncorrelated）
    bw = np.random.normal(0.0, sigma_bu, size=(h, w)).astype(dtype)
    # 行常数项
    br_line = np.random.normal(0.0, sigma_br, size=(h, 1)).astype(dtype)
    br = np.repeat(br_line, w, axis=1)
    # 列常数项
    bc_col = np.random.normal(0.0, sigma_bc, size=(1, w)).astype(dtype)
    bc = np.repeat(bc_col, h, axis=0)
    return bw + br + bc

def add_fpn_noise(img_uint8, sigma_bu=5.0, sigma_br=5.0, sigma_bc=5.0, global_scale=1.0):
    """
    对灰度/彩色图像添加FPN。
    - 对彩色：使用同一张FPN模板叠加至各通道（单传感器假设）
    """
    assert img_uint8.dtype == np.uint8
    if img_uint8.ndim == 2:
        h, w = img_uint8.shape
        f = img_uint8.astype(np.float32)
        b = make_fpn_template(h, w, sigma_bu, sigma_br, sigma_bc)
        noisy = f + global_scale * b
        return np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        h, w, c = img_uint8.shape
        f = img_uint8.astype(np.float32)
        b = make_fpn_template(h, w, sigma_bu, sigma_br, sigma_bc)  # 单传感器：同一模板
        for ch in range(c):
            f[..., ch] += global_scale * b
        return np.clip(f, 0, 255).astype(np.uint8)

# ================================
# 2) progressive sigma（随epoch增强）
#    轻(3)->中(5)->重(8/12)
# ================================
def get_progressive_sigmas(epoch, max_epoch):
    r = epoch / max_epoch if max_epoch > 0 else 0.0
    if r < 0.3:
        # 轻度：S=3
        s = np.random.uniform(0, 7.5)
    elif r < 0.7:
        # 中度：S=5
        s = np.random.uniform(0, 7.5)
    else:
        # 重度：S=8~12 之间采样
        s = np.random.uniform(0, 7.5)
    # 行/列/像素独立项使用同一数量级（与论文合成设置一致）
    return (s, s, s)

# ================================
# 3) custom trainer for detect
# ================================
class FPNTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        imgs = batch["img"]  # (B,C,H,W) in [0,1]
        device, dtype = imgs.device, imgs.dtype
        epoch = getattr(self, "epoch", 0)
        max_epoch = getattr(self.args, "epochs", 300)

        with torch.no_grad():
            for i in range(imgs.shape[0]):
                if np.random.rand() < 0.5:  # 50%概率应用FPN
                    img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    sigma_bu, sigma_br, sigma_bc = get_progressive_sigmas(epoch, max_epoch)
                    aug_np = add_fpn_noise(img_np, sigma_bu=sigma_bu, sigma_br=sigma_br, sigma_bc=sigma_bc, global_scale=1.0)
                    aug = torch.from_numpy(aug_np.astype(np.float32) / 255.).permute(2, 0, 1)
                    imgs[i] = aug.to(device=device, dtype=dtype).contiguous()
        batch["img"] = imgs
        return batch

# ================================
# 4) train
# ================================
if __name__ == "__main__":
    model = YOLO("yolo11-mac2-p245.yaml")  # cfg path
    model.train(
        data="/content/drive/MyDrive/yolov13-solar/dataset/data.yaml",
        epochs=300,
        batch=16,
        imgsz=640,
        scale=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.1,
        device="0",
        name="YOLO11-mac2-p245-noprogFPN",
        patience=30,
        trainer=FPNTrainer,  # ✅ 用FPN增强的task-aware trainer
    )
