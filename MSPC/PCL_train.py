import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

# =============== FPN 生成与叠加 ===============
def make_fpn_template(h, w, sigma_bu=5.0, sigma_br=5.0, sigma_bc=5.0, dtype=np.float32):
    bw = np.random.normal(0.0, sigma_bu, size=(h, w)).astype(dtype)
    br_line = np.random.normal(0.0, sigma_br, size=(h, 1)).astype(dtype)
    br = np.repeat(br_line, w, axis=1)
    bc_col = np.random.normal(0.0, sigma_bc, size=(1, w)).astype(dtype)
    bc = np.repeat(bc_col, h, axis=0)
    return bw + br + bc

def add_fpn_noise(img_uint8, sigma_bu=5.0, sigma_br=5.0, sigma_bc=5.0, global_scale=1.0):
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
        b = make_fpn_template(h, w, sigma_bu, sigma_br, sigma_bc)
        for ch in range(c):
            f[..., ch] += global_scale * b
        return np.clip(f, 0, 255).astype(np.uint8)

# =============== 渐进强度 ===============
def get_progressive_sigmas(epoch, max_epoch):
    r = epoch / max_epoch if max_epoch > 0 else 0.0
    if r < 0.3:
        s = np.random.uniform(0.0, 2.5)
    elif r < 0.7:
        s = np.random.uniform(2.5, 5.0)
    else:
        s = np.random.uniform(5.0, 7.5)
    return (s, s, s)

# =============== Robust 期 Trainer（带FPN增强） ===============
class FPNTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        imgs = batch["img"]  # (B,C,H,W) in [0,1]
        device, dtype = imgs.device, imgs.dtype
        epoch = getattr(self, "epoch", 0)
        max_epoch = getattr(self.args, "epochs", 300)
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                if np.random.rand() < 0.5:
                    img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    sigma_bu, sigma_br, sigma_bc = get_progressive_sigmas(epoch, max_epoch)
                    aug_np = add_fpn_noise(img_np, sigma_bu=sigma_bu, sigma_br=sigma_br, sigma_bc=sigma_bc, global_scale=1.0)
                    aug = torch.from_numpy(aug_np.astype(np.float32) / 255.).permute(2, 0, 1)
                    imgs[i] = aug.to(device=device, dtype=dtype).contiguous()
        batch["img"] = imgs
        return batch

# =============== Cooldown Trainer（关闭强增强） ===============
class CleanCooldownTrainer(DetectionTrainer):
    pass  # 直接用父类预处理（只剩最基础/弱增强）

# =============== (可选) Re-BN 重新校准 ===============
def rebn_calibrate(yolo_model: YOLO, clean_images_dir: str, imgsz: int = 640, max_images: int = 512):
    mdl = yolo_model.model
    mdl.train()  # 让BN以train模式更新running stats
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(clean_images_dir, e)))
    paths = paths[:max_images]
    if not paths:
        print("[ReBN] No images found, skip BN recalibration.")
        return
    device = next(mdl.parameters()).device
    with torch.no_grad():
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            scale = min(imgsz / h, imgsz / w)
            nh, nw = int(h * scale), int(w * scale)
            im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            top = (imgsz - nh) // 2
            left = (imgsz - nw) // 2
            canvas[top:top+nh, left:left+nw] = im_resized
            tensor = torch.from_numpy(canvas[:, :, ::-1].transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            _ = mdl(tensor)
    print("[ReBN] BatchNorm running stats refreshed with clean images.")

# =============== 训练：鲁棒 → 冷却微调 → （可选）Re-BN ===============
if __name__ == "__main__":
    cfg = "yolo11-mac2-p245.yaml"
    data_yaml = "/content/drive/MyDrive/yolov13-solar/dataset/data.yaml"
    imgsz = 640
    batch = 16
    device = "0"

    robust_epochs = 300
    cooldown_epochs = 15  # 10~20

    run_name_robust = "YOLO11-mac2-p245-progFPNv2"
    run_name_cooldn = "YOLO11-mac2-p245-progFPNv2-cooldn"

    # 阶段一：鲁棒训练（带渐进FPN）
    model = YOLO(cfg)
    model.train(
        data=data_yaml,
        epochs=robust_epochs,
        batch=batch,
        imgsz=imgsz,
        scale=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.1,
        device=device,
        name=run_name_robust,
        patience=30,
        trainer=FPNTrainer,   # ✅ 自定义FPN增强
    )

    robust_ckpt = f"runs/detect/{run_name_robust}/weights/best.pt"
    if not os.path.exists(robust_ckpt):
        robust_ckpt = f"runs/detect/{run_name_robust}/weights/last.pt"

    # 阶段二：Cooldown 微调（关闭强增强，LR 降 10×）
    model_cool = YOLO(robust_ckpt)
    model_cool.train(
        data=data_yaml,
        epochs=cooldown_epochs,
        batch=batch,
        imgsz=imgsz,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        perspective=0.0, scale=0.0, shear=0.0,
        flipud=0.0, fliplr=0.0,
        degrees=0.0, translate=0.0,
        lr0=1e-4, lrf=1e-4,   # ← 比你主训期小10×
        cos_lr=False,
        device=device,
        name=run_name_cooldn,
        patience=0,
        trainer=CleanCooldownTrainer,
    )

    # （可选）Re-BN 校准 + 评测
    clean_val_images = "/content/drive/MyDrive/yolov13-solar/dataset/valid/images"
    if os.path.isdir(clean_val_images):
        rebn_model = YOLO(f"runs/detect/{run_name_cooldn}/weights/best.pt")
        rebn_calibrate(rebn_model, clean_val_images, imgsz=imgsz, max_images=512)
        rebn_model.val(data=data_yaml, imgsz=imgsz, device=device, batch=batch, name=f"{run_name_cooldn}-rebn-eval")
    else:
        print("[ReBN] Skipped: clean validation image folder not found:", clean_val_images)

    print("✅ Robust training + Cooldown finetune done (Re-BN optional).")
