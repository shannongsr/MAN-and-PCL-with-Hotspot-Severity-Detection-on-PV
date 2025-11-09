#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a pre-split subset with selectable post-processing:
- Original NMS (none)
- Soft-NMS (gaussian/linear)
- WBF

This version patches ultralytics.utils.ops.non_max_suppression so that
custom post-processing is guaranteed to be applied during validation.

Usage examples:

# 1) 原版 NMS
python eval_nms.py \
  --weights runs/detect/xxx/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --project runs/test_eval --name part_001 \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --postproc none

# 2) Soft-NMS（Gaussian）
python eval_nms.py \
  --weights runs/detect/xxx/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --project runs/test_eval --name part_001_snms \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --postproc softnms --snms-method gaussian --snms-sigma 0.5 \
  --max-det 300

# 3) Soft-NMS（Linear）
python eval_nms.py \
  --weights runs/detect/xxx/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --project runs/test_eval --name part_001_snms_lin \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --postproc softnms --snms-method linear --snms-Nt 0.5 \
  --max-det 300

# 4) WBF
python eval_nms.py \
  --weights runs/detect/xxx/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --project runs/test_eval --name part_001_wbf \
  --imgsz 640 --batch 4 --conf 0.001 --iou 0.6 \
  --postproc wbf --wbf-iou 0.55 --max-det 300
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple

import yaml
import numpy as np
import torch
from ultralytics import YOLO


# -------------------------- 基础工具 --------------------------

def _resolve(p: Path) -> Path:
    return Path(p).expanduser().resolve()

def _write_yaml(obj: dict, p: Path):
    p = _resolve(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


# -------------------------- 算法实现 --------------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [M,4], b: [N,4], xyxy"""
    M, N = a.shape[0], b.shape[0]
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, 1e-9, None)).astype(np.float32)

def soft_nms_per_class(
    boxes: np.ndarray, scores: np.ndarray,
    method: str = "gaussian", Nt: float = 0.5, sigma: float = 0.5,
    score_thresh: float = 1e-3, max_det: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS for one class. Return kept boxes/scores (sorted)."""
    keep_boxes, keep_scores = [], []
    idxs = scores.argsort()[::-1]
    boxes = boxes.copy()
    scores = scores.copy()
    while idxs.size > 0 and len(keep_boxes) < max_det:
        i = idxs[0]
        keep_boxes.append(boxes[i])
        keep_scores.append(scores[i])
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = iou_xyxy(boxes[i:i+1], boxes[rest]).reshape(-1)
        if method == "linear":
            decay = np.ones_like(ious)
            m = ious > Nt
            decay[m] = 1.0 - ious[m]
            scores[rest] *= decay
        else:  # gaussian
            scores[rest] *= np.exp(-(ious * ious) / max(1e-9, sigma))
        rest = rest[scores[rest] >= score_thresh]
        if rest.size == 0:
            break
        rest = rest[np.argsort(scores[rest])[::-1]]
        idxs = rest
    if len(keep_boxes) == 0:
        return boxes[:0], scores[:0]
    return np.stack(keep_boxes, 0), np.asarray(keep_scores)

def wbf_per_class(
    boxes: np.ndarray, scores: np.ndarray,
    iou_thr: float = 0.55, skip_box_thr: float = 1e-3,
    max_det: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified WBF (single-model). Cluster by IoU>=thr, weighted-average by scores."""
    sel = scores >= skip_box_thr
    boxes, scores = boxes[sel], scores[sel]
    if boxes.size == 0:
        return boxes, scores
    order = scores.argsort()[::-1]
    boxes, scores = boxes[order], scores[order]
    clusters = []
    for i in range(len(boxes)):
        assigned = False
        for k, idxs in enumerate(clusters):
            if iou_xyxy(boxes[i:i+1], boxes[idxs]).max() >= iou_thr:
                clusters[k].append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
    fused_boxes, fused_scores = [], []
    for idxs in clusters:
        bs = boxes[idxs]
        sc = scores[idxs]
        w = sc / (sc.sum() + 1e-9)
        xyxy = (bs * w[:, None]).sum(0)
        fused_boxes.append(xyxy)
        fused_scores.append(sc.max())
        if len(fused_boxes) >= max_det:
            break
    return np.stack(fused_boxes, 0), np.asarray(fused_scores)


# -------------------------- 关键：补丁 non_max_suppression --------------------------

_ORIG_NMS_FN = None  # 保存原始 NMS 函数引用（全局）

def install_nms_patch(mode: str, args):
    """
    Monkey-patch ultralytics.utils.ops.non_max_suppression so our postproc is guaranteed to run.
    """
    global _ORIG_NMS_FN
    try:
        from ultralytics.utils import ops
    except Exception as e:
        print("[Warn] Cannot import ultralytics.utils.ops:", e)
        return False

    if not hasattr(ops, "non_max_suppression"):
        print("[Warn] ultralytics.utils.ops has no non_max_suppression")
        return False

    if _ORIG_NMS_FN is None:
        _ORIG_NMS_FN = ops.non_max_suppression  # 只保存一次

    def _patched_non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
        **kwargs
    ):
        """
        调用原始 NMS 得到每张图的 det 张量列表（[N,6]: x1,y1,x2,y2,conf,cls）。
        若选择 softnms/wbf，则先用极大 iou_thres=0.99 做最小抑制，
        再在 det 上按类别执行 Soft-NMS/WBF，返回替换后的 det。
        """
        # 当启用自定义后处理时，把原生 iou 推到很高，仅做置信度筛与排序
        iou_used = iou_thres
        if mode in ("softnms", "wbf"):
            iou_used = 0.99

        dets = _ORIG_NMS_FN(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_used,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det,    # 原生也会限一下；我们后面还会再限
            nc=nc,
            max_time_img=max_time_img,
            max_nms=max_nms,
            max_wh=max_wh,
            in_place=in_place,
            rotated=rotated,
            **kwargs
        )

        if mode not in ("softnms", "wbf"):
            return dets  # 原版 NMS 直接返回

        # ---- 对每张图做自定义后处理 ----
        new_dets = []
        for det in dets:
            # det: [N,6] -> (可能 N=0)
            if det is None or det.numel() == 0:
                new_dets.append(det)
                continue

            device = det.device
            dtype  = det.dtype

            xyxy = det[:, 0:4].detach().cpu().numpy()
            conf = det[:, 4].detach().cpu().numpy()
            cls  = det[:, 5].detach().cpu().numpy().astype(int)

            out_xyxy, out_conf, out_cls = [], [], []
            for c in np.unique(cls):
                m = (cls == c)
                b_c, s_c = xyxy[m], conf[m]
                if b_c.shape[0] == 0:
                    continue
                if mode == "softnms":
                    method = "gaussian" if args.snms_method == "gaussian" else "linear"
                    b2, s2 = soft_nms_per_class(
                        b_c, s_c,
                        method=method, Nt=args.snms_Nt, sigma=args.snms_sigma,
                        score_thresh=conf_thres, max_det=args.max_det
                    )
                else:  # wbf
                    b2, s2 = wbf_per_class(
                        b_c, s_c,
                        iou_thr=args.wbf_iou, skip_box_thr=conf_thres,
                        max_det=args.max_det
                    )
                if b2.shape[0] == 0:
                    continue
                out_xyxy.append(b2)
                out_conf.append(s2)
                out_cls.append(np.full(b2.shape[0], c, dtype=int))

            if len(out_xyxy) == 0:
                new_dets.append(torch.empty((0, 6), device=device, dtype=dtype))
                continue

            out_xyxy = np.concatenate(out_xyxy, 0)
            out_conf = np.concatenate(out_conf, 0)
            out_cls  = np.concatenate(out_cls, 0)

            # 统一排序 + 裁剪
            order = out_conf.argsort()[::-1][:args.max_det]
            out_xyxy, out_conf, out_cls = out_xyxy[order], out_conf[order], out_cls[order]

            det_new = torch.zeros((out_xyxy.shape[0], 6), device=device, dtype=dtype)
            det_new[:, 0:4] = torch.as_tensor(out_xyxy, device=device, dtype=dtype)
            det_new[:, 4]   = torch.as_tensor(out_conf, device=device, dtype=dtype)
            det_new[:, 5]   = torch.as_tensor(out_cls,  device=device, dtype=dtype)

            new_dets.append(det_new)

        return new_dets

    # 安装补丁
    ops.non_max_suppression = _patched_non_max_suppression
    print(f"[Info] Patched ultralytics.utils.ops.non_max_suppression for mode: {mode}")
    return True


# -------------------------- 主流程 --------------------------

def main():
    ap = argparse.ArgumentParser("Evaluate subset with selectable post-processing (NMS/Soft-NMS/WBF)")
    ap.add_argument("--weights", required=True, type=str, help="Path to .pt weights")
    ap.add_argument("--data", required=True, type=str, help="Path to the subset's data.yaml")
    ap.add_argument("--project", type=str, default="runs/test_eval", help="Ultralytics project dir")
    ap.add_argument("--name", type=str, default="subset_eval", help="Ultralytics run name")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.001, help="Score threshold (prefilter)")
    ap.add_argument("--iou", type=float, default=0.6, help="Original NMS IoU (used when --postproc none)")
    ap.add_argument("--out-label-dir", type=str, default="pred_labels_subset", help="Where to collect predicted labels")
    ap.add_argument("--no-clear-out", action="store_true", help="Do not clear OUT_LABEL_DIR before copy")

    # 自定义后处理选择与超参
    ap.add_argument("--postproc", choices=["none", "softnms", "wbf"], default="none",
                    help="Post-processing: original NMS (none), Soft-NMS or WBF")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections per image after postproc")

    # Soft-NMS 参数
    ap.add_argument("--snms-method", choices=["gaussian", "linear"], default="gaussian")
    ap.add_argument("--snms-Nt", type=float, default=0.5, help="Linear Soft-NMS IoU threshold")
    ap.add_argument("--snms-sigma", type=float, default=0.5, help="Gaussian Soft-NMS sigma")

    # WBF 参数
    ap.add_argument("--wbf-iou", type=float, default=0.55, help="IoU threshold to cluster boxes in WBF")

    args = ap.parse_args()

    weights = _resolve(Path(args.weights))
    data_yaml = _resolve(Path(args.data))
    project = _resolve(Path(args.project))
    out_label_dir = _resolve(Path(args.out_label_dir))

    # 读一下 data.yaml（提示）
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        print("[Info] 子集 data.yaml 加载成功。test 指向：", d.get("test"))
    except Exception as e:
        print("[Warn] 子集 data.yaml 读取失败，但继续评估：", e)

    # 安装 NMS 补丁（保证真的生效）
    ok = install_nms_patch(args.postproc, args)
    if not ok:
        print("[Warn] Failed to patch non_max_suppression. Falling back to original NMS.")
        args.postproc = "none"

    # 进行评测
    model = YOLO(str(weights))
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,     # 注意：当启用 softnms/wbf 时，我们在补丁里会把 iou_thres 覆盖为 0.99
        save_txt=True,
        save_conf=True,
        project=str(project),
        name=args.name,
        verbose=True,
    )

    # 兼容不同版本字段
    metrics_dict = {}
    try:
        metrics_dict = results.results_dict
    except Exception:
        try:
            m = results.metrics
            metrics_dict = {
                "mAP50-95": getattr(m, "map", None),
                "mAP50": getattr(m, "map50", None),
                "mAP75": getattr(m, "map75", None),
                "P": getattr(m, "mp", None),
                "R": getattr(m, "mr", None),
            }
        except Exception:
            pass

    print("\n===== Evaluation Metrics (this subset) =====")
    print(metrics_dict)

    # 收集预测 labels
    pred_label_src = project / args.name / "labels"
    print(f"\nPredicted label txt files are at: {pred_label_src}")

    if pred_label_src.is_dir():
        out_label_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_clear_out:
            for f in out_label_dir.iterdir():
                if f.is_file():
                    f.unlink()
        copied = 0
        for f in pred_label_src.iterdir():
            if f.is_file() and f.suffix.lower() == ".txt":
                shutil.copy2(f, out_label_dir / f.name)
                copied += 1
        print(f"Copied {copied} label files to: {out_label_dir}")
    else:
        print("WARNING: labels folder not found. Make sure save_txt=True and split='test' worked.")

    print("\n[Done] Subset evaluation completed.")


if __name__ == "__main__":
    main()
