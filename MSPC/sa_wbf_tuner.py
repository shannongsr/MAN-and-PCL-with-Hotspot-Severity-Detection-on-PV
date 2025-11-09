#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulated Annealing (SA) tuner for WBF post-processing hyperparameters on Ultralytics val().

Optimizes (conf, wbf_iou, max_det) to maximize a chosen metric (default: mAP50-95).

Usage example:
python sa_wbf_tuner.py \
  --weights runs/detect/xxx/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --imgsz 640 --batch 4 \
  --metric map5095 \
  --iters 60 --temp0 1.0 --alpha 0.90 --steps-per-temp 5 \
  --conf-range 0.001 0.3 \
  --wbf-iou-range 0.3 0.8 \
  --max-det-range 100 500 \
  --seed 42 \
  --project runs/sa_wbf --name coffee_part001_sa

Outputs:
- A CSV log with all trials.
- A YAML file containing the best hyperparameters.
"""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from ultralytics import YOLO


# ----------------------------- WBF + NMS Patch -----------------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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

def wbf_per_class(
    boxes: np.ndarray, scores: np.ndarray,
    iou_thr: float = 0.55, skip_box_thr: float = 1e-3,
    max_det: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
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
        fused_scores.append(sc.max())  # 可改为其他策略
        if len(fused_boxes) >= max_det:
            break
    return np.stack(fused_boxes, 0), np.asarray(fused_scores)

_ORIG_NMS_FN = None

def install_wbf_patch(conf_thres: float, iou_thr_wbf: float, max_det_out: int):
    """
    Monkey-patch ultralytics.utils.ops.non_max_suppression:
    - 先调用原生 NMS，但把 iou_thres 拉到 0.99（尽量不抑制）
    - 再对每张图按类别执行 WBF（使用 conf_thres/ iou_thr_wbf / max_det_out）
    """
    global _ORIG_NMS_FN
    from ultralytics.utils import ops
    if _ORIG_NMS_FN is None:
        _ORIG_NMS_FN = ops.non_max_suppression

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
        # 放宽原生 nms 的 iou（几乎不删框），仅做置信度筛与排序
        base = _ORIG_NMS_FN(
            prediction,
            conf_thres=conf_thres,
            iou_thres=0.99,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det_out,
            nc=nc,
            max_time_img=max_time_img,
            max_nms=max_nms,
            max_wh=max_wh,
            in_place=in_place,
            rotated=rotated,
            **kwargs
        )
        # WBF
        new_dets = []
        for det in base:
            if det is None or det.numel() == 0:
                new_dets.append(det)
                continue
            device, dtype = det.device, det.dtype
            xyxy = det[:, 0:4].detach().cpu().numpy()
            conf = det[:, 4].detach().cpu().numpy()
            cls  = det[:, 5].detach().cpu().numpy().astype(int)

            out_xyxy, out_conf, out_cls = [], [], []
            for c in np.unique(cls):
                m = (cls == c)
                b_c, s_c = xyxy[m], conf[m]
                if b_c.shape[0] == 0:
                    continue
                b2, s2 = wbf_per_class(
                    b_c, s_c,
                    iou_thr=iou_thr_wbf,
                    skip_box_thr=conf_thres,
                    max_det=max_det_out
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

            order = out_conf.argsort()[::-1][:max_det_out]
            out_xyxy, out_conf, out_cls = out_xyxy[order], out_conf[order], out_cls[order]

            out = torch.zeros((out_xyxy.shape[0], 6), device=device, dtype=dtype)
            out[:, 0:4] = torch.as_tensor(out_xyxy, device=device, dtype=dtype)
            out[:, 4]   = torch.as_tensor(out_conf, device=device, dtype=dtype)
            out[:, 5]   = torch.as_tensor(out_cls,  device=device, dtype=dtype)
            new_dets.append(out)
        return new_dets

    ops.non_max_suppression = _patched_non_max_suppression


# ----------------------------- SA Utilities -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def propose(current, ranges, step_scales):
    """
    邻域提案：对 conf/wbf_iou 做高斯微扰；max_det 做整数扰动。
    step_scales: dict like {'conf':0.05, 'wbf_iou':0.05, 'max_det':50}
    """
    conf = clamp(
        current['conf'] + random.gauss(0, step_scales['conf']),
        ranges['conf'][0], ranges['conf'][1]
    )
    wbf_iou = clamp(
        current['wbf_iou'] + random.gauss(0, step_scales['wbf_iou']),
        ranges['wbf_iou'][0], ranges['wbf_iou'][1]
    )
    max_det = int(round(
        clamp(current['max_det'] + random.gauss(0, step_scales['max_det']),
              ranges['max_det'][0], ranges['max_det'][1])
    ))
    return {'conf': conf, 'wbf_iou': wbf_iou, 'max_det': max_det}

def accept_prob(old, new, T):
    if new >= old:
        return 1.0
    return math.exp((new - old) / max(1e-9, T))

def evaluate(model, data, imgsz, batch, conf, wbf_iou, max_det, project, name, metric_key):
    """
    执行一次 val，返回目标分数（越大越好）。
    - 先安装 WBF 补丁（以 conf / wbf_iou / max_det 为参数）
    - 再调用 model.val()
    - 读取 metrics
    """
    install_wbf_patch(conf_thres=conf, iou_thr_wbf=wbf_iou, max_det_out=max_det)
    results = model.val(
        data=str(data),
        split='val',
        imgsz=imgsz,
        batch=batch,
        conf=conf,     # 作为预筛阈值，同时补丁内部也会用到
        iou=0.6,       # 不重要，补丁里已把原生 iou 设置为 0.99
        save_txt=False,
        save_conf=False,
        project=str(project),
        name=name,
        verbose=False,
    )
    # 兼容不同版本的 metrics 访问
    score = None
    # 优先从 results.results_dict
    if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
        rd = results.results_dict
        # 统一 metric_key 名称映射
        mapping = {
            'map5095': ('metrics/mAP50-95(B)', 'mAP50-95', 'map'),
            'map50':   ('metrics/mAP50(B)', 'mAP50', 'map50'),
            'map75':   ('metrics/mAP75(B)', 'mAP75', 'map75'),
            'precision': ('metrics/precision(B)', 'P', 'mp'),
            'recall':    ('metrics/recall(B)', 'R', 'mr'),
        }
        keys = mapping.get(metric_key, ('mAP50-95', 'map'))
        for k in keys:
            if k in rd and rd[k] is not None:
                score = float(rd[k])
                break
    if score is None:
        try:
            m = results.metrics
            if metric_key == 'map5095':
                score = float(getattr(m, 'map', None))
            elif metric_key == 'map50':
                score = float(getattr(m, 'map50', None))
            elif metric_key == 'map75':
                score = float(getattr(m, 'map75', None))
            elif metric_key == 'precision':
                score = float(getattr(m, 'mp', None))
            elif metric_key == 'recall':
                score = float(getattr(m, 'mr', None))
        except Exception:
            score = None
    return score if score is not None else -1.0


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser("Simulated Annealing tuner for WBF (conf, wbf_iou, max_det)")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)

    ap.add_argument("--metric", choices=["map5095","map50","map75","precision","recall"],
                    default="map5095", help="Optimize target metric")

    ap.add_argument("--iters", type=int, default=60, help="Total proposals (rough upper bound)")
    ap.add_argument("--steps-per-temp", type=int, default=5, help="Proposals per temperature level")
    ap.add_argument("--temp0", type=float, default=1.0, help="Initial temperature")
    ap.add_argument("--alpha", type=float, default=0.9, help="Cooling factor per temperature level")
    ap.add_argument("--seed", type=int, default=0)

    # ranges
    ap.add_argument("--conf-range", nargs=2, type=float, default=[0.001, 0.4])
    ap.add_argument("--wbf-iou-range", nargs=2, type=float, default=[0.3, 0.8])
    ap.add_argument("--max-det-range", nargs=2, type=int,   default=[100, 500])

    # initial guess (optional). If not set, use mid of ranges.
    ap.add_argument("--init-conf", type=float, default=None)
    ap.add_argument("--init-wbf-iou", type=float, default=None)
    ap.add_argument("--init-max-det", type=int, default=None)

    # step scales
    ap.add_argument("--step-conf", type=float, default=0.05)
    ap.add_argument("--step-wbf-iou", type=float, default=0.05)
    ap.add_argument("--step-max-det", type=float, default=50)

    ap.add_argument("--project", type=Path, default=Path("runs/sa_wbf"))
    ap.add_argument("--name", type=str, default="sa_run")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project = args.project / args.name
    project.mkdir(parents=True, exist_ok=True)
    log_csv = project / "sa_log.csv"
    best_yaml = project / "best_wbf.yaml"

    # 初始化点
    conf_lo, conf_hi = args.conf_range
    iou_lo, iou_hi = args.wbf_iou_range
    md_lo, md_hi   = args.max_det_range
    current = {
        'conf': args.init_conf if args.init_conf is not None else 0.5*(conf_lo + conf_hi),
        'wbf_iou': args.init_wbf_iou if args.init_wbf_iou is not None else 0.5*(iou_lo + iou_hi),
        'max_det': args.init_max_det if args.init_max_det is not None else int(round(0.5*(md_lo + md_hi))),
    }
    # 限幅
    current['conf']    = clamp(current['conf'], conf_lo, conf_hi)
    current['wbf_iou'] = clamp(current['wbf_iou'], iou_lo, iou_hi)
    current['max_det'] = int(round(clamp(current['max_det'], md_lo, md_hi)))

    # 步长
    step_scales = {'conf': args.step_conf, 'wbf_iou': args.step_wbf_iou, 'max_det': args.step_max_det}
    ranges = {'conf': (conf_lo, conf_hi), 'wbf_iou': (iou_lo, iou_hi), 'max_det': (md_lo, md_hi)}

    # 载入模型（一次加载，多次评测）
    model = YOLO(str(args.weights))

    # 初始评测
    cur_score = evaluate(
        model, args.data, args.imgsz, args.batch,
        current['conf'], current['wbf_iou'], current['max_det'],
        project, f"sa_eval_init", args.metric
    )

    best = dict(current)
    best_score = cur_score

    # 日志表头
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter","T","conf","wbf_iou","max_det","score","accept","is_best","run_name"])

    T = args.temp0
    it = 0
    temp_level = 0
    while it < args.iters:
        temp_level += 1
        for k in range(args.steps_per_temp):
            if it >= args.iters:
                break
            it += 1
            cand = propose(current, ranges, step_scales)
            run_name = f"sa_eval_t{temp_level}_i{it}"
            new_score = evaluate(
                model, args.data, args.imgsz, args.batch,
                cand['conf'], cand['wbf_iou'], cand['max_det'],
                project, run_name, args.metric
            )
            ap = accept_prob(cur_score, new_score, T)
            acc = (random.random() < ap)
            if acc:
                current = cand
                cur_score = new_score
            if new_score > best_score:
                best = cand
                best_score = new_score
                is_best = True
            else:
                is_best = False
            # 记录日志
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([it, f"{T:.6f}",
                            f"{cand['conf']:.6f}", f"{cand['wbf_iou']:.6f}", cand['max_det'],
                            f"{new_score:.6f}", int(acc), int(is_best), run_name])
        # 降温
        T *= args.alpha

    # 保存最优超参
    out = {
        "metric": args.metric,
        "best_score": float(best_score),
        "best_params": {
            "conf": float(best['conf']),
            "wbf_iou": float(best['wbf_iou']),
            "max_det": int(best['max_det']),
        },
        "ranges": {
            "conf": list(args.conf_range),
            "wbf_iou": list(args.wbf_iou_range),
            "max_det": list(args.max_det_range),
        },
        "sa": {
            "iters": args.iters,
            "steps_per_temp": args.steps_per_temp,
            "temp0": args.temp0,
            "alpha": args.alpha,
            "seed": args.seed
        },
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project),
    }
    with open(best_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    print("\n===== SA Finished =====")
    print(f"Best {args.metric}: {best_score:.6f}")
    print(f"Best params: conf={best['conf']:.6f}, wbf_iou={best['wbf_iou']:.6f}, max_det={best['max_det']}")
    print(f"Log CSV: {log_csv}")
    print(f"Best YAML: {best_yaml}")


if __name__ == "__main__":
    main()
