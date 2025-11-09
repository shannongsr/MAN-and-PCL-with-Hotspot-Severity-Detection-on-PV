#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulated Annealing (SA) tuner for Soft-NMS (Linear) post-processing on Ultralytics val().

Optimizes (conf, Nt, max_det) to maximize a chosen metric (default: mAP50-95).
Fair-comparison protocol matches the WBF SA tuner:
- Monkey-patch ultralytics.utils.ops.non_max_suppression so Soft-NMS truly applies.
- For custom postproc we first run original NMS with IoU=0.99 (minimal suppression)
  then apply class-wise Soft-NMS(Linear), finally sort & cap by max_det.

Usage example:
python sa_softnms_linear_tuner.py \
  --weights runs/detect/xxx/weights/best.pt \
  --data test_splits/test_part_001/data.yaml \
  --imgsz 640 --batch 4 \
  --metric map5095 \
  --iters 60 --temp0 1.0 --alpha 0.90 --steps-per-temp 5 \
  --conf-range 0.001 0.3 \
  --Nt-range 0.3 0.7 \
  --max-det-range 100 500 \
  --seed 42 \
  --project runs/sa_snms --name coffee_part001_sa

Outputs:
- CSV log of all trials (same columns as WBF tuner)
- best_softnms_linear.yaml (same schema as WBF tuner -> easy to compare)
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


# ----------------------------- Soft-NMS (Linear) + NMS Patch -----------------------------

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

def soft_nms_linear_per_class(
    boxes: np.ndarray, scores: np.ndarray,
    Nt: float = 0.5, score_thresh: float = 1e-3, max_det: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear Soft-NMS for one class.
    s_j <- s_j * (1 - IoU(i, j)) if IoU(i,j) > Nt; otherwise unchanged.
    Keep top 'max_det' after iterative decay; drop boxes < score_thresh.
    """
    keep_boxes, keep_scores = [], []
    idxs = scores.argsort()[::-1]  # high -> low
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
        decay = np.ones_like(ious, dtype=np.float32)
        m = ious > Nt
        decay[m] = 1.0 - ious[m]
        scores[rest] *= decay

        rest = rest[scores[rest] >= score_thresh]
        if rest.size == 0:
            break
        rest = rest[np.argsort(scores[rest])[::-1]]
        idxs = rest  # next round excludes i (already kept)

    if len(keep_boxes) == 0:
        return boxes[:0], scores[:0]
    return np.stack(keep_boxes, 0), np.asarray(keep_scores)


_ORIG_NMS_FN = None

def install_softnms_linear_patch(conf_thres: float, Nt: float, max_det_out: int):
    """
    Monkey-patch ultralytics.utils.ops.non_max_suppression:
    - Run original NMS once with IoU=0.99 (nearly no suppression) to get candidates
      under the same prefilter & pipeline as WBF tuner.
    - Then apply class-wise Linear Soft-NMS with (Nt, conf_thres, max_det_out).
    - Return det lists [N,6] of (x1,y1,x2,y2,conf,cls).
    """
    global _ORIG_NMS_FN
    from ultralytics.utils import ops
    if _ORIG_NMS_FN is None:
        _ORIG_NMS_FN = ops.non_max_suppression  # store original once

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
        # First pass: minimally suppressive NMS to keep Ultralytics' prefiltering behaviors
        base = _ORIG_NMS_FN(
            prediction,
            conf_thres=conf_thres,
            iou_thres=0.99,  # nearly no suppression, same policy as WBF tuner
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det_out,  # allow enough candidates for soft-nms
            nc=nc,
            max_time_img=max_time_img,
            max_nms=max_nms,
            max_wh=max_wh,
            in_place=in_place,
            rotated=rotated,
            **kwargs
        )

        # Second pass: apply our Linear Soft-NMS by class
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
                b2, s2 = soft_nms_linear_per_class(
                    b_c, s_c,
                    Nt=Nt,
                    score_thresh=conf_thres,
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

            # Global sorting + cap
            order = out_conf.argsort()[::-1][:max_det_out]
            out_xyxy, out_conf, out_cls = out_xyxy[order], out_conf[order], out_cls[order]

            out = torch.zeros((out_xyxy.shape[0], 6), device=device, dtype=dtype)
            out[:, 0:4] = torch.as_tensor(out_xyxy, device=device, dtype=dtype)
            out[:, 4]   = torch.as_tensor(out_conf, device=device, dtype=dtype)
            out[:, 5]   = torch.as_tensor(out_cls,  device=device, dtype=dtype)
            new_dets.append(out)

        return new_dets

    ops.non_max_suppression = _patched_non_max_suppression


# ----------------------------- SA Utilities (same as WBF tuner) -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def propose(current, ranges, step_scales):
    """
    Neighborhood proposal: gaussian noise for conf/Nt; integer noise for max_det.
    """
    conf = clamp(
        current['conf'] + random.gauss(0, step_scales['conf']),
        ranges['conf'][0], ranges['conf'][1]
    )
    Nt = clamp(
        current['Nt'] + random.gauss(0, step_scales['Nt']),
        ranges['Nt'][0], ranges['Nt'][1]
    )
    max_det = int(round(
        clamp(current['max_det'] + random.gauss(0, step_scales['max_det']),
              ranges['max_det'][0], ranges['max_det'][1])
    ))
    return {'conf': conf, 'Nt': Nt, 'max_det': max_det}

def accept_prob(old, new, T):
    if new >= old:
        return 1.0
    return math.exp((new - old) / max(1e-9, T))

def evaluate(model, data, imgsz, batch, conf, Nt, max_det, project, name, metric_key):
    """
    One validation with current hyperparams; return target metric.
    Patches NMS to Soft-NMS(Linear) using given (conf, Nt, max_det).
    """
    install_softnms_linear_patch(conf_thres=conf, Nt=Nt, max_det_out=max_det)
    results = model.val(
        data=str(data),
        split='val',
        imgsz=imgsz,
        batch=batch,
        conf=conf,     # prefilter threshold (also used inside patch)
        iou=0.6,       # irrelevant; patch internally uses iou=0.99 for the base NMS pass
        save_txt=False,
        save_conf=False,
        project=str(project),
        name=name,
        verbose=False,
    )
    # Robust metric fetch (same style as WBF tuner)
    score = None
    if hasattr(results, 'results_dict') and isinstance(results.results_dict, dict):
        rd = results.results_dict
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


# ----------------------------- Main (fair with WBF tuner) -----------------------------

def main():
    ap = argparse.ArgumentParser("Simulated Annealing tuner for Soft-NMS (Linear): (conf, Nt, max_det)")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)

    ap.add_argument("--metric", choices=["map5095","map50","map75","precision","recall"],
                    default="map5095", help="Optimize target metric")

    ap.add_argument("--iters", type=int, default=60, help="Total proposals")
    ap.add_argument("--steps-per-temp", type=int, default=5, help="Proposals per temperature level")
    ap.add_argument("--temp0", type=float, default=1.0, help="Initial temperature")
    ap.add_argument("--alpha", type=float, default=0.9, help="Cooling factor")
    ap.add_argument("--seed", type=int, default=0)

    # Ranges (parallel to WBF tuner for fairness)
    ap.add_argument("--conf-range", nargs=2, type=float, default=[0.001, 0.3])
    ap.add_argument("--Nt-range",   nargs=2, type=float, default=[0.3, 0.7])
    ap.add_argument("--max-det-range", nargs=2, type=int, default=[100, 500])

    # Initial guess (optional)
    ap.add_argument("--init-conf", type=float, default=None)
    ap.add_argument("--init-Nt",   type=float, default=None)
    ap.add_argument("--init-max-det", type=int, default=None)

    # Step scales (parallel to WBF tuner)
    ap.add_argument("--step-conf", type=float, default=0.05)
    ap.add_argument("--step-Nt",   type=float, default=0.05)
    ap.add_argument("--step-max-det", type=float, default=50)

    ap.add_argument("--project", type=Path, default=Path("runs/sa_snms"))
    ap.add_argument("--name", type=str, default="sa_run")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project = args.project / args.name
    project.mkdir(parents=True, exist_ok=True)
    log_csv = project / "sa_log.csv"
    best_yaml = project / "best_softnms_linear.yaml"

    # Init point
    conf_lo, conf_hi = args.conf_range
    Nt_lo, Nt_hi = args.Nt_range
    md_lo, md_hi = args.max_det_range
    current = {
        'conf': args.init_conf if args.init_conf is not None else 0.5*(conf_lo + conf_hi),
        'Nt':   args.init_Nt   if args.init_Nt   is not None else 0.5*(Nt_lo + Nt_hi),
        'max_det': args.init_max_det if args.init_max_det is not None else int(round(0.5*(md_lo + md_hi))),
    }
    current['conf']    = clamp(current['conf'], conf_lo, conf_hi)
    current['Nt']      = clamp(current['Nt'], Nt_lo, Nt_hi)
    current['max_det'] = int(round(clamp(current['max_det'], md_lo, md_hi)))

    step_scales = {'conf': args.step_conf, 'Nt': args.step_Nt, 'max_det': args.step_max_det}
    ranges = {'conf': (conf_lo, conf_hi), 'Nt': (Nt_lo, Nt_hi), 'max_det': (md_lo, md_hi)}

    # Load model once
    model = YOLO(str(args.weights))

    # Initial evaluation
    cur_score = evaluate(
        model, args.data, args.imgsz, args.batch,
        current['conf'], current['Nt'], current['max_det'],
        project, "sa_eval_init", args.metric
    )
    best = dict(current)
    best_score = cur_score

    # CSV header (same columns as WBF tuner)
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter","T","conf","Nt","max_det","score","accept","is_best","run_name"])

    T = args.temp0
    it = 0
    temp_level = 0
    while it < args.iters:
        temp_level += 1
        for _ in range(args.steps_per_temp):
            if it >= args.iters:
                break
            it += 1
            cand = propose(current, ranges, step_scales)
            run_name = f"sa_eval_t{temp_level}_i{it}"
            new_score = evaluate(
                model, args.data, args.imgsz, args.batch,
                cand['conf'], cand['Nt'], cand['max_det'],
                project, run_name, args.metric
            )
            ap = accept_prob(cur_score, new_score, T)
            accepted = (random.random() < ap)
            if accepted:
                current = cand
                cur_score = new_score
            is_best = False
            if new_score > best_score:
                best, best_score = cand, new_score
                is_best = True
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([it, f"{T:.6f}",
                            f"{cand['conf']:.6f}", f"{cand['Nt']:.6f}", cand['max_det'],
                            f"{new_score:.6f}", int(accepted), int(is_best), run_name])
        T *= args.alpha  # cool down

    # Save best hyperparams (same schema style as WBF tuner)
    out = {
        "metric": args.metric,
        "best_score": float(best_score),
        "best_params": {
            "conf": float(best['conf']),
            "Nt": float(best['Nt']),
            "max_det": int(best['max_det']),
        },
        "ranges": {
            "conf": list(args.conf_range),
            "Nt": list(args.Nt_range),
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

    print("\n===== SA (Soft-NMS Linear) Finished =====")
    print(f"Best {args.metric}: {best_score:.6f}")
    print(f"Best params: conf={best['conf']:.6f}, Nt={best['Nt']:.6f}, max_det={best['max_det']}")
    print(f"Log CSV: {log_csv}")
    print(f"Best YAML: {best_yaml}")


if __name__ == "__main__":
    main()
