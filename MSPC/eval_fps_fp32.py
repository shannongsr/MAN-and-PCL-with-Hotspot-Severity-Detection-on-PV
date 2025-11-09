#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch FP32 evaluation + GPU power sampling (NVML).
- Ultralytics YOLOv8 native .val() with half=False
- Power sampling via pynvml at ~20 Hz (avg W, net W, FPS/W, J/img)

Usage:
python eval_fp32_power.py \
  --weights runs/detect/YOLO11_300_solar/weights/best.pt \
  --data dataset/data.yaml \
  --imgsz 640 --device cuda \
  --split test --batch 1 \
  --warmup 1 --repeats 3
"""

import argparse
import time
import threading
from pathlib import Path
from ultralytics import YOLO


# -------- NVML power sampler --------
class PowerSampler:
    def __init__(self, device_index: int = 0, hz: float = 20.0):
        self.enabled = False
        self.device_index = device_index
        self.period = 1.0 / max(1e-6, hz)
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
        try:
            import pynvml
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(device_index)
            self.enabled = True
        except Exception:
            self.enabled = False
            self.nvml = None
            self.handle = None

    def _read_power(self):
        try:
            mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
            return mw / 1000.0
        except Exception:
            return None

    def _loop(self):
        next_t = time.time()
        while not self._stop.is_set():
            t = time.time()
            if t >= next_t:
                p = self._read_power()
                if p is not None:
                    self.samples.append((t, p))
                next_t += self.period
            else:
                time.sleep(min(0.001, next_t - t))

    def start(self):
        if not self.enabled:
            return
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def summarize(self, t0: float, t1: float):
        if not self.enabled or not self.samples:
            return None, None
        data = [(t, w) for (t, w) in self.samples if t0 <= t <= t1]
        if len(data) < 2:
            return None, None
        energy_j = 0.0
        for i in range(1, len(data)):
            t0i, w0 = data[i - 1]
            t1i, w1 = data[i]
            dt = max(0.0, t1i - t0i)
            energy_j += 0.5 * (w0 + w1) * dt
        duration = max(1e-9, data[-1][0] - data[0][0])
        avg_w = energy_j / duration
        return avg_w, energy_j

    def __del__(self):
        try:
            if self.nvml:
                self.nvml.nvmlShutdown()
        except Exception:
            pass


def _res(p: str) -> Path:
    return Path(p).expanduser().resolve()


def parse_args():
    ap = argparse.ArgumentParser("Evaluate YOLO (PyTorch FP32) + GPU power metrics")
    ap.add_argument("--weights", required=True, type=str, help="Path to .pt weights")
    ap.add_argument("--data", required=True, type=str, help="Path to dataset data.yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Eval split")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for val")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs (not timed)")
    ap.add_argument("--repeats", type=int, default=3, help="Timed runs")
    ap.add_argument("--idle_seconds", type=float, default=3.0, help="Seconds to measure idle baseline power")
    ap.add_argument("--power_hz", type=float, default=20.0, help="Sampling frequency (Hz)")
    return ap.parse_args()


def measure_idle_power(ps: PowerSampler, seconds: float):
    if not ps.enabled or seconds <= 0:
        return None
    ps.start()
    t0 = time.time()
    time.sleep(seconds)
    t1 = time.time()
    ps.stop()
    avg_w, _ = ps.summarize(t0, t1)
    return avg_w


def eval_once(model, data_yaml, imgsz, device, split, batch, conf, iou, verbose=True):
    return model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        split=split,
        batch=batch,
        conf=conf,
        iou=iou,
        half=False,          # <<—— FP32：明确关闭半精度
        verbose=verbose
    )


def main():
    args = parse_args()
    pt_path = _res(args.weights)
    data_yaml = _res(args.data)

    if not pt_path.exists() or pt_path.suffix.lower() != ".pt":
        raise SystemExit("[ERR] --weights must be a .pt file")
    if not data_yaml.exists():
        raise SystemExit("[ERR] data.yaml not found")

    model = YOLO(str(pt_path))

    # Power sampler
    ps = PowerSampler(device_index=0, hz=args.power_hz)

    # Idle baseline
    idle_w = measure_idle_power(ps, args.idle_seconds)
    if idle_w is not None:
        print(f"[POWER] Idle baseline (avg over {args.idle_seconds:.1f}s): {idle_w:.2f} W")
    else:
        print("[POWER] NVML unavailable -> power metrics disabled.")

    # Warmup
    for i in range(max(0, args.warmup)):
        print(f"[Warmup] {i+1}/{args.warmup}")
        _ = eval_once(model, data_yaml, args.imgsz, args.device, args.split,
                      args.batch, args.conf, args.iou, verbose=False)

    # Timed runs
    wall_times, inf_ms_list = [], []
    avg_w_list, net_w_list, energy_j_list, net_energy_j_list = [], [], [], []
    last_results = None

    for r in range(max(1, args.repeats)):
        print(f"\n[Run {r+1}/{args.repeats}] Starting timed eval (FP32)...")
        if ps.enabled:
            ps.start()
        t0 = time.time()
        results = eval_once(model, data_yaml, args.imgsz, args.device, args.split,
                            args.batch, args.conf, args.iou, verbose=True)
        t1 = time.time()
        if ps.enabled:
            ps.stop()

        wall = t1 - t0
        wall_times.append(wall)
        last_results = results

        try:
            inf_ms = results.speed["inference"]
            inf_ms_list.append(inf_ms)
            print(f"[Speed] inference={inf_ms:.2f} ms/img -> FPS≈{1000.0/inf_ms:.2f} | wall={wall:.2f}s")
        except Exception:
            print(f"[Speed] wall={wall:.2f}s (Ultralytics speed dict unavailable)")

        if ps.enabled:
            avg_w, energy_j = ps.summarize(t0, t1)
            if (avg_w is not None) and (energy_j is not None):
                avg_w_list.append(avg_w)
                energy_j_list.append(energy_j)
                if idle_w is not None:
                    net_w = max(0.0, avg_w - idle_w)
                    net_energy = max(0.0, energy_j - idle_w * wall)
                    net_w_list.append(net_w)
                    net_energy_j_list.append(net_energy)
                print(f"[Power] avg={avg_w:.2f} W"
                      + (f" | net={avg_w - idle_w:.2f} W" if idle_w is not None else ""))
            else:
                print("[Power] Not enough samples to summarize.")

    # Summary
    print("\n========== SUMMARY (FP32) ==========")
    if inf_ms_list:
        avg_ms = sum(inf_ms_list) / len(inf_ms_list)
        print(f"[FPS]  inference avg: {avg_ms:.2f} ms/img  ->  {1000.0/avg_ms:.2f} FPS")
    if wall_times:
        avg_wall = sum(wall_times) / len(wall_times)
        print(f"[Time] wall-clock avg per run: {avg_wall:.2f} s")

    if avg_w_list:
        mean_w = sum(avg_w_list) / len(avg_w_list)
        print(f"[Power] avg power: {mean_w:.2f} W")
        if inf_ms_list:
            fps = 1000.0 / (sum(inf_ms_list) / len(inf_ms_list))
            print(f"[Eff ] FPS/W (gross): {fps/mean_w:.4f} (1/W)")

        if energy_j_list and last_results is not None:
            n_imgs = getattr(last_results, "nimgs", None)
            if n_imgs is None:
                metrics = getattr(last_results, "metrics", None)
                if metrics and hasattr(metrics, "images"):
                    n_imgs = metrics.images
            if n_imgs:
                mean_energy = sum(energy_j_list) / len(energy_j_list)
                j_per_img = mean_energy / n_imgs
                print(f"[Ener] energy per image (gross): {j_per_img:.2f} J/img")

    if avg_w_list and net_w_list and idle_w is not None:
        mean_w_net = sum(net_w_list) / len(net_w_list)
        print(f"[Power] net power (minus idle): {mean_w_net:.2f} W")
        if inf_ms_list:
            fps = 1000.0 / (sum(inf_ms_list) / len(inf_ms_list))
            print(f"[Eff ] FPS/W (net): {fps/mean_w_net:.4f} (1/W)")

        if net_energy_j_list and last_results is not None:
            n_imgs = getattr(last_results, "nimgs", None)
            if n_imgs is None:
                metrics = getattr(last_results, "metrics", None)
                if metrics and hasattr(metrics, "images"):
                    n_imgs = metrics.images
            if n_imgs:
                mean_net_energy = sum(net_energy_j_list) / len(net_energy_j_list)
                j_per_img_net = mean_net_energy / n_imgs
                print(f"[Ener] energy per image (net): {j_per_img_net:.2f} J/img")

    if last_results is not None:
        try:
            print("\n[Metrics] results_dict:")
            print(last_results.results_dict)
        except Exception:
            pass


if __name__ == "__main__":
    main()
