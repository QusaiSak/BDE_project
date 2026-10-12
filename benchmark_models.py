"""Latency benchmark: ONNX Runtime CPU, batch 1, ONE thread, the same test frames
for every operating point, warm-up excluded. The adaptive system is timed THROUGH the
controller, so its overhead (softmax, branching, passing features) is included.
Every threshold of the sweep (raw and temperature-scaled) is timed, on the FULL test set
by default (--n 0), so accuracies here equal the evaluation accuracies.
Writes results/improved/benchmark_results.csv.

  python benchmark_models.py [--n 0] [--warmup 100] [--force]
"""
import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from utils import (IMPROVED_DIR, MODEL_DIR, RESULT_DIR, AdaptiveONNXController, adaptive_paths, file_size_mb, guard_overwrite,
                   hard_exit, load_exit_macs, load_selected_config, load_split, load_stats, load_temperatures,
                   make_session, normalize, read_csv, write_csv)


def stats(ms):
    ms = np.asarray(ms)
    return {"mean_ms": f"{ms.mean():.3f}", "std_ms": f"{ms.std():.3f}", "p50_ms": f"{np.percentile(ms, 50):.3f}",
            "p95_ms": f"{np.percentile(ms, 95):.3f}", "p99_ms": f"{np.percentile(ms, 99):.3f}",
            "max_ms": f"{ms.max():.3f}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="test frames to time; 0 = full test set")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = os.path.join(IMPROVED_DIR, "benchmark_results.csv")
    guard_overwrite(out, a.force)

    mean, std, _ = load_stats()
    xu, yt = load_split("test")
    a.n = a.n or len(yt)
    idx = np.random.default_rng(a.seed).choice(len(yt), a.n, replace=False)
    x, y = normalize(xu[idx], mean, std), yt[idx]
    frames = [x[i:i + 1] for i in range(a.n)]
    macs = {r["model"]: int(r["macs"]) for r in read_csv(os.path.join(RESULT_DIR, "pruning_results.csv")) if r["split"] == "test"}
    cum = load_exit_macs()
    paths = adaptive_paths()
    rows = []

    def run(name, fn, kind, avg_macs, size):
        for i in range(a.warmup):
            fn(frames[i % a.n])
        ms, correct = [], 0
        for f, t in tqdm(list(zip(frames, y)), desc=name, leave=False):
            t0 = time.perf_counter()
            p = fn(f)
            ms.append((time.perf_counter() - t0) * 1e3)
            correct += int(p == t)
        r = {"operating_point": name, "kind": kind, "avg_macs": int(avg_macs), "onnx_mb": f"{size:.3f}",
             **stats(ms), "subset_accuracy": f"{correct / a.n:.4f}", "n": a.n, "threads": 1, "batch": 1}
        rows.append(r)
        print(f"{name:34s} mean {r['mean_ms']} ms  p95 {r['p95_ms']} ms  p99 {r['p99_ms']}  MACs {int(avg_macs):,}")
        return ms

    for name, rel in (("baseline", "baseline.onnx"), ("pruned_25", "pruned/pruned_25.onnx"),
                      ("pruned_50", "pruned/pruned_50.onnx")):
        s = make_session(os.path.join(MODEL_DIR, rel), threads=1)
        run(name, lambda f, s=s: int(s.run(None, {"image": f})[0].argmax()), "static", macs[name],
            file_size_mb(os.path.join(MODEL_DIR, rel)))

    temps = load_temperatures()
    thr, use_t = load_selected_config()
    ctrl = AdaptiveONNXController(paths, temps if use_t else (1.0, 1.0, 1.0), threads=1)
    for k in (1, 2, 3):  # forced-depth operating points of the SAME shared network
        run(f"adaptive_forced_exit{k}", lambda f, k=k: ctrl.predict(f, threshold=2.0, max_exit=k)[0],
            "adaptive (fixed depth)", cum[k - 1], file_size_mb(*paths[:k]))

    # adaptive with its real controller at EVERY swept threshold (raw and temperature-scaled);
    # model time is recorded separately so the controller overhead can be isolated
    sel_name = f"adaptive_thr{thr}_{'temperature' if use_t else 'none'}"
    for cal, tt in (("none", (1.0, 1.0, 1.0)), ("temperature", temps)):
        ctrl = AdaptiveONNXController(paths, tt, threads=1)
        for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            exits, model_ms = [], []

            def ctrl_fn(f, t=t, ctrl=ctrl, exits=exits, model_ms=model_ms):
                p, e, _ = ctrl.predict(f, t)
                exits.append(e)
                model_ms.append(ctrl.last_model_time * 1e3)
                return p
            name = f"adaptive_thr{t}_{cal}"
            tot = run(name, ctrl_fn, "adaptive (controller)", 0, file_size_mb(*paths))
            ex = np.array(exits[a.warmup:])
            rows[-1]["avg_macs"] = int(round(sum((ex == k).mean() * cum[k - 1] for k in (1, 2, 3))))
            rows[-1]["exit_shares"] = "/".join(f"{(ex == k).mean():.3f}" for k in (1, 2, 3))
            over = np.asarray(tot) - np.asarray(model_ms[a.warmup:])
            rows[-1]["controller_overhead_ms"] = f"{over.mean():.4f}"
            rows[-1]["selected_on_val"] = int(name == sel_name)
            if name == sel_name:
                print(f"controller overhead (total - ORT run time): {over.mean():.4f} ms/frame "
                      f"({100 * over.mean() / np.mean(tot):.1f}% of adaptive latency)")
    fields = list(rows[-1].keys())  # last row has every column
    write_csv(out, rows, a.force, fieldnames=fields)


if __name__ == "__main__":
    main()
    hard_exit()
