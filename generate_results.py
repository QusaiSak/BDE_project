"""Consolidate TEST results of every operating point -> results/improved/final_results.csv,
results/improved/matched_comparison.csv and results/improved/pareto_frontier.png (accuracy vs fvcore MACs,
Pareto-optimal points marked; latency panel as supporting evidence).

  python generate_results.py [--force]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import IMPROVED_DIR, MODEL_DIR, RESULT_DIR, adaptive_paths, file_size_mb, guard_overwrite, read_csv, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_csv = os.path.join(IMPROVED_DIR, "final_results.csv")
    out_cmp = os.path.join(IMPROVED_DIR, "matched_comparison.csv")
    out_png = os.path.join(IMPROVED_DIR, "pareto_frontier.png")
    for p in (out_csv, out_cmp, out_png):
        guard_overwrite(p, a.force)

    R = lambda n: read_csv(os.path.join(RESULT_DIR, n))
    bench = {r["operating_point"]: r for r in read_csv(os.path.join(IMPROVED_DIR, "benchmark_results.csv"))}
    paths = adaptive_paths()
    pts = []
    for r in R("static_results.csv"):
        if r["split"] == "test":
            rel = "baseline.onnx" if r["model"] == "baseline" else f"pruned/{r['model']}.onnx"
            pts.append({"operating_point": r["model"], "family": "static", "avg_macs": int(r["macs"]),
                        "accuracy": float(r["accuracy"]), "macro_f1": float(r["macro_f1"]),
                        "onnx_mb": file_size_mb(os.path.join(MODEL_DIR, rel)), "bench": r["model"]})
    for r in R("adaptive_results.csv"):
        if r["split"] != "test" or "controller" in r["threshold"]:
            continue
        if r["threshold"].startswith("exit"):
            k = int(r["threshold"][4])
            pts.append({"operating_point": f"adaptive_{r['threshold']}", "family": "adaptive fixed depth",
                        "avg_macs": int(r["avg_macs"]), "accuracy": float(r["accuracy"]),
                        "macro_f1": float(r["macro_f1"]), "onnx_mb": file_size_mb(*paths[:k]),
                        "bench": f"adaptive_forced_exit{k}"})
        else:
            pts.append({"operating_point": f"adaptive_thr{r['threshold']}_{r['calibration']}",
                        "family": f"adaptive early-exit ({r['calibration']})", "avg_macs": int(r["avg_macs"]),
                        "accuracy": float(r["accuracy"]), "macro_f1": float(r["macro_f1"]),
                        "onnx_mb": file_size_mb(*paths), "selected_on_val": r["selected"],
                        "exit_shares": f"{r['share_exit1']}/{r['share_exit2']}/{r['share_exit3']}",
                        "bench": f"adaptive_thr{r['threshold']}_{r['calibration']}"})
    M, A = np.array([p["avg_macs"] for p in pts]), np.array([p["accuracy"] for p in pts])
    for i, p in enumerate(pts):
        dom = any((M[j] <= M[i]) and (A[j] >= A[i]) and ((M[j] < M[i]) or (A[j] > A[i])) for j in range(len(pts)))
        p["pareto_optimal"] = int(not dom)
    base = next(p for p in pts if p["operating_point"] == "baseline")
    rows = []
    for p in sorted(pts, key=lambda p: p["avg_macs"]):
        b = bench.get(p.get("bench") or "", {})
        rows.append({"operating_point": p["operating_point"], "family": p["family"], "avg_macs": p["avg_macs"],
                     "macs_vs_baseline": f"{p['avg_macs'] / base['avg_macs']:.3f}", "accuracy": f"{p['accuracy']:.4f}",
                     "macro_f1": f"{p['macro_f1']:.4f}", "onnx_mb": f"{p['onnx_mb']:.3f}",
                     "latency_mean_ms": b.get("mean_ms", "not yet measured"),
                     "latency_p95_ms": b.get("p95_ms", "not yet measured"),
                     "exit_shares": p.get("exit_shares", ""), "selected_on_val": p.get("selected_on_val", ""),
                     "pareto_optimal": p["pareto_optimal"], "split": "test"})
    write_csv(out_csv, rows, a.force)

    # ---- matched-FLOPs / matched-accuracy comparison vs every static model
    ada = [p for p in pts if p["family"].startswith("adaptive")]
    cmp_rows = []
    print("\nAdaptive vs static (TEST):")
    for s in [p for p in pts if p["family"] == "static"]:
        under = [p for p in ada if p["avg_macs"] <= s["avg_macs"]]
        best = max(under, key=lambda p: p["accuracy"]) if under else None
        match = [p for p in ada if p["accuracy"] >= s["accuracy"] - 0.005]
        cheap = min(match, key=lambda p: p["avg_macs"]) if match else None
        r = {"static_model": s["operating_point"], "static_macs": s["avg_macs"], "static_acc": f"{s['accuracy']:.4f}",
             "best_adaptive_at_or_below_macs": best["operating_point"] if best else "none",
             "its_macs": best["avg_macs"] if best else "", "its_acc": f"{best['accuracy']:.4f}" if best else "",
             "acc_delta_pts": f"{100 * (best['accuracy'] - s['accuracy']):+.2f}" if best else "",
             "cheapest_adaptive_within_0.5pt": cheap["operating_point"] if cheap else "none",
             "its_macs_2": cheap["avg_macs"] if cheap else "",
             "macs_saving_vs_static": f"{1 - cheap['avg_macs'] / s['avg_macs']:+.3f}" if cheap else ""}
        cmp_rows.append(r)
        cheap_m = f"{cheap['avg_macs'] / 1e6:.1f}M MACs" if cheap else "-"
        print(f"  {s['operating_point']:10s} ({s['avg_macs'] / 1e6:.1f}M MACs, acc {s['accuracy']:.4f}): "
              f"best adaptive at <= MACs = {r['best_adaptive_at_or_below_macs']} acc {r['its_acc']} ({r['acc_delta_pts']} pts); "
              f"cheapest adaptive within 0.5pt = {r['cheapest_adaptive_within_0.5pt']} ({cheap_m})")
    write_csv(out_cmp, cmp_rows, a.force)

    # ---- plot
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.5))
    style = {"static": ("s", "#d62828"), "adaptive fixed depth": ("^", "#6c757d"),
             "adaptive early-exit (none)": ("o", "#90be6d"), "adaptive early-exit (temperature)": ("o", "#277da1")}
    for fam, (mk, col) in style.items():
        fp = sorted([p for p in pts if p["family"] == fam], key=lambda p: p["avg_macs"])
        if not fp:
            continue
        ax.plot([p["avg_macs"] / 1e6 for p in fp], [p["accuracy"] for p in fp], mk + ("-" if "early" in fam else ""),
                c=col, ms=7, lw=1, label=fam)
        for p in fp:
            lab = p["operating_point"].replace("adaptive_", "").replace("_temperature", "").replace("_none", "")
            ax.annotate(lab, (p["avg_macs"] / 1e6, p["accuracy"]), fontsize=6.5, xytext=(4, -9), textcoords="offset points")
    par = sorted([p for p in pts if p["pareto_optimal"]], key=lambda p: p["avg_macs"])
    ax.step([p["avg_macs"] / 1e6 for p in par], [p["accuracy"] for p in par], where="post", c="k", lw=0.8, ls=":",
            label="Pareto frontier (all points)")
    ax.scatter([p["avg_macs"] / 1e6 for p in par], [p["accuracy"] for p in par], s=150, facecolors="none",
               edgecolors="k", lw=1.2, label="Pareto-optimal")
    sel = [p for p in pts if p.get("selected_on_val") == "1"]
    ax.scatter([p["avg_macs"] / 1e6 for p in sel], [p["accuracy"] for p in sel], marker="*", s=300, c="gold",
               edgecolors="k", zorder=6, label="val-selected threshold")
    ax.set(xlabel="average MACs per frame (millions, fvcore, batch 1, 64x64)", ylabel="test accuracy",
           title="Accuracy vs FLOPs (TEST)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    for name, r in bench.items():
        if r["kind"] == "adaptive (controller)" and r.get("selected_on_val") != "1":
            continue  # keep the latency panel readable: statics, fixed depths, selected controller
        mk = "s" if r["kind"] == "static" else "^" if "fixed" in r["kind"] else "*"
        bx.errorbar(float(r["mean_ms"]), float(r["subset_accuracy"]), fmt=mk, ms=9 if mk != "*" else 14,
                    label=name)
        bx.annotate(f"p95 {float(r['p95_ms']):.2f}", (float(r["mean_ms"]), float(r["subset_accuracy"])), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")
    bx.set(xlabel="mean latency (ms, ORT CPU, 1 thread, batch 1)", ylabel="test accuracy (benchmarked frames)",
           title="Supporting evidence: latency")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", os.path.relpath(out_png))


if __name__ == "__main__":
    main()
