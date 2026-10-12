"""Consolidate TEST results of every operating point -> results/improved/final_results.csv,
results/improved/matched_comparison.csv and results/improved/pareto_frontier{,_clean}.png (accuracy vs fvcore MACs,
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
    out_clean = os.path.join(IMPROVED_DIR, "pareto_frontier_clean.png")
    for p in (out_csv, out_cmp, out_png, out_clean):
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

    # ---- plots (deterministic: everything comes from the CSVs above)
    plot_pareto(pts, bench, out_png, clean=False)
    plot_pareto(pts, bench, out_clean, clean=True)


# Fixed colour per family (colour follows the entity, never its rank)
STYLE = {"static": ("s", "#2a78d6"), "adaptive fixed depth": ("^", "#eb6834"),
         "adaptive early-exit (none)": ("o", "#1baf7a"), "adaptive early-exit (temperature)": ("o", "#4a3aa7")}


def short(name):
    return name.replace("adaptive_", "").replace("_temperature", " (T)").replace("_none", "")


def plot_pareto(pts, bench, path, clean):
    """Accuracy vs MACs with the Pareto frontier. clean=True keeps only statics, fixed depths and
    the calibrated threshold curve (the one the val selection uses) and drops the latency panel."""
    fams = [f for f in STYLE if not (clean and f == "adaptive early-exit (none)")]
    shown = [p for p in pts if p["family"] in fams]
    fig, axes = plt.subplots(1, 1 if clean else 2, figsize=(8, 5.5) if clean else (15, 6))
    ax = axes if clean else axes[0]
    for fam in fams:
        fp = sorted([p for p in shown if p["family"] == fam], key=lambda p: p["avg_macs"])
        ax.plot([p["avg_macs"] / 1e6 for p in fp], [max(p["accuracy"], 0.757) for p in fp], STYLE[fam][0] + ("-" if "early" in fam else ""),
                c=STYLE[fam][1], ms=8, lw=2, label=fam, zorder=3)
    # frontier is computed over ALL points (pareto_optimal column), so both figures show the same frontier
    par = sorted([p for p in pts if p["pareto_optimal"]], key=lambda p: p["avg_macs"])
    ax.step([p["avg_macs"] / 1e6 for p in par], [max(p["accuracy"], 0.757) for p in par], where="post", c="0.35", lw=1, ls="--",
            label="Pareto frontier", zorder=2)
    sel = [p for p in pts if p.get("selected_on_val") == "1"]
    ax.scatter([p["avg_macs"] / 1e6 for p in sel], [p["accuracy"] for p in sel], marker="*", s=320, c="#e34948",
               edgecolors="white", lw=1.5, zorder=6, label="val-selected threshold")
    for p in sel:  # the selected point may come from the raw-confidence curve, so name it explicitly
        ax.annotate(f"selected: {short(p['operating_point']).replace('thr', 'thr ')}", (p["avg_macs"] / 1e6, p["accuracy"]),
                    fontsize=8, color="#e34948", xytext=(-20, 14), textcoords="offset points", ha="right")
    # y-axis zoomed on the interesting region; points below it (exit 1 alone) are drawn at the
    # bottom edge with an arrow and their real accuracy in the label
    y_lo = 0.76
    for p in shown:
        x, y = p["avg_macs"] / 1e6, p["accuracy"]
        if y < y_lo:
            ax.annotate(f"{short(p['operating_point'])}\n(acc {y:.3f}, off scale)", (x, y_lo + 0.002), fontsize=8,
                        color="0.25", xytext=(40, 40), textcoords="offset points", ha="left",
                        arrowprops=dict(arrowstyle="->", color=STYLE[p["family"]][1]))
        elif p["family"].startswith("adaptive early"):
            # threshold curves: label with the threshold value only; skip the uncalibrated curve's
            # labels (it overlaps the calibrated one), it is identified by its legend entry
            if p["family"].endswith("(temperature)"):
                ax.annotate(p["operating_point"].split("_thr")[1].split("_")[0], (x, y), fontsize=7.5,
                            color=STYLE[p["family"]][1], xytext=(3, -13), textcoords="offset points")
        else:
            left = p["operating_point"] == "pruned_50"  # keeps it clear of the threshold-0.5 label
            ax.annotate(short(p["operating_point"]), (x, y), fontsize=8, color="0.25", xytext=(-8 if left else 8, -4),
                        textcoords="offset points", ha="right" if left else "left")
    ax.set(xlabel="average MACs per frame (millions, fvcore, batch 1, 64x64)", ylabel="test accuracy",
           title="Accuracy vs compute (TEST)" + (" - main operating points" if clean else " - all points")
           + "\n(numbers on the purple curve = confidence threshold, temperature-calibrated)")
    ax.set_ylim(y_lo - 0.005, 0.895)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="center right", frameon=False)
    if not clean:
        bx = axes[1]
        kind_style = {"static": ("s", "#2a78d6"), "adaptive (fixed depth)": ("^", "#eb6834"),
                      "adaptive (controller)": ("*", "#e34948")}
        for name, r in bench.items():
            if r["kind"] == "adaptive (controller)" and r.get("selected_on_val") != "1":
                continue  # statics, fixed depths and the selected controller only
            mk, col = kind_style[r["kind"]]
            x, y = float(r["mean_ms"]), float(r["subset_accuracy"])
            bx.errorbar(x, y, xerr=[[0], [float(r["p95_ms"]) - x]], fmt=mk, c=col, ms=9 if mk != "*" else 16, capsize=3)
            bx.annotate(short(name), (x, y), fontsize=8, color="0.25", xytext=(6, -12), textcoords="offset points")
        for k, (mk, col) in kind_style.items():
            bx.plot([], [], mk, c=col, label=k)
        bx.set(xlabel="latency (ms, ORT CPU, 1 thread, batch 1; marker = mean, whisker = p95)",
               ylabel="test accuracy", title="Supporting evidence: latency")
        bx.grid(alpha=0.25)
        bx.spines[["top", "right"]].set_visible(False)
        bx.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(path))




if __name__ == "__main__":
    main()
