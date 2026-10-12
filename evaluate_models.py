import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import (CLASSES, MODEL_DIR, RESULT_DIR, AdaptiveONNXController, adaptive_paths, classification_metrics,
                   ece_score, fit_temperature, guard_overwrite, hard_exit, load_split, load_stats, onnx_exit_logits,
                   onnx_logits, read_csv, simulate_exits, softmax, write_csv)

THRESHOLDS = [0.70, 0.80, 0.90, 0.95, 0.99]
STATIC = {"baseline": "baseline.onnx", "pruned_25": "pruned/pruned_25.onnx", "pruned_50": "pruned/pruned_50.onnx"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="adaptive")
    ap.add_argument("--tolerance", type=float, default=0.01, help="allowed val-accuracy drop vs exit-3-only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = {k: os.path.join(RESULT_DIR, f"{k}.csv") for k in
           ("adaptive_results", "exit_distribution", "static_results", "calibration_results",
            "per_class_results", "failure_modes")}
    png = os.path.join(RESULT_DIR, "reliability.png")
    for p in list(out.values()) + [png]:
        guard_overwrite(p, a.force)

    mean, std, _ = load_stats()
    data = {}
    for s in ("val", "test"):
        x, y = load_split(s)
        data[s] = (((x.astype(np.float32) / 255 - mean) / std).transpose(0, 3, 1, 2).copy(), y)

    # ------------------------------------------------------------------ static models
    static_rows, per_class_rows, calib_rows = [], [], []
    macs_static = {r["model"]: int(r["macs"]) for r in read_csv(os.path.join(RESULT_DIR, "pruning_results.csv"))
                   if r["split"] == "test"}
    for name, rel in STATIC.items():
        for s in ("val", "test"):
            lg = onnx_logits(os.path.join(MODEL_DIR, rel), data[s][0])
            m = classification_metrics(data[s][1], lg.argmax(1))
            ece, _ = ece_score(softmax(lg), data[s][1])
            static_rows.append({"model": name, "split": s, "accuracy": f"{m['accuracy']:.4f}",
                                "macro_f1": f"{m['macro_f1']:.4f}", "ece": f"{ece:.4f}", "macs": macs_static[name],
                                "worst_class": CLASSES[int(np.nanargmin(m["per_class"]))],
                                "worst_class_acc": f"{np.nanmin(m['per_class']):.4f}"})
            if s == "test":
                per_class_rows.append({"model": name, **{c: f"{v:.4f}" for c, v in zip(CLASSES, m["per_class"])}})
    write_csv(out["static_results"], static_rows, a.force)

    # ------------------------------------------------------------------ adaptive exits
    paths = adaptive_paths(a.tag)
    cum = [int(r["cumulative_macs"]) for r in read_csv(os.path.join(RESULT_DIR, "exit_flops.csv"))]
    base_macs = macs_static["baseline"]
    logits = {s: onnx_exit_logits(paths, data[s][0]) for s in ("val", "test")}
    temps = [fit_temperature(logits["val"][k], data["val"][1]) for k in range(3)]
    print("temperatures fitted on val:", [round(t, 3) for t in temps])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for k in range(3):
        for s in ("val", "test"):
            y = data[s][1]
            e_raw, rel_raw = ece_score(softmax(logits[s][k]), y)
            e_cal, rel_cal = ece_score(softmax(logits[s][k], temps[k]), y)
            m = classification_metrics(y, logits[s][k].argmax(1))
            mc = softmax(logits[s][k]).max(1).mean()
            calib_rows.append({"model": a.tag, "split": s, "exit": k + 1, "temperature": f"{temps[k]:.4f}",
                               "accuracy": f"{m['accuracy']:.4f}", "macro_f1": f"{m['macro_f1']:.4f}",
                               "mean_conf_raw": f"{mc:.4f}", "ece_raw": f"{e_raw:.4f}", "ece_temperature": f"{e_cal:.4f}"})
            if s == "test":
                per_class_rows.append({"model": f"{a.tag}_exit{k + 1}_only",
                                       **{c: f"{v:.4f}" for c, v in zip(CLASSES, m["per_class"])}})
            if s == "val":
                ax = axes[k]
                for rel, lab, mk in ((rel_raw, f"raw (ECE {e_raw:.3f})", "o-"), (rel_cal, f"T={temps[k]:.2f} (ECE {e_cal:.3f})", "s-")):
                    ax.plot([r[2] for r in rel], [r[3] for r in rel], mk, ms=4, label=lab)
                ax.plot([0, 1], [0, 1], "k--", lw=0.8)
                ax.set(title=f"Exit {k + 1} reliability (val)", xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"wrote results/reliability.png")
    write_csv(out["calibration_results"], calib_rows, a.force)

    # ------------------------------------------------------------------ threshold sweep
    rows, dist_rows = [], []
    for s in ("val", "test"):
        y = data[s][1]
        exit3_correct = logits[s][2].argmax(1) == y
        for calib, T in (("none", (1.0, 1.0, 1.0)), ("temperature", tuple(temps))):
            for thr in THRESHOLDS:
                pred, ex = simulate_exits(logits[s], thr, T)
                m = classification_metrics(y, pred)
                share = [float((ex == k).mean()) for k in (1, 2, 3)]
                avg = float(sum(p * c for p, c in zip(share, cum)))
                early = ex < 3
                premature = float((early & (pred != y) & exit3_correct).mean())
                rows.append({"split": s, "calibration": calib, "threshold": thr, "accuracy": f"{m['accuracy']:.4f}",
                             "macro_f1": f"{m['macro_f1']:.4f}", "avg_macs": int(round(avg)),
                             "avg_macs_vs_baseline": f"{avg / base_macs:.3f}",
                             "share_exit1": f"{share[0]:.4f}", "share_exit2": f"{share[1]:.4f}",
                             "share_exit3": f"{share[2]:.4f}", "premature_exit_rate": f"{premature:.4f}",
                             "worst_class": CLASSES[int(np.nanargmin(m["per_class"]))],
                             "worst_class_acc": f"{np.nanmin(m['per_class']):.4f}", "selected": 0})
                for k in (1, 2, 3):
                    sel = ex == k
                    dist_rows.append({"split": s, "calibration": calib, "threshold": thr, "exit": k,
                                      "n": int(sel.sum()), "share": f"{sel.mean():.4f}",
                                      "accuracy_at_exit": f"{(pred[sel] == y[sel]).mean():.4f}" if sel.any() else "",
                                      "cumulative_macs": cum[k - 1]})
        # reference rows: each exit alone
        for k in range(3):
            m = classification_metrics(y, logits[s][k].argmax(1))
            rows.append({"split": s, "calibration": "n/a", "threshold": f"exit{k + 1}_only",
                         "accuracy": f"{m['accuracy']:.4f}", "macro_f1": f"{m['macro_f1']:.4f}",
                         "avg_macs": cum[k], "avg_macs_vs_baseline": f"{cum[k] / base_macs:.3f}",
                         "share_exit1": int(k == 0), "share_exit2": int(k == 1), "share_exit3": int(k == 2),
                         "premature_exit_rate": "", "worst_class": CLASSES[int(np.nanargmin(m["per_class"]))],
                         "worst_class_acc": f"{np.nanmin(m['per_class']):.4f}", "selected": 0})

    # ------------------------------------------------------------------ selection on VAL
    ref = float(next(r for r in rows if r["split"] == "val" and r["threshold"] == "exit3_only")["accuracy"])
    cands = [r for r in rows if r["split"] == "val" and r["calibration"] != "n/a"
             and float(r["accuracy"]) >= ref - a.tolerance]
    if not cands:
        raise RuntimeError("no threshold within tolerance on val; widen --tolerance explicitly")
    best = min(cands, key=lambda r: (r["avg_macs"], -float(r["accuracy"])))
    for r in rows:
        if r["calibration"] == best["calibration"] and r["threshold"] == best["threshold"]:
            r["selected"] = 1
    print(f"exit-3-only val acc {ref:.4f}; selected on VAL: calibration={best['calibration']} "
          f"threshold={best['threshold']} val acc {best['accuracy']} avg MACs {best['avg_macs']:,}")
    sel_T = tuple(temps) if best["calibration"] == "temperature" else (1.0, 1.0, 1.0)
    sel_thr = float(best["threshold"])

    # ------------------------------------------------------------------ real controller on test
    ctrl = AdaptiveONNXController(paths, sel_T, threads=1)
    xt, yt = data["test"]
    cp, ce = np.empty(len(yt), int), np.empty(len(yt), int)
    for i in tqdm(range(len(yt)), desc="controller per-frame (test)"):
        cp[i], ce[i], _ = ctrl.predict(xt[i:i + 1], sel_thr)
    vp, ve = simulate_exits(logits["test"], sel_thr, sel_T)
    same = float((cp == vp).mean()), float((ce == ve).mean())
    print(f"per-frame controller vs vectorised sweep on test: pred agreement {same[0]:.5f}, exit agreement {same[1]:.5f}")
    assert min(same) > 0.999, "controller disagrees with sweep"
    mc = classification_metrics(yt, cp)
    print(f"controller test acc {mc['accuracy']:.4f} macro-F1 {mc['macro_f1']:.4f}")
    per_class_rows.append({"model": f"{a.tag}_selected", **{c: f"{v:.4f}" for c, v in zip(CLASSES, mc["per_class"])}})
    per_class_rows.append({"model": f"{a.tag}_selected_exit1_share",
                           **{c: f"{(ce[yt == i] == 1).mean():.4f}" for i, c in enumerate(CLASSES)}})
    per_class_rows.append({"model": f"{a.tag}_selected_avg_exit",
                           **{c: f"{ce[yt == i].mean():.3f}" for i, c in enumerate(CLASSES)}})
    rows.append({"split": "test", "calibration": best["calibration"], "threshold": f"{sel_thr}_per_frame_controller",
                 "accuracy": f"{mc['accuracy']:.4f}", "macro_f1": f"{mc['macro_f1']:.4f}",
                 "avg_macs": int(round(sum((ce == k).mean() * cum[k - 1] for k in (1, 2, 3)))),
                 "avg_macs_vs_baseline": "", "share_exit1": f"{(ce == 1).mean():.4f}",
                 "share_exit2": f"{(ce == 2).mean():.4f}", "share_exit3": f"{(ce == 3).mean():.4f}",
                 "premature_exit_rate": "", "worst_class": CLASSES[int(np.nanargmin(mc["per_class"]))],
                 "worst_class_acc": f"{np.nanmin(mc['per_class']):.4f}", "selected": 0})
    write_csv(out["adaptive_results"], rows, a.force)
    write_csv(out["exit_distribution"], dist_rows, a.force)
    write_csv(out["per_class_results"], per_class_rows, a.force)

    # ------------------------------------------------------------------ slide-43 failure modes
    sv = next(r for r in rows if r["split"] == "val" and r["selected"] == 1)
    st = next(r for r in rows if r["split"] == "test" and r["selected"] == 1)
    ex_shares = [float(st[f"share_exit{k}"]) for k in (1, 2, 3)]
    p25 = macs_static["pruned_25"]
    ws = {r["check"]: r["result"] for r in read_csv(os.path.join(RESULT_DIR, "weight_sharing.csv"))}
    fm = [
        {"failure_mode": "premature exit", "metric": "test frames exiting early WRONG although exit 3 is right",
         "value": st["premature_exit_rate"]},
        {"failure_mode": "premature exit", "metric": "test accuracy loss vs exit-3-only (pts)",
         "value": f"{100 * (float(next(r for r in rows if r['split'] == 'test' and r['threshold'] == 'exit3_only')['accuracy']) - float(st['accuracy'])):.2f}"},
        {"failure_mode": "exit imbalance", "metric": "test exit shares (1/2/3)", "value": "/".join(f"{s:.3f}" for s in ex_shares)},
        {"failure_mode": "exit imbalance", "metric": "max single-exit share", "value": f"{max(ex_shares):.3f}"},
        {"failure_mode": "calibration drift", "metric": "exit-1 share val -> test", "value": f"{sv['share_exit1']} -> {st['share_exit1']}"},
        {"failure_mode": "calibration drift", "metric": "accuracy val -> test", "value": f"{sv['accuracy']} -> {st['accuracy']}"},
        {"failure_mode": "calibration drift", "metric": "ECE exit1/2/3 raw val",
         "value": "/".join(r["ece_raw"] for r in calib_rows if r["split"] == "val")},
        {"failure_mode": "calibration drift", "metric": "ECE exit1/2/3 temperature val -> test",
         "value": "/".join(r["ece_temperature"] for r in calib_rows if r["split"] == "val") + " -> "
                  + "/".join(r["ece_temperature"] for r in calib_rows if r["split"] == "test")},
        {"failure_mode": "aux-head overhead", "metric": "MACs exit3 path / pruned_25 MACs", "value": f"{cum[2] / p25:.3f}"},
        {"failure_mode": "aux-head overhead", "metric": "params 3-exit model / pruned_25",
         "value": f"{int(ws['params of whole 3-exit model (conv w + BN affine + linear)']) / int(ws['params of the 25%-pruned static model (for overhead)']):.3f}"},
    ]
    write_csv(out["failure_modes"], fm, a.force)
    for r in fm:
        print(f"  [{r['failure_mode']}] {r['metric']}: {r['value']}")


if __name__ == "__main__":
    main()
    hard_exit()
