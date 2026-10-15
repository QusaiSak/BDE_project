"""SIMULATED battery-powered roadside camera. Every simulated frame is a real image passed
through the real ONNX model / AdaptiveONNXController (no precomputed logits, no cache;
--memo is an opt-in in-process memo and defaults OFF).

Energy per frame = MACs actually executed (exit reached) x --nj-per-mac
                 + fixed sensor/ISP/idle energy (--frame-mj)
                 + controller overhead (measured ms in benchmark_results.csv x --cpu-w)
All energy constants are ASSUMED, not measured on hardware.

Policies (same battery, same frame order, same seeds):
  1 static baseline   2 static pruned_25   3 static pruned_50
  4 adaptive, fixed val-selected threshold
  5 adaptive, battery-aware threshold  thr(b) = T_MIN + (T_MAX - T_MIN) * b  (+ optional exit cap)
  6 switching between separate ONNX models by battery level, with a switching cost (slide 67)
  7 adaptive, lifetime-target ("budget") controller: every frame it divides the energy it can
    still use (battery + forecast solar harvest) by the frames left until the required lifetime
    T, and picks the threshold whose expected MACs (measured on VALIDATION) fit that budget.
    No tuned parameters; it is evaluated as a MISSION: run T hours, frames after the battery
    dies count as missed, score = correct decisions / frames required in T.

Step 1 (VALIDATION stream): grid over T_MIN / T_MAX / cap -> choose the battery-aware policy.
Step 2 (TEST stream, >= 20 seeds): all policies -> summary, log, plots.
Step 3 (TEST stream, same seeds): mission evaluation for several required lifetimes T,
        statics / fixed / linear vs the lifetime-target controller -> simulation_mission.csv.
Outputs go to results/improved/simulation_battery/ (or simulation_solar/ with --solar).

  python simulate_edge_stream.py [--seeds 20] [--battery-j 500] [--fps 0.5] [--solar] [--force]
"""
import argparse
import itertools
import os
import time
from multiprocessing import get_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import (CLASSES, IMPROVED_DIR, MODEL_DIR, RESULT_DIR, AdaptiveONNXController, adaptive_paths, classification_metrics,
                   file_size_mb, guard_overwrite, hard_exit, load_exit_macs, load_selected_config, load_split,
                   load_stats, load_temperatures, make_session, normalize, onnx_exit_logits, read_csv, simulate_exits, write_csv, STATIC)

BINS = np.linspace(0, 1, 11)
G = {}  # per-worker state


# --------------------------------------------------------------------------- worker
def init_worker(split, memo):
    mean, std, _ = load_stats()
    xu, y = load_split(split)
    G.update(x=normalize(xu, mean, std), y=y, memo={} if memo else None,
             sess={n: make_session(os.path.join(MODEL_DIR, p), threads=1) for n, p in STATIC.items()},
             macs={r["model"]: int(r["macs"]) for r in read_csv(os.path.join(RESULT_DIR, "pruning_results.csv"))
                   if r["split"] == "test"},
             mb={n: file_size_mb(os.path.join(MODEL_DIR, p)) for n, p in STATIC.items()},
             cum=load_exit_macs())
    thr, use_t = load_selected_config()
    G["temps"] = load_temperatures() if use_t else (1.0, 1.0, 1.0)
    G["ctrl"] = AdaptiveONNXController(adaptive_paths(), G["temps"], threads=1)


def run_static(name, i):
    s = G["sess"][name]
    return int(s.run(None, {"image": G["x"][i:i + 1]})[0].argmax()), G["macs"][name]


def run_adaptive(i, thr, max_exit):
    pred, ex, confs = G["ctrl"].predict(G["x"][i:i + 1], thr, max_exit)
    return pred, ex, confs, G["cum"][ex - 1]


def solar_w(t_s, c):
    """Solar power (W) at time t_s: half a sine wave from 06:00 to 18:00, zero at night."""
    if not c["solar"]:
        return 0.0
    h = (c["start_hour"] + t_s / 3600.0) % 24
    return c["solar_peak_w"] * max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0


def budget_threshold(budget_macs, c):
    """Highest threshold whose expected MACs per frame (validation curve) fit the budget."""
    g, m = c["curve_thr"], c["curve_macs"]
    k = int(np.searchsorted(m, budget_macs, side="right")) - 1
    return float(g[max(k, 0)])


def simulate(task):
    """One (policy, seed) run until the battery cannot pay for the next frame."""
    pol, cfg, seed, c, keep_log = task
    rng = np.random.default_rng(seed)
    n = len(G["y"])
    dt = 1.0 / c["fps"]
    max_frames = int(c["max_hours"] * 3600 * c["fps"])
    if pol == "adaptive_budget":  # a mission of exactly target_h hours
        max_frames = int(round(cfg["target_h"] * 3600 * c["fps"]))
        harvest_cum = np.concatenate([[0.0], np.cumsum([solar_w(f * dt, c) * dt for f in range(max_frames)])])
    E0 = E = c["battery_j"]
    over_j = c["ctrl_overhead_ms"] * 1e-3 * c["cpu_w"]
    order = np.concatenate([rng.permutation(n) for _ in range(max_frames // n + 1)])[:max_frames]
    preds, trues, exits, bat, macs_l, thr_l, log = [], [], [], [], [], [], []
    current, switches, t, spent = None, 0, 0.0, 0.0
    for f, i in enumerate(order):
        b = E / E0  # battery fraction 0..1
        # each policy decides which model / threshold to use for this frame
        thr, ex, confs = np.nan, 0, []
        if pol in STATIC:
            pred, macs = run_static(pol, i)
            extra = 0.0
        elif pol == "switching":
            # bag of models: pick a model by battery band; loading a new one costs energy
            name ="baseline" if b > cfg["hi"] else "pruned_25" if b > cfg["lo"] else "pruned_50"
            extra = 0.0
            if current is not None and name != current:
                extra = G["mb"][name] * c["switch_j_per_mb"]
                switches += 1
            current = name
            pred, macs = run_static(name, i)
        else:
            if pol == "adaptive_fixed":
                thr, cap = cfg["thr"], 3
            elif pol == "adaptive_budget":
                # battery + forecast harvest until T, minus one worst-case frame kept in reserve
                avail = E + harvest_cum[max_frames] - harvest_cum[f] - c["reserve_j"]
                budget_j = avail / (max_frames - f)
                budget_macs = (budget_j - c["frame_mj"] * 1e-3 - over_j) / (c["nj_per_mac"] * 1e-9)
                thr, cap = budget_threshold(budget_macs, c), 3
            else:
                # battery-aware: thr(b) = T_MIN + (T_MAX - T_MIN) * b  -> low battery, lower thr, earlier exits
                thr = cfg["t_min"] + (cfg["t_max"] - cfg["t_min"]) * b
                cap = cfg["cap_exit"] if b < cfg["cap_below"] else 3
            key = (round(thr, 4), cap, int(i))
            if G["memo"] is not None and key in G["memo"]:
                pred, ex, confs, macs = G["memo"][key]
            else:
                pred, ex, confs, macs = run_adaptive(i, thr, cap)
                if G["memo"] is not None:
                    G["memo"][key] = (pred, ex, confs, macs)
            extra = over_j  # measured controller overhead per frame
        # energy model (ASSUMED constants): compute + fixed sensor/idle + controller/switch overhead
        cost = macs * c["nj_per_mac"] * 1e-9 + c["frame_mj"] * 1e-3 + extra
        if cost > E:
            break
        spent += cost
        # solar harvest refills the battery, capped at full charge (surplus is wasted)
        E = min(E0, E - cost + solar_w(t, c) * dt)
        t += dt
        y = int(G["y"][i])
        preds.append(pred), trues.append(y), exits.append(ex), bat.append(b), macs_l.append(macs), thr_l.append(thr)
        if keep_log:
            log.append({"policy": pol, "seed": seed, "frame": f, "t_s": f"{t:.1f}", "battery_frac": f"{b:.4f}",
                        "threshold": "" if np.isnan(thr) else f"{thr:.4f}", "exit": ex if ex else "",
                        "model": current if pol == "switching" else (pol if pol in STATIC else "adaptive"),
                        "conf1": f"{confs[0]:.4f}" if len(confs) > 0 else "",
                        "conf2": f"{confs[1]:.4f}" if len(confs) > 1 else "",
                        "conf3": f"{confs[2]:.4f}" if len(confs) > 2 else "",
                        "macs": macs, "energy_j": f"{cost:.5f}", "test_index": int(i), "true": CLASSES[y],
                        "pred": CLASSES[pred], "correct": int(pred == y)})
    preds, trues, exits, bat = map(np.asarray, (preds, trues, exits, bat))
    m = classification_metrics(trues, preds)
    correct = int((preds == trues).sum())
    res = {"policy": pol, "seed": seed, "frames": len(preds), "lifetime_h": len(preds) * dt / 3600,
           "alive_at_cap": int(len(preds) == max_frames), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
           "correct_per_charge": correct, "energy_used_j": spent, "j_per_correct": spent / max(correct, 1),
           "avg_macs": float(np.mean(macs_l)), "switches": switches,
           "share_exit1": float((exits == 1).mean()), "share_exit2": float((exits == 2).mean()),
           "share_exit3": float((exits == 3).mean()), "_corr": (preds == trues).astype(np.uint8)}
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (bat > lo) & (bat <= hi) if lo > 0 else (bat >= lo) & (bat <= hi)
        if s.any():
            bins.append({"policy": pol, "seed": seed, "bin_lo": lo, "bin_hi": hi, "frames": int(s.sum()),
                         "accuracy": float((preds[s] == trues[s]).mean()),
                         "share_exit1": float((exits[s] == 1).mean()), "share_exit2": float((exits[s] == 2).mean()),
                         "share_exit3": float((exits[s] == 3).mean()),
                         "mean_threshold": float(np.nanmean(np.asarray(thr_l)[s])) if pol.startswith("adaptive") else ""})
    return res, bins, log


def run_tasks(tasks, split, memo, workers, desc):
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=init_worker, initargs=(split, memo)) as pool:
        return list(tqdm(pool.imap(simulate, tasks), total=len(tasks), desc=desc))


def agg(results, keys):
    out = []
    pols = list(dict.fromkeys(r["policy_label"] for r in results))
    for p in pols:
        rs = [r for r in results if r["policy_label"] == p]
        row = {"policy": p, "n_seeds": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs], dtype=float)
            row[f"{k}_mean"], row[f"{k}_std"] = f"{v.mean():.5f}", f"{v.std():.5f}"
        out.append(row)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--tune-seeds", type=int, default=5)
    ap.add_argument("--battery-j", type=float, default=500.0, help="ASSUMED usable charge (J), scaled-down")
    ap.add_argument("--nj-per-mac", type=float, default=1.0, help="ASSUMED compute energy (nJ/MAC)")
    ap.add_argument("--frame-mj", type=float, default=30.0, help="ASSUMED sensor+ISP+idle energy per frame (mJ)")
    ap.add_argument("--cpu-w", type=float, default=1.0, help="ASSUMED CPU power to convert controller ms to J")
    ap.add_argument("--switch-j-per-mb", type=float, default=0.5, help="ASSUMED cost of loading a model (J/MB)")
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--solar", action="store_true")
    ap.add_argument("--solar-peak-w", type=float, default=0.06, help="ASSUMED peak harvest (W)")
    ap.add_argument("--start-hour", type=float, default=8.0)
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--acc-tolerance", type=float, default=0.01, help="val accuracy allowed below fixed policy")
    ap.add_argument("--memo", action="store_true", help="opt-in in-process memo (default OFF)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-suffix", default="", help="e.g. _solar: keeps other scenarios' outputs intact")
    ap.add_argument("--out-dir", default=None, help="default results/improved/simulation_{battery,solar}")
    ap.add_argument("--targets", type=float, nargs="+", default=None,
                    help="required lifetimes (h) for the mission evaluation; default depends on --solar")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    names = ["simulation_log", "simulation_summary", "simulation_runs", "simulation_battery_bins",
             "simulation_policy_sweep", "simulation_mission"]
    od = a.out_dir or os.path.join(IMPROVED_DIR, "simulation_solar" if a.solar else "simulation_battery")
    targets = a.targets or ([10, 12, 14, 16, 20, 24] if a.solar else [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    out = {k: os.path.join(od, f"{k}{a.out_suffix}.csv") for k in names}
    pngs = {k: os.path.join(od, f"{k}{a.out_suffix}.png")
            for k in ("simulation", "simulation_examples", "simulation_pareto", "simulation_mission")}
    for p in list(out.values()) + list(pngs.values()):
        guard_overwrite(p, a.force)

    bench = read_csv(os.path.join(IMPROVED_DIR, "benchmark_results.csv"))
    ctrl_ms = float(next(r for r in bench if r.get("selected_on_val") == "1")["controller_overhead_ms"])
    c = {"battery_j": a.battery_j, "nj_per_mac": a.nj_per_mac, "frame_mj": a.frame_mj, "cpu_w": a.cpu_w,
         "switch_j_per_mb": a.switch_j_per_mb, "fps": a.fps, "solar": a.solar, "solar_peak_w": a.solar_peak_w,
         "start_hour": a.start_hour, "max_hours": a.max_hours, "ctrl_overhead_ms": ctrl_ms}
    print("ASSUMED constants:", {k: v for k, v in c.items()})
    thr_sel, use_t = load_selected_config()
    # energy model of the budget controller: expected MACs per frame vs threshold, VALIDATION only
    mean, std, _ = load_stats()
    lv = onnx_exit_logits(adaptive_paths(), normalize(load_split("val")[0], mean, std))
    temps = load_temperatures() if use_t else (1.0, 1.0, 1.0)
    cum = np.array(load_exit_macs(), dtype=float)
    g = np.append(np.round(np.linspace(0.0, 1.0, 201), 3), 1.01)  # 1.01 = always exit 3
    m = np.array([cum[simulate_exits(lv, t, temps)[1] - 1].mean() for t in g])
    c["curve_thr"], c["curve_macs"] = g, np.maximum.accumulate(m)
    c["reserve_j"] = cum[-1] * c["nj_per_mac"] * 1e-9 + c["frame_mj"] * 1e-3 + ctrl_ms * 1e-3 * c["cpu_w"]

    # ---------------- step 1: tune the battery-aware policy on VALIDATION
    # 5 T_MIN x 3 T_MAX x 3 cap options = 45 configs. Keep the one with the most correct decisions
    # per charge whose val accuracy is within --acc-tolerance of the fixed-threshold policy.
    grid = []
    for t_min, t_max, (cap_exit, cap_below) in itertools.product([0.4, 0.5, 0.6, 0.7, 0.8], [0.9, 0.95, 0.99],
                                                                  [(3, 0.0), (2, 0.2), (1, 0.1)]):
        grid.append({"t_min": t_min, "t_max": t_max, "cap_exit": cap_exit, "cap_below": cap_below})
    base = [("baseline", {}), ("pruned_25", {}), ("pruned_50", {}), ("adaptive_fixed", {"thr": thr_sel})]
    tasks = [(p, cfg, 1000 + s, c, False) for p, cfg in base + [("adaptive_battery", g) for g in grid]
             for s in range(a.tune_seeds)]
    t0 = time.time()
    res = run_tasks(tasks, "val", a.memo, a.workers, "VAL sweep")
    for (p, cfg, _, _, _), (r, _, _) in zip(tasks, res):
        r["policy_label"] = p if p != "adaptive_battery" else \
            f"battery tmin={cfg['t_min']} tmax={cfg['t_max']} cap={cfg['cap_exit']}@{cfg['cap_below']}"
        r["cfg"] = cfg
    sweep = agg([r for r, _, _ in res], ["lifetime_h", "accuracy", "macro_f1", "correct_per_charge", "j_per_correct"])
    fixed_acc = float(next(r for r in sweep if r["policy"] == "adaptive_fixed")["accuracy_mean"])
    cands = [r for r in sweep if r["policy"].startswith("battery") and float(r["accuracy_mean"]) >= fixed_acc - a.acc_tolerance]
    chosen = max(cands, key=lambda r: float(r["correct_per_charge_mean"]))
    cfg_by_label = {r["policy_label"]: r["cfg"] for r, _, _ in res}
    best_cfg = cfg_by_label[chosen["policy"]]
    for r in sweep:
        r["split"] = "val"
        r["selected"] = int(r["policy"] == chosen["policy"])
    write_csv(out["simulation_policy_sweep"], sweep, a.force)
    print(f"VAL: fixed-threshold acc {fixed_acc:.4f}; selected battery-aware policy: {chosen['policy']} "
          f"(val acc {float(chosen['accuracy_mean']):.4f}, correct/charge {float(chosen['correct_per_charge_mean']):.0f}) "
          f"[{time.time() - t0:.0f}s]")

    # Pareto plot of the val sweep: lifetime vs accuracy
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bat = [r for r in sweep if r["policy"].startswith("battery")]
    L, A = np.array([float(r["lifetime_h_mean"]) for r in bat]), np.array([float(r["accuracy_mean"]) for r in bat])
    ax.scatter(L, A, s=18, c="#7f9cc4", label="battery-aware grid (T_MIN,T_MAX,cap)")
    pareto = [i for i in range(len(L)) if not any((L[j] >= L[i]) & (A[j] >= A[i]) & ((L[j] > L[i]) | (A[j] > A[i])) for j in range(len(L)))]
    pareto = sorted(pareto, key=lambda i: L[i])
    ax.plot(L[pareto], A[pareto], "-", c="#1f4e8c", lw=1.5, label="Pareto front (grid)")
    for r in sweep:
        if not r["policy"].startswith("battery"):
            ax.scatter(float(r["lifetime_h_mean"]), float(r["accuracy_mean"]), marker="s", s=50, label=r["policy"])
    ax.scatter(float(chosen["lifetime_h_mean"]), float(chosen["accuracy_mean"]), marker="*", s=220, c="red",
               zorder=5, label="selected (val)")
    ax.set(xlabel="lifetime per charge (h, SIMULATED)", ylabel="accuracy", title="Battery policy sweep on VALIDATION")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(pngs["simulation_pareto"], dpi=140)

    # ---------------- step 2: all policies on TEST, >= 20 seeds
    pols = [("baseline", {}), ("pruned_25", {}), ("pruned_50", {}), ("adaptive_fixed", {"thr": thr_sel}),
            ("adaptive_battery", best_cfg), ("switching", {"hi": 2 / 3, "lo": 1 / 3})]
    tasks = [(p, cfg, s, c, s == 0) for p, cfg in pols for s in range(a.seeds)]
    res = run_tasks(tasks, "test", a.memo, a.workers, "TEST policies")
    runs, bins, log = [], [], []
    for r, b, lg in res:
        r["policy_label"] = r["policy"]
        runs.append(r)
        bins += b
        log += lg
    keys = ["frames", "lifetime_h", "accuracy", "macro_f1", "correct_per_charge", "j_per_correct", "avg_macs",
            "share_exit1", "share_exit2", "share_exit3", "switches"]
    summary = agg(runs, keys)
    mem = {"adaptive_fixed": file_size_mb(*adaptive_paths()), "adaptive_battery": file_size_mb(*adaptive_paths()),
           "switching": sum(file_size_mb(os.path.join(MODEL_DIR, p)) for p in STATIC.values())}
    for r in summary:
        r["resident_model_mb"] = f"{mem.get(r['policy'], file_size_mb(os.path.join(MODEL_DIR, STATIC.get(r['policy'], 'baseline.onnx')))):.3f}"
        r["config"] = str(dict(pols)[r["policy"]])
        r["split"] = "test"
    write_csv(out["simulation_runs"], [{k: v for k, v in r.items() if k not in ("policy_label", "_corr")} for r in runs],
              a.force)
    write_csv(out["simulation_summary"], summary, a.force)
    write_csv(out["simulation_battery_bins"], bins, a.force)
    write_csv(out["simulation_log"], log, a.force)
    print("\nTEST (mean +/- std over %d seeds, SIMULATED battery):" % a.seeds)
    for r in summary:
        print(f"  {r['policy']:17s} life {float(r['lifetime_h_mean']):6.2f}+/-{float(r['lifetime_h_std']):.2f} h  "
              f"acc {float(r['accuracy_mean']):.4f}+/-{float(r['accuracy_std']):.4f}  "
              f"F1 {float(r['macro_f1_mean']):.4f}  correct/charge {float(r['correct_per_charge_mean']):7.0f}  "
              f"J/correct {float(r['j_per_correct_mean']):.4f}")

    # ---------------- plots
    S = {r["policy"]: r for r in summary}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for p, _ in pols:
        lg = [l for l in log if l["policy"] == p]
        ax[0, 0].plot([float(l["t_s"]) / 3600 for l in lg], [float(l["battery_frac"]) for l in lg], label=p)
    ax[0, 0].set(xlabel="time (h)", ylabel="battery fraction", title="Battery drain (seed 0, SIMULATED)")
    ax[0, 0].legend(fontsize=8)
    for p, mk in (("adaptive_battery", "o-"), ("adaptive_fixed", "s--")):
        bb = [b for b in bins if b["policy"] == p]
        centers = sorted({(b["bin_lo"] + b["bin_hi"]) / 2 for b in bb})
        for k, col in ((1, "#2a9d8f"), (3, "#e76f51")):
            ax[0, 1].plot(centers, [np.mean([b[f"share_exit{k}"] for b in bb if (b["bin_lo"] + b["bin_hi"]) / 2 == x]) for x in centers],
                          mk, c=col, label=f"{p} exit{k} share")
    ax[0, 1].set(xlabel="battery fraction", ylabel="share of frames", title="Exit share vs battery (test, all seeds)")
    ax[0, 1].legend(fontsize=7)
    for p, _ in pols:
        if p == "switching":  # per-band it just repeats a static model; kept in the summary only
            continue
        bb = [b for b in bins if b["policy"] == p]
        centers = sorted({(b["bin_lo"] + b["bin_hi"]) / 2 for b in bb})
        ax[1, 0].plot(centers, [np.mean([b["accuracy"] for b in bb if (b["bin_lo"] + b["bin_hi"]) / 2 == x]) for x in centers],
                      "o-", ms=3, label=p)
    ax[1, 0].set(xlabel="battery fraction", ylabel="accuracy", title="Accuracy vs battery (test, all seeds)")
    ax[1, 0].legend(fontsize=8)
    names_p = [p for p, _ in pols]
    xx = np.arange(len(names_p))
    ax2 = ax[1, 1].twinx()
    ax[1, 1].bar(xx - 0.2, [float(S[p]["correct_per_charge_mean"]) for p in names_p], 0.4,
                 yerr=[float(S[p]["correct_per_charge_std"]) for p in names_p], color="#457b9d", label="correct decisions / charge")
    ax2.bar(xx + 0.2, [float(S[p]["accuracy_mean"]) for p in names_p], 0.4,
            yerr=[float(S[p]["accuracy_std"]) for p in names_p], color="#e9c46a", label="accuracy")
    ax[1, 1].set_xticks(xx, names_p, rotation=25, fontsize=8)
    ax[1, 1].set(ylabel="correct decisions per charge", title=f"Per charge, mean+/-std over {a.seeds} seeds")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(min(float(S[p]["accuracy_mean"]) for p in names_p) - 0.05, 1.0)
    h1, l1 = ax[1, 1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1, 1].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.suptitle("SIMULATED battery (energy constants ASSUMED); every frame = real ONNX inference", fontsize=10)
    fig.tight_layout()
    fig.savefig(pngs["simulation"], dpi=140)

    # example frames from the battery-aware run: high vs low battery
    xu, _ = load_split("test")
    lg = [l for l in log if l["policy"] == "adaptive_battery"]
    hi = [l for l in lg if float(l["battery_frac"]) > 0.8][:6]
    lo = [l for l in lg if float(l["battery_frac"]) < 0.2][:6]
    fig, axs = plt.subplots(2, 6, figsize=(14, 5.4))
    for row, sel, tag in ((0, hi, "battery > 80%"), (1, lo, "battery < 20%")):
        for j in range(6):
            axx = axs[row, j]
            axx.axis("off")
            if j < len(sel):
                l = sel[j]
                axx.imshow(xu[int(l["test_index"])])
                axx.set_title(f"{tag}\nthr {l['threshold']} exit {l['exit']}\n{l['pred']} ({'OK' if l['correct'] == 1 else 'true ' + l['true']})",
                              fontsize=7, color="green" if l["correct"] == 1 else "red")
    fig.suptitle("Battery-aware policy: same controller, different battery -> different threshold/exit (seed 0)")
    fig.tight_layout()
    fig.savefig(pngs["simulation_examples"], dpi=140)
    print("wrote", ", ".join(os.path.relpath(p) for k, p in pngs.items() if k != "simulation_mission"))

    # ---------------- step 3: mission evaluation (required lifetime T) on TEST, same seeds
    # Policies that do not know T are run once (step 2) and cut at T; the budget controller is run per T.
    fps = a.fps
    btasks = [("adaptive_budget", {"target_h": T}, s, c, False) for T in targets for s in range(a.seeds)]
    bres = run_tasks(btasks, "test", a.memo, a.workers, "TEST missions")
    by = {}
    for r in runs:
        by.setdefault(r["policy"], []).append(r)
    mission = []
    for T in targets:
        need = int(round(T * 3600 * fps))
        rows_T = {}
        for p, rs in list(by.items()) + [("adaptive_budget", [r for (_, cfg, _, _, _), (r, _, _) in zip(btasks, bres)
                                                               if cfg["target_h"] == T])]:
            # mission score: correct decisions within the first `need` frames / need (dead = missed)
            sc = np.array([r["_corr"][:need].sum() / need for r in rs])
            cov = np.array([min(r["frames"], need) / need for r in rs])
            alive = np.array([r["_corr"][:need].mean() for r in rs])
            rows_T[p] = {"target_h": T, "policy": p, "n_seeds": len(rs),
                         "mission_score_mean": f"{sc.mean():.5f}", "mission_score_std": f"{sc.std():.5f}",
                         "coverage_mean": f"{cov.mean():.5f}", "survived_share": f"{(cov >= 1.0).mean():.2f}",
                         "accuracy_while_alive_mean": f"{alive.mean():.5f}", "_sc": sc.mean()}
        best_static = max(("baseline", "pruned_25", "pruned_50"), key=lambda p: rows_T[p]["_sc"])
        for p, r in rows_T.items():
            r["best_static"] = best_static
            r["delta_vs_best_static_pts"] = f"{100 * (r['_sc'] - rows_T[best_static]['_sc']):+.2f}"
            mission.append({k: v for k, v in r.items() if k != "_sc"})
    write_csv(out["simulation_mission"], mission, a.force)
    print("\nMISSION (TEST): score = correct decisions / frames required in T (dead frames count as missed)")
    for T in targets:
        rT = {r["policy"]: r for r in mission if r["target_h"] == T}
        print(f"  T={T:5.1f} h  " + "  ".join(f"{p}:{float(rT[p]['mission_score_mean']):.3f}" for p in rT)
              + f"  | budget vs best static ({rT['adaptive_budget']['best_static']}): "
                f"{rT['adaptive_budget']['delta_vs_best_static_pts']} pts")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5))
    for p in list(by) + ["adaptive_budget"]:
        if p == "switching":
            continue
        rs = [r for r in mission if r["policy"] == p]
        st = dict(lw=2.5, c="k", marker="*", ms=10) if p == "adaptive_budget" else dict(marker="o", ms=4)
        ax.plot([r["target_h"] for r in rs], [float(r["mission_score_mean"]) for r in rs], label=p, **st)
        bx.plot([r["target_h"] for r in rs], [float(r["coverage_mean"]) for r in rs], label=p, **st)
    ax.set(xlabel="required lifetime T (h)", ylabel="correct decisions / frames required",
           title="Mission score (TEST, mean over seeds)")
    bx.set(xlabel="required lifetime T (h)", ylabel="share of the mission the camera was alive", title="Coverage")
    for z in (ax, bx):
        z.grid(alpha=0.3)
        z.legend(fontsize=8)
    fig.suptitle("SIMULATED " + ("battery + solar" if a.solar else "battery only") + " (energy constants ASSUMED)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(pngs["simulation_mission"], dpi=140)

    # ---------------- sanity checks
    cal = [r for r in read_csv(os.path.join(RESULT_DIR, "calibration_results.csv")) if r["split"] == "test"]
    ea = [float(r["accuracy"]) for r in sorted(cal, key=lambda r: r["exit"])]
    print("\nSANITY exit accuracy exit1<exit2<exit3 (test):", "PASS" if ea[0] < ea[1] < ea[2] else "FAIL", ea)
    bb = [b for b in bins if b["policy"] == "adaptive_battery"]
    low = np.mean([b["share_exit1"] for b in bb if b["bin_hi"] <= 0.35])
    high = np.mean([b["share_exit1"] for b in bb if b["bin_lo"] >= 0.65])
    print(f"SANITY exit-1 share falls as battery rises: low-battery {low:.3f} vs high-battery {high:.3f}:",
          "PASS" if low > high else "FAIL")
    life = [float(S[p]["lifetime_h_mean"]) for p in ("baseline", "pruned_25", "pruned_50")]
    print("SANITY static lifetime ordering follows MACs (baseline < pruned_25 < pruned_50):",
          "PASS" if life[0] < life[1] < life[2] else "FAIL", [round(v, 2) for v in life])


if __name__ == "__main__":
    main()
    hard_exit()
