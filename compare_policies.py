"""Head-to-head comparison of the runtime policies in BOTH simulated scenarios
(battery only, battery + solar):

  baseline            static full MobileNetV2 (reference)
  adaptive_fixed      early exit, fixed val-selected threshold
  adaptive_battery    early exit, battery-aware threshold thr(b)
  switching           "bag of models": switch baseline / pruned_25 / pruned_50 by battery level
  adaptive_budget     early exit, lifetime-target controller (mission evaluation only)

pruned_25 / pruned_50 are kept in the table as static references.
Reads only the CSVs written by simulate_edge_stream.py (no inference, no randomness), so the
outputs are exactly reproducible. Writes to results/improved/comparison/:
  policy_comparison.csv / .md   table: mean +/- std over seeds and delta vs baseline
  comparison_bars.png           lifetime, accuracy, correct decisions/charge, J/correct
  comparison_tradeoff.png       lifetime vs accuracy per scenario
  comparison_mission.png        mission score vs required lifetime T per scenario

  python compare_policies.py [--force]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import IMPROVED_DIR, guard_overwrite, read_csv, write_csv

SCENARIOS = {"battery": "simulation_battery", "solar": "simulation_solar"}
FOCUS = ["baseline", "adaptive_fixed", "adaptive_battery", "switching"]
REFERENCE = ["pruned_25", "pruned_50"]
LABEL = {"baseline": "baseline", "adaptive_fixed": "adaptive (fixed thr)", "adaptive_battery": "adaptive (battery-aware)",
         "switching": "bag of models (switching)", "adaptive_budget": "adaptive (lifetime budget)",
         "pruned_25": "pruned_25", "pruned_50": "pruned_50"}
# fixed colour per policy (never by rank); scenarios get their own pair for the grouped bars
COLOR = {"baseline": "#2a78d6", "adaptive_fixed": "#1baf7a", "adaptive_battery": "#4a3aa7",
         "switching": "#eb6834", "adaptive_budget": "#e34948", "pruned_25": "#9a9a9a", "pruned_50": "#6b6b6b"}
SCEN_COLOR = {"battery": "#2a78d6", "solar": "#eda100"}
# (csv column, display name, higher is better?)
METRICS = [("lifetime_h", "lifetime per charge (h)", True), ("accuracy", "accuracy", True),
           ("macro_f1", "macro-F1", True), ("correct_per_charge", "correct decisions per charge", True),
           ("j_per_correct", "energy per correct decision (J)", False)]


def load(scen):
    d = os.path.join(IMPROVED_DIR, SCENARIOS[scen])
    summary = {r["policy"]: r for r in read_csv(os.path.join(d, "simulation_summary.csv"))}
    mission = read_csv(os.path.join(d, "simulation_mission.csv"))
    return summary, mission


def style_axis(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    od = os.path.join(IMPROVED_DIR, "comparison")
    out = {k: os.path.join(od, k) for k in ("policy_comparison.csv", "policy_comparison.md", "comparison_bars.png",
                                            "comparison_tradeoff.png", "comparison_mission.png")}
    for p in out.values():
        guard_overwrite(p, a.force)
    data = {s: load(s) for s in SCENARIOS}

    # ---- table: every metric as mean +/- std, plus relative change vs the static baseline
    rows = []
    for scen, (S, _) in data.items():
        base = S["baseline"]
        for p in FOCUS + REFERENCE:
            r = {"scenario": scen, "policy": p, "role": "compared" if p in FOCUS else "static reference",
                 "n_seeds": S[p]["n_seeds"]}
            for k, _, _ in METRICS:
                m, sd, b = float(S[p][f"{k}_mean"]), float(S[p][f"{k}_std"]), float(base[f"{k}_mean"])
                r[f"{k}_mean"], r[f"{k}_std"] = f"{m:.4f}", f"{sd:.4f}"
                # accuracy-type metrics as point differences, the others as % change
                r[f"{k}_vs_baseline"] = (f"{100 * (m - b):+.2f} pts" if k in ("accuracy", "macro_f1")
                                         else f"{100 * (m / b - 1):+.1f}%")
            r["avg_macs_M"] = f"{float(S[p]['avg_macs_mean']) / 1e6:.1f}"
            r["resident_model_mb"] = S[p]["resident_model_mb"]
            r["model_switches"] = f"{float(S[p]['switches_mean']):.0f}"
            rows.append(r)
    write_csv(out["policy_comparison.csv"], rows, a.force)

    md = ["# Policy comparison (TEST stream, mean ± std over seeds, SIMULATED energy)", ""]
    for scen in SCENARIOS:
        md += [f"## {scen.capitalize()} scenario", "",
               "| Policy | Lifetime (h) | Accuracy | Macro-F1 | Correct / charge | J / correct | Avg MACs (M) | Model MB | Switches |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["scenario"] == scen):
            cell = lambda k, nd: (f"{float(r[k + '_mean']):.{nd}f} ± {float(r[k + '_std']):.{nd}f}"
                                  + ("" if r["policy"] == "baseline" else f" ({r[k + '_vs_baseline']})"))
            name = LABEL[r["policy"]] + (" *(ref)*" if r["role"] != "compared" else "")
            md.append(f"| {name} | {cell('lifetime_h', 2)} | {cell('accuracy', 3)} | {cell('macro_f1', 3)} | "
                      f"{cell('correct_per_charge', 0)} | {cell('j_per_correct', 3)} | {r['avg_macs_M']} | "
                      f"{r['resident_model_mb']} | {r['model_switches']} |")
        md.append("")
        # mission table: score per required lifetime, incl. the budget controller
        _, mission = data[scen]
        Ts = sorted({float(m["target_h"]) for m in mission})
        pols = FOCUS + ["adaptive_budget"]
        md += [f"### Mission score, {scen} (correct decisions / frames required in T; dead frames count as missed)", "",
               "| T (h) | " + " | ".join(LABEL[p] for p in pols) + " |", "|---" * (len(pols) + 1) + "|"]
        for T in Ts:
            sc = {m["policy"]: float(m["mission_score_mean"]) for m in mission if float(m["target_h"]) == T}
            best = max(sc[p] for p in pols)
            md.append(f"| {T:g} | " + " | ".join(f"**{sc[p]:.3f}**" if sc[p] == best else f"{sc[p]:.3f}" for p in pols)
                      + " |")
        md.append("")
    with open(out["policy_comparison.md"], "w") as f:
        f.write("\n".join(md))
    print("wrote", os.path.relpath(out["policy_comparison.md"]))

    # ---- grouped bars: one small multiple per metric (no dual axes), battery vs solar side by side
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    x = np.arange(len(FOCUS))
    for ax, (k, title, hib) in zip(axes, [METRICS[0], METRICS[1], METRICS[3], METRICS[4]]):
        for j, (scen, (S, _)) in enumerate(data.items()):
            v = [float(S[p][f"{k}_mean"]) for p in FOCUS]
            e = [float(S[p][f"{k}_std"]) for p in FOCUS]
            ax.bar(x + (j - 0.5) * 0.38, v, 0.36, yerr=e, capsize=2, color=SCEN_COLOR[scen], edgecolor="white",
                   lw=1, label=scen)
        ax.set_xticks(x, [LABEL[p].replace(" (", "\n(") for p in FOCUS], fontsize=8)
        ax.set_title(title + (" ↑" if hib else " ↓"), fontsize=10)
        if k == "accuracy":  # zoom: differences are a few points
            lo = min(float(S[p]["accuracy_mean"]) for S, _ in data.values() for p in FOCUS)
            ax.set_ylim(lo - 0.03, 0.9)
        style_axis(ax)
    axes[0].legend(title="scenario", fontsize=8, frameon=False)
    fig.suptitle("Policies on the TEST stream (mean ± std over seeds; SIMULATED energy, constants ASSUMED)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out["comparison_bars.png"], dpi=150)
    plt.close(fig)

    # ---- trade-off: lifetime vs accuracy, one panel per scenario (lifetime scales differ)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (scen, (S, _)) in zip(axes, data.items()):
        for p in FOCUS + REFERENCE:
            lx, ly = float(S[p]["lifetime_h_mean"]), float(S[p]["accuracy_mean"])
            ax.errorbar(lx, ly, xerr=float(S[p]["lifetime_h_std"]), yerr=float(S[p]["accuracy_std"]),
                        fmt="o" if p in FOCUS else "s", ms=9 if p in FOCUS else 7, c=COLOR[p], capsize=2,
                        mec="white", mew=1.5, label=LABEL[p] + ("" if p in FOCUS else " (ref)"))
            # below-left for the two labels that would otherwise sit on a neighbour
            dy = -14 if p in ("adaptive_battery", "switching") else 5
            ax.annotate(LABEL[p], (lx, ly), fontsize=8, color="0.25", xytext=(-7 if p == "switching" else 7, dy),
                        textcoords="offset points", ha="right" if p == "switching" else "left")
        ax.set(xlabel="lifetime per charge (h)", ylabel="accuracy",
               title=f"{scen.capitalize()}: up and to the right is better")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.18)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out["comparison_tradeoff.png"], dpi=150)
    plt.close(fig)

    # ---- mission score vs required lifetime T
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (scen, (_, mission)) in zip(axes, data.items()):
        for p in FOCUS + ["adaptive_budget"]:
            ms = sorted((m for m in mission if m["policy"] == p), key=lambda m: float(m["target_h"]))
            ax.plot([float(m["target_h"]) for m in ms], [float(m["mission_score_mean"]) for m in ms], "o-", c=COLOR[p],
                    lw=2, ms=6, mec="white", label=LABEL[p])
        ax.set(xlabel="required lifetime T (h)", ylabel="correct decisions / frames required",
               title=f"Mission score, {scen}")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out["comparison_mission.png"], dpi=150)
    plt.close(fig)
    print("wrote", ", ".join(os.path.relpath(p) for k, p in out.items() if k.endswith(".png")))


if __name__ == "__main__":
    main()
