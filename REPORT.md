# Battery-Aware Adaptive Vehicle Classification for Traffic Cameras at the Edge

Big Data at the Edge, adaptive inference project. All numbers below are copied from `results/*.csv`
(ONNX Runtime, CPU). The battery is **SIMULATED**, and every energy constant is **ASSUMED**, not
measured on hardware. Anything not run is marked *not yet measured*.

> **Status of numbers.** This report reflects the run with the 5-threshold sweep
> (0.70–0.99) and the drain-only battery. The code has since been changed (sweep now 0.50–0.99,
> legend fix, `--solar` scenario with `--out-suffix`). Sections marked **[re-run pending]**
> will change after re-running `evaluate_models → benchmark_models → generate_results →
> simulate_edge_stream`, and `simulate_edge_stream --solar --out-suffix _solar`.

## 1. Scenario and research question
A roadside camera powered by a battery (optionally solar) classifies road users into 11
MIO-TCD classes. Two things vary at runtime:
- **input difficulty**: background and pedestrians are easy; similar trucks and vans are hard;
- **system budget**: remaining battery.

Slide 52 recommends combining both knobs when both vary.

**Question:** does confidence-based early exit on a pruned shared backbone, with a
battery-aware threshold, beat static compression on accuracy-vs-energy, and is the extra
complexity worth it?

## 2. Data
MIO-TCD-Classification, labeled `train/` only (519,164 crops; the 129,795 test crops have no labels and are unused).
- Capped at 6,000 per class. Stratified 70/10/20 split, seed 42: 37,196 / 5,313 / 10,628.
- Images resized to 64×64. Mean and std computed on train only (`data/stats.csv`).
- Augmentation: random crop (pad 4) plus horizontal flip. Flipping is class-safe for road users.
- Accuracy and macro-F1 are both reported, because the classes are imbalanced (1,751–6,000 per class).

## 3. Toolbox choices
| Family | Choice | Why |
|---|---|---|
| Compression (3.1) | Structured channel pruning (torch-pruning DepGraph, L2 magnitude), 25% and 50% | Physically smaller tensors, so real latency gains on CPU |
| Adaptive (3.2) | 3-exit early-exit network on the shared 25%-pruned backbone | Per-frame difficulty varies a lot in traffic scenes |
| Resource awareness | Battery-aware exit threshold | Budgeted inference driven by the battery (slide 25) |

**Baseline:** torchvision MobileNetV2, random init.
- The stem stride was changed from 2 to 1, so a 64×64 input ends on a 4×4 map instead of 2×2. This keeps spatial detail for the small crops.
- Cost: 102.19 M MACs, 2.24 M params (fvcore, CPU, batch 1, 64×64; the same method is used for every point).
- Training: SGD with Nesterov momentum, cosine LR, label smoothing 0.1, 50 epochs, class-balanced CE.
- *The plain-vs-balanced comparison on val was not run (the baseline was trained with `--balanced on`).*

**Pruning:** each model starts from `baseline.onnx`.
- An ONNX→PyTorch loader rebuilds the network from the folded Conv weights (BN set to an exact identity) and reproduces the baseline val accuracy exactly (tolerance 1e-4).
- BN statistics are then re-estimated without changing the function, followed by 15 fine-tuning epochs at lr 0.01.

**Early-exit design:**
- Exits sit after `features[3]` (24→18 channels, 32×32), after `features[10]` (48 channels, 8×8), and at the end.
- Channel counts were probed with a dummy tensor, not read from metadata.
- Each head is a 1×1 conv → BN → ReLU6 → avgpool → linear. Exit 3 is the original classifier.
- Training: joint loss 0.5/0.5/1.0, 30 epochs.
- Epoch selection uses the **weighted mean of all exit val accuracies** (0.25/0.25/0.5), not exit 3 alone, so exit 1 is not sacrificed.

**Runtime loop (slide 13):**
1. SENSE: a frame arrives.
2. ESTIMATE: compute the softmax confidence at the current exit.
3. CHOOSE: stop if confidence ≥ threshold(battery).
4. EXECUTE: otherwise run only the next stage on this frame's feature map.
5. UPDATE: log the exit, the energy used and the battery level.

This is implemented as `AdaptiveONNXController` (`utils.py`) over three chained ONNX graphs.

## 4. Shared weights (`results/weight_sharing.csv`)
- The exit modules reference the same Parameter objects and the same memory (`data_ptr`) as the trained network. Check: PASS.
- The three ONNX files are a **disjoint partition** of one network: 13,127 + 135,071 + 1,130,651 = 1,278,849 weights. Exit 2 and exit 3 continue from the previous exit's feature map, so nothing is duplicated.
- Overhead vs the static pruned_25 model: params ×1.009, full-path MACs ×1.053 (the cost of the aux heads, slide 43).

## 5. Operating points (TEST) [re-run pending]
| Point | Avg MACs | ×baseline | Acc | Macro-F1 | Mean / p95 latency (ms) |
|---|---|---|---|---|---|
| baseline | 102.19 M | 1.000 | 0.8815 | 0.8733 | 2.170 / 2.271 |
| pruned_25 | 60.24 M | 0.589 | 0.8362 | 0.8200 | 1.626 / 1.683 |
| pruned_50 | 29.22 M | 0.286 | 0.7826 | 0.7507 | 1.034 / 1.058 |
| adaptive exit 1 only | 19.72 M | 0.193 | 0.4991 | 0.3989 | 0.658 / 0.674 |
| adaptive exit 2 only | 35.41 M | 0.347 | 0.8064 | 0.7764 | 1.107 / 1.143 |
| adaptive exit 3 only | 63.41 M | 0.621 | 0.8695 | 0.8564 | 1.694 / 1.758 |
| **adaptive, thr 0.70 (val-selected)** | **46.71 M** | **0.457** | **0.8649** | **0.8515** | **1.336 / 1.712** |

- Selection (val only): pick the cheapest threshold whose val accuracy is within 1 point of exit-3-only (0.8635). The result was raw confidence at 0.70 (val 0.8577). 0.70 was the lowest value tested, so the sweep has been extended to 0.50/0.60.
- The per-frame controller reproduces the vectorised sweep on 100% of test frames.
- Controller overhead: 0.03 ms per frame (slide 34: overhead counted).

### Pareto interpretation (`results/pareto_frontier.png`)
- **Adaptive dominates pruned_25:** +2.9 points at 22% fewer MACs. At equal accuracy, temperature-calibrated threshold 0.7 needs 39.8 M MACs, 34% less than pruned_25.
- **It does not reach the baseline:** the best adaptive point is 1.2 points below it, at 0.59× the MACs. The 25% pruning caps the ceiling.
- **Below about 30 M MACs, pruned_50 wins:** exit 1 alone is far too weak (49.9%). The frontier from 30 to 63 M MACs is entirely adaptive points, and both ends are static or fixed-depth points.

## 6. Exit behaviour and slide-43 failure modes (`failure_modes.csv`, `per_class_results.csv`)
- **Exit distribution (test, selected):** 11.1% / 42.3% / 46.6%. No single exit takes most frames, so there is no exit-imbalance problem.
- **Which classes exit at exit 1:** 50.5% of background and 33.6% of pedestrian frames, but under 1% of car, pickup and single-unit truck frames. The controller spends compute where the classes are visually similar.
- **Premature exit:** 0.72% of test frames exit early with a wrong answer that exit 3 would have got right. The total accuracy cost vs exit 3 is 0.46 points.
- **Calibration:** label smoothing makes the exits under-confident (val ECE 0.114 / 0.105 / 0.044 raw). Temperatures fitted on val (0.66 / 0.68 / 0.83) bring ECE down to 0.013 / 0.013 / 0.025. **No drift:** test ECE is 0.013 / 0.009 / 0.025, and the exit-1 share is 0.111 on val vs 0.111 on test.
- **Tail and per-class results:** the worst class everywhere is non-motorized vehicle (baseline 0.74, adaptive 0.59, pruned_50 0.25). Exit 1 never predicts it correctly (0.00) and gets motorcycle right 2% of the time.

## 7. Battery simulation [re-run pending]
**Setup:**
- Every frame is a real test image run through the real ONNX model or controller; there is no logit cache.
- Battery 500 J, 0.5 fps, 1 nJ/MAC, 30 mJ per frame for sensor/ISP/idle, controller overhead = measured ms × 1 W (all ASSUMED).
- 20 seeds; every policy sees the same frame order.
- The battery-aware rule is thr(b) = T_MIN + (T_MAX − T_MIN)·b. It was tuned on the **val** stream (5 seeds, 45 configurations) by maximising correct decisions per charge while keeping val accuracy within 1 point of the fixed policy. The selection was T_MIN 0.5, T_MAX 0.9, no cap.

| Policy (test, mean ± std, 20 seeds) | Lifetime (h) | Accuracy | Macro-F1 | Correct / charge | J / correct |
|---|---|---|---|---|---|
| static baseline | 2.10 ± 0.00 | 0.882 ± 0.004 | 0.874 | 3336 | 0.150 |
| static pruned_25 | 3.08 ± 0.00 | 0.837 ± 0.004 | 0.821 | 4635 | 0.108 |
| static pruned_50 | 4.69 ± 0.00 | 0.783 ± 0.002 | 0.751 | **6608** | **0.076** |
| adaptive, fixed thr | 3.62 ± 0.00 | 0.865 ± 0.002 | 0.852 | 5640 | 0.089 |
| adaptive, battery-aware | 3.63 ± 0.01 | 0.862 ± 0.002 | 0.848 | 5625 | 0.089 |
| switching separate models | 3.26 ± 0.00 | 0.821 ± 0.004 | 0.801 | 4821 | 0.104 |

**Sanity checks:** all PASS.
- Exit accuracy rises with depth.
- Exit-1 share falls as the battery rises, from 0.21 at low battery to 0.03 at full.
- Static lifetimes follow MACs.

**Honest reading:**
- **Adaptive vs static models:** adaptive gives 72% more lifetime than the baseline for −1.7 points of accuracy. It also gets more lifetime *and* more accuracy than pruned_25 and than the switching policy (slide 67: separate models also cost more memory, 15.6 MB resident vs 4.9 MB).
- **pruned_50 has the most correct decisions per charge.** It is the cheapest model, but it is 8 points less accurate and fails on non-motorized vehicles (0.25).
- **The battery-aware threshold gave no measurable gain over the fixed threshold** when the battery only drains. Lifetime depends only on the average energy per frame, and the battery-aware rule only moves compute in time: it spends more at high charge and less at low charge. The val constraint kept its average cost at the fixed policy's level. The adaptation itself works (exit-3 share goes from 0.26 to 0.72 as battery rises), but in a drain-only scenario that doesn't buy anything.
- **Next test:** with harvesting (`--solar`, ASSUMED 0.06 W peak, day/night cycle), a fixed policy either wastes the surplus when the battery is full or dies at night. That is where the battery-aware threshold should help. Results: *not yet measured* (`results/simulation_summary_solar.csv` after the run).

## 8. What did not work and how we responded
- **Earlier attempt:** selecting the epoch on exit-3 accuracy alone left exit 1 at about 34%. Switching to weighted-mean selection gave val accuracies 0.494 / 0.799 / 0.864, rising with depth. Exit 1 is still weak.
- **Exit-1 ablations** (split points 5/12, distillation from the baseline, which PDF §3.1 allows): *not yet measured*, left as future work.
- **Plain vs class-balanced loss on val:** *not yet measured*.
- **MPS training was slow:** about 98 s per epoch, and mixed precision and channels_last did not help (measured). This limited the number of ablations.
- **Drain-only battery-aware policy ≈ fixed threshold** (section 7).

## 9. Reflection (edge / big data)
- The camera sees a stream where most frames are easy. Early exit turns that skew into savings: 54% fewer MACs than the baseline, and 1.6× faster on one CPU thread.
- Adaptivity is not free: aux heads add +5% MACs on the full path. Dynamic paths raise the latency spread (p95 1.71 ms vs mean 1.34 ms), which matters if a hard deadline applies (slide 52: avoid adaptivity under strict worst-case latency).
- A static model is simpler and wins on raw energy per correct decision. The adaptive system is worth its complexity when accuracy above about 0.85 is required at under 0.5× the baseline's compute. That is the range where no static model in this study lives.

## 10. Presentation outline (grading weights)
1. **Scenario and motivation (Novelty 10%):** battery/solar roadside camera; two sources of variability.
2. **System design (Design 20%):** the slide-13 loop diagram, pruning + early exit, the battery threshold rule, why a battery-aware threshold rather than temporal gating (it needs no inter-frame state, and its decision cost is negligible).
3. **Shared weights (15%):** the 3-file disjoint partition, the `data_ptr` proof, +0.9% params.
4. **Correct adaptive behaviour (20%):** the controller-vs-sweep match, exit share per class, exit share vs battery plot, `simulation_examples.png`.
5. **Pareto frontier (15%):** `pareto_frontier.png`; adaptive dominates pruned_25 but loses to pruned_50 below 30 M MACs; latency panel.
6. **Trade-offs, failures, reflection (Presentation 20%):** slide-43 table, weak exit 1, battery-aware ≈ fixed when the battery only drains (plus the solar result), pruned_50 wins correct-per-charge, limitations (simulated energy, CPU-only, 64×64 crops not full frames).

## Reproduce
```
python data_prepare.py --root .
python train_baseline.py --balanced on   # as run; --balanced auto also compares plain CE
python train_pruned.py
python train_adaptive.py
python evaluate_models.py --force
python benchmark_models.py --force
python generate_results.py --force
python simulate_edge_stream.py --force
python simulate_edge_stream.py --solar --out-suffix _solar
```
