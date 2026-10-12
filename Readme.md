# Battery-Aware Adaptive Vehicle Classification for Traffic Cameras at the Edge

Big Data at the Edge, adaptive inference project. All numbers below are copied from `results/*.csv`
(ONNX Runtime, CPU). The battery is **SIMULATED**, and every energy constant is **ASSUMED**, not
measured on hardware. 

> **Status of numbers.** Every number below comes from the current run: threshold sweep 0.50–0.99,
> battery-only and battery + solar simulations, mission evaluation, and the policy comparison
> (`results/improved/comparison/`).

## 1. Scenario and research question

A roadside camera powered by a battery (optionally solar) classifies road users into 11
MIO-TCD classes. Two things vary at runtime:

- **input difficulty**: background and pedestrians are easy; similar trucks and vans are hard;
- **system budget**: remaining battery.

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


| Family             | Choice                                                                         | Why                                                      |
| ------------------ | ------------------------------------------------------------------------------ | -------------------------------------------------------- |
| Compression (3.1)  | Structured channel pruning (torch-pruning DepGraph, L2 magnitude), 25% and 50% | Physically smaller tensors, so real latency gains on CPU |
| Adaptive (3.2)     | 3-exit early-exit network on the shared 25%-pruned backbone                    | Per-frame difficulty varies a lot in traffic scenes      |
| Resource awareness | Battery-aware exit threshold                                                   | Budgeted inference driven by the battery (slide 25)      |


**Baseline:** torchvision MobileNetV2, random init.

- The stem stride was changed from 2 to 1, so a 64×64 input ends on a 4×4 map instead of 2×2. This keeps spatial detail for the small crops.
- Cost: 102.19 M MACs, 2.24 M params (fvcore, CPU, batch 1, 64×64; the same method is used for every point).
- Training: SGD with Nesterov momentum, cosine LR, label smoothing 0.1, 50 epochs, class-balanced CE.
- *The plain-vs-balanced comparison on val was not run (the baseline was trained with* `--balanced on`*).*

**Pruning:** each model starts from `baseline.onnx`.

- An ONNX→PyTorch loader rebuilds the network from the folded Conv weights (BN set to an exact identity) and reproduces the baseline val accuracy exactly (tolerance 1e-4).
- BN statistics are then re-estimated without changing the function, followed by 15 fine-tuning epochs at lr 0.01.

**Early-exit design:**

- Exits sit after `features[3]` (24→18 channels, 32×32), after `features[10]` (48 channels, 8×8), and at the end.
- Channel counts were probed with a dummy tensor, not read from metadata.
- Each head is a 1×1 conv → BN → ReLU6 → avgpool → linear. Exit 3 is the original classifier.
- Training: joint loss 0.5/0.5/1.0, 30 epochs.
- Epoch selection uses the **weighted mean of all exit val accuracies** (0.25/0.25/0.5), not exit 3 alone, so exit 1 is not sacrificed.

**Runtime loop:**

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



## 5. Operating points (TEST) (`results/improved/final_results.csv`)


| Point                                 | Avg MACs    | ×baseline | Acc        | Macro-F1   | Mean / p95 latency (ms) |
| ------------------------------------- | ----------- | --------- | ---------- | ---------- | ----------------------- |
| baseline                              | 102.19 M    | 1.000     | 0.8815     | 0.8733     | 2.121 / 2.168           |
| pruned_25                             | 60.24 M     | 0.589     | 0.8362     | 0.8200     | 1.587 / 1.617           |
| pruned_50                             | 29.22 M     | 0.286     | 0.7826     | 0.7507     | 1.002 / 1.016           |
| adaptive exit 1 only                  | 19.72 M     | 0.193     | 0.4991     | 0.3989     | 0.646 / 0.658           |
| adaptive exit 2 only                  | 35.41 M     | 0.347     | 0.8064     | 0.7764     | 1.086 / 1.106           |
| adaptive exit 3 only                  | 63.41 M     | 0.621     | 0.8695     | 0.8564     | 1.648 / 1.677           |
| **adaptive, thr 0.70 (val-selected)** | **46.71 M** | **0.457** | **0.8649** | **0.8515** | **1.300 / 1.662**       |


- Selection (val only): pick the cheapest threshold whose val accuracy is within 1 point of exit-3-only (0.8635). The result was raw confidence at 0.70 (val 0.8577). The sweep covers 0.50–0.99 for both raw and temperature-scaled confidence, so 0.70 is not an edge-of-grid artefact.
- The per-frame controller reproduces the vectorised sweep on 100% of test frames.
- Controller overhead: 0.01 ms per frame (slide 34: overhead counted).



### Pareto interpretation (`results/improved/pareto_frontier_clean.png`; all points: `pareto_frontier.png`)

Both figures are generated by `generate_results.py` from the CSVs only, so re-running gives identical images.
The y-axis is zoomed to 0.76–0.90; exit 1 alone (0.499) is drawn as an off-scale arrow.

- **Adaptive dominates pruned_25:** +2.9 points at 22% fewer MACs (selected point). At equal accuracy (within 0.5 pt), raw threshold 0.5 needs 38.5 M MACs, 36% less than pruned_25 (`matched_comparison.csv`).
- **It does not reach the baseline:** the best adaptive point is 1.2 points below it, at 0.59× the MACs. The 25% pruning caps the ceiling.
- **Below about 30 M MACs, pruned_50 wins:** exit 1 alone is far too weak (49.9%). From about 32 to 63 M MACs the frontier is made of adaptive points; both ends are static or fixed-depth points.



## 6. Exit behaviour and failure modes (`failure_modes.csv`, `per_class_results.csv`)

- **Exit distribution (test, selected):** 11.1% / 42.3% / 46.6%. No single exit takes most frames, so there is no exit-imbalance problem.
- **Which classes exit at exit 1:** 50.5% of background and 33.6% of pedestrian frames, but under 1% of car, pickup and single-unit truck frames. The controller spends compute where the classes are visually similar.
- **Premature exit:** 0.72% of test frames exit early with a wrong answer that exit 3 would have got right. The total accuracy cost vs exit 3 is 0.46 points.
- **Calibration:** label smoothing makes the exits under-confident (val ECE 0.114 / 0.105 / 0.044 raw). Temperatures fitted on val (0.66 / 0.68 / 0.83) bring ECE down to 0.013 / 0.013 / 0.025. **No drift:** test ECE is 0.013 / 0.009 / 0.025, and the exit-1 share is 0.111 on val vs 0.111 on test.
- **Tail and per-class results:** the worst class everywhere is non-motorized vehicle (baseline 0.74, adaptive 0.59, pruned_50 0.25). Exit 1 never predicts it correctly (0.00) and gets motorcycle right 2% of the time.



## 7. Battery simulation and policy comparison (`results/improved/simulation_*`, `results/improved/comparison/`)

**Setup:**

- Every frame is a real test image run through the real ONNX model or controller; there is no logit cache.
- Battery 500 J, 0.5 fps, 1 nJ/MAC, 30 mJ per frame for sensor/ISP/idle, controller overhead = measured ms × 1 W (all ASSUMED).
- Solar scenario: an ASSUMED 0.06 W peak with a sine day curve (06:00–18:00), starting at 08:00. Surplus beyond a full battery is wasted.
- 20 seeds; every policy sees the same frame order.
- **Policies:** the static baseline; adaptive with the fixed val-selected threshold; adaptive with the battery-aware threshold thr(b) = T_MIN + (T_MAX − T_MIN)·b; a **bag of models** that switches between baseline / pruned_25 / pruned_50 at 2/3 and 1/3 battery and pays a load cost (ASSUMED 0.5 J/MB); and, for missions only, a lifetime-budget controller. pruned_25 and pruned_50 are shown as static references.
- The battery-aware rule was tuned on the **val** stream (5 seeds, 45 configurations) by maximising correct decisions per charge while keeping val accuracy within 1 point of the fixed policy. Selected: battery-only T_MIN 0.5, T_MAX 0.9, no cap; solar T_MIN 0.4, T_MAX 0.9, exit cap 2 below 20% battery.

**Head-to-head (TEST, mean ± std over 20 seeds; Δ vs baseline)** (`policy_comparison.md`, `comparison_bars.png`, `comparison_tradeoff.png`)


| Scenario | Policy                    | Lifetime (h) | Accuracy             | Macro-F1 | Correct / charge | J / correct | Model MB |
| -------- | ------------------------- | ------------ | -------------------- | -------- | ---------------- | ----------- | -------- |
| battery  | baseline                  | 2.10         | 0.882 ± 0.004        | 0.874    | 3336             | 0.150       | 8.5      |
| battery  | adaptive, fixed           | 3.62 (+72%)  | 0.865 ± 0.002 (−1.7) | 0.852    | 5640 (+69%)      | 0.089       | 4.9      |
| battery  | adaptive, battery-aware   | 3.63 (+73%)  | 0.862 ± 0.002 (−2.0) | 0.848    | 5625 (+69%)      | 0.089       | 4.9      |
| battery  | bag of models (switching) | 3.26 (+55%)  | 0.821 ± 0.004 (−6.1) | 0.801    | 4821 (+45%)      | 0.104       | 15.6     |
| solar    | baseline                  | 8.19         | 0.881 ± 0.001        | 0.873    | 12993            | 0.150       | 8.5      |
| solar    | adaptive, fixed           | 12.35 (+51%) | 0.865 ± 0.001 (−1.7) | 0.852    | 19231 (+48%)     | 0.089       | 4.9      |
| solar    | adaptive, battery-aware   | 12.40 (+51%) | 0.862 ± 0.001 (−2.0) | 0.847    | 19226 (+48%)     | 0.095       | 4.9      |
| solar    | bag of models (switching) | 11.07 (+35%) | 0.836 ± 0.001 (−4.6) | 0.820    | 16655 (+28%)     | 0.122       | 15.6     |


References: pruned_25 3.08 h / 0.837 (solar 11.56 h), pruned_50 4.69 h / 0.783 (solar 13.73 h, the most correct decisions per charge in both scenarios).

**Mission score** (correct decisions / frames required in a lifetime T; frames after the battery dies count as missed; `comparison_mission.png`):


| Scenario, T  | baseline | adaptive fixed | adaptive battery-aware | bag of models | adaptive budget |
| ------------ | -------- | -------------- | ---------------------- | ------------- | --------------- |
| battery, 3 h | 0.618    | 0.866          | 0.865                  | 0.824         | **0.870**       |
| battery, 4 h | 0.463    | 0.783          | 0.781                  | 0.670         | **0.843**       |
| battery, 5 h | 0.371    | 0.627          | 0.625                  | 0.536         | **0.659**       |
| solar, 12 h  | 0.602    | 0.865          | 0.864                  | 0.771         | **0.869**       |
| solar, 16 h  | 0.451    | **0.668**      | 0.668                  | 0.578         | 0.565           |
| solar, 24 h  | 0.301    | **0.445**      | 0.445                  | 0.386         | 0.307           |


**Sanity checks:** all PASS.

- Exit accuracy rises with depth.
- Exit-1 share falls as the battery rises.
- Static lifetimes follow MACs.

**Honest reading:**

- **Adaptive vs baseline:** +72% lifetime (battery) / +51% (solar) for −1.7 points of accuracy.
- **Adaptive vs bag of models:** the shared early-exit network beats switching between separate models on lifetime *and* accuracy in both scenarios, and needs 4.9 MB resident instead of 15.6 MB (slide 67). Switching spends most of the charge on the expensive baseline while the battery is high, then drops to pruned_50; with solar it swaps models 58 times per run, which adds load energy (J/correct 0.122 vs 0.089).
- **Battery-aware vs fixed threshold:** no measurable gain in either scenario. Lifetime depends on the average energy per frame, and the val constraint keeps the battery-aware rule at the fixed policy's average cost; it only moves compute in time (the exit-1 share is higher at low battery). The assumed solar harvest is small relative to the load, so the battery rarely reaches full and there is little surplus to exploit.
- **Lifetime-budget controller:** best whenever the required lifetime is reachable (battery up to 5 h, solar up to 12–14 h). For solar missions of 16 h or more it is worse than the fixed threshold: its harvest forecast makes it spend energy in the day that the night then lacks. This is a limitation of the current forecast, not of early exit.
- **pruned_50 has the most correct decisions per charge.** It is the cheapest model, but it is 8 points less accurate and fails on non-motorized vehicles (0.25).



## 8. What did not work and how we responded

- **Earlier attempt:** selecting the epoch on exit-3 accuracy alone left exit 1 at about 34%. Switching to weighted-mean selection gave val accuracies 0.494 / 0.799 / 0.864, rising with depth. Exit 1 is still weak.
- **Exit-1 distillation from the baseline** was tried and made exit 1 worse (val 0.465 vs 0.494), so it was removed from the code. Split-point ablations: *not yet measured*.
- **MPS training was slow:** about 98 s per epoch, and mixed precision and channels_last did not help (measured). This limited the number of ablations.
- **Battery-aware threshold ≈ fixed threshold**, with and without solar (section 7).
- **Lifetime-budget controller over-spends on long solar missions** (≥ 16 h), section 7.



## 9. Reflection (edge / big data)

- The camera sees a stream where most frames are easy. Early exit turns that skew into savings: 54% fewer MACs than the baseline, and 1.6× faster on one CPU thread.
- Adaptivity is not free: aux heads add +5% MACs on the full path. Dynamic paths raise the latency spread (p95 1.66 ms vs mean 1.30 ms), which matters if a hard deadline applies (slide 52: avoid adaptivity under strict worst-case latency).
- A static model is simpler and wins on raw energy per correct decision. The adaptive system is worth its complexity when accuracy above about 0.85 is required at under 0.5× the baseline's compute. That is the range where no static model in this study lives.



## Reproduce

```
python data_prepare.py --root [data_dir]
python train_baseline.py --balanced on   # as run; --balanced auto also compares plain CE
python train_pruned.py
python train_adaptive.py
python evaluate_models.py --force
python benchmark_models.py --force
python generate_results.py --force
python simulate_edge_stream.py --force
python simulate_edge_stream.py --solar
python compare_policies.py --force       # baseline vs adaptive vs bag of models, both scenarios
```

