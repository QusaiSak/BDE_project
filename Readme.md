# Battery-Aware Early-Exit Vehicle Classification

A roadside camera on a battery (optionally solar) classifies road users into 11 MIO-TCD classes.
We compare a static MobileNetV2, pruned versions of it, and a 3-exit early-exit network whose
confidence threshold depends on remaining battery.

> Numbers are ONNX Runtime on CPU. The battery is **simulated** and all energy constants are **assumed**.

## Data

- MIO-TCD-Classification `train/`, capped at 6,000 per class.
- Stratified 70/10/20 split (seed 42): 37,196 / 5,313 / 10,628 images, resized to 64×64.

## Training flow

```
1. Baseline MobileNetV2        → sets the accuracy ceiling (frozen, never retrained)
          ↓
2. Prune 25% / 50% + fine-tune → static competitors to beat
          ↓
3. 3-exit model on pruned_25   → easy images leave early
          ↓
4. Temperature calibration     → honest confidence for the exit rule
          ↓
5. Threshold sweep 0.50–0.99   → pick operating point on VAL (no test leakage)
          ↓
6. Latency benchmark (ONNX)    → check fewer MACs = less real time
          ↓
7. Battery + solar simulation  → lifetime and correct predictions per charge
```

All costs are fvcore MACs at batch size 1, measured the same way for every model.

## Models trained


| Model                 | How                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Baseline              | MobileNetV2 (stem stride 1), random init, 50 epochs, SGD + cosine LR, label smoothing 0.1, class-balanced CE |
| pruned_25 / pruned_50 | Structured L2 channel pruning (torch-pruning) of the baseline, 15 fine-tune epochs                           |
| Adaptive (early exit) | 3 exits on the shared pruned_25 backbone, 30 epochs, joint loss                                              |




## Formulas

**Joint training loss** (3 exits):

```
L = 0.5·CE(exit1) + 0.5·CE(exit2) + 1.0·CE(exit3)
```

**Exit rule** (per frame): stop at exit *k* if

```
max softmax(z_k) ≥ threshold
```

**Battery-aware threshold** (b = battery fraction, 0..1):

```
thr(b) = T_MIN + (T_MAX − T_MIN) · b
```

Low battery → lower threshold → more frames exit early → less energy.

**Energy per frame** (assumed constants):

```
E = MACs · 1 nJ  +  30 mJ (sensor/idle)  +  controller_ms · 1 W
```

**Temperature scaling** (calibration):

```
p = softmax(z / T)
```



## Temperature scaling: why

The exit rule trusts the softmax confidence, so that confidence has to mean something.
Label smoothing made every exit **under-confident**: a frame might show 0.6 confidence while
being right far more often than 60% of the time. Temperature scaling fixes this by dividing the logits by one
number T per exit, fitted on val. With T < 1, confidence gets sharper. The predicted class stays the same.


| Exit | T    | ECE before | ECE after (val) | ECE (test) |
| ---- | ---- | ---------- | --------------- | ---------- |
| 1    | 0.66 | 0.114      | 0.013           | 0.013      |
| 2    | 0.68 | 0.105      | 0.013           | 0.009      |
| 3    | 0.83 | 0.044      | 0.025           | 0.025      |


ECE (expected calibration error) measures how far confidence is from actual accuracy. Lower is better.
The test values match val, so the calibration holds on unseen data.

We swept the threshold with both raw and temperature-scaled confidence. Raw confidence at 0.70 came out
cheapest within the accuracy target, so the final controller uses raw confidence. The calibrated version
mainly shows that the exits' confidence is reliable and doesn't drift from val to test.

## How the exits were chosen

- **Placement:** exits after `features[3]` (32×32 map), after `features[10]` (8×8 map), and at the end.
Each head: 1×1 conv → BN → ReLU6 → avgpool → linear.
- **Epoch selection:** weighted val accuracy `0.25·acc1 + 0.25·acc2 + 0.5·acc3`, so exit 1 isn't sacrificed
(selecting on exit 3 alone left exit 1 at ~34%).
- **Threshold:** swept 0.50–0.99 on **val**; picked the cheapest threshold within 1 point of exit-3-only
accuracy → **0.70**.
- **Battery-aware T_MIN / T_MAX:** tuned on val stream (45 configs). Battery: 0.5 / 0.9. Solar: 0.4 / 0.9.

Resulting exit split on test: **11% / 42% / 47%**. Easy classes (background, pedestrian) leave at exit 1;
cars and trucks go deeper.

## Benchmarks (test set)


| Model                   | Avg MACs   | Accuracy  | Macro-F1  | Latency (ms) |
| ----------------------- | ---------- | --------- | --------- | ------------ |
| baseline                | 102.2 M    | 0.882     | 0.873     | 2.12         |
| pruned_25               | 60.2 M     | 0.836     | 0.820     | 1.59         |
| pruned_50               | 29.2 M     | 0.783     | 0.751     | 1.00         |
| **adaptive (thr 0.70)** | **46.7 M** | **0.865** | **0.852** | **1.30**     |


Adaptive uses 54% fewer MACs than the baseline for −1.7 points accuracy, and beats pruned_25 on both cost and accuracy.

## Accuracy vs compute: adaptive vs each static model

This comparison comes from the **threshold sweep**. A static model has one fixed cost and one accuracy.
The early-exit model gives a different trade-off at every threshold. We ran the test set at every threshold
from 0.50 to 0.99, with raw and temperature-scaled confidence. Each run gave one point: average MACs and accuracy.
Because those points cover a range of costs, we could put an adaptive point next to each static model
at the same compute or at the same accuracy.

Average MACs for an adaptive point = the mean over test frames of the MACs actually run. Frames that go deeper cost more.

Each static model is compared to the swept points in two ways:

1. **Same compute:** the best adaptive accuracy using **≤ the static model's MACs**.
   `Δacc = acc_adaptive − acc_static`
2. **Same accuracy:** the cheapest adaptive point within **0.5 points** of the static accuracy.
   `MAC saving = 1 − MACs_adaptive / MACs_static`

| Static model | Static acc | Best adaptive at ≤ same MACs | Cheapest adaptive at ≈ same acc | Verdict |
| --- | --- | --- | --- | --- |
| baseline (102 M) | 88.2% | 86.95% at 60 M (−1.2 pts) | none reaches it | −1.2 pts for 41% fewer MACs |
| pruned_25 (60 M) | 83.6% | 86.95% at 59 M (**+3.3 pts**) | 38.5 M (**−36% MACs**) | adaptive wins on both |
| pruned_50 (29 M) | 78.3% | none useful (exit 1 alone: 49.9%) | 32.6 M (+12% MACs) | pruned_50 wins / tie |

**In short:** at the same compute, adaptive is 3.3 points better than pruned_25. At the same accuracy, it needs
36% fewer MACs. It gets within 1.2 points of the baseline at 41% less compute. It does not beat pruned_50,
because exit 1 alone is too weak at ~29 M MACs.

Source: `results/improved/matched_comparison.csv`, made by `generate_results.py`.

## How the comparison was done

- Battery simulation: 500 J battery, 0.5 fps, every frame is a real test image through the real ONNX model.
- 20 seeds, all policies see the same frame order.
- Policies: static baseline, adaptive (fixed thr), adaptive (battery-aware), and a **bag of models**
that switches baseline → pruned_25 → pruned_50 as the battery drops.


| Scenario | Policy                  | Lifetime (h) | Accuracy | Model MB |
| -------- | ----------------------- | ------------ | -------- | -------- |
| battery  | baseline                | 2.10         | 0.882    | 8.5      |
| battery  | adaptive, fixed         | 3.62 (+72%)  | 0.865    | 4.9      |
| battery  | adaptive, battery-aware | 3.63 (+73%)  | 0.862    | 4.9      |
| battery  | bag of models           | 3.26 (+55%)  | 0.821    | 15.6     |
| solar    | baseline                | 8.19         | 0.881    | 8.5      |
| solar    | adaptive, fixed         | 12.35 (+51%) | 0.865    | 4.9      |
| solar    | bag of models           | 11.07 (+35%) | 0.836    | 15.6     |


**Takeaways**

- Early exit gives +50–70% battery life for ~1.7 points of accuracy.
- It beats switching between separate models on both lifetime and accuracy, with 3× less memory.
- Battery-aware threshold ≈ fixed threshold in practice (no measurable gain).

Full results: `results/improved/`.

## How to run

**1. Download the dataset first.** Get **MIO-TCD-Classification** from the
[MIO-TCD website](https://tcd.miovision.com/) and unzip it so the folder looks like:

```
<data_dir>/MIO-TCD-Classification/train/<class_name>/*.jpg
```

**2. Run the pipeline** (`<data_dir>` = the folder that contains `MIO-TCD-Classification/`):

```bash
python data_prepare.py --root <data_dir>
python train_baseline.py --balanced on
python train_pruned.py
python train_adaptive.py
python evaluate_models.py --force
python benchmark_models.py --force
python generate_results.py --force
python simulate_edge_stream.py --force
python simulate_edge_stream.py --solar
python compare_policies.py --force
```

