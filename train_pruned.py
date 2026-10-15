"""Structured channel pruning (25% / 50%) of baseline.onnx, then BN re-estimation and
fine-tuning; exports models/pruned/pruned_XX.onnx and results/pruning_results.csv.

  python train_pruned.py [--ratios 0.25 0.5] [--epochs 15] [--force]
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (MODEL_DIR, RESULT_DIR, classification_metrics, count_macs, count_params, export_onnx,
                   file_size_mb, fit, get_device, guard_overwrite, hard_exit, load_mobilenetv2_from_onnx,
                   load_split, load_stats, make_tensors, normalize, onnx_logits, predict_torch, prune_model,
                   reestimate_bn, set_seed, write_csv)


def conv_channels(model):
    return [m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    names = {r: f"pruned_{int(round(r * 100))}" for r in a.ratios}
    res_path = os.path.join(RESULT_DIR, "pruning_results.csv")
    for p in [res_path] + [os.path.join(MODEL_DIR, "pruned", f"{n}.onnx") for n in names.values()]:
        guard_overwrite(p, a.force)

    device = get_device()
    mean, std, size = load_stats()
    xv_u8, yv = load_split("val")
    xt_u8, yt = load_split("test")
    xv, xt = normalize(xv_u8, mean, std), normalize(xt_u8, mean, std)
    base_path = os.path.join(MODEL_DIR, "baseline.onnx")
    example = torch.zeros(1, 3, size, size)

    # ---- loader check: rebuilt PyTorch model must reproduce the ONNX baseline
    acc_onnx = classification_metrics(yv, onnx_logits(base_path, xv).argmax(1))["accuracy"]
    rebuilt = load_mobilenetv2_from_onnx(base_path)
    acc_torch = classification_metrics(yv, predict_torch(rebuilt, torch.from_numpy(xv))[0].argmax(1).numpy())["accuracy"]
    print(f"baseline val acc  ONNX {acc_onnx:.5f}  rebuilt-from-ONNX PyTorch {acc_torch:.5f}")
    assert abs(acc_onnx - acc_torch) <= 1e-4, "ONNX->PyTorch loader does not reproduce the baseline"
    base_macs, base_params = count_macs(rebuilt), count_params(rebuilt)
    base_test = classification_metrics(yt, onnx_logits(base_path, xt).argmax(1))
    rows = [{"model": "baseline", "ratio": 0.0, "stage": "reference", "split": "test",
             "accuracy": f"{base_test['accuracy']:.4f}", "macro_f1": f"{base_test['macro_f1']:.4f}",
             "macs": base_macs, "params": base_params, "macs_vs_base": "1.000", "params_vs_base": "1.000",
             "onnx_mb": f"{file_size_mb(base_path):.3f}"}]

    train = make_tensors("train", mean, std, device)
    for r, name in names.items():
        # every ratio starts from the same trained baseline weights
        set_seed(a.seed)
        model = load_mobilenetv2_from_onnx(base_path)
        before = conv_channels(model)
        # remove the r fraction of channels with the smallest L2 norm (tensors physically shrink)
        model = prune_model(model, r, example)
        after = conv_channels(model)
        print(f"{name}: conv channels {sum(before)} -> {sum(after)}  (stem {before[0]}->{after[0]}, "
              f"last {before[-1]}->{after[-1]})")
        pv = predict_torch(model, torch.from_numpy(xv))[0].argmax(1).numpy()
        pre = classification_metrics(yv, pv)
        print(f"{name}: val acc right after pruning (no fine-tune) {pre['accuracy']:.4f}")

        # pruning drops accuracy; make BN trainable again, then fine-tune to recover
        model = model.to(device)
        reestimate_bn(model, train[0])
        xvd, yvd = torch.from_numpy(xv).to(device), yv

        def loss_fn(m, x, y):
            return F.cross_entropy(m(x), y, label_smoothing=0.1)

        def score_fn(m):
            p = predict_torch(m, xvd)[0].argmax(1).numpy()
            rr = classification_metrics(yvd, p)
            return rr["macro_f1"], {"val_acc": rr["accuracy"]}

        fit(model, loss_fn, score_fn, train, a.epochs, a.lr, a.wd, a.bs, desc=name)
        # export the fine-tuned model and record accuracy before/after fine-tuning plus its cost
        model = model.cpu().eval()
        path = os.path.join(MODEL_DIR, "pruned", f"{name}.onnx")
        export_onnx(model, path, example, ["image"], ["logits"], a.force, check_inputs=torch.from_numpy(xv[:64]))
        macs, params = count_macs(model), count_params(model)
        rows.append({"model": name, "ratio": r, "stage": "pruned_no_finetune", "split": "val",
                     "accuracy": f"{pre['accuracy']:.4f}", "macro_f1": f"{pre['macro_f1']:.4f}",
                     "macs": macs, "params": params, "macs_vs_base": f"{macs / base_macs:.3f}",
                     "params_vs_base": f"{params / base_params:.3f}", "onnx_mb": ""})
        for split, x, y in (("val", xv, yv), ("test", xt, yt)):
            m = classification_metrics(y, onnx_logits(path, x).argmax(1))
            rows.append({"model": name, "ratio": r, "stage": "finetuned_onnx", "split": split,
                         "accuracy": f"{m['accuracy']:.4f}", "macro_f1": f"{m['macro_f1']:.4f}",
                         "macs": macs, "params": params, "macs_vs_base": f"{macs / base_macs:.3f}",
                         "params_vs_base": f"{params / base_params:.3f}", "onnx_mb": f"{file_size_mb(path):.3f}"})
            print(f"{name} ONNX {split}: acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f}  "
                  f"MACs {macs:,} ({macs / base_macs:.2f}x)  params {params:,}")
    write_csv(res_path, rows, a.force)


if __name__ == "__main__":
    main()
    hard_exit()
