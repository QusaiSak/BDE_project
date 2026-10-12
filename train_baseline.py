"""Train the MobileNetV2 baseline (random init, stride-1 stem for 64x64 crops), optionally
compare plain vs class-balanced CE on val macro-F1, export the winner to models/baseline.onnx.

  python train_baseline.py [--balanced auto|on|off] [--epochs 50] [--force]
"""
import argparse
import os

import torch
import torch.nn.functional as F

from utils import (MODEL_DIR, RESULT_DIR, build_mobilenetv2, classification_metrics, count_macs, count_params,
                   export_onnx, file_size_mb, fit, get_device, guard_overwrite, load_stats, make_tensors,
                   normalize, load_split, onnx_logits, predict_torch, set_seed, write_csv, CLASSES, hard_exit)


def train_variant(balanced, a, train, val, device):
    set_seed(a.seed)
    model = build_mobilenetv2().to(device)
    w = None
    if balanced:
        cnt = torch.bincount(train[1], minlength=len(CLASSES)).float()
        w = (cnt.sum() / (len(cnt) * cnt)).to(device)

    def loss_fn(m, x, y):
        return F.cross_entropy(m(x), y, weight=w, label_smoothing=0.1)

    def score_fn(m):
        pred = predict_torch(m, val[0])[0].argmax(1).numpy()
        r = classification_metrics(val[1].cpu().numpy(), pred)
        return r["macro_f1"], {"val_acc": r["accuracy"]}

    hist = fit(model, loss_fn, score_fn, train, a.epochs, a.lr, a.wd, a.bs,
               desc="baseline-" + ("balanced" if balanced else "plain"))
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    onnx_path = os.path.join(MODEL_DIR, "baseline.onnx")
    res_path = os.path.join(RESULT_DIR, "baseline_results.csv")
    for p in (onnx_path, res_path, os.path.join(RESULT_DIR, "baseline_history.csv")):
        guard_overwrite(p, a.force)

    device = get_device()
    print("device:", device)
    mean, std, size = load_stats()
    train, val = make_tensors("train", mean, std, device), make_tensors("val", mean, std, device)

    variants = {"auto": [False, True], "on": [True], "off": [False]}[a.balanced]
    trained, hist_rows = {}, []
    for b in variants:
        model, hist = train_variant(b, a, train, val, device)
        trained[b] = (model, max(h["val_score"] for h in hist))
        hist_rows += [{"variant": "balanced" if b else "plain", **h} for h in hist]
    chosen = max(trained, key=lambda b: trained[b][1])
    print({("balanced" if b else "plain"): round(s, 4) for b, (_, s) in trained.items()},
          "-> selected on val macro-F1:", "balanced" if chosen else "plain")
    write_csv(os.path.join(RESULT_DIR, "baseline_history.csv"), hist_rows, a.force)

    model = trained[chosen][0].cpu().eval()
    xv, yv = load_split("val")
    xt, yt = load_split("test")
    xv, xt = normalize(xv, mean, std), normalize(xt, mean, std)
    export_onnx(model, onnx_path, torch.zeros(1, 3, size, size), ["image"], ["logits"], a.force,
                check_inputs=torch.from_numpy(xv[:64]))
    macs, params = count_macs(model, (1, 3, size, size)), count_params(model)
    rows = []
    for b, (_, s) in trained.items():
        rows.append({"model": "baseline", "variant": "balanced" if b else "plain", "selected": int(b == chosen),
                     "split": "val", "accuracy": "", "macro_f1": f"{s:.4f}", "source": "pytorch (selection)",
                     "macs": "", "params": "", "onnx_mb": ""})
    for split, x, y in (("val", xv, yv), ("test", xt, yt)):
        r = classification_metrics(y, onnx_logits(onnx_path, x).argmax(1))
        rows.append({"model": "baseline", "variant": "balanced" if chosen else "plain", "selected": 1,
                     "split": split, "accuracy": f"{r['accuracy']:.4f}", "macro_f1": f"{r['macro_f1']:.4f}",
                     "source": "onnxruntime", "macs": macs, "params": params,
                     "onnx_mb": f"{file_size_mb(onnx_path):.3f}"})
        print(f"ONNX {split}: acc {r['accuracy']:.4f} macro-F1 {r['macro_f1']:.4f}")
    write_csv(res_path, rows, a.force)
    print(f"baseline MACs {macs:,}  params {params:,}")


if __name__ == "__main__":
    main()
    hard_exit()
