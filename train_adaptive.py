"""Train the 3-exit early-exit network on the 25%-pruned backbone, export it as three
chained ONNX graphs (exit k only runs stage k on the previous exit's feature map),
prove the weights are shared, and record cumulative MACs per exit.

  python train_adaptive.py [--tag adaptive] [--splits 3 10] [--epochs 30] [--force]
"""
import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from utils import (EXIT_IO, MODEL_DIR, RESULT_DIR, EarlyExitMobileNetV2, Exit1, Exit2, Exit3, adaptive_exit_macs,
                   adaptive_paths, classification_metrics, count_macs, count_params, export_onnx, fit, get_device,
                   guard_overwrite, hard_exit, load_mobilenetv2_from_onnx, load_split, load_stats, make_tensors,
                   normalize, onnx_exit_logits, predict_torch, reestimate_bn, set_seed, write_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="adaptive", help="artifact name; ablations get their own tag")
    ap.add_argument("--init", default=os.path.join(MODEL_DIR, "pruned", "pruned_25.onnx"))
    ap.add_argument("--splits", type=int, nargs=2, default=[3, 10], help="feature indices after which exits 1,2 sit")
    ap.add_argument("--head-ch", type=int, default=128)
    ap.add_argument("--loss-weights", type=float, nargs=3, default=[0.5, 0.5, 1.0])
    ap.add_argument("--select-weights", type=float, nargs=3, default=[0.25, 0.25, 0.5])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    sfx = "" if a.tag == "adaptive" else f"_{a.tag}"
    paths = adaptive_paths(a.tag)
    out_csv = {k: os.path.join(RESULT_DIR, f"{k}{sfx}.csv") for k in ("exit_flops", "adaptive_training", "weight_sharing")}
    for p in paths + list(out_csv.values()):
        guard_overwrite(p, a.force)

    set_seed(a.seed)
    device = get_device()
    mean, std, size = load_stats()
    train = make_tensors("train", mean, std, device)
    xv_u8, yv = load_split("val")
    xv = normalize(xv_u8, mean, std)
    xvd = torch.from_numpy(xv).to(device)

    backbone = load_mobilenetv2_from_onnx(a.init).to(device)
    reestimate_bn(backbone, train[0])  # function-preserving, makes BN trainable
    model = EarlyExitMobileNetV2(backbone, a.splits, a.head_ch).to(device)
    print(f"[{a.tag}] splits after features {a.splits} (+ final {len(model.features) - 1}); "
          f"probed feature shapes {model.feat_shapes}")

    lw = a.loss_weights

    # joint loss: weighted sum of the label-smoothed CE of every exit (deep supervision)
    def loss_fn(m, x, y):
        return sum(w * F.cross_entropy(o, y, label_smoothing=0.1) for w, o in zip(lw, m(x)))

    # epoch selection on the WEIGHTED MEAN of all exit accuracies (exit 3 alone starves exit 1)
    sw = np.array(a.select_weights) / sum(a.select_weights)

    def score_fn(m):
        accs = [classification_metrics(yv, o.argmax(1).numpy())["accuracy"] for o in predict_torch(m, xvd)]
        return float(np.dot(sw, accs)), {f"val_acc_exit{k + 1}": v for k, v in enumerate(accs)}

    hist = fit(model, loss_fn, score_fn, train, a.epochs, a.lr, a.wd, a.bs, desc=a.tag)
    best = max(hist, key=lambda h: h["val_score"])
    print(f"[{a.tag}] best epoch {best['epoch']} (weighted mean {sw.round(3).tolist()}): "
          + ", ".join(f"exit{k} {best[f'val_acc_exit{k}']:.4f}" for k in (1, 2, 3)))
    write_csv(out_csv["adaptive_training"], [{"tag": a.tag, **h} for h in hist], a.force)

    # ---- chained ONNX export (CPU). Exit modules wrap the SAME submodules (no copies).
    model = model.cpu().eval()
    e1, e2, e3 = Exit1(model), Exit2(model), Exit3(model)
    xs = torch.from_numpy(xv[:64])
    with torch.no_grad():
        f1 = model.stage1(xs)
        f2 = model.stage2(f1)
    for mod, path, (ins, outs), ex, chk in ((e1, paths[0], EXIT_IO[0], torch.zeros(1, 3, size, size), xs),
                                             (e2, paths[1], EXIT_IO[1], f1[:1], f1),
                                             (e3, paths[2], EXIT_IO[2], f2[:1], f2)):
        export_onnx(mod, path, ex, ins, outs, a.force, check_inputs=chk)

    # chain == PyTorch on the whole val set
    onnx_out = onnx_exit_logits(paths, xv)
    torch_out = predict_torch(model, torch.from_numpy(xv))
    diffs = [float(np.abs(o - t.numpy()).max()) for o, t in zip(onnx_out, torch_out)]
    agree = [float((o.argmax(1) == t.numpy().argmax(1)).mean()) for o, t in zip(onnx_out, torch_out)]
    print(f"chain vs PyTorch on val: max|diff| per exit {[f'{d:.1e}' for d in diffs]}, argmax agreement {agree}")
    assert max(diffs) < 1e-3 and min(agree) > 0.999

    # ---- weight-sharing proof
    names = dict(model.named_parameters())
    for wrap in (e1, e2, e3):
        for n, p in wrap.named_parameters():
            assert any(p is q for q in names.values()), "exit module holds a copy"
    ptr_ok = all(p.data_ptr() == q.data_ptr() for p, q in zip(model.stage1.parameters(), model.features[: a.splits[0] + 1].parameters()))
    inits = [{i.name: numpy_helper.to_array(i) for i in onnx.load(p).graph.initializer} for p in paths]
    n_onnx = [int(sum(v.size for v in d.values())) for d in inits]
    # Every value of the full 3-exit model lives in exactly one exit file (disjoint partition)
    full_path_params = count_params(model)
    rows = [
        {"check": "exit wrappers reference the training model's own Parameter objects", "result": "PASS"},
        {"check": "stage1 params data_ptr == features[0..s1] data_ptr (same memory)", "result": "PASS" if ptr_ok else "FAIL"},
        {"check": "exit1.onnx scalar weights (incl. folded conv bias)", "result": n_onnx[0]},
        {"check": "exit2.onnx scalar weights (only stage2+head2; reuses exit1 feature)", "result": n_onnx[1]},
        {"check": "exit3.onnx scalar weights (only stage3+classifier; reuses exit2 feature)", "result": n_onnx[2]},
        {"check": "sum over the three files", "result": sum(n_onnx)},
        {"check": "params of whole 3-exit model (conv w + BN affine + linear)", "result": full_path_params},
        {"check": "params of the 25%-pruned static model (for overhead)", "result": count_params(load_mobilenetv2_from_onnx(a.init))},
    ]
    write_csv(out_csv["weight_sharing"], rows, a.force)

    # ---- cumulative fvcore MACs per exit
    cum, parts = adaptive_exit_macs(model)
    p25 = count_macs(load_mobilenetv2_from_onnx(a.init))
    onnx_val = [classification_metrics(yv, o.argmax(1)) for o in onnx_out]
    frows = []
    for k in range(3):
        frows.append({"exit": k + 1, "after_feature_index": [a.splits[0], a.splits[1], len(model.features) - 1][k],
                      "cumulative_macs": cum[k], "macs_vs_pruned_25": f"{cum[k] / p25:.3f}",
                      "val_acc_onnx": f"{onnx_val[k]['accuracy']:.4f}", "val_macro_f1_onnx": f"{onnx_val[k]['macro_f1']:.4f}",
                      **{f"part_{n}": v for n, v in parts.items()}})
    write_csv(out_csv["exit_flops"], frows, a.force)
    for r in frows:
        print(f"exit{r['exit']}: cumulative MACs {r['cumulative_macs']:,} ({r['macs_vs_pruned_25']}x pruned_25)  "
              f"val acc {r['val_acc_onnx']}")
    acc = [o["accuracy"] for o in onnx_val]
    print("SANITY exit accuracy increases with depth:", "PASS" if acc[0] < acc[1] < acc[2] else "FAIL", acc)


if __name__ == "__main__":
    main()
    hard_exit()
