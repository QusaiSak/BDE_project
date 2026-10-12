"""Shared helpers for the whole pipeline: paths, data loading/augmentation, metrics,
calibration, model construction (MobileNetV2 + pruning + early exits), ONNX export and
reload, and the runtime early-exit controller used by evaluation, benchmark and simulation.
"""
import copy
import csv
import os
import time
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from onnx import numpy_helper
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
IMPROVED_DIR = os.path.join(RESULT_DIR, "improved")  # benchmark / final results / simulations

CLASSES = ["articulated_truck", "bicycle", "bus", "car", "motorcycle",
           "non-motorized_vehicle", "pedestrian", "pickup_truck",
           "single_unit_truck", "work_van", "background"]
NUM_CLASSES = len(CLASSES)
OPSET = 18
# The three static operating points (paths relative to MODEL_DIR)
STATIC = {"baseline": "baseline.onnx", "pruned_25": "pruned/pruned_25.onnx", "pruned_50": "pruned/pruned_50.onnx"}


# --------------------------------------------------------------------------- misc
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def guard_overwrite(path, force):
    """Refuse to silently overwrite an existing artifact."""
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite it")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_csv(path, rows, force, fieldnames=None):
    guard_overwrite(path, force)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- data
def load_stats():
    s = {r["key"]: r["value"] for r in read_csv(os.path.join(DATA_DIR, "stats.csv"))}
    mean = np.array([float(s[f"mean_{c}"]) for c in "rgb"], dtype=np.float32)
    std = np.array([float(s[f"std_{c}"]) for c in "rgb"], dtype=np.float32)
    return mean, std, int(s["image_size"])


def decode_image(args):
    path, size = args
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_split(split):
    """Decoded split as (uint8 N,H,W,3, int64 labels). Cache written by data_prepare.py."""
    d = np.load(os.path.join(CACHE_DIR, f"{split}.npz"))
    return d["x"], d["y"]


def normalize(x_uint8, mean, std):
    """uint8 NHWC -> float32 NCHW normalized (the preprocessing used at eval)."""
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def gpu_augment(x, pad=4):
    """Random crop (reflect pad 4) + horizontal flip on a normalized NCHW batch.
    Left/right flip is class-safe: a mirrored truck/bike/pedestrian keeps its class."""
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i[:, None] + torch.arange(h, device=x.device)[None])  # n,h
    cols = (j[:, None] + torch.arange(w, device=x.device)[None])  # n,w
    idx_n = torch.arange(n, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[idx_n, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out.contiguous()


# --------------------------------------------------------------------------- metrics
def classification_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = [float((y_pred[y_true == k] == k).mean()) if (y_true == k).any() else float("nan")
                 for k in range(NUM_CLASSES)]
    return {"accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)))),
            "per_class": per_class}


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece_score(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rel = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = float((pred[m] == y[m]).mean()), float(conf[m].mean())
            ece += m.mean() * abs(acc - c)
            rel.append((lo, hi, c, acc, int(m.sum())))
    return float(ece), rel


def fit_temperature(logits, y):
    """Single scalar temperature minimising NLL (fit on VALIDATION only)."""
    lt, yt = torch.tensor(logits, dtype=torch.float64), torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


# --------------------------------------------------------------------------- models
def build_mobilenetv2(num_classes=NUM_CLASSES):
    """torchvision MobileNetV2 (width 1.0, random init) with the stem stride 2->1.
    At 64x64 the stock net would end on a 2x2 map (stride 32); a stride-1 stem keeps
    a 4x4 final map (stride 16), the usual small-input adaptation."""
    m = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
    m.features[0][0].stride = (1, 1)
    return m


def count_params(model):
    """conv weights + BN affine + linear weights/bias (BN running stats excluded; the conv
    bias that BN folding creates is excluded so ONNX-rebuilt models count the same)."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            n += mod.weight.numel()
        elif isinstance(mod, nn.Linear):
            n += mod.weight.numel() + mod.bias.numel()
        elif isinstance(mod, nn.BatchNorm2d) and mod.affine:
            n += mod.weight.numel() + mod.bias.numel()
    return n


def count_macs(module, input_shape=(1, 3, 64, 64), example=None):
    """The ONE FLOPs method: fvcore, CPU, batch 1 (fvcore counts one MAC per FLOP)."""
    from fvcore.nn import FlopCountAnalysis
    module = copy.deepcopy(module).cpu().eval()
    x = example if example is not None else torch.zeros(*input_shape)
    with torch.no_grad():
        fa = FlopCountAnalysis(module, x)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return int(fa.total())


def file_size_mb(*paths):
    return sum(os.path.getsize(p) for p in paths) / 2 ** 20


# --------------------------------------------------------------------------- training
def make_tensors(split, mean, std, device):
    x, y = load_split(split)
    return torch.from_numpy(normalize(x, mean, std)).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def predict_torch(model, x, bs=1024):
    """Returns list of logits tensors (one per model output)."""
    model.eval()
    outs = None
    for i in range(0, len(x), bs):
        o = model(x[i:i + bs])
        o = o if isinstance(o, (tuple, list)) else (o,)
        outs = [[] for _ in o] if outs is None else outs
        for k, t in enumerate(o):
            outs[k].append(t.float().cpu())
    return [torch.cat(o) for o in outs]


# Shared training loop for baseline, pruned and early-exit models.
def fit(model, loss_fn, score_fn, train, epochs, lr, wd=1e-4, bs=256, desc="train", sampler_weights=None):
    """SGD+nesterov, cosine LR. Keeps the best-on-val weights IN MEMORY (never on disk).
    loss_fn(model, x, y) -> loss ; score_fn(model) -> (score, info_dict)."""
    xtr, ytr = train
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        if sampler_weights is not None:
            perm = torch.multinomial(sampler_weights, len(xtr), replacement=True)
        else:
            perm = torch.randperm(len(xtr), device=xtr.device)
        tot, n = 0.0, 0
        for i in tqdm(range(len(xtr) // bs), desc=f"{desc} ep{ep}/{epochs}", leave=False):
            idx = perm[i * bs:(i + 1) * bs]
            x = gpu_augment(xtr[idx])
            loss = loss_fn(model, x, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        score, info = score_fn(model)
        history.append({"epoch": ep, "train_loss": tot / n, "val_score": score, **info})
        tag = ""
        if score > best:
            best, best_state, tag = score, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), " *best"
        print(f"[{desc}] ep {ep:3d} loss {tot / n:.4f} val_score {score:.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in info.items()) + tag, flush=True)
    model.load_state_dict(best_state)
    return history


# --------------------------------------------------------------------------- ONNX
def make_session(path, threads=None):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def export_onnx(model, path, example, input_names, output_names, force, check_inputs=None):
    """Export from CPU (opset 18, BN folded), run onnx.checker and verify ORT == PyTorch."""
    guard_overwrite(path, force)
    model = copy.deepcopy(model).cpu().eval()
    dyn = {n: {0: "batch"} for n in input_names + output_names}
    with torch.no_grad():
        torch.onnx.export(model, example, path, opset_version=OPSET, input_names=input_names,
                          output_names=output_names, dynamic_axes=dyn, do_constant_folding=True,
                          dynamo=False)
    m = onnx.load(path)
    onnx.checker.check_model(m, full_check=True)
    names = [v for n in m.graph.node for v in n.output] + [i.name for i in m.graph.input]
    assert len(names) == len(set(names)), f"{path}: tensor names not unique (SSA violated)"
    sess = make_session(path)
    xs = check_inputs if check_inputs is not None else example
    xs = xs if isinstance(xs, tuple) else (xs,)
    with torch.no_grad():
        ref = model(*xs)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    got = sess.run(None, {n: x.numpy() for n, x in zip(input_names, xs)})
    diff = max(float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref))
    assert diff < 1e-3, f"{path}: ONNX vs PyTorch max abs diff {diff}"
    print(f"exported {os.path.relpath(path, ROOT)}  checker OK  max|onnx-torch|={diff:.2e}")
    return diff


def onnx_logits(path, x, bs=512):
    sess = make_session(path)
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i:i + bs]})[0] for i in range(0, len(x), bs)])


# ---- ONNX -> PyTorch: pruning and early-exit training start from the exported ONNX files,
# so the exact deployed weights are reused instead of separate PyTorch checkpoints.
def onnx_conv_params(path):
    """Ordered (weight, bias) of every Conv node, and (W, b) of the final Gemm."""
    g = onnx.load(path).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:  # identical weights may be deduplicated into Constant / Identity aliases
        if n.op_type == "Constant":
            init[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
        elif n.op_type == "Identity" and n.input[0] in init:
            init[n.output[0]] = init[n.input[0]]
    convs, gemm = [], None
    for n in g.node:
        if n.op_type == "Conv":
            w = init[n.input[1]]
            b = init[n.input[2]] if len(n.input) > 2 else np.zeros(w.shape[0], np.float32)
            convs.append((w, b))
        elif n.op_type == "Gemm":
            gemm = (init[n.input[1]], init[n.input[2]])
    return convs, gemm


def _identity_bn(c):
    bn = nn.BatchNorm2d(c, eps=0.0)
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
    return bn


def load_mobilenetv2_from_onnx(path):
    """Rebuild a PyTorch MobileNetV2 from a BN-folded ONNX file.
    Channel counts come from the ONNX weights (works for pruned models). Each Conv gets
    the folded weight+bias and the following BN is set to an exact identity
    (eps=0, mean 0, var 1, gamma 1, beta 0)."""
    convs, (lw, lb) = onnx_conv_params(path)
    # walk MobileNetV2 convs in forward order; ONNX Conv nodes appear in the same order
    model = build_mobilenetv2()
    slots = []  # (parent_sequential, index) of every conv in forward (DFS) order
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            parent_name, idx = name.rsplit(".", 1)
            slots.append((model.get_submodule(parent_name), int(idx)))
    assert len(slots) == len(convs), f"conv count mismatch {len(slots)} vs {len(convs)}"
    for (parent, i), (w, b) in zip(slots, convs):
        t = parent[i]
        out_c, in_per_g = w.shape[0], w.shape[1]
        groups = out_c if t.groups > 1 else 1
        conv = nn.Conv2d(in_per_g * groups, out_c, t.kernel_size, t.stride, t.padding, groups=groups, bias=True)
        conv.weight.data = torch.from_numpy(w.copy())
        conv.bias.data = torch.from_numpy(b.copy())
        parent[i] = conv
        assert isinstance(parent[i + 1], nn.BatchNorm2d)
        parent[i + 1] = _identity_bn(out_c)
    lin = nn.Linear(lw.shape[1], lw.shape[0])
    lin.weight.data, lin.bias.data = torch.from_numpy(lw.copy()), torch.from_numpy(lb.copy())
    model.classifier[1] = lin
    return model.eval()


@torch.no_grad()
def reestimate_bn(model, x, n_batches=40, bs=256):
    """Make identity BNs trainable without changing the eval-mode function:
    measure each BN input's mean/var on training data, then set
    running_mean=mu, running_var=var, gamma=sqrt(var+eps), beta=mu  =>  BN(z)=z in eval,
    and ~z in train mode (batch stats ~ population stats)."""
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    sums = {id(b): [0.0, 0.0, 0] for b in bns}

    def hook(b):
        def f(mod, inp, out):
            z = inp[0].float()  # per-batch reduction on device (MPS has no float64) ...
            s = sums[id(b)]     # ... accumulated in float64 on CPU
            s[0] = s[0] + z.sum((0, 2, 3)).cpu().double()
            s[1] = s[1] + (z * z).sum((0, 2, 3)).cpu().double()
            s[2] += z.numel() // z.shape[1]
        return f
    hs = [b.register_forward_hook(hook(b)) for b in bns]
    perm = torch.randperm(len(x))[: n_batches * bs].to(x.device)
    for i in range(0, len(perm), bs):
        model(x[perm[i:i + bs]])
    for h in hs:
        h.remove()
    for b in bns:
        s0, s1, n = sums[id(b)]
        mu = s0 / n
        var = (s1 / n - mu * mu).clamp_min(1e-6)
        b.eps = 1e-5
        b.running_mean.copy_(mu.float())
        b.running_var.copy_(var.float())
        b.weight.copy_((var + b.eps).sqrt().float())
        b.bias.copy_(mu.float())


def prune_model(model, ratio, example):
    """Structured channel pruning: torch-pruning DepGraph, L2 magnitude, uniform ratio.
    The classifier output (11 classes) is never pruned. Physically shrinks tensors."""
    import torch_pruning as tp
    model = model.cpu().eval()
    pruner = tp.pruner.MagnitudePruner(model, example, importance=tp.importance.MagnitudeImportance(p=2),
                                       pruning_ratio=ratio, ignored_layers=[model.classifier[1]])
    pruner.step()
    return model


# --------------------------------------------------------------------------- early exit
def exit_head(c_in, c_mid, num_classes=NUM_CLASSES):
    return nn.Sequential(nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU6(inplace=True),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_mid, num_classes))


class EarlyExitMobileNetV2(nn.Module):
    """Three exits on ONE shared backbone. Exit k reuses stage 1..k of the same features;
    exit 3 is the backbone's original classifier."""

    def __init__(self, backbone, splits=(3, 10), head_ch=128):
        super().__init__()
        self.features, self.classifier = backbone.features, backbone.classifier
        self.splits = tuple(splits)
        s1, s2 = self.splits
        self.stage1 = self.features[: s1 + 1]
        self.stage2 = self.features[s1 + 1: s2 + 1]
        self.stage3 = self.features[s2 + 1:]
        with torch.no_grad():  # probe real (possibly pruned) channel counts
            dev = next(backbone.parameters()).device
            was = self.training
            self.eval()
            f1 = self.stage1(torch.zeros(1, 3, 64, 64, device=dev))
            f2 = self.stage2(f1)
            self.train(was)
        self.feat_shapes = (tuple(f1.shape[1:]), tuple(f2.shape[1:]))
        self.head1 = exit_head(f1.shape[1], head_ch)
        self.head2 = exit_head(f2.shape[1], head_ch)

    def final(self, f):
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(self.stage3(f), 1), 1))

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        return self.head1(f1), self.head2(f2), self.final(f2)


# Deployment wrappers: each exports ONE stage. Exit k+1 takes exit k's feature map as input,
# so the three ONNX files together hold every weight exactly once.
class Exit1(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.stage, self.head = m.stage1, m.head1

    def forward(self, x):
        f = self.stage(x)
        return self.head(f), f


class Exit2(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.stage, self.head = m.stage2, m.head2

    def forward(self, f):
        g = self.stage(f)
        return self.head(g), g


class Exit3(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, f):
        return self.m.final(f)


EXIT_IO = [(["image"], ["exit1_logits", "s1_feat"]),
           (["s1_feat_in"], ["exit2_logits", "s2_feat"]),
           (["s2_feat_in"], ["exit3_logits"])]


def adaptive_paths(tag="adaptive"):
    d = os.path.join(MODEL_DIR, tag)
    return [os.path.join(d, f"exit{k}.onnx") for k in (1, 2, 3)]


def adaptive_exit_macs(model):
    """Cumulative fvcore MACs to REACH each exit, including the heads already evaluated
    on the way (a sample exiting at 3 also paid for heads 1 and 2)."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        f1 = m.stage1(x)
        f2 = m.stage2(f1)
    s1 = count_macs(m.stage1, example=x)
    h1 = count_macs(m.head1, example=f1)
    s2 = count_macs(m.stage2, example=f1)
    h2 = count_macs(m.head2, example=f2)
    s3 = count_macs(Exit3(m), example=f2)
    parts = {"stage1": s1, "head1": h1, "stage2": s2, "head2": h2, "stage3_cls": s3}
    cum = [s1 + h1, s1 + h1 + s2 + h2, s1 + h1 + s2 + h2 + s3]
    return cum, parts


def onnx_exit_logits(paths, x, bs=512):
    """Run the chained ONNX exits on every sample (for offline analysis / calibration)."""
    sess = [make_session(p) for p in paths]
    outs = [[], [], []]
    for i in range(0, len(x), bs):
        l1, f1 = sess[0].run(None, {"image": x[i:i + bs]})
        l2, f2 = sess[1].run(None, {"s1_feat_in": f1})
        (l3,) = sess[2].run(None, {"s2_feat_in": f2})
        for k, l in enumerate((l1, l2, l3)):
            outs[k].append(l)
    return [np.concatenate(o) for o in outs]


def simulate_exits(logits, threshold, temps=(1.0, 1.0, 1.0), max_exit=3):
    """Vectorised equivalent of the controller rule (used for sweeps; the per-frame
    controller is checked against it in evaluate_models.py)."""
    n = len(logits[0])
    pred = np.empty(n, dtype=np.int64)
    ex = np.full(n, max_exit, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    for k in range(max_exit):
        p = softmax(logits[k], temps[k])
        stop = ~done & ((p.max(1) >= threshold) | (k == max_exit - 1))
        pred[stop], ex[stop] = p.argmax(1)[stop], k + 1
        done |= stop
    return pred, ex


class AdaptiveONNXController:
    """Runtime loop (slide 13): SENSE frame -> ESTIMATE confidence at the current exit ->
    CHOOSE stop/continue vs threshold -> EXECUTE only the next stage -> UPDATE log.
    Only unsure samples continue; the last allowed exit always returns.
    The intermediate feature map of THIS frame is passed to the next stage."""

    def __init__(self, paths=None, temps=(1.0, 1.0, 1.0), threads=1, exit_macs=None):
        paths = paths or adaptive_paths()
        self.sess = [make_session(p, threads) for p in paths]
        self.temps = tuple(float(t) for t in temps)
        self.exit_macs = exit_macs
        self.last_model_time = 0.0

    @staticmethod
    def _conf(logits, T):
        z = logits[0] / T
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        k = int(p.argmax())
        return k, float(p[k])

    # threshold > 1 forces the frame down to max_exit (fixed-depth operating points)
    def predict(self, image, threshold, max_exit=3):
        """image: float32 (1,3,H,W) normalized. Returns (pred, exit, [conf per evaluated exit])."""
        t_model = 0.0
        t0 = time.perf_counter()
        l1, f1 = self.sess[0].run(None, {"image": image})
        t_model += time.perf_counter() - t0
        pred, c = self._conf(l1, self.temps[0])
        confs = [c]
        if c >= threshold or max_exit == 1:
            self.last_model_time = t_model
            return pred, 1, confs
        t0 = time.perf_counter()
        l2, f2 = self.sess[1].run(None, {"s1_feat_in": f1})
        t_model += time.perf_counter() - t0
        pred, c = self._conf(l2, self.temps[1])
        confs.append(c)
        if c >= threshold or max_exit == 2:
            self.last_model_time = t_model
            return pred, 2, confs
        t0 = time.perf_counter()
        (l3,) = self.sess[2].run(None, {"s2_feat_in": f2})
        t_model += time.perf_counter() - t0
        pred, c = self._conf(l3, self.temps[2])
        confs.append(c)
        self.last_model_time = t_model
        return pred, 3, confs


def load_temperatures(tag="adaptive"):
    p = os.path.join(RESULT_DIR, "calibration_results.csv")
    rows = [r for r in read_csv(p) if r["model"] == tag and r["split"] == "val"]
    return tuple(float(next(r for r in rows if r["exit"] == str(k))["temperature"]) for k in (1, 2, 3))


def load_exit_macs():
    rows = read_csv(os.path.join(RESULT_DIR, "exit_flops.csv"))
    return [int(next(r for r in rows if r["exit"] == str(k))["cumulative_macs"]) for k in (1, 2, 3)]


def load_selected_config():
    """Val-selected (threshold, calibrated?) written by evaluate_models.py."""
    rows = read_csv(os.path.join(RESULT_DIR, "adaptive_results.csv"))
    r = next(r for r in rows if r["selected"] == "1" and r["split"] == "val")
    return float(r["threshold"]), r["calibration"] == "temperature"


def hard_exit():
    """onnxruntime + torch can abort at interpreter exit on macOS."""
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
