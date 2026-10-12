"""Verify MIO-TCD-Classification on disk, cap per class, stratified 70/10/20 split,
compute train-only normalization stats, and decode resized images into a RAM cache.

  python data_prepare.py --root . [--max-per-class 6000] [--image-size 64] [--force]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from utils import CACHE_DIR, CLASSES, DATA_DIR, SPLIT_DIR, decode_image, guard_overwrite, write_csv

EXPECTED = {"articulated_truck": 10346, "bicycle": 2284, "bus": 10316, "car": 260518, "motorcycle": 1982,
            "non-motorized_vehicle": 1751, "pedestrian": 6262, "pickup_truck": 50906,
            "single_unit_truck": 5120, "work_van": 9679, "background": 160000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder containing MIO-TCD-Classification/")
    ap.add_argument("--max-per-class", type=int, default=6000)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    train_dir = os.path.join(a.root, "MIO-TCD-Classification", "train")
    assert os.path.isdir(train_dir), f"missing {train_dir}"
    found = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    assert set(found) == set(CLASSES), f"class folders differ: {set(found) ^ set(CLASSES)}"

    # global source_index = position in the sorted list of all labeled training files
    files, counts, src = {}, [], 0
    for c in CLASSES:
        names = sorted(f for f in os.listdir(os.path.join(train_dir, c)) if f.lower().endswith(".jpg"))
        files[c] = [(src + i, n) for i, n in enumerate(names)]
        src += len(names)
        status = "OK" if len(names) == EXPECTED[c] else f"MISMATCH (expected {EXPECTED[c]})"
        print(f"{c:24s} {len(names):7d}  {status}")
        counts.append(len(names))
    print(f"total {sum(counts)} (expected {sum(EXPECTED.values())})")

    rng = np.random.default_rng(a.seed)
    splits = {"train": [], "val": [], "test": []}
    rows_counts = []
    for c, n_orig in zip(CLASSES, counts):
        items = files[c]
        keep = sorted(rng.choice(len(items), size=min(a.max_per_class, len(items)), replace=False))
        chosen = [items[i] for i in keep]
        perm = rng.permutation(len(chosen))
        n_tr, n_va = int(round(0.7 * len(chosen))), int(round(0.1 * len(chosen)))
        parts = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
        for s, idx in parts.items():
            for i in sorted(idx):
                si, name = chosen[i]
                splits[s].append({"source_index": si, "image_id": os.path.splitext(name)[0], "class": c,
                                  "path": os.path.relpath(os.path.join(train_dir, c, name), a.root)})
        rows_counts.append({"class": c, "original": n_orig, "used": len(chosen),
                            "train": len(parts["train"]), "val": len(parts["val"]), "test": len(parts["test"])})

    for s, rows in splits.items():
        write_csv(os.path.join(SPLIT_DIR, f"{s}.csv"), rows, a.force)
    write_csv(os.path.join(DATA_DIR, "class_counts.csv"), rows_counts, a.force)

    os.makedirs(CACHE_DIR, exist_ok=True)
    arrays = {}
    with Pool(a.workers) as pool:
        for s, rows in splits.items():
            path = os.path.join(CACHE_DIR, f"{s}.npz")
            guard_overwrite(path, a.force)
            jobs = [(os.path.join(a.root, r["path"]), a.image_size) for r in rows]
            x = np.stack(list(tqdm(pool.imap(decode_image, jobs, chunksize=64), total=len(jobs), desc=f"decode {s}")))
            y = np.array([CLASSES.index(r["class"]) for r in rows], dtype=np.int64)
            np.savez(path, x=x, y=y)
            arrays[s] = x

    xt = arrays["train"].astype(np.float64) / 255.0
    mean, std = xt.mean((0, 1, 2)), xt.std((0, 1, 2))
    stats = [{"key": "image_size", "value": a.image_size}, {"key": "seed", "value": a.seed},
             {"key": "max_per_class", "value": a.max_per_class}]
    stats += [{"key": f"mean_{ch}", "value": f"{m:.6f}"} for ch, m in zip("rgb", mean)]
    stats += [{"key": f"std_{ch}", "value": f"{s:.6f}"} for ch, s in zip("rgb", std)]
    write_csv(os.path.join(DATA_DIR, "stats.csv"), stats, a.force)
    print({s: len(r) for s, r in splits.items()}, "mean", mean.round(4), "std", std.round(4), "(train only)")


if __name__ == "__main__":
    main()
