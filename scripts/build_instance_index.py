"""Build a sharded per-instance index from DamageTriage-Bench masks.

Output: revision/instance_index/shard_N.json, each a list of
  {tile, split, cls, x0, y0, x1, y1, area}
one entry for each building instance (connected component of one class color).

Used by: annotation-sample generation (Track C), per-event analyses (Track B),
ambiguity/error correlation (R3.4). Shardable + resumable like the audit.

  python scripts/build_instance_index.py --shard I --nshards 8
"""
import argparse
import json
import os

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("DINOV3_DATA_ROOT", os.path.join(ROOT, "data"))
COLOR_TO_CAT = {
    (255, 255, 255): 0,
    (0, 255, 83): 1,
    (246, 255, 11): 2,
    (255, 138, 18): 3,
    (255, 0, 0): 4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--damage-dir",
                    default=os.path.join(DATA_ROOT, "tiles_1024", "damage"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "photo_splits.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "instance_index"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not 0 <= args.shard < args.nshards:
        raise SystemExit("--shard must be in [0, --nshards)")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"shard_{args.shard}.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists, skipping")
        return

    with open(args.splits) as f:
        split_all = json.load(f)["all"]
    split_of = {t: s for s, ids in split_all.items() for t in ids}
    ids = sorted(split_of)
    mine = [t for i, t in enumerate(ids) if i % args.nshards == args.shard]

    rows = []
    for k, tid in enumerate(mine):
        bgr = cv2.imread(os.path.join(args.damage_dir, tid + ".png"),
                         cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        mask = bgr[:, :, ::-1]
        for color, cat in COLOR_TO_CAT.items():
            sel = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
            if not sel.any():
                continue
            n, lab, stats, _ = cv2.connectedComponentsWithStats(
                sel.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                rows.append({"tile": tid, "split": split_of[tid], "cls": int(cat),
                             "x0": int(x), "y0": int(y),
                             "x1": int(x + w), "y1": int(y + h),
                             "area": int(area)})
            del lab, stats, sel
        del mask, bgr
        if (k + 1) % 200 == 0:
            print(f"shard {args.shard}: {k+1}/{len(mine)}", flush=True)

    with open(out_path, "w") as f:
        json.dump(rows, f)
    print(f"shard {args.shard} done: {len(mine)} tiles, {len(rows)} instances")


if __name__ == "__main__":
    main()
