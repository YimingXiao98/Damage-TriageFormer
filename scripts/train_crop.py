"""Train the building-crop classifiers used in the revised paper.

Standard CNN/ViT classifiers on crops of individual buildings, the conventional
formulation prior work uses: each building instance is cropped from the tile
(bbox + margin), resized to 224, and classified into the 5 damage-typology
classes. The gated DINOv3 option is the paper's primary crop configuration;
ResNet-50, ViT-B/16, and plain DINOv3 provide crop baselines. Checkpoints are
selected by validation macro F1.

Stage 1 (--prepare) extracts crops to a directory once (node-local $TMPDIR in
Slurm). Stage 2 (default) trains and evaluates.

  python scripts/build_instance_index.py
  python scripts/train_crop.py --prepare --crops /path/to/crops
  python scripts/train_crop.py --arch dinov3_vitl16 --gated \
      --crops /path/to/crops --name crop_dtf
"""
import argparse
import json
import os
import time
from collections import defaultdict
from glob import glob

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("DINOV3_DATA_ROOT", os.path.join(ROOT, "data"))
NUM_CLASSES = 5
MARGIN = 32
MIN_AREA = 30
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_instances(index_dir):
    rows = []
    paths = sorted(glob(os.path.join(index_dir, "shard_*.json")))
    if not paths:
        raise FileNotFoundError(
            f"no shard_*.json files under {index_dir}; run "
            "scripts/build_instance_index.py first")
    for path in paths:
        with open(path) as f:
            rows += json.load(f)
    return [r for r in rows if r["cls"] >= 0 and r["area"] >= MIN_AREA]


def prepare(crops_dir, image_dir, index_dir):
    os.makedirs(crops_dir, exist_ok=True)
    rows = load_instances(index_dir)
    by_tile = defaultdict(list)
    for i, r in enumerate(rows):
        by_tile[r["tile"]].append((i, r))
    meta = {}
    t0 = time.time()
    for k, (tile, items) in enumerate(sorted(by_tile.items())):
        img = cv2.imread(os.path.join(image_dir, tile + ".png"),
                         cv2.IMREAD_COLOR)
        if img is None:
            continue
        for i, r in items:
            x0 = max(0, r["x0"] - MARGIN); y0 = max(0, r["y0"] - MARGIN)
            x1 = min(1024, r["x1"] + MARGIN); y1 = min(1024, r["y1"] + MARGIN)
            crop = cv2.resize(img[y0:y1, x0:x1], (224, 224),
                              interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(crops_dir, f"{i:06d}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            meta[i] = {"tile": r["tile"], "cls": r["cls"]}
        if (k + 1) % 500 == 0:
            print(f"prepare: {k+1}/{len(by_tile)} tiles "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(crops_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"prepared {len(meta)} crops -> {crops_dir}")


class CropDS(Dataset):
    def __init__(self, crops_dir, ids, labels, train):
        self.dir, self.ids, self.labels, self.train = crops_dir, ids, labels, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        img = cv2.imread(os.path.join(self.dir, f"{i:06d}.jpg"),
                         cv2.IMREAD_COLOR)[:, :, ::-1]
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), \
            self.labels[k]


def macro_f1(gt, pred, k=NUM_CLASSES):
    f1s = []
    for c in range(k):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)), f1s


def evaluate(model, loader, dev):
    model.eval()
    gts, prs = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(dev, non_blocking=True))
            prs.append(out.argmax(1).cpu().numpy())
            gts.append(y.numpy())
    return macro_f1(np.concatenate(gts), np.concatenate(prs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--images",
                    default=os.path.join(DATA_ROOT, "tiles_1024", "images"))
    ap.add_argument("--index-dir", default=os.path.join(ROOT, "instance_index"))
    ap.add_argument("--arch", choices=["resnet50", "vit_b_16", "dinov3_vitl16"],
                    default="resnet50")
    ap.add_argument("--dump-only", action="store_true",
                    help="skip training; load best.pth and dump per-instance "
                         "val+test predictions for paired analysis")
    ap.add_argument("--gated", action="store_true",
                    help="dinov3 arch only: two-stage gated head + aux "
                         "severity (Damage-TriageFormer crop mode)")
    ap.add_argument("--splits", default=os.path.join(ROOT, "photo_splits.json"))
    ap.add_argument("--name", default="crop_baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.prepare:
        prepare(args.crops, args.images, args.index_dir)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    meta = {int(k): v for k, v in
            json.load(open(os.path.join(args.crops, "meta.json"))).items()}
    split_all = json.load(open(args.splits))["all"]
    split_of = {t: s for s, ids in split_all.items() for t in ids}

    ids = {s: [] for s in ("train", "val", "test")}
    labels = {s: [] for s in ("train", "val", "test")}
    for i, m in meta.items():
        s = split_of.get(m["tile"])
        if s:
            ids[s].append(i)
            labels[s].append(m["cls"])
    for s in ids:
        print(f"{s}: {len(ids[s])} crops")

    dev = "cuda"
    import torchvision.models as tvm
    if args.arch == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif args.arch == "vit_b_16":
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    else:  # dinov3_vitl16: same backbone as the tile model, crop-conditioned
        from transformers import AutoModel

        class DinoCropClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.vit = AutoModel.from_pretrained(
                    "facebook/dinov3-vitl16-pretrain-lvd1689m")
                self.head = nn.Linear(self.vit.config.hidden_size, NUM_CLASSES)

            def forward(self, x):
                out = self.vit(pixel_values=x)
                cls = out.last_hidden_state[:, 0]
                return self.head(cls)

        class GatedDinoCrop(nn.Module):
            """Damage-TriageFormer (crop mode): DINOv3-L backbone with the
            paper's two-stage gated head + training-only aux severity head."""
            returns_probs = True

            def __init__(self):
                super().__init__()
                self.vit = AutoModel.from_pretrained(
                    "facebook/dinov3-vitl16-pretrain-lvd1689m")
                d = self.vit.config.hidden_size
                self.gate = nn.Linear(d, 1)
                self.leaf = nn.Linear(d, NUM_CLASSES - 1)
                self.aux = nn.Linear(d, 1)

            def heads(self, x):
                cls = self.vit(pixel_values=x).last_hidden_state[:, 0]
                return (self.gate(cls).squeeze(-1), self.leaf(cls),
                        torch.sigmoid(self.aux(cls)).squeeze(-1))

            def forward(self, x):
                g_logit, leaf_logits, _ = self.heads(x)
                g = torch.sigmoid(g_logit).unsqueeze(1)
                q = torch.softmax(leaf_logits, dim=-1)
                return torch.cat([1 - g, g * q], dim=1)   # (N,5) probs

        model = GatedDinoCrop() if args.gated else DinoCropClassifier()
    model = model.to(dev)

    cnt = np.bincount(labels["train"], minlength=NUM_CLASSES)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = w / w.sum() * NUM_CLASSES
    print("class counts:", cnt.tolist(), " weights:", np.round(w, 3).tolist())
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32,
                                                   device=dev),
                               label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    mk = lambda s, tr: DataLoader(CropDS(args.crops, ids[s], labels[s], tr),
                                  batch_size=args.batch, shuffle=tr,
                                  num_workers=args.workers, pin_memory=True,
                                  drop_last=tr)
    tl, vl, sl = mk("train", True), mk("val", False), mk("test", False)

    out_dir = os.path.join(os.environ.get("DINOV3_RUNS_ROOT", "./runs"), args.name)
    os.makedirs(out_dir, exist_ok=True)
    if args.dump_only:
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth")))
        dump_predictions(model, args, ids, labels, out_dir, dev)
        return
    best = -1.0
    if args.gated:
        # mirror the tile recipe: unweighted gate BCE, weighted 4-way leaf CE
        # (leaf weight 2.0), aux severity smooth-L1 (weight 0.5)
        leaf_w = torch.tensor(w[1:] / w[1:].sum() * (NUM_CLASSES - 1),
                              dtype=torch.float32, device=dev)
        leaf_crit = nn.CrossEntropyLoss(weight=leaf_w, label_smoothing=0.1)
        sev_targets = torch.tensor([0.0, 0.3, 0.7, 0.5, 1.0], device=dev)

    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if args.gated:
                g_logit, leaf_logits, aux_pred = model.heads(x)
                gate_t = (y > 0).float()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    g_logit, gate_t)
                dmg = y > 0
                if dmg.any():
                    loss = loss + 2.0 * leaf_crit(leaf_logits[dmg], y[dmg] - 1)
                loss = loss + 0.5 * nn.functional.smooth_l1_loss(
                    aux_pred, sev_targets[y])
            else:
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        vf1, _ = evaluate(model, vl, dev)
        print(f"ep {ep+1}/{args.epochs} loss {tot/max(n,1):.4f} "
              f"val macroF1 {vf1:.4f}", flush=True)
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth")))
    vf1, vper = evaluate(model, vl, dev)
    tf1, tper = evaluate(model, sl, dev)
    res = {"arch": args.arch, "splits": args.splits, "epochs": args.epochs,
           "val_macro_f1": vf1, "val_per_class_f1": vper,
           "test_macro_f1": tf1, "test_per_class_f1": tper}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    dump_predictions(model, args, ids, labels, out_dir, dev)


def dump_predictions(model, args, ids, labels, out_dir, dev):
    """Per-instance val/test dumps (tile, bbox, gt, probs) for paired analysis;
    schema-compatible with revision/analyze_dump.py."""
    all_rows = load_instances(args.index_dir)  # crop id == filtered row index
    model.eval()
    for s in ("val", "test"):
        dl = DataLoader(CropDS(args.crops, ids[s], labels[s], False),
                        batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        probs = []
        with torch.no_grad():
            for x, _ in dl:
                out = model(x.to(dev))
                if not getattr(model, "returns_probs", False):
                    out = torch.softmax(out, dim=-1)
                probs.append(out.cpu())
        probs = torch.cat(probs).numpy()
        rows = []
        for k, i in enumerate(ids[s]):
            r = all_rows[i]
            rows.append({"tile": r["tile"], "gt": int(r["cls"]),
                         "bbox": [r["x0"], r["y0"], r["x1"], r["y1"]],
                         "probs": [round(float(p), 6) for p in probs[k]]})
        out = os.path.join(out_dir, f"dump_{s}.json")
        with open(out, "w") as f:
            json.dump({"checkpoint": os.path.join(out_dir, "best.pth"),
                       "splits": args.splits, "split_key": s, "rows": rows}, f)
        print(f"dumped {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
