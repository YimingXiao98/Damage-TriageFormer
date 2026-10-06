"""Round-2 crop-mode experiments (R2 second-round comments 1 and 4).

Extends scripts/train_crop.py with the controls of the revised paper's
Table 6 and the crop-mode leave-one-event-out models:

  --variant     dinov3 head: plain (flat 5-class CE), gated (two-stage gate +
                leaf, no aux), aux (flat CE + aux severity), gated_aux
                (Damage-TriageFormer crop mode = revision-1 --gated)
  --recipe      crop: revision-1 crop recipe (full fine-tuning, one LR,
                cosine, no LA/EMA). tile: the tile-mode training recipe applied
                to crops (last 4 blocks + all LayerNorms trainable, backbone
                LR x0.1, head LR 5e-5, logit adjustment tau=1 on gate/leaf,
                EMA 0.9995, 30 epochs, batch 32, 2-epoch warm-up + cosine)
  --crop-mode   upsampled: revision-1 crops (bbox + 32 px margin resized to
                224x224). native: the same bbox + margin rectangle kept at
                native pixel scale (downsampled only if larger than 224),
                centred on a 224x224 mean-colour canvas, so each ViT patch
                covers 16 native pixels as in tile mode
  --seed        training seed

With --variant gated_aux --recipe crop --crop-mode upsampled --seed 42 this
reproduces the revision-1 crop_dtf configuration; with --variant plain it
reproduces crop_dinov3; --arch resnet50/vit_b_16 reproduce the ImageNet crop
baselines.

  python scripts/train_crop_r2.py --prepare --crops /path/to/crops [--crop-mode native] \
      --index-dir instance_index
  python scripts/train_crop_r2.py --crops /path/to/crops --index-dir instance_index \
      --variant gated_aux --recipe crop --seed 43 --name r2_crop_dtf_s43
"""
import argparse
import copy
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.environ.get("DINOV3_DATA_ROOT", os.path.join(ROOT, "data"))
IMG_DIR = os.path.join(DATA_ROOT, "tiles_1024", "images")
INDEX_DIR = os.path.join(ROOT, "instance_index")
NUM_CLASSES = 5
MARGIN = 32
MIN_AREA = 30
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_BGR = tuple(int(round(v * 255)) for v in MEAN[::-1])
SEV_TARGETS = [0.0, 0.3, 0.7, 0.5, 1.0]


def load_instances():
    rows = []
    for shard in range(8):
        rows += json.load(open(os.path.join(INDEX_DIR,
                                            f"shard_{shard}.json")))
    return [r for r in rows if r["cls"] >= 0 and r["area"] >= MIN_AREA]


def make_crop(img, r, mode):
    x0 = max(0, r["x0"] - MARGIN); y0 = max(0, r["y0"] - MARGIN)
    x1 = min(1024, r["x1"] + MARGIN); y1 = min(1024, r["y1"] + MARGIN)
    patch = img[y0:y1, x0:x1]
    if mode == "upsampled":
        return cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    h, w = patch.shape[:2]
    s = min(1.0, SIZE / max(h, w))
    if s < 1.0:
        patch = cv2.resize(patch, (max(1, round(w * s)), max(1, round(h * s))),
                           interpolation=cv2.INTER_AREA)
        h, w = patch.shape[:2]
    canvas = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
    canvas[:] = PAD_BGR
    oy, ox = (SIZE - h) // 2, (SIZE - w) // 2
    canvas[oy:oy + h, ox:ox + w] = patch
    return canvas


def prepare(crops_dir, mode):
    os.makedirs(crops_dir, exist_ok=True)
    rows = load_instances()
    by_tile = defaultdict(list)
    for i, r in enumerate(rows):
        by_tile[r["tile"]].append((i, r))
    meta = {}
    t0 = time.time()
    for k, (tile, items) in enumerate(sorted(by_tile.items())):
        img = cv2.imread(os.path.join(IMG_DIR, tile + ".png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for i, r in items:
            cv2.imwrite(os.path.join(crops_dir, f"{i:06d}.jpg"),
                        make_crop(img, r, mode), [cv2.IMWRITE_JPEG_QUALITY, 90])
            meta[i] = {"tile": r["tile"], "cls": r["cls"]}
        if (k + 1) % 500 == 0:
            print(f"prepare[{mode}]: {k+1}/{len(by_tile)} tiles "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(crops_dir, "meta.json"), "w") as f:
        json.dump({"crop_mode": mode, "meta": meta}, f)
    print(f"prepared {len(meta)} {mode} crops -> {crops_dir}")


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


def predict(model, loader, dev):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            out = model(x.to(dev, non_blocking=True)).float()
            if not getattr(model, "returns_probs", False):
                out = torch.softmax(out, dim=-1)
            probs.append(out.cpu())
    return torch.cat(probs).numpy()


def evaluate(model, loader, labels, dev):
    probs = predict(model, loader, dev)
    return macro_f1(np.asarray(labels), probs.argmax(1))


class DinoCrop(nn.Module):
    """DINOv3-L on a building crop. Heads by variant: flat 5-way classifier
    (plain, aux), two-stage gate + 4-way leaf (gated, gated_aux), and a
    training-only aux severity regressor (aux, gated_aux)."""

    def __init__(self, variant):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m")
        d = self.vit.config.hidden_size
        self.variant = variant
        self.gated = variant in ("gated", "gated_aux")
        self.use_aux = variant in ("aux", "gated_aux")
        self.returns_probs = self.gated
        if self.gated:
            self.gate = nn.Linear(d, 1)
            self.leaf = nn.Linear(d, NUM_CLASSES - 1)
        else:
            self.head = nn.Linear(d, NUM_CLASSES)
        if self.use_aux:
            self.aux = nn.Linear(d, 1)

    def heads(self, x):
        cls = self.vit(pixel_values=x).last_hidden_state[:, 0]
        aux = torch.sigmoid(self.aux(cls)).squeeze(-1) if self.use_aux else None
        if self.gated:
            return self.gate(cls).squeeze(-1), self.leaf(cls), aux
        return None, self.head(cls), aux

    def forward(self, x):
        g_logit, logits, _ = self.heads(x)
        if not self.gated:
            return logits
        g = torch.sigmoid(g_logit).unsqueeze(1)
        return torch.cat([1 - g, g * torch.softmax(logits, dim=-1)], dim=1)


def partial_finetune(model, unfreeze_blocks=4):
    """Tile-recipe freezing: last N blocks + every LayerNorm trainable."""
    vit = model.vit
    for p in vit.parameters():
        p.requires_grad_(False)
    layers = vit.layer if hasattr(vit, "layer") else vit.encoder.layer
    for blk in list(layers)[len(layers) - unfreeze_blocks:]:
        for p in blk.parameters():
            p.requires_grad_(True)
    for m in vit.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
    n_tr = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in vit.parameters())
    print(f"[partial FT] backbone trainable {n_tr:,}/{n_all:,} "
          f"({100*n_tr/n_all:.1f}%)")


def main():
    global IMG_DIR, INDEX_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--crop-mode", choices=["upsampled", "native"],
                    default="upsampled")
    ap.add_argument("--arch", choices=["resnet50", "vit_b_16", "dinov3_vitl16"],
                    default="dinov3_vitl16")
    ap.add_argument("--variant", choices=["plain", "gated", "aux", "gated_aux"],
                    default="gated_aux")
    ap.add_argument("--recipe", choices=["crop", "tile"], default="crop")
    ap.add_argument("--splits", default=os.path.join(ROOT, "photo_splits.json"))
    ap.add_argument("--image-dir", default=IMG_DIR)
    ap.add_argument("--index-dir", default=INDEX_DIR,
                    help="output of scripts/build_instance_index.py")
    ap.add_argument("--name", required=False, default="r2_crop")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug-n", type=int, default=0,
                    help="smoke test: keep only the first N crops per split")
    args = ap.parse_args()
    IMG_DIR, INDEX_DIR = args.image_dir, args.index_dir

    if args.prepare:
        prepare(args.crops, args.crop_mode)
        return

    tile = args.recipe == "tile"
    if args.arch == "dinov3_vitl16":
        d_ep, d_bs, d_lr = (30, 32, 5e-5) if tile else (20, 48, 3e-5)
    else:
        if tile:
            raise SystemExit("--recipe tile is defined for dinov3 only")
        d_ep, d_bs, d_lr = 20, 256, 1e-4
    epochs = args.epochs or d_ep
    bs = args.batch or d_bs
    lr = args.lr or d_lr

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    blob = json.load(open(os.path.join(args.crops, "meta.json")))
    assert blob["crop_mode"] == args.crop_mode, (blob["crop_mode"], args.crop_mode)
    meta = {int(k): v for k, v in blob["meta"].items()}
    split_of = {t: s for s, tids in json.load(open(args.splits))["all"].items()
                for t in tids}

    ids = {s: [] for s in ("train", "val", "test")}
    labels = {s: [] for s in ("train", "val", "test")}
    for i in sorted(meta):
        s = split_of.get(meta[i]["tile"])
        if s:
            ids[s].append(i)
            labels[s].append(meta[i]["cls"])
    if args.debug_n:
        for s in ids:
            keep = np.random.RandomState(0).permutation(len(ids[s]))[:args.debug_n]
            ids[s] = [ids[s][k] for k in sorted(keep)]
            labels[s] = [labels[s][k] for k in sorted(keep)]
    for s in ids:
        print(f"{s}: {len(ids[s])} crops")

    dev = "cuda"
    import torchvision.models as tvm
    if args.arch == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        variant = "plain"
    elif args.arch == "vit_b_16":
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
        variant = "plain"
    else:
        model = DinoCrop(args.variant)
        variant = args.variant
        if tile:
            partial_finetune(model, 4)
    model = model.to(dev)
    gated = variant in ("gated", "gated_aux")
    use_aux = variant in ("aux", "gated_aux")

    cnt = np.bincount(labels["train"], minlength=NUM_CLASSES)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = w / w.sum() * NUM_CLASSES
    print("class counts:", cnt.tolist(), " weights:", np.round(w, 3).tolist())
    flat_crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=dev),
        label_smoothing=0.1)
    leaf_crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w[1:] / w[1:].sum() * (NUM_CLASSES - 1),
                            dtype=torch.float32, device=dev),
        label_smoothing=0.1)
    sev_t = torch.tensor(SEV_TARGETS, device=dev)

    # Logit adjustment (tile recipe only): shift gate by log-odds of the
    # damaged prior and leaf logits by log conditional priors, training only.
    gate_off = leaf_off = None
    if tile and gated:
        p_d = cnt[1:].sum() / cnt.sum()
        gate_off = float(math.log(p_d) - math.log(1 - p_d))
        leaf_off = torch.log(torch.tensor(cnt[1:] / cnt[1:].sum(),
                                          dtype=torch.float32, device=dev))
        print(f"[LA] tau=1 gate_off={gate_off:.3f} leaf_off="
              f"{leaf_off.cpu().numpy().round(3).tolist()}")

    if tile:
        head_p = [p for n, p in model.named_parameters()
                  if not n.startswith("vit.") and p.requires_grad]
        bb_p = [p for n, p in model.named_parameters()
                if n.startswith("vit.") and p.requires_grad]
        opt = torch.optim.AdamW([{"params": head_p, "lr": lr},
                                 {"params": bb_p, "lr": lr * 0.1}],
                                weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    mk = lambda s, tr: DataLoader(CropDS(args.crops, ids[s], labels[s], tr),
                                  batch_size=bs, shuffle=tr,
                                  num_workers=args.workers, pin_memory=True,
                                  drop_last=tr, persistent_workers=tr)
    tl, vl, sl = mk("train", True), mk("val", False), mk("test", False)
    steps_per_ep = len(tl)
    if tile:
        warm = 2 * steps_per_ep
        total = epochs * steps_per_ep
        lam = lambda t: (t + 1) / warm if t < warm else \
            0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ema = None
    if tile:
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

    out_dir = os.path.join(os.environ.get("DINOV3_RUNS_ROOT", "./runs"), args.name)
    os.makedirs(out_dir, exist_ok=True)
    config = {"arch": args.arch, "variant": variant, "recipe": args.recipe,
              "crop_mode": args.crop_mode, "seed": args.seed, "epochs": epochs,
              "batch": bs, "lr": lr, "splits": args.splits,
              "la": gate_off is not None, "ema": ema is not None,
              "selection": "best val macro F1 over epochs"
                           + (" (EMA weights)" if ema is not None else "")}
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    print(json.dumps(config))

    best, best_ep, hist = -1.0, -1, []
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if True:  # fp32 throughout, matching the revision-1 crop runs
                if args.arch != "dinov3_vitl16":
                    loss = flat_crit(model(x).float(), y)
                else:
                    g_logit, logits, aux_pred = model.heads(x)
                    logits = logits.float()
                    if gated:
                        g_logit = g_logit.float()
                        if gate_off is not None:
                            g_logit = g_logit + gate_off
                        loss = F.binary_cross_entropy_with_logits(
                            g_logit, (y > 0).float())
                        dmg = y > 0
                        if dmg.any():
                            ll = logits[dmg]
                            if leaf_off is not None:
                                ll = ll + leaf_off
                            loss = loss + 2.0 * leaf_crit(ll, y[dmg] - 1)
                    else:
                        loss = flat_crit(logits, y)
                    if use_aux:
                        loss = loss + 0.5 * F.smooth_l1_loss(
                            aux_pred.float(), sev_t[y])
            loss.backward()
            opt.step()
            if tile:
                sched.step()
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(0.9995).add_(pm.detach(), alpha=0.0005)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            tot += loss.item() * len(y); n += len(y)
        if not tile:
            sched.step()
        eval_model = ema if ema is not None else model
        vf1, _ = evaluate(eval_model, vl, labels["val"], dev)
        hist.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_macro_f1": vf1})
        print(f"ep {ep+1}/{epochs} loss {tot/max(n,1):.4f} val macroF1 {vf1:.4f}",
              flush=True)
        if vf1 > best:
            best, best_ep = vf1, ep + 1
            torch.save(eval_model.state_dict(), os.path.join(out_dir, "best.pth"))

    final = ema if ema is not None else model
    final.load_state_dict(torch.load(os.path.join(out_dir, "best.pth")))
    res = dict(config)
    res["best_epoch"] = best_ep
    res["history"] = hist
    for s, loader in (("val", vl), ("test", sl)):
        probs = predict(final, loader, dev)
        f1, per = macro_f1(np.asarray(labels[s]), probs.argmax(1))
        res[f"{s}_macro_f1"], res[f"{s}_per_class_f1"] = f1, per
        res[f"{s}_n"] = len(labels[s])
        dump_rows(probs, ids[s], s, out_dir, args)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))


def dump_rows(probs, split_ids, s, out_dir, args):
    """Per-instance dump, schema-compatible with revision/analyze_dump.py."""
    all_rows = load_instances()   # crop id i == i-th filtered index row
    rows = []
    for k, i in enumerate(split_ids):
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
