"""Predict building damage for a new event with the released crop-mode model.

Inputs are post-event imagery (one or more georeferenced GeoTIFF/COG files,
8-bit RGB, at native resolution) and building footprints (any vector format
GeoPandas reads). Every footprint is cropped exactly as in training (pixel
bounding box plus a 32-pixel margin, clipped to the image, resized to 224x224
with INTER_AREA, JPEG quality-90 round trip, RGB, ImageNet normalization) and
classified by one checkpoint or by the mean probabilities of several (seed
ensemble). The output is one row per footprint: five class probabilities, the
argmax label, its probability, and the probability of any damage.

  python scripts/predict_event.py \
      --images $PROJECT/data/raw/<event>/imagery \
      --footprints $PROJECT/data/raw/<event>/footprints.gpkg \
      --checkpoints $PROJECT/runs/crop_dtf/best.pth \
                    $PROJECT/runs/crop_dtf_s43/best.pth \
                    $PROJECT/runs/crop_dtf_s44/best.pth \
      --out-dir $PROJECT/runs/<event>/<date>

Outputs in --out-dir: predictions.gpkg (footprints with predictions),
predictions.csv (same, without geometry), summary.json, and quicklook.png.
Footprints that cannot be scored keep empty predictions and a `skip_reason`.
"""
import argparse
import glob
import json
import os
import time

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window
from shapely.geometry import box
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 5
CLASS_NAMES = ["Undamaged", "Partial Roof Damage", "Total Roof Damage",
               "Partial Structural Damage", "Total Structural Collapse"]
CLASS_COLORS = [(255, 255, 255), (255, 221, 51), (255, 140, 0), (186, 85, 211), (215, 25, 28)]
BACKBONE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MARGIN = 32
SIZE = 224
MIN_AREA = 30
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GatedDinoCrop(nn.Module):
    """Crop-mode Damage-TriageFormer (same as scripts/train_crop.py --gated)."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(BACKBONE)
        d = self.vit.config.hidden_size
        self.gate = nn.Linear(d, 1)
        self.leaf = nn.Linear(d, NUM_CLASSES - 1)
        self.aux = nn.Linear(d, 1)

    def forward(self, x):
        cls = self.vit(pixel_values=x).last_hidden_state[:, 0]
        g = torch.sigmoid(self.gate(cls))
        q = torch.softmax(self.leaf(cls), dim=-1)
        return torch.cat([1 - g, g * q], dim=1)


def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
                out += glob.glob(os.path.join(p, "**", ext), recursive=True)
        else:
            out += glob.glob(p)
    return sorted(set(out))


def plan_crops(images, fp, min_area):
    """Assign each footprint to the image that contains its centroid and compute
    its pixel bounding box there, as in the training instance index."""
    plan = {}          # footprint index -> (image, x0, y0, x1, y1, area_px)
    reason = {}
    for path in images:
        with rasterio.open(path) as src:
            if src.count < 3:
                raise SystemExit(f"{path}: expected RGB imagery, found {src.count} band(s)")
            if src.dtypes[0] != "uint8":
                raise SystemExit(f"{path}: expected 8-bit imagery, found {src.dtypes[0]}")
            if src.crs is None:
                raise SystemExit(f"{path}: no CRS; imagery must be georeferenced")
            loc = fp.to_crs(src.crs) if fp.crs != src.crs else fp
            px_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
            inv = ~src.transform
            l, b, r, t = src.bounds
            cand = loc.sindex.query(box(l, b, r, t), predicate="intersects")
            for k in cand:
                if k in plan:
                    continue
                geom = loc.geometry.iloc[k]
                c = geom.centroid
                if not (l <= c.x <= r and b <= c.y <= t):
                    continue
                parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                xs, ys = [], []
                for part in parts:
                    for gx, gy in part.exterior.coords:
                        px, py = inv * (gx, gy)
                        xs.append(px); ys.append(py)
                x0, x1 = int(np.floor(min(xs) + 1e-6)), int(np.ceil(max(xs) - 1e-6))
                y0, y1 = int(np.floor(min(ys) + 1e-6)), int(np.ceil(max(ys) - 1e-6))
                area = geom.area / px_area
                if area < min_area:
                    reason[k] = f"footprint smaller than {min_area} image pixels"
                    continue
                plan[k] = (path, x0, y0, x1, y1, float(area))
                reason.pop(k, None)
    for k in range(len(fp)):
        if k not in plan and k not in reason:
            reason[k] = "centroid outside the imagery"
    return plan, reason


class FootprintCrops(Dataset):
    def __init__(self, items, jpeg):
        self.items, self.jpeg, self.handles = items, jpeg, {}

    def __len__(self):
        return len(self.items)

    def _src(self, path):
        if path not in self.handles:
            self.handles[path] = rasterio.open(path)
        return self.handles[path]

    def __getitem__(self, i):
        k, (path, x0, y0, x1, y1, _) = self.items[i]
        src = self._src(path)
        xa, ya = max(0, x0 - MARGIN), max(0, y0 - MARGIN)
        xb, yb = min(src.width, x1 + MARGIN), min(src.height, y1 + MARGIN)
        rgb = src.read((1, 2, 3), window=Window(xa, ya, xb - xa, yb - ya))   # (3,h,w)
        nodata_frac = float((rgb.max(axis=0) == 0).mean())
        bgr = np.ascontiguousarray(rgb.transpose(1, 2, 0)[:, :, ::-1])
        crop = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if self.jpeg:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        x = (crop[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), k, nodata_frac


def quicklook(gdf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if gdf.crs is not None and gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    done = gdf[gdf["label"].notna()]
    if gdf["label"].isna().any():
        gdf[gdf["label"].isna()].plot(ax=ax, facecolor="none", edgecolor="#9a9a9a", linewidth=0.2)
    for c in range(NUM_CLASSES):
        sub = done[done["label"] == c]
        if len(sub):
            sub.plot(ax=ax, color=np.array(CLASS_COLORS[c]) / 255, edgecolor="#333333", linewidth=0.2)
    ax.legend(handles=[Patch(facecolor=np.array(CLASS_COLORS[c]) / 255, edgecolor="#333333",
                             label=f"{CLASS_NAMES[c]} ({int((done['label'] == c).sum())})")
                       for c in range(NUM_CLASSES)], loc="lower left", fontsize=7, frameon=True)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="GeoTIFF/COG files, globs, or directories")
    ap.add_argument("--footprints", required=True, help="building footprints (GPKG, GeoJSON, SHP, ...)")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one checkpoint, or several to average (seed ensemble)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-area", type=float, default=MIN_AREA,
                    help="skip footprints smaller than this many image pixels (training used 30)")
    ap.add_argument("--no-jpeg", action="store_true",
                    help="skip the JPEG quality-90 round trip applied to training crops")
    ap.add_argument("--max-nodata", type=float, default=0.5,
                    help="skip crops whose fraction of all-zero pixels exceeds this")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N footprints (testing)")
    ap.add_argument("--no-quicklook", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    images = list_images(args.images)
    if not images:
        raise SystemExit("no GeoTIFF imagery found")
    fp = gpd.read_file(args.footprints)
    fp = fp[fp.geometry.notna() & ~fp.geometry.is_empty].reset_index(drop=True)
    fp = fp[fp.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    print(f"{len(images)} image(s), {len(fp)} footprint polygons", flush=True)

    plan, reason = plan_crops(images, fp, args.min_area)
    items = sorted(plan.items())
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} footprints to score; {len(reason)} skipped", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(FootprintCrops(items, not args.no_jpeg), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers, pin_memory=dev.type == "cuda")
    probs_sum = {}
    nodata = {}
    for ci, ckpt in enumerate(args.checkpoints):
        model = GatedDinoCrop()
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.to(dev).eval()
        n = 0
        with torch.no_grad():
            for x, ks, nd in loader:
                p = model(x.to(dev, non_blocking=True)).float().cpu().numpy()
                for k, pk, f in zip(ks.tolist(), p, nd.tolist()):
                    probs_sum[k] = probs_sum.get(k, 0) + pk
                    nodata[k] = f
                n += len(ks)
        print(f"checkpoint {ci + 1}/{len(args.checkpoints)} ({ckpt}): {n} crops, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    m = len(args.checkpoints)
    cols = {f"p{c}": np.full(len(fp), np.nan) for c in range(NUM_CLASSES)}
    label = np.full(len(fp), np.nan)
    src_img = np.full(len(fp), None, dtype=object)
    skip = np.full(len(fp), None, dtype=object)
    for k, r in reason.items():
        skip[k] = r
    for k, (path, *_rest) in items:
        if nodata.get(k, 0) > args.max_nodata:
            skip[k] = f"more than {args.max_nodata:.0%} of the crop is no-data"
            continue
        p = probs_sum[k] / m
        for c in range(NUM_CLASSES):
            cols[f"p{c}"][k] = p[c]
        label[k] = int(np.argmax(p))
        src_img[k] = os.path.basename(path)
    out = fp.copy()
    for c in range(NUM_CLASSES):
        out[f"p{c}"] = cols[f"p{c}"]
    out["label"] = label
    out["label_name"] = [CLASS_NAMES[int(v)] if not np.isnan(v) else None for v in label]
    out["conf"] = out[[f"p{c}" for c in range(NUM_CLASSES)]].max(axis=1)
    out["p_damaged"] = 1 - out["p0"]
    out["n_models"] = np.where(np.isnan(label), 0, m)
    out["src_image"] = src_img
    out["skip_reason"] = skip

    out.to_file(os.path.join(args.out_dir, "predictions.gpkg"), driver="GPKG")
    out.drop(columns="geometry").to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    scored = out[out["label"].notna()]
    summary = {
        "images": images, "footprints": args.footprints, "checkpoints": args.checkpoints,
        "n_footprints": int(len(out)), "n_scored": int(len(scored)),
        "n_skipped": int(len(out) - len(scored)),
        "skip_reasons": {str(k): int(v) for k, v in out["skip_reason"].value_counts().items()},
        "class_counts": {CLASS_NAMES[c]: int((scored["label"] == c).sum()) for c in range(NUM_CLASSES)},
        "mean_conf": float(scored["conf"].mean()) if len(scored) else None,
        "jpeg_round_trip": not args.no_jpeg, "min_area_px": args.min_area,
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    if not args.no_quicklook and len(scored):
        try:
            quicklook(out, os.path.join(args.out_dir, "quicklook.png"))
        except Exception as e:   # the predictions are already written
            print(f"quicklook skipped: {e}", flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
