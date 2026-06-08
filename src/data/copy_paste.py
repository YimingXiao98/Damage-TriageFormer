"""
Copy-paste augmentation for building damage assessment.

Extracts individual building instances (image crop + mask) organized by damage
class, then pastes rare-class instances onto training tiles at random locations
to balance the class distribution.

Reference: Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation
Method for Instance Segmentation", CVPR 2021.

Rescue-recipe changes over the original implementation (2026-04-13 attempt had
gradient explosions):
  - Alpha feathering: Gaussian-blurred paste mask → soft alpha blending so the
    boundary has no hard pixel step (prevents activation spikes).
  - Hard cap on total pastes per tile (default 2) so loss multipliers can't
    compound across 4-5 pastes in a single sample.
  - Shared-memory paste probability so the main process can ramp it linearly
    over warmup epochs (0 → max_prob) without restarting workers.
"""

import os
import random
import pickle
import time
import multiprocessing as mp
import numpy as np
import cv2
from tqdm import tqdm
from src.data.preprocessing import make_sample
from src.config import NUM_CLASSES, CATEGORIES


BANK_CACHE_FILENAME = "instance_bank.pkl"


class InstanceBank:
    """
    Pre-extracted building instance crops organized by damage class.

    Each entry is (image_crop, mask_crop) where:
    - image_crop: uint8 RGB array, shape (h, w, 3), tight bounding box
    - mask_crop:  uint8 binary array, shape (h, w), same bbox
    """

    def __init__(self, triplets=None, cache_dir=None, min_area=30):
        self.bank = {cls: [] for cls in range(NUM_CLASSES)}
        self.min_area = min_area

        cache_path = os.path.join(cache_dir, BANK_CACHE_FILENAME) if cache_dir else None

        if cache_path and os.path.exists(cache_path):
            self._load(cache_path)
        elif triplets is not None:
            self._build(triplets)
            if cache_path:
                self._save(cache_path)
        else:
            print("[InstanceBank] Warning: no triplets and no cache — bank is empty")

        for cls in range(NUM_CLASSES):
            print(f"  [InstanceBank] Class {cls} ({CATEGORIES[cls]}): "
                  f"{len(self.bank[cls])} instances")

    def _build(self, triplets):
        print(f"[InstanceBank] Building instance bank from {len(triplets)} tiles...")
        t0 = time.perf_counter()

        for img_path, label_p, mask_p, damage_p in tqdm(
                triplets, desc="[InstanceBank] Extracting instances"):
            try:
                sample = make_sample(img_path, label_p, mask_p, damage_p)
            except Exception:
                continue

            img = sample["image_rgb"]  # uint8 (H, W, 3)
            for inst_mask, cls in zip(sample["inst_masks"], sample["inst_sev"]):
                if cls < 0:
                    continue
                area = int(inst_mask.sum())
                if area < self.min_area:
                    continue

                ys, xs = np.where(inst_mask > 0)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1

                self.bank[cls].append((
                    img[y0:y1, x0:x1].copy(),
                    inst_mask[y0:y1, x0:x1].copy(),
                ))

        elapsed = time.perf_counter() - t0
        total = sum(len(v) for v in self.bank.values())
        print(f"[InstanceBank] Extracted {total} instances in {elapsed:.1f}s")

    def _save(self, cache_path):
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.bank, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"[InstanceBank] Saved cache ({size_mb:.1f} MB): {cache_path}")
        except Exception as e:
            print(f"[InstanceBank] Warning: could not save cache: {e}")

    def _load(self, cache_path):
        print(f"[InstanceBank] Loading from cache: {cache_path}")
        t0 = time.perf_counter()
        with open(cache_path, "rb") as f:
            self.bank = pickle.load(f)
        total = sum(len(v) for v in self.bank.values())
        print(f"[InstanceBank] Loaded {total} instances in "
              f"{time.perf_counter() - t0:.1f}s")

    def sample(self, cls, n=1):
        """Return n random (img_crop, mask_crop) from *cls* (with replacement)."""
        items = self.bank[cls]
        if not items:
            return []
        return [random.choice(items) for _ in range(n)]

    def class_counts(self):
        return {cls: len(items) for cls, items in self.bank.items()}


class CopyPasteAugmentation:
    """
    Training-time augmentation that pastes rare-class instances onto tiles.

    Uses a shared ``mp.Value('d')`` for paste_prob so the main process can
    update it (e.g. linearly ramp from 0 → max_prob over the first few epochs)
    and forked DataLoader workers see the change immediately.
    """

    def __init__(self, instance_bank, num_train_tiles,
                 paste_prob=0.3, max_paste_per_class=3,
                 max_overlap=0.3, max_attempts=10,
                 max_total_pastes=2, feather_radius=3):
        self.bank = instance_bank
        self._paste_prob = mp.Value('d', float(paste_prob))
        self.max_overlap = max_overlap
        self.max_attempts = max_attempts
        self.max_total_pastes = int(max_total_pastes)
        self.feather_radius = int(feather_radius)

        counts = instance_bank.class_counts()

        # Only paste the two weakest classes: Total Roof (2) and Partial
        # Structural (3). Class 0 (Undamaged) and 4 (Total Structural) are
        # already well-represented / high-performing; class 1 (Partial Roof)
        # is the most common damaged class.
        paste_eligible = {2, 3}

        damaged = {c: counts.get(c, 0) for c in range(1, NUM_CLASSES)
                   if counts.get(c, 0) > 0}
        if damaged:
            sorted_counts = sorted(damaged.values())
            target = sorted_counts[len(sorted_counts) // 2]  # median
        else:
            target = 0
        effective_tiles = max(num_train_tiles * paste_prob, 1)

        self.paste_rates = {}  # cls -> expected pastes per tile (float)
        for cls in paste_eligible:
            n_have = counts.get(cls, 0)
            if n_have == 0:
                continue
            deficit = target - n_have
            if deficit <= 0:
                continue
            rate = min(deficit / effective_tiles, max_paste_per_class)
            self.paste_rates[cls] = rate

        print(f"[CopyPaste] Target instance count per damaged class: {target} "
              f"(median)")
        print(f"[CopyPaste] Paste-eligible classes: "
              f"{[CATEGORIES[c] for c in sorted(paste_eligible)]}")
        print(f"[CopyPaste] paste_prob={paste_prob} (shared, ramped from main), "
              f"max_total_pastes={max_total_pastes}, "
              f"feather_radius={feather_radius}, "
              f"effective_tiles={effective_tiles:.0f}")
        for cls in sorted(self.paste_rates):
            rate = self.paste_rates[cls]
            print(f"  Class {cls} ({CATEGORIES[cls]}): ~{rate:.2f} pastes/tile  "
                  f"(bank={counts[cls]}, deficit={target - counts[cls]})")

    @property
    def paste_prob(self):
        return float(self._paste_prob.value)

    def set_paste_prob(self, value):
        """Update the shared paste probability (visible to forked workers)."""
        with self._paste_prob.get_lock():
            self._paste_prob.value = float(value)

    def __call__(self, image, sem_mask, inst_masks, inst_sev):
        """
        image:      (H, W, 3) float32 [0, 1]
        sem_mask:   (H, W) uint8
        inst_masks: list of (H, W) uint8
        inst_sev:   list of int
        """
        p = self.paste_prob
        if p <= 0.0 or random.random() > p:
            return image, sem_mask, inst_masks, inst_sev

        h, w = image.shape[:2]
        image = image.copy()
        sem_mask = sem_mask.copy()
        inst_masks = list(inst_masks)
        inst_sev = list(inst_sev)

        # Build shuffled queue of (class, crop) candidates, then apply up to
        # max_total_pastes. This cap is the core of the rescue recipe: with
        # the old code a single tile could absorb 4-5 pastes when per-class
        # rates were high, and loss multipliers (class_weights × focal ×
        # hard-class boost) compounded per paste → gradient explosion.
        queue = []
        for cls, rate in self.paste_rates.items():
            n = int(rate)
            if random.random() < (rate - n):
                n += 1
            for crop in self.bank.sample(cls, n):
                queue.append((cls, crop))
        random.shuffle(queue)

        pasted = 0
        for cls, (img_crop, mask_crop) in queue:
            if pasted >= self.max_total_pastes:
                break
            if self._paste_one(image, sem_mask, inst_masks, inst_sev,
                               img_crop, mask_crop, cls, h, w):
                pasted += 1

        return image, sem_mask, inst_masks, inst_sev

    # ------------------------------------------------------------------
    def _paste_one(self, image, sem_mask, inst_masks, inst_sev,
                   img_crop, mask_crop, cls, h, w):
        img_crop, mask_crop = _augment_crop(img_crop, mask_crop)
        ch, cw = mask_crop.shape
        if ch >= h or cw >= w:
            return False

        for _ in range(self.max_attempts):
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)

            existing = sem_mask[y0:y0 + ch, x0:x0 + cw]
            overlap = float((existing * mask_crop).sum())
            if overlap / max(float(mask_crop.sum()), 1) > self.max_overlap:
                continue

            # Soft alpha blend (feathered edges) — avoids the hard pixel step
            # on paste boundaries that produced large activation gradients in
            # the original attempt.
            mask_bin = (mask_crop > 0).astype(np.float32)
            if self.feather_radius > 0:
                k = 2 * self.feather_radius + 1
                alpha = cv2.GaussianBlur(mask_bin, (k, k),
                                         sigmaX=self.feather_radius)
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                alpha = mask_bin
            alpha_3d = alpha[..., None]

            crop_f = img_crop.astype(np.float32) / 255.0
            region = image[y0:y0 + ch, x0:x0 + cw]
            image[y0:y0 + ch, x0:x0 + cw] = (
                alpha_3d * crop_f + (1.0 - alpha_3d) * region
            )

            # Segmentation masks use the hard binary mask (not feathered) so
            # the supervision signal stays tight. Only the RGB paste is soft.
            sem_mask[y0:y0 + ch, x0:x0 + cw] = np.maximum(
                sem_mask[y0:y0 + ch, x0:x0 + cw], mask_crop)

            new_inst = np.zeros((h, w), dtype=np.uint8)
            new_inst[y0:y0 + ch, x0:x0 + cw] = mask_crop
            inst_masks.append(new_inst)
            inst_sev.append(cls)
            return True

        return False


def _augment_crop(img_crop, mask_crop):
    """Random flip / 90-degree rotation of a building crop."""
    if random.random() < 0.5:
        img_crop = np.fliplr(img_crop).copy()
        mask_crop = np.fliplr(mask_crop).copy()
    if random.random() < 0.5:
        img_crop = np.flipud(img_crop).copy()
        mask_crop = np.flipud(mask_crop).copy()
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        img_crop = np.rot90(img_crop, k).copy()
        mask_crop = np.rot90(mask_crop, k).copy()
    return img_crop, mask_crop
