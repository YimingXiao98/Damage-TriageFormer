import os
import re
import numpy as np
import cv2
from PIL import Image
from src.config import COLOR_TO_CAT, FILENAME_REGEX, IGNORE_INDEX

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def read_tif_rgb(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return arr


def read_tif_gray(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def extract_signature(name: str):
    """
    Extract a canonical signature by removing type-specific keywords.
    """
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]
    s = base.lower()
    for kw in ['images', 'image', 'labels', 'label', 'masks', 'mask', 'damage']:
        s = s.replace(kw, '')
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s


def mask_to_instances(mask_bin, min_area=30, dilate=0):
    mb = mask_bin.astype(np.uint8)
    if dilate > 0:
        mb = cv2.dilate(mb, np.ones((dilate, dilate), np.uint8), iterations=1)
    n, cc = cv2.connectedComponents(mb)
    out = []
    for lab in range(1, n):
        inst = (cc == lab).astype(np.uint8)
        if inst.sum() >= min_area:
            out.append(inst)
    return out


def process_unified_mask(damage_path):
    """
    Process a unified mask (colored polygons).
    Returns:
        sem_mask: Binary mask of all buildings (non-black).
        inst_masks: List of instance masks.
        inst_sev: List of severity classes.
    """
    dmg_img = read_tif_rgb(damage_path)  # H,W,3

    # 1. Create Binary Mask (Any non-black pixel)
    # Check if pixel is NOT black (sum > 0 is simple approximation for uint8)
    # Be careful with artifacts, but strict (0,0,0) check is best.
    is_not_black = np.any(dmg_img > 0, axis=-1)
    sem_mask = is_not_black.astype(np.uint8)

    # 2. Extract Instances
    inst_masks = mask_to_instances(sem_mask, min_area=30)

    # 3. Assign Severity
    inst_sev = []
    for m in inst_masks:
        # Get pixels for this instance
        pixels = dmg_img[m == 1]

        # Find majority color
        # We filter out black just in case, though mask shouldn't have it
        # We assume the instance is mostly one color

        # Convert to tuples for counting
        colors = [tuple(p) for p in pixels]
        if not colors:
            inst_sev.append(IGNORE_INDEX)
            continue

        # Count
        from collections import Counter
        counts = Counter(colors)
        most_common_color = counts.most_common(1)[0][0]

        # Map to class
        cls = COLOR_TO_CAT.get(most_common_color, IGNORE_INDEX)

        inst_sev.append(cls)

    return sem_mask, inst_masks, inst_sev


def process_legacy_mask(label_path, mask_path):
    """
    Process legacy mask (binary mask + dot labels).
    """
    lab = read_tif_rgb(label_path)
    msk = read_tif_gray(mask_path)
    msk_bin = (msk > 0).astype(np.uint8)

    insts = mask_to_instances(msk_bin, min_area=30)

    ys, xs, classes = [], [], []
    nonblack = np.any(lab != np.array([0, 0, 0]), axis=-1)
    y_all, x_all = np.where(nonblack)
    for y, x in zip(y_all, x_all):
        c = tuple(lab[y, x])
        cls = COLOR_TO_CAT.get(c, IGNORE_INDEX)
        if cls != IGNORE_INDEX:
            ys.append(y)
            xs.append(x)
            classes.append(cls)

    sev = []
    for m in insts:
        if len(ys) > 0:
            inside = np.where(m[ys, xs] == 1)[0]
            if len(inside) > 0:
                vals = np.array(classes)[inside]
                cls = int(np.bincount(vals).argmax())
            else:
                cls = IGNORE_INDEX
        else:
            cls = IGNORE_INDEX
        sev.append(cls)

    return msk_bin, insts, sev


def make_sample(image_path, label_path=None, mask_path=None, damage_path=None):
    """
    Create a sample.
    Supports two modes:
    1. Unified: Provide damage_path (colored polygons).
    2. Legacy: Provide label_path (dots) and mask_path (binary).
    """
    img = read_tif_rgb(image_path)

    if damage_path:
        # Unified Mode
        sem, insts, sev = process_unified_mask(damage_path)
    else:
        # Legacy Mode
        sem, insts, sev = process_legacy_mask(label_path, mask_path)

    return {
        "image_rgb": img,
        "sem_mask": sem,
        "inst_masks": insts,
        "inst_sev": sev
    }
