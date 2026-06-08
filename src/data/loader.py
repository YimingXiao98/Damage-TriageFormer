import os
import glob
import time
import math
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
from src.data.preprocessing import extract_signature
from src.data.dataset import BuildingSegDataset
from src.data.balanced_sampler import create_balanced_sampler
from src.config import BATCH_SIZE, IM_SIZE, COLOR_TO_CAT, DATA_ROOT, TILE_IMAGE_DIR, TILE_DAMAGE_DIR
import json
import multiprocessing
from functools import partial


class DistributedWeightedSampler(Sampler):
    """Distributed sampler that supports per-sample weights.

    Combines DistributedSampler's cross-rank sharding with
    WeightedRandomSampler's weighted sampling. Each rank draws its
    own weighted sample from the full dataset, with non-overlapping
    indices guaranteed by rank-specific seeding.
    """

    def __init__(self, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, seed=0):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.total_size = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + self.rank)
        indices = torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement,
            generator=g)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


TILE_TRIPLET_CACHE_NAME = "tile_triplets_cache.json"


def list_tile_triplets(tile_image_dir=TILE_IMAGE_DIR, tile_damage_dir=TILE_DAMAGE_DIR,
                       compute_weights=True):
    """
    List triplets from pre-cropped 1024x1024 tiles.

    Tiles have matching filenames in images/ and damage/ directories.
    Returns triplets in unified mask format: (image_path, None, None, damage_path).

    If `compute_weights` is True, derives per-tile sample weights from
    `class_index.json` (Total Roof present -> 10x, Partial Structural present -> 8x).
    Avoids decoding every PNG (used to take ~12 min for 7.5k tiles).
    Returns weights == [] if `compute_weights` is False (caller supplies its own
    sampler, e.g. BalancedClassSampler).

    Caches the triplet list + per-tile weights in
    ``<tiles_parent>/tile_triplets_cache.json`` and reuses it when the cache
    file is newer than both tile dirs. On a shared filesystem, this skips a
    glob over thousands of files + a membership test per file on every run.
    """
    t0 = time.perf_counter()

    if not os.path.exists(tile_image_dir):
        print(
            f"[Tile loader] Warning: Tile image dir not found: {tile_image_dir}")
        return [], []

    # Try cache first.
    cache_path = os.path.join(os.path.dirname(tile_image_dir),
                              TILE_TRIPLET_CACHE_NAME)
    cache_hit = _load_tile_triplet_cache(cache_path, tile_image_dir,
                                         tile_damage_dir, compute_weights)
    if cache_hit is not None:
        triplets, weights = cache_hit
        print(f"[Tile loader] Cache hit ({cache_path}): "
              f"{len(triplets)} tile triplets "
              f"(weights={'yes' if weights else 'skipped'}) "
              f"in {time.perf_counter() - t0:.2f}s")
        return triplets, weights

    # --- cold scan ---
    t_scan = time.perf_counter()
    image_files = sorted(glob.glob(os.path.join(tile_image_dir, "*.png")))
    print(f"[Tile loader] Found {len(image_files)} tile images in "
          f"{tile_image_dir} (glob took {time.perf_counter() - t_scan:.2f}s)")

    # Build a set of damage basenames in one readdir, instead of calling
    # os.path.exists() once per image (7472 metadata round-trips on GPFS).
    t_dmg = time.perf_counter()
    try:
        damage_set = {fn for fn in os.listdir(tile_damage_dir)
                      if fn.endswith(".png")}
    except FileNotFoundError:
        print(f"[Tile loader] Warning: Tile damage dir not found: "
              f"{tile_damage_dir}")
        return [], []
    print(f"[Tile loader] Listed {len(damage_set)} damage tiles in "
          f"{time.perf_counter() - t_dmg:.2f}s")

    t_pair = time.perf_counter()
    triplets = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        if basename in damage_set:
            triplets.append((img_path, None, None,
                             os.path.join(tile_damage_dir, basename)))
    print(f"[Tile loader] Paired {len(triplets)} triplets in "
          f"{time.perf_counter() - t_pair:.2f}s")

    weights = []
    if compute_weights:
        t_w = time.perf_counter()
        # Derive per-tile rare-class weights from class_index.json instead of
        # decoding every PNG. Schema: keys "0".."3" map to model classes 1..4
        # (matches BalancedClassSampler convention).
        class_index_path = os.path.join(
            os.path.dirname(tile_image_dir), "class_index.json")
        rare_total_roof = set()       # model class 2
        rare_partial_struct = set()   # model class 3
        if os.path.exists(class_index_path):
            with open(class_index_path) as f:
                class_index = json.load(f)
            rare_total_roof = set(class_index.get("1", []))
            rare_partial_struct = set(class_index.get("2", []))
        else:
            print(f"[Tile loader] Warning: {class_index_path} not found; "
                  "using uniform weights")

        for img_path, _, _, _ in triplets:
            basename = os.path.basename(img_path)
            w = 1.0
            if basename in rare_total_roof:
                w += 9.0      # 10x total
            if basename in rare_partial_struct:
                w += 7.0      # 8x total
            weights.append(w)
        print(f"[Tile loader] Computed weights in "
              f"{time.perf_counter() - t_w:.2f}s")

    _save_tile_triplet_cache(cache_path, tile_image_dir, tile_damage_dir,
                             triplets, weights)

    print(f"[Tile loader] Loaded {len(triplets)} tile triplets "
          f"(total {time.perf_counter() - t0:.2f}s)")
    return triplets, weights


def _load_tile_triplet_cache(cache_path, tile_image_dir, tile_damage_dir,
                             compute_weights):
    """
    Return (triplets, weights) if the cache at ``cache_path`` is usable for the
    current (image_dir, damage_dir, compute_weights) request, else None.

    Invalidates when either tile directory has been modified (mtime > cache
    mtime) or when the cache was written for different directories.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        cache_mtime = os.path.getmtime(cache_path)
        # stat the tile dirs: their mtime bumps whenever a file is added /
        # removed, which is what would invalidate the triplet list.
        img_mtime = os.path.getmtime(tile_image_dir)
        dmg_mtime = os.path.getmtime(tile_damage_dir) \
            if os.path.exists(tile_damage_dir) else 0
        if max(img_mtime, dmg_mtime) > cache_mtime:
            print(f"[Tile loader] Cache stale (dir modified after cache write); "
                  f"rescanning")
            return None

        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("image_dir") != tile_image_dir or \
                cache.get("damage_dir") != tile_damage_dir:
            print(f"[Tile loader] Cache dir mismatch; rescanning")
            return None

        # JSON turns tuples into lists — restore triplet shape explicitly so
        # downstream .append / iteration stays consistent with the cold path.
        triplets = [(t[0], t[1], t[2], t[3]) for t in cache["triplets"]]
        weights = cache.get("weights", []) if compute_weights else []
        # If the caller wants weights but the cache was written without them,
        # fall through to a rescan so we re-derive them.
        if compute_weights and not weights and cache.get("weights_computed"):
            # Cached with compute_weights but deliberately empty — unusual;
            # trust it.
            pass
        elif compute_weights and not cache.get("weights_computed"):
            print(f"[Tile loader] Cache lacks weights; rescanning")
            return None
        return triplets, weights
    except Exception as e:
        print(f"[Tile loader] Cache read error ({e}); rescanning")
        return None


def _save_tile_triplet_cache(cache_path, tile_image_dir, tile_damage_dir,
                             triplets, weights):
    try:
        payload = {
            "image_dir": tile_image_dir,
            "damage_dir": tile_damage_dir,
            "weights_computed": bool(weights),
            "triplets": triplets,
            "weights": weights,
        }
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, cache_path)
        print(f"[Tile loader] Wrote cache {cache_path}")
    except Exception as e:
        print(f"[Tile loader] Warning: could not write cache "
              f"{cache_path}: {e}")


def get_unique_files(directory, prefer_png=True):
    """
    Get unique files from directory, preferring PNG over TIF when both exist.
    This avoids duplicates when both formats are present.
    """
    if not os.path.exists(directory):
        return []

    tif_files = glob.glob(os.path.join(directory, "*.tif"))
    png_files = glob.glob(os.path.join(directory, "*.png"))

    if prefer_png and png_files:
        # Use PNG files, they're smaller and faster to load
        return sorted(png_files)
    elif tif_files:
        return sorted(tif_files)
    else:
        return sorted(png_files)


def _process_single_image(args):
    """
    Worker function for parallel processing.
    Args:
        args: tuple of (img_path, use_unified, damage_lookup, label_lookup, mask_lookup, drop_if_empty_mask)
    Returns:
        (triplet, weight, classes_in_sample, dropped_missing, dropped_empty)
        or None if skipped
    """
    img_path, use_unified, damage_lookup, label_lookup, mask_lookup, drop_if_empty_mask = args

    sig = extract_signature(img_path)
    if not sig:
        return None

    triplet = None
    weight = 1.0
    classes = set()
    d_missing = 0
    d_empty = 0

    if use_unified:
        damage_path = damage_lookup.get(sig)
        if not damage_path:
            return (None, 0, set(), 1, 0)  # missing

        # Check if empty (black)
        if drop_if_empty_mask:
            try:
                dmg_arr = np.array(Image.open(damage_path))
                if np.all(dmg_arr == 0):
                    return (None, 0, set(), 0, 1)  # empty
            except Exception as e:
                # print(f"[WARN] Could not read {damage_path}: {e}")
                return (None, 0, set(), 1, 0)  # error -> treat as missing

        triplet = (img_path, None, None, damage_path)

        # Calculate weight
        try:
            if dmg_arr.ndim == 3:
                if dmg_arr.shape[-1] == 4:
                    dmg_rgb = dmg_arr[..., :3]
                else:
                    dmg_rgb = dmg_arr

                # Define colors for all classes (Match config.py 5-class)
                class_colors = {
                    0: np.array([255, 255, 255], dtype=np.uint8), # Undamaged
                    1: np.array([0, 255, 83], dtype=np.uint8), # Partial Roof
                    2: np.array([246, 255, 11], dtype=np.uint8), # Total Roof
                    3: np.array([255, 138, 18], dtype=np.uint8), # Partial Struct
                    4: np.array([255, 0, 0], dtype=np.uint8), # Total Struct
                }

                for cls_id, color in class_colors.items():
                    if np.any(np.all(dmg_rgb == color, axis=-1)):
                        classes.add(cls_id)

                if 2 in classes: # Total Roof (Rare)
                    weight += 9.0  # 10x total
                if 3 in classes: # Partial Struct (Rare)
                    weight += 7.0  # 8x total

        except Exception:
            pass

    else:
        # Legacy Mode
        label_path = label_lookup.get(sig)
        mask_path = mask_lookup.get(sig)

        if not label_path or not mask_path:
            return (None, 0, set(), 1, 0)

        if drop_if_empty_mask:
            try:
                msk_arr = np.array(Image.open(mask_path))
                if np.all(msk_arr == 0):
                    return (None, 0, set(), 0, 1)
            except Exception:
                return (None, 0, set(), 1, 0)

        triplet = (img_path, label_path, mask_path, None)

    return (triplet, weight, classes, d_missing, d_empty)


def list_triplets_filtered(image_dir, label_dir, mask_dir, damage_dir=None, drop_if_empty_mask=True):
    # Get image files (images are usually TIF, don't convert those)
    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.tif")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )

    # Check for Unified Masks first - prefer PNG if available
    damage_files = get_unique_files(
        damage_dir, prefer_png=True) if damage_dir else []

    use_unified = len(damage_files) > 0

    if use_unified:
        # Report format being used
        fmt = "PNG" if damage_files[0].endswith('.png') else "TIF"
        print(
            f"[Triplet builder] Found {len(damage_files)} unified damage masks ({fmt}). Using Unified Mode.")
    else:
        print(
            "[Triplet builder] No unified masks found. Using Legacy Mode (Label + Mask).")
        label_files = get_unique_files(label_dir, prefer_png=True)
        mask_files = get_unique_files(mask_dir, prefer_png=True)

    triplets = []
    sample_weights = []
    class_presence = []  # Track which classes are present in each sample
    dropped_missing = 0
    dropped_empty = 0

    # Cache file path
    cache_file = os.path.join(DATA_ROOT, "dataset_cache.json")

    # Try to load from cache
    if os.path.exists(cache_file):
        print(f"[Triplet builder] Loading from cache: {cache_file}")
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                # Verify cache matches current mode (Unified vs Legacy)
                if cache_data.get("mode") == ("unified" if use_unified else "legacy"):
                    print(
                        f"[Triplet builder] Cache hit! Loaded {len(cache_data['triplets'])} samples.")
                    return cache_data['triplets'], cache_data['weights']
                else:
                    print("[Triplet builder] Cache mode mismatch. Rescanning...")
        except Exception as e:
            print(f"[Triplet builder] Error reading cache: {e}. Rescanning...")

    from tqdm import tqdm

    # Build lookups (O(N) instead of O(N^2))
    print("[Triplet builder] Building file lookups...")
    damage_lookup = {extract_signature(
        f): f for f in damage_files} if use_unified else {}
    label_lookup = {extract_signature(
        f): f for f in label_files} if not use_unified else {}
    mask_lookup = {extract_signature(
        f): f for f in mask_files} if not use_unified else {}

    # Prepare arguments for parallel processing
    tasks = [
        (img, use_unified, damage_lookup,
         label_lookup, mask_lookup, drop_if_empty_mask)
        for img in image_files
    ]

    print(
        f"[Triplet builder] Scanning {len(tasks)} files with {multiprocessing.cpu_count()} workers...")

    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(_process_single_image, tasks, chunksize=10),
            total=len(tasks),
            desc="[Triplet builder] Scanning dataset"
        ))

    # Aggregate results
    for res in results:
        if not res:
            continue
        t, w, c, dm, de = res
        if t:
            triplets.append(t)
            sample_weights.append(w)
            class_presence.append(c)
        dropped_missing += dm
        dropped_empty += de

    print(
        f"[Triplet builder] kept={len(triplets)} | dropped_empty={dropped_empty} | dropped_missing={dropped_missing}")

    # Print class distribution info
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for classes in class_presence:
        for c in classes:
            class_counts[c] += 1
    print(f"[Triplet builder] Samples per class: {class_counts}")

    print(f"[Triplet builder] Samples per class: {class_counts}")

    # Save to cache
    try:
        cache_data = {
            "mode": "unified" if use_unified else "legacy",
            "triplets": triplets,
            "weights": sample_weights,
            "class_counts": class_counts
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        print(f"[Triplet builder] Saved scan results to {cache_file}")
    except Exception as e:
        print(f"[Triplet builder] Warning: Could not save cache: {e}")

    return triplets, sample_weights


def collate_fn(batch):
    """
    Custom collate function to handle variable number of instances.
    """
    images = []
    sem_masks = []
    inst_masks = []
    inst_sevs = []
    image_paths = []

    for item in batch:
        images.append(item['image'])
        sem_masks.append(item['sem_mask'])
        inst_masks.append(item['inst_masks'])  # Keep as tensor (N, H, W)
        inst_sevs.append(item['inst_sev'])    # Keep as tensor (N,)
        image_paths.append(item['image_path'])

    return {
        "image": torch.stack(images, dim=0),       # (B, 3, H, W)
        "sem_mask": torch.stack(sem_masks, dim=0),  # (B, 1, H, W)
        "inst_masks": inst_masks,                  # List of (N, H, W)
        "inst_sev": inst_sevs,                     # List of (N,)
        "image_path": image_paths
    }


def get_dataloaders(image_dir, label_dir, mask_dir, damage_dir=None,
                    batch_size=BATCH_SIZE, image_size=IM_SIZE, num_workers=2,
                    augment_train=True, use_tiles=False, use_balanced_sampling=False,
                    samples_per_class=3000, use_copy_paste=False,
                    copy_paste_initial_prob=0.0,
                    copy_paste_max_pastes=2,
                    copy_paste_feather=3,
                    rank=0, world_size=1, no_weighted_sampler=False,
                    stratified_splits_path=None,
                    val_split_key="val"):
    """
    Create train and validation dataloaders.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels (legacy mode)
        mask_dir: Directory containing masks (legacy mode)
        damage_dir: Directory containing unified damage masks
        batch_size: Batch size
        image_size: Image resize target
        num_workers: Number of data loading workers
        augment_train: Whether to apply augmentations to training data
        use_tiles: Whether to use pre-cropped tiles (default: False)
        use_balanced_sampling: Whether to use balanced class sampling (limits overrepresented classes)
        samples_per_class: Target samples per class for balanced sampling
        rank: Process rank for distributed training (default: 0)
        world_size: Total number of processes for distributed training (default: 1)

    Returns:
        train_loader, val_loader, train_sampler, val_sampler
        (samplers are None when world_size == 1)
    """
    t_dl_start = time.perf_counter()
    if use_tiles:
        print(f"[DataLoader] Using tiled dataset (size={image_size})")
        # Per-tile weights are only consumed by WeightedRandomSampler below;
        # skip computing them when the balanced sampler will replace them.
        triplets, weights = list_tile_triplets(
            compute_weights=not use_balanced_sampling)
    else:
        triplets, weights = list_triplets_filtered(
            image_dir, label_dir, mask_dir, damage_dir)
    t_after_list = time.perf_counter()
    print(f"[DataLoader] list_triplets took {t_after_list - t_dl_start:.2f}s")

    # Create full dataset (without augmentation for splitting)
    full_ds = BuildingSegDataset(triplets, image_size, augment=False)


    if stratified_splits_path is not None and os.path.exists(stratified_splits_path):
        # Stratified-by-event split. The JSON file maps split names to
        # tile-id lists (filename basenames without the .png suffix).
        # We filter `triplets` by membership.
        import json as _json
        with open(stratified_splits_path) as _f:
            _splits = _json.load(_f)
        train_ids = set(_splits["all"]["train"])
        val_ids   = set(_splits["all"][val_split_key])

        def _tile_id(triplet):
            img_path = triplet[0]
            return os.path.splitext(os.path.basename(img_path))[0]

        # Compute the corresponding indices into `triplets` so that
        # downstream code (which subsets `weights` by index) keeps working.
        train_indices = [i for i, t in enumerate(triplets) if _tile_id(t) in train_ids]
        val_indices   = [i for i, t in enumerate(triplets) if _tile_id(t) in val_ids]
        train_triplets = [triplets[i] for i in train_indices]
        val_triplets   = [triplets[i] for i in val_indices]

        if rank == 0:
            print(f"[DataLoader] Stratified split from {stratified_splits_path}: "
                  f"train={len(train_triplets)}, {val_split_key}={len(val_triplets)}")
        if len(train_triplets) == 0 or len(val_triplets) == 0:
            raise ValueError(
                f"Stratified split produced empty fold(s); check that tile IDs "
                f"in {stratified_splits_path} match files in {TILE_IMAGE_DIR}.")
    else:
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train

        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(
            range(len(full_ds)), [n_train, n_val], generator=generator
        )
        train_indices = list(train_indices)
        val_indices = list(val_indices)

        # Create separate datasets for train (with augmentation) and val (without)
        train_triplets = [triplets[i] for i in train_indices]
        val_triplets = [triplets[i] for i in val_indices]

    # Build copy-paste augmentation (instance bank) if requested
    copy_paste_aug = None
    if use_copy_paste and augment_train:
        from src.data.copy_paste import InstanceBank, CopyPasteAugmentation
        cache_dir = os.path.dirname(TILE_IMAGE_DIR)  # tiles_1024/
        bank = InstanceBank(train_triplets, cache_dir=cache_dir)
        copy_paste_aug = CopyPasteAugmentation(
            bank, num_train_tiles=len(train_triplets),
            paste_prob=copy_paste_initial_prob,
            max_total_pastes=copy_paste_max_pastes,
            feather_radius=copy_paste_feather)

    train_ds = BuildingSegDataset(
        train_triplets, image_size, augment=augment_train,
        copy_paste=copy_paste_aug)
    val_ds = BuildingSegDataset(val_triplets, image_size, augment=False)
    print(f"[DataLoader] split+dataset ctor took "
          f"{time.perf_counter() - t_after_list:.2f}s")

    # Choose sampling strategy
    distributed = world_size > 1
    train_sampler = None
    val_sampler = None

    if distributed:
        # Validation always uses plain DistributedSampler
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False)

        if no_weighted_sampler:
            # Instance-balanced natural-distribution sampling. Kang ICLR 2020
            # shows this produces the best feature representations for long-tail
            # classification. Pairs with post-hoc / train-time logit adjustment.
            if rank == 0:
                print(f"[DataLoader] Using plain DistributedSampler (natural distribution, world_size={world_size})")
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            # Training: use DistributedWeightedSampler to preserve rare-class
            # upweighting that WeightedRandomSampler provides in single-GPU mode.
            train_weights = [weights[i] for i in train_indices]
            weight_sum = sum(train_weights)
            train_weights = [w / weight_sum * len(train_weights) for w in train_weights]

            if rank == 0:
                print(f"[DataLoader] Using DistributedWeightedSampler (world_size={world_size})")
            train_sampler = DistributedWeightedSampler(
                train_weights, num_samples=len(train_weights),
                num_replicas=world_size, rank=rank, replacement=True, seed=42)
        sampler = train_sampler
    else:
        if use_balanced_sampling and use_tiles:
            # Use balanced class sampler (limits overrepresented classes)
            class_index_file = os.path.join(DATA_ROOT, "tiles_1024", "class_index.json")
            if not os.path.exists(class_index_file):
                print(f"[WARNING] Balanced sampling requested but class_index.json not found at {class_index_file}")
                print("[WARNING] Falling back to weighted sampling")
                use_balanced_sampling = False
            else:
                print(f"[DataLoader] Using balanced class sampling (samples_per_class={samples_per_class})")
                sampler = create_balanced_sampler(
                    train_triplets,
                    class_index_file,
                    samples_per_class=samples_per_class,
                    seed=42
                )

        if not use_balanced_sampling:
            # Create WeightedRandomSampler for training
            train_weights = [weights[i] for i in train_indices]

            # Normalize weights for numerical stability
            weight_sum = sum(train_weights)
            train_weights = [w / weight_sum *
                             len(train_weights) for w in train_weights]

            sampler = WeightedRandomSampler(
                train_weights, num_samples=len(train_weights), replacement=True)

    t_loader = time.perf_counter()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Shuffle must be False with sampler
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,  # None for single-GPU, DistributedSampler for DDP
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(
        f"[DataLoader] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"[DataLoader] Train augmentation: {augment_train}")
    print(f"[DataLoader] DataLoader ctor took "
          f"{time.perf_counter() - t_loader:.2f}s")
    print(f"[DataLoader] get_dataloaders total "
          f"{time.perf_counter() - t_dl_start:.2f}s")

    return train_loader, val_loader, train_sampler, val_sampler
