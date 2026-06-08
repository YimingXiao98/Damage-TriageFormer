import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from src.config import NUM_CLASSES, IGNORE_INDEX
from src.data.preprocessing import process_unified_mask, process_legacy_mask


def worker_count_classes(triplet):
    """
    Worker function to count classes in a single sample.
    Does NOT load the satellite image.
    """
    _, label_path, mask_path, damage_path = triplet

    try:
        if damage_path:
            _, _, inst_sev = process_unified_mask(damage_path)
        else:
            _, _, inst_sev = process_legacy_mask(label_path, mask_path)

        # Convert to numpy array
        sev = np.array(inst_sev, dtype=np.int64)

        # Filter ignored
        valid_sev = sev[sev != IGNORE_INDEX]

        if len(valid_sev) == 0:
            return np.zeros(NUM_CLASSES, dtype=np.int64)

        # Count
        counts = np.bincount(valid_sev, minlength=NUM_CLASSES)
        return counts

    except Exception as e:
        # print(f"Error processing {damage_path or mask_path}: {e}")
        return np.zeros(NUM_CLASSES, dtype=np.int64)


def calculate_class_weights(dataset, num_classes=NUM_CLASSES, num_workers=8):
    """
    Iterates through the dataset to calculate class weights based on pixel counts.
    Uses multiprocessing to speed up reading.
    """
    print(
        f"Calculating dynamic class weights from training data (Workers={num_workers})...")

    # Initialize counts
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # Handle Subset (from random_split)
    if isinstance(dataset, torch.utils.data.Subset):
        triplets = [dataset.dataset.triplets[i] for i in dataset.indices]
    else:
        triplets = dataset.triplets

    # Use multiprocessing
    with multiprocessing.Pool(num_workers) as pool:
        # chunksize can be tuned, but default is usually fine
        results = list(tqdm(pool.imap_unordered(worker_count_classes, triplets, chunksize=10),
                            total=len(triplets), desc="Scanning dataset"))

    # Aggregate results
    for res in results:
        class_counts += res

    print(f"Class counts: {class_counts}")

    # Avoid division by zero
    class_counts = class_counts.astype(np.float32) + 1e-6

    # Calculate inverse frequency with dampening (sqrt)
    # This prevents rare classes from having exploding weights
    weights = 1.0 / np.sqrt(class_counts)

    # Normalize weights so they sum to num_classes (or mean is 1)
    weights = weights / weights.sum() * num_classes

    print(f"Calculated weights: {weights}")

    return torch.from_numpy(weights).float()
