"""
Balanced Class Sampler for Damage Assessment.

Balances classes via downsampling overrepresented AND oversampling
underrepresented classes to a target count:
- Overrepresented classes are randomly downsampled (without replacement)
- Underrepresented classes are oversampled (with replacement)
"""

import json
import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class BalancedClassSampler(Sampler):
    """
    Sampler that limits overrepresented classes for balanced training.

    Args:
        triplets: List of (image_path, label_path, mask_path, damage_path) tuples
        class_index_file: Path to class_index.json mapping class -> image filenames
        samples_per_class: Target number of samples per class (None = use minimum)
        seed: Random seed for reproducibility
    """

    def __init__(self, triplets, class_index_file, samples_per_class=None, seed=42):
        self.triplets = triplets
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load class index
        with open(class_index_file, 'r') as f:
            class_index = json.load(f)

        # Build mapping from filename to indices in triplets
        self.filename_to_indices = defaultdict(list)
        for idx, triplet in enumerate(triplets):
            # Extract filename from image path
            image_path = triplet[0]
            filename = os.path.basename(image_path)
            self.filename_to_indices[filename].append(idx)

        # Build class -> triplet indices mapping
        # Note: class_index uses old 4-class system (0-3), we need 5-class (0-4)
        # Mapping: class_index class -> actual damage class
        # 0 -> 1 (Partial Roof)
        # 1 -> 2 (Total Roof)
        # 2 -> 3 (Partial Structural)
        # 3 -> 4 (Total Structural)
        # Undamaged (0) are tiles not in any class list

        self.class_to_indices = defaultdict(list)
        all_indexed_files = set()

        for class_str, filenames in class_index.items():
            # Map old class index to new 5-class system
            old_class = int(class_str)
            new_class = old_class + 1  # 0->1, 1->2, 2->3, 3->4

            for filename in filenames:
                if filename in self.filename_to_indices:
                    indices = self.filename_to_indices[filename]
                    self.class_to_indices[new_class].extend(indices)
                    all_indexed_files.add(filename)

        # Find undamaged samples (not in any damage class)
        for filename, indices in self.filename_to_indices.items():
            if filename not in all_indexed_files:
                self.class_to_indices[0].extend(indices)

        # Print class distribution
        print("\n[BalancedSampler] Class distribution in dataset:")
        for cls in sorted(self.class_to_indices.keys()):
            count = len(self.class_to_indices[cls])
            class_names = ["Undamaged", "Partial Roof", "Total Roof",
                          "Partial Structural", "Total Structural"]
            print(f"  Class {cls} ({class_names[cls]}): {count:,} samples")

        # Determine samples per class
        if samples_per_class is None:
            # Use the median to avoid being dominated by rare or common classes
            class_counts = [len(indices) for indices in self.class_to_indices.values()]
            samples_per_class = int(np.median(class_counts))

        self.samples_per_class = samples_per_class
        print(f"\n[BalancedSampler] Target samples per class: {self.samples_per_class:,}")

        # Build balanced index list with oversampling for rare classes
        self.balanced_indices = []
        for cls in sorted(self.class_to_indices.keys()):
            indices = self.class_to_indices[cls]

            if len(indices) == 0:
                print(f"  Class {cls}: No samples available, skipping")
                continue
            elif len(indices) == samples_per_class:
                # Exact match
                selected = indices
                print(f"  Class {cls}: Using all {len(indices):,} samples (exact match)")
            elif len(indices) < samples_per_class:
                # OVERSAMPLE with replacement to reach target
                selected = self.rng.choice(
                    indices,
                    size=samples_per_class,
                    replace=True
                ).tolist()
                print(f"  Class {cls}: Oversampled {len(indices):,} -> {len(selected):,} samples "
                      f"({samples_per_class / len(indices):.1f}x)")
            else:
                # Downsample without replacement for common classes
                selected = self.rng.choice(
                    indices,
                    size=samples_per_class,
                    replace=False
                ).tolist()
                print(f"  Class {cls}: Downsampled {len(indices):,} -> {len(selected):,} samples")

            self.balanced_indices.extend(selected)

        # Shuffle the balanced indices
        self.rng.shuffle(self.balanced_indices)

        print(f"\n[BalancedSampler] Total balanced samples: {len(self.balanced_indices):,}")
        print(f"[BalancedSampler] Original dataset: {len(triplets):,}")

    def __iter__(self):
        # Re-shuffle at each epoch
        indices = self.balanced_indices.copy()
        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.balanced_indices)


def create_balanced_sampler(triplets, class_index_file, samples_per_class=3000, seed=42):
    """
    Helper function to create a balanced sampler.

    Args:
        triplets: List of dataset triplets
        class_index_file: Path to class_index.json
        samples_per_class: Target samples per class (default: 3000)
        seed: Random seed

    Returns:
        BalancedClassSampler instance
    """
    return BalancedClassSampler(triplets, class_index_file, samples_per_class, seed)
