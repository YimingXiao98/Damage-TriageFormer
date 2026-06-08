import torch
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
from src.config import IM_SIZE
from src.data.preprocessing import make_sample

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Augmentations:
    """
    Data augmentation pipeline for satellite imagery.
    
    Designed for building damage assessment:
    - Geometric transforms preserve spatial relationships
    - Color augments simulate different lighting/sensor conditions
    - All transforms are applied consistently to image and masks
    """
    
    def __init__(self, 
                 p_hflip=0.5,
                 p_vflip=0.5,
                 p_rotate90=0.5,
                 p_color=0.3,
                 p_blur=0.1,
                 p_noise=0.1,
                 brightness_range=0.2,
                 contrast_range=0.2,
                 saturation_range=0.2):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rotate90 = p_rotate90
        self.p_color = p_color
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
    
    def __call__(self, image, sem_mask, inst_masks):
        """
        Apply augmentations to image and masks.
        
        Args:
            image: (H, W, 3) float32 image [0, 1]
            sem_mask: (H, W) uint8 semantic mask
            inst_masks: List of (H, W) uint8 instance masks
        Returns:
            Augmented image, sem_mask, inst_masks
        """
        # Horizontal flip
        if random.random() < self.p_hflip:
            image = np.fliplr(image).copy()
            sem_mask = np.fliplr(sem_mask).copy()
            inst_masks = [np.fliplr(m).copy() for m in inst_masks]
        
        # Vertical flip
        if random.random() < self.p_vflip:
            image = np.flipud(image).copy()
            sem_mask = np.flipud(sem_mask).copy()
            inst_masks = [np.flipud(m).copy() for m in inst_masks]
        
        # 90-degree rotation
        if random.random() < self.p_rotate90:
            k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
            image = np.rot90(image, k).copy()
            sem_mask = np.rot90(sem_mask, k).copy()
            inst_masks = [np.rot90(m, k).copy() for m in inst_masks]
        
        # Color augmentations (only on image)
        if random.random() < self.p_color:
            image = self._color_jitter(image)
        
        # Gaussian blur (only on image)
        if random.random() < self.p_blur:
            image = self._gaussian_blur(image)
        
        # Additive noise (only on image)
        if random.random() < self.p_noise:
            image = self._add_noise(image)
        
        return image, sem_mask, inst_masks
    
    def _color_jitter(self, image):
        """Apply random color jittering."""
        # Brightness
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness_range, self.brightness_range)
            image = np.clip(image + delta, 0, 1)
        
        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 1)
        
        # Saturation (in HSV space)
        if random.random() < 0.5:
            factor = random.uniform(1 - self.saturation_range, 1 + self.saturation_range)
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        return image.astype(np.float32)
    
    def _gaussian_blur(self, image):
        """Apply random Gaussian blur."""
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    def _add_noise(self, image):
        """Add random Gaussian noise."""
        std = random.uniform(0.01, 0.03)
        noise = np.random.randn(*image.shape).astype(np.float32) * std
        return np.clip(image + noise, 0, 1).astype(np.float32)


class BuildingSegDataset(Dataset):
    """
    Dataset for building segmentation and damage assessment.
    
    Args:
        triplets: List of (image_path, label_path, mask_path, damage_path) tuples
        image_size: Output image size
        augment: Whether to apply augmentations (set True for training)
        augmentation_config: Dict of augmentation parameters
    """
    
    def __init__(self, triplets, image_size=IM_SIZE, augment=False,
                 augmentation_config=None, copy_paste=None):
        self.triplets = triplets
        self.size = image_size
        self.augment = augment
        self.copy_paste = copy_paste  # CopyPasteAugmentation or None

        # Initialize augmentations
        if augment:
            aug_config = augmentation_config or {}
            self.aug = Augmentations(
                p_hflip=aug_config.get('p_hflip', 0.5),
                p_vflip=aug_config.get('p_vflip', 0.5),
                p_rotate90=aug_config.get('p_rotate90', 0.5),
                p_color=aug_config.get('p_color', 0.3),
                p_blur=aug_config.get('p_blur', 0.1),
                p_noise=aug_config.get('p_noise', 0.1),
            )
        else:
            self.aug = None

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        image_p, label_p, mask_p, damage_p = self.triplets[i]
        s = make_sample(image_p, label_p, mask_p, damage_p)

        img = s["image_rgb"].astype(np.float32) / 255.0
        sem = s["sem_mask"].astype(np.uint8)
        inst_masks = s["inst_masks"]
        inst_sev = s["inst_sev"]

        # Copy-paste rare-class instances (before standard augmentations so
        # pasted buildings also get flipped / rotated / colour-jittered).
        if self.copy_paste is not None:
            img, sem, inst_masks, inst_sev = self.copy_paste(
                img, sem, inst_masks, inst_sev)

        # Apply augmentations (before resize for better quality)
        if self.augment and self.aug is not None:
            img, sem, inst_masks = self.aug(img, sem, inst_masks)

        # Resize
        img_r = cv2.resize(img, (self.size, self.size),
                           interpolation=cv2.INTER_LINEAR)
        sem_r = cv2.resize(sem, (self.size, self.size),
                           interpolation=cv2.INTER_NEAREST)

        # Normalize
        img_r = (img_r - MEAN) / STD

        # To Tensor
        img_t = torch.from_numpy(img_r).permute(2, 0, 1).float()      # (3,H,W)
        sem_t = torch.from_numpy(sem_r).float().unsqueeze(0)          # (1,H,W)

        inst_ts = []
        for m in inst_masks:
            m_r = cv2.resize(m, (self.size, self.size),
                             interpolation=cv2.INTER_NEAREST)
            inst_ts.append(torch.from_numpy(m_r).float())

        if len(inst_ts) == 0:
            inst_stack = torch.zeros((0, self.size, self.size), dtype=torch.float32)
        else:
            inst_stack = torch.stack(inst_ts, 0)                      # (N,H,W)

        sev = torch.tensor(inst_sev, dtype=torch.long)
        if len(sev) == 0:
            sev = torch.zeros((0,), dtype=torch.long)

        return {
            "image": img_t,               # (3,H,W)
            "sem_mask": sem_t,            # (1,H,W)
            "inst_masks": inst_stack,     # (N,H,W)
            "inst_sev": sev,              # (N,)
            "image_path": image_p
        }
