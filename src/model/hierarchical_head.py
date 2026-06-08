"""
Hierarchical Damage Classification Head.

Exploits the 2x2 structure of damage classes:
    - Type: Roof (0,1) vs Structural (2,3)
    - Extent: Partial (0,2) vs Total (1,3)

This decomposition helps with rare classes since within each type,
partial vs total is ~50/50 balanced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.heads import ResidualMLP, SpatialAttentionPooling, MultiScaleROIPooling


# Class mappings
# 0=Undamaged -> Type 0
# 1=Partial Roof, 2=Total Roof -> Type 1
# 3=Partial Struct, 4=Total Struct -> Type 2
# 0=None, 1=Roof, 2=Structural
CLASS_TO_TYPE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
CLASS_TO_EXTENT = {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}    # 0=Partial/None, 1=Total

TYPE_NAMES = ["No Damage", "Roof", "Structural"]
EXTENT_NAMES = ["Partial", "Total"]


def class_to_type_extent(class_labels: torch.Tensor) -> tuple:
    """
    Convert 4-class labels to binary type and extent labels.

    Args:
        class_labels: (N,) tensor of class indices [0-3]

    Returns:
        type_labels: (N,) tensor [0=None, 1=Roof, 2=Structural]
        extent_labels: (N,) tensor [0=Partial, 1=Total]
    """
    device = class_labels.device
    type_labels = torch.zeros_like(class_labels)
    extent_labels = torch.zeros_like(class_labels)

    for cls, typ in CLASS_TO_TYPE.items():
        type_labels[class_labels == cls] = typ

    for cls, ext in CLASS_TO_EXTENT.items():
        extent_labels[class_labels == cls] = ext

    return type_labels, extent_labels


def type_extent_to_class(type_pred: torch.Tensor, extent_pred: torch.Tensor) -> torch.Tensor:
    """
    Combine type and extent predictions into 4-class prediction.

    Args:
        type_pred: (N,) tensor [0=Roof, 1=Structural]
        extent_pred: (N,) tensor [0=Partial, 1=Total]

    Returns:
        class_pred: (N,) tensor [0-4]
    """
    # 0 -> 0
    # 1 -> 1 + extent (1+0=1, 1+1=2)
    # 2 -> 3 + extent (3+0=3, 3+1=4)

    # Logic: if type==0: 0. else: (type-1)*2 + 1 + extent
    # Simplified:
    # T=0 -> 0
    # T=1 -> 1 + E
    # T=2 -> 3 + E

    # Vectorized:
    # 0 where type==0
    # ((type-1)*2 + 1 + extent) where type > 0

    out = torch.zeros_like(type_pred)
    mask = (type_pred > 0)
    out[mask] = (type_pred[mask] - 1) * 2 + 1 + extent_pred[mask]
    return out


class HierarchicalDamageHead(nn.Module):
    """
    Hierarchical damage classification with separate Type and Extent heads.

    Architecture:
        Backbone features
            ↓
        Shared feature processing (multi-scale + attention pooling)
            ↓
        Shared MLP layers
            ↓
        ┌──────────────┬──────────────┐
        │  Type Head   │  Extent Head │
        │ (Roof/Struct)│ (Part/Total) │
        └──────────────┴──────────────┘

    Final class = type * 2 + extent
    """

    def __init__(self, in_ch, hidden_dim=512, num_layers=2, dropout=0.3,
                 use_attention=True, use_multiscale=True):
        super().__init__()
        self.in_ch = in_ch
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale

        # Multi-scale feature extraction
        if use_multiscale:
            self.multiscale = MultiScaleROIPooling(in_ch)

        # Attention pooling
        if use_attention:
            self.attn_pool = SpatialAttentionPooling(in_ch)

        # Shared feature processing
        self.input_proj = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Shared residual blocks
        self.shared_blocks = nn.ModuleList([
            ResidualMLP(hidden_dim, expansion=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Type head: No Damage (0) vs Roof (1) vs Structural (2)
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Changed 2 -> 3
        )

        # Extent head: Partial (0) vs Total (1)
        self.extent_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pool_instance_features(self, features, mask):
        """Simple masked average pooling for a single instance."""
        mask = mask.unsqueeze(0)  # (1, h, w)
        weight_sum = mask.sum() + 1e-6
        pooled = (features * mask).sum(dim=(1, 2)) / weight_sum
        return pooled

    def forward(self, f, inst_masks):
        """
        Args:
            f: (B, C, H, W) backbone features
            inst_masks: List of (N_i, H, W) instance masks per image

        Returns:
            type_logits: List of (N_i, 3) type logits per image
                        (0=No Damage, 1=Roof, 2=Structural)
            extent_logits: List of (N_i, 2) extent logits per image
                          (0=Partial, 1=Total)
        """
        B, C, H, W = f.shape

        # Apply multi-scale feature extraction
        if self.use_multiscale:
            f = self.multiscale(f)

        type_outs = []
        extent_outs = []

        for b in range(B):
            masks_b = inst_masks[b]

            if masks_b.numel() == 0:
                type_outs.append(torch.zeros(0, 3, device=f.device))
                extent_outs.append(torch.zeros(0, 2, device=f.device))
                continue

            # Resize masks to feature map size
            N = masks_b.shape[0]
            masks_ds = F.interpolate(
                masks_b.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

            # Pool features per instance
            if self.use_attention:
                feats = self.attn_pool(f[b], masks_ds)
            else:
                feats = []
                for i in range(N):
                    pooled = self._pool_instance_features(f[b], masks_ds[i])
                    feats.append(pooled)
                feats = torch.stack(feats, dim=0) if feats else torch.zeros(
                    0, C, device=f.device)

            if feats.shape[0] == 0:
                type_outs.append(torch.zeros(0, 3, device=f.device))
                extent_outs.append(torch.zeros(0, 2, device=f.device))
                continue

            # Shared processing
            x = self.input_proj(feats)
            for block in self.shared_blocks:
                x = block(x)

            # Separate heads
            type_logits = self.type_head(x)
            extent_logits = self.extent_head(x)

            type_outs.append(type_logits)
            extent_outs.append(extent_logits)

        return type_outs, extent_outs

    def predict_classes(self, f, inst_masks):
        """
        Get 4-class predictions by combining type and extent.

        Returns:
            class_preds: List of (N_i,) class predictions per image
        """
        type_outs, extent_outs = self.forward(f, inst_masks)

        class_preds = []
        for type_logits, extent_logits in zip(type_outs, extent_outs):
            if type_logits.shape[0] == 0:
                class_preds.append(torch.zeros(
                    0, dtype=torch.long, device=type_logits.device))
                continue

            type_pred = type_logits.argmax(dim=1)
            extent_pred = extent_logits.argmax(dim=1)
            class_pred = type_extent_to_class(type_pred, extent_pred)
            class_preds.append(class_pred)

        return class_preds
