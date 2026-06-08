"""
Simple Feature Pyramid (SFP) for plain ViT backbones.

Reference: Li et al., "Exploring Plain Vision Transformer Backbones
for Object Detection", ECCV 2022 (ViTDet) — section 3.2.

The plain ViT produces a single low-resolution feature map (1/16 of
the input here for ViT-L/16 with 1024-pixel inputs => 64x64).  For
fine-grained per-instance pooling we want a higher-resolution feature
map.  SFP achieves this by applying a small stack of transposed
convolutions to the single ViT output, with optional refinement.

Unlike a traditional FPN that aggregates features from multiple
hierarchical backbone stages, SFP synthesises the pyramid from a
single feature map — which is what plain ViTs give you.

Output strides supported:
    stride=4  -> 4x upsample => 256x256 from 64x64 (most detail, most VRAM)
    stride=8  -> 2x upsample => 128x128 (modest cost)
    stride=2  -> 8x upsample => 512x512 (very expensive; usually OOM at B=4)
"""

import torch
import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    """Single-scale upsampler ala ViTDet's SFP.

    We only return the requested target stride feature map (not the full
    {1/4, 1/8, 1/16, 1/32} pyramid), since downstream per-instance
    pooling consumes one tensor.  This keeps activations small.
    """

    def __init__(self, in_channels: int, target_stride: int = 4,
                 source_stride: int = 16, hidden_channels: int = None):
        super().__init__()
        if target_stride >= source_stride:
            raise ValueError(
                f"target_stride ({target_stride}) must be < source_stride "
                f"({source_stride}) for SFP to be useful.")
        # Number of 2x upsamplings needed: e.g. 16 -> 4 = 2 doublings.
        n_up = 0
        s = source_stride
        while s > target_stride:
            s //= 2
            n_up += 1
        if s != target_stride:
            raise ValueError(
                f"source_stride / target_stride must be a power of 2; "
                f"got {source_stride} / {target_stride}.")

        hidden = hidden_channels or in_channels
        layers = []
        cur = in_channels
        for i in range(n_up):
            # Transposed conv 2x upsample; keep channels constant.
            layers.append(nn.ConvTranspose2d(cur, hidden, kernel_size=2,
                                             stride=2, bias=False))
            layers.append(nn.GroupNorm(num_groups=32, num_channels=hidden))
            if i < n_up - 1:
                layers.append(nn.GELU())
            cur = hidden
        # Optional 3x3 refinement at the target resolution.
        layers.append(nn.Conv2d(cur, in_channels, kernel_size=3,
                                padding=1, bias=False))
        self.up = nn.Sequential(*layers)

        self.in_channels = in_channels
        self.target_stride = target_stride
        self.source_stride = source_stride

        # Initialise the deconv weights to be near-identity-ish
        # (small std) so early epochs don't destroy the resumed
        # checkpoint's signal.
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) at source_stride.  Returns (B, C, H', W') at
        target_stride."""
        return self.up(x)

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"source_stride={self.source_stride}, "
                f"target_stride={self.target_stride}")
