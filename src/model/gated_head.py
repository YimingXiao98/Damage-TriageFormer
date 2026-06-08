"""
Gated Damage Classification Head (xView2-winning pattern).

Replaces the Type × Extent decomposition with:
    Head 1: Binary Damaged vs Undamaged (gate)
    Head 2: 4-way softmax over damaged leaves {P-Roof, T-Roof, P-Struct, T-Struct}

Motivation (Priority 4.1 in the research survey):
    The Type × Extent decomposition is structurally flawed — Undamaged has
    no meaningful Extent, so the extent head wastes capacity learning that
    constraint. The leaf F1 is capped by F1(type) × F1(extent) because a
    mistake on either cascades. xView2 winner and every strong follow-up
    use a gate + conditional leaves instead.

Inference:
    P(class=0 | x) = 1 - σ(gate_logit)
    P(class=k | x) = σ(gate_logit) * softmax(leaf_logits)[k-1]   (k ∈ {1,2,3,4})

Training:
    gate_loss = BCE over all valid samples (target = 1 if class > 0)
    leaf_loss = CE over damaged samples only (target = class - 1 in {0..3})
    total = seg_loss + gate_weight * gate_loss + leaf_weight * leaf_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.heads import ResidualMLP, SpatialAttentionPooling, MultiScaleROIPooling


# Leaf index -> 5-class index mapping
# Leaf 0 -> class 1 (Partial Roof)
# Leaf 1 -> class 2 (Total Roof)
# Leaf 2 -> class 3 (Partial Structural)
# Leaf 3 -> class 4 (Total Structural)


def gated_5class_probs(gate_logits: torch.Tensor,
                       leaf_logits: torch.Tensor) -> torch.Tensor:
    """Compose the gate + conditional leaf head into a 5-class distribution."""
    if gate_logits.numel() == 0:
        return torch.zeros(0, 5, dtype=torch.float32, device=gate_logits.device)

    p_damaged = torch.sigmoid(gate_logits)
    p_leaf = torch.softmax(leaf_logits, dim=1) if leaf_logits.numel() > 0 else \
        torch.zeros(gate_logits.shape[0], 4, dtype=torch.float32, device=gate_logits.device)

    probs = torch.zeros(gate_logits.shape[0], 5,
                        dtype=p_leaf.dtype, device=gate_logits.device)
    probs[:, 0] = 1.0 - p_damaged
    probs[:, 1:] = p_damaged.unsqueeze(1) * p_leaf
    return probs


def predict_5class_from_gate(gate_logits: torch.Tensor,
                             leaf_logits: torch.Tensor,
                             threshold: float = None) -> torch.Tensor:
    """
    Decode gated predictions into the final 5-class label.

    Default behavior uses the true joint 5-class argmax implied by:
        p(0)   = 1 - sigmoid(gate)
        p(k>0) = sigmoid(gate) * softmax(leaf)[k-1]

    Passing ``threshold`` keeps the old hard-gate behavior around for
    calibration sweeps / ablations.
    """
    if gate_logits.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=gate_logits.device)

    if threshold is not None:
        damaged = torch.sigmoid(gate_logits) > threshold
        leaf_pred = leaf_logits.argmax(dim=1) if leaf_logits.numel() > 0 else \
            torch.zeros(gate_logits.shape[0], dtype=torch.long, device=gate_logits.device)
        out = torch.zeros_like(leaf_pred)
        out[damaged] = leaf_pred[damaged] + 1
        return out

    return gated_5class_probs(gate_logits, leaf_logits).argmax(dim=1)


class GatedDamageHead(nn.Module):
    """Gate + conditional 4-way damage classifier."""

    def __init__(self, in_ch, hidden_dim=512, num_layers=2, dropout=0.3,
                 use_attention=True, use_multiscale=True,
                 use_aux_severity=False):
        super().__init__()
        self.in_ch = in_ch
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        self.use_aux_severity = use_aux_severity

        if use_multiscale:
            self.multiscale = MultiScaleROIPooling(in_ch)
        if use_attention:
            self.attn_pool = SpatialAttentionPooling(in_ch)

        self.input_proj = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_blocks = nn.ModuleList([
            ResidualMLP(hidden_dim, expansion=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Gate: 1 logit for "is damaged"
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 4-way damage leaf classifier
        self.leaf_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),
        )

        # Auxiliary severity-regression head (optional). Predicts a continuous
        # damage severity score in [0, 1] per instance. Trained with a
        # monotonic class-derived target so the shared embeddings get pulled
        # apart along an interpretable severity axis. Targets the diagnosed
        # PR/TR confusion: PR target=0.3 vs TR target=0.7 (gap of 0.4 vs the
        # standard one-hot near-equal distance).
        if use_aux_severity:
            self.aux_severity_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1),
            )
        else:
            self.aux_severity_head = None

        self._init_weights()

    def _init_weights(self):
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
        mask = mask.unsqueeze(0)
        weight_sum = mask.sum() + 1e-6
        pooled = (features * mask).sum(dim=(1, 2)) / weight_sum
        return pooled

    def encode_instances(self, f, inst_masks):
        """
        Encode each instance mask into the shared hidden representation that
        feeds both the gate and leaf heads.

        Returns:
            List of (N_i, hidden_dim) embeddings per image.
        """
        B, C, H, W = f.shape
        if self.use_multiscale:
            f = self.multiscale(f)

        embeddings = []
        for b in range(B):
            masks_b = inst_masks[b]
            if masks_b.numel() == 0:
                embeddings.append(
                    torch.zeros(0, self.input_proj[0].out_features, device=f.device))
                continue

            N = masks_b.shape[0]
            masks_ds = F.interpolate(
                masks_b.unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).squeeze(1)

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
                embeddings.append(
                    torch.zeros(0, self.input_proj[0].out_features, device=f.device))
                continue

            x = self.input_proj(feats)
            for block in self.shared_blocks:
                x = block(x)
            embeddings.append(x)

        return embeddings

    def classify_embeddings(self, embeddings):
        """
        Apply the gate and leaf heads to precomputed shared embeddings.

        Args:
            embeddings: List of (N_i, hidden_dim) tensors.
        Returns:
            gate_logits, leaf_logits lists aligned with ``embeddings``.
        """
        gate_outs = []
        leaf_outs = []
        for x in embeddings:
            if x.numel() == 0:
                gate_outs.append(torch.zeros(0, device=x.device))
                leaf_outs.append(torch.zeros(0, 4, device=x.device))
                continue

            gate_logits = self.gate_head(x).squeeze(-1)
            leaf_logits = self.leaf_head(x)
            gate_outs.append(gate_logits)
            leaf_outs.append(leaf_logits)
        return gate_outs, leaf_outs

    def classify_aux_embeddings(self, embeddings):
        """Apply aux-severity head to precomputed embeddings."""
        outs = []
        for x in embeddings:
            if x.numel() == 0:
                outs.append(torch.zeros(0, device=x.device))
                continue
            outs.append(self.aux_severity_head(x).squeeze(-1))
        return outs

    def forward(self, f, inst_masks):
        """
        Returns:
            (gate_list, leaf_list)                 if use_aux_severity is False
            (gate_list, leaf_list, aux_list)       if use_aux_severity is True
                aux_list: List of (N_i,) raw severity logits per image
                          (apply sigmoid for [0,1] interpretation).
        """
        embeddings = self.encode_instances(f, inst_masks)
        gate_list, leaf_list = self.classify_embeddings(embeddings)
        if self.use_aux_severity:
            aux_list = self.classify_aux_embeddings(embeddings)
            return gate_list, leaf_list, aux_list
        return gate_list, leaf_list
