import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import IGNORE_INDEX, NUM_CLASSES
from src.model.hierarchical_head import class_to_type_extent


def dice_loss(logits, targets, eps=1e-6):
    """Dice loss for binary segmentation."""
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum(dim=(2, 3))
    den = (p * p).sum(dim=(2, 3)) + (targets * targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Focal Loss down-weights well-classified examples and focuses training
    on hard, misclassified examples. This is particularly effective for
    class imbalance problems.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights tensor of shape (num_classes,) or None for uniform
        gamma: Focusing parameter. gamma=0 reduces to CE, gamma=2 is common
        ignore_index: Label to ignore in loss computation
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-1, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, C) raw predictions
            targets: (N,) class indices
        """
        # Filter out ignored indices
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        num_classes = logits.shape[-1]

        # Apply label smoothing
        if self.label_smoothing > 0:
            # Create smoothed one-hot targets
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(
                1), 1.0 - self.label_smoothing)

        # Compute softmax probabilities with numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Get probability of true class
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            focal_weight = focal_weight * alpha_t

        # Compute final focal loss
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing for better generalization.
    """

    def __init__(self, smoothing=0.1, weight=None, ignore_index=-1):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        num_classes = logits.shape[-1]
        log_probs = F.log_softmax(logits, dim=-1)

        # Standard CE component
        nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Smoothing component (uniform distribution)
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        # Apply class weights if provided
        if self.weight is not None:
            weight_t = self.weight.to(logits.device).gather(0, targets)
            loss = loss * weight_t
        return loss.mean()


def calculate_loss(seg_logits, sev_logits, sem, sev, class_weights, device,
                   use_focal=True, focal_gamma=2.0, label_smoothing=0.1):
    """
    Calculate combined segmentation and severity loss.

    Args:
        seg_logits: Segmentation logits (B, 1, H, W)
        sev_logits: Severity logits (N, num_classes)
        sem: Semantic mask targets (B, 1, H, W)
        sev: Severity targets (N,)
        class_weights: Optional class weights tensor
        device: torch device
        use_focal: Whether to use focal loss for severity
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor

    Returns:
        total_loss, seg_loss, sev_loss
    """
    # === Segmentation Loss ===
    # Use Dice + BCE for robust binary segmentation
    dl = dice_loss(seg_logits, sem).mean()
    bl = F.binary_cross_entropy_with_logits(seg_logits, sem)
    seg_loss = dl + bl

    # === Severity Loss ===
    sev_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Check for valid inputs
    if sev_logits is None or sev_logits.numel() == 0 or sev.numel() == 0:
        loss = seg_loss + sev_loss
        return loss, seg_loss, sev_loss

    if sev_logits.shape[0] != sev.shape[0]:
        loss = seg_loss + sev_loss
        return loss, seg_loss, sev_loss

    # Filter out ignored samples before loss computation
    valid_mask = sev != IGNORE_INDEX
    if not valid_mask.any():
        loss = seg_loss + sev_loss
        return loss, seg_loss, sev_loss

    if use_focal:
        # Focal Loss - better for class imbalance
        focal_fn = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing
        )
        sev_loss = focal_fn(sev_logits, sev)
    else:
        # Standard Cross Entropy with label smoothing
        ls_fn = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            weight=class_weights,
            ignore_index=IGNORE_INDEX
        )
        sev_loss = ls_fn(sev_logits, sev)

    # Let NaN propagate — the batch-level skip in engine.py handles it,
    # and a NaN counter aborts the epoch if too many batches NaN in a row.
    # Silent NaN-to-zero replacement here was masking genuine instability.
    loss = seg_loss + sev_loss
    return loss, seg_loss, sev_loss


def calculate_hierarchical_loss(seg_logits, type_logits, extent_logits, sem,
                                class_labels, device, class_weights=None, focal_gamma=2.0,
                                label_smoothing=0.1, type_weight=1.0, extent_weight=2.0,
                                hard_class_boost=True, la_tau=0.0,
                                type_priors=None, extent_priors=None):
    """
    Calculate combined segmentation and hierarchical damage loss with cost-sensitive learning.

    Converts 5-class labels to binary type/extent labels and computes
    separate losses with asymmetric penalties for rare classes.

    Uses boosted per-sample weights for the hardest classes (C2=Total Roof, C3=Partial Structural)
    to complement oversampling.

    Args:
        seg_logits: Segmentation logits (B, 1, H, W)
        type_logits: Type classification logits (N, 3) - No Damage vs Roof vs Structural
        extent_logits: Extent classification logits (N, 2) - Partial vs Total
        sem: Semantic mask targets (B, 1, H, W)
        class_labels: Original 5-class labels (N,)
        device: torch device
        class_weights: Per-class weights for cost-sensitive learning
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        type_weight: Weight for type loss (default: 1.0)
        extent_weight: Weight for extent loss (default: 2.0, higher to focus on rare classes)

    Returns:
        total_loss, seg_loss, type_loss, extent_loss
    """
    # === Segmentation Loss ===
    dl = dice_loss(seg_logits, sem).mean()
    bl = F.binary_cross_entropy_with_logits(seg_logits, sem)
    seg_loss = dl + bl

    # Initialize damage losses
    type_loss = torch.tensor(0.0, device=device, requires_grad=True)
    extent_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Check for valid inputs
    if type_logits is None or type_logits.numel() == 0:
        loss = seg_loss + type_loss + extent_loss
        return loss, seg_loss, type_loss, extent_loss

    if class_labels.numel() == 0:
        loss = seg_loss + type_loss + extent_loss
        return loss, seg_loss, type_loss, extent_loss

    # Filter out ignored samples
    valid_mask = class_labels != IGNORE_INDEX
    if not valid_mask.any():
        loss = seg_loss + type_loss + extent_loss
        return loss, seg_loss, type_loss, extent_loss

    valid_type_logits = type_logits[valid_mask]
    valid_extent_logits = extent_logits[valid_mask]
    valid_class_labels = class_labels[valid_mask]

    # Convert 5-class to binary type/extent labels
    # Type: 0 -> 0 (None), 1,2 -> 1 (Roof), 3,4 -> 2 (Structural)
    type_labels = torch.zeros_like(valid_class_labels)
    type_labels[(valid_class_labels == 1) | (valid_class_labels == 2)] = 1
    type_labels[(valid_class_labels == 3) | (valid_class_labels == 4)] = 2

    # Extent: 1,3 -> 0 (Partial), 2,4 -> 1 (Total)
    extent_labels = torch.zeros_like(valid_class_labels)
    extent_labels[(valid_class_labels == 2) | (valid_class_labels == 4)] = 1

    # Mask for extent loss: Only compute for Damaged buildings (Class > 0)
    # Undamaged buildings (Class 0) have undefined extent
    extent_mask = (valid_class_labels > 0)

    # Apply per-sample class weights if provided. Optional 1.5× extra boost
    # for C2 (Total Roof) and C3 (Partial Structural). The boost stacks on
    # top of inverse-sqrt class_weights (~3× for rare classes) and focal
    # (1-p)^γ, which together have been linked to NaN instability — disable
    # via `hard_class_boost=False` to run a clean imbalance experiment.
    sample_weights = None
    if class_weights is not None:
        sample_weights = class_weights.to(device).gather(0, valid_class_labels)
        if hard_class_boost:
            boost = torch.ones_like(sample_weights)
            boost[valid_class_labels == 2] = 1.5  # Total Roof
            boost[valid_class_labels == 3] = 1.5  # Partial Structural
            sample_weights = sample_weights * boost

    # Logit Adjustment (Menon ICLR 2021): shift logits by tau * log(prior)
    # during training so the model learns balanced probabilities.
    # At inference, use raw logits (no further adjustment needed).
    if la_tau > 0.0 and type_priors is not None:
        type_offset = la_tau * torch.log(
            type_priors.to(device).clamp(min=1e-10))
        type_logits_for_loss = valid_type_logits + type_offset.unsqueeze(0)
    else:
        type_logits_for_loss = valid_type_logits

    # Type loss (3-class classification: None/Roof/Structural)
    type_ce = F.cross_entropy(type_logits_for_loss, type_labels,
                              label_smoothing=label_smoothing, reduction='none')
    if sample_weights is not None:
        type_ce = type_ce * sample_weights
    type_ce = type_ce.mean()

    # Extent loss (binary classification: Partial/Total)
    # Only compute for damaged buildings (Class > 0)
    if extent_mask.any():
        masked_extent_logits = valid_extent_logits[extent_mask]
        masked_extent_labels = extent_labels[extent_mask]

        # Apply LA to extent logits (using extent priors conditional on damaged)
        if la_tau > 0.0 and extent_priors is not None:
            extent_offset = la_tau * torch.log(
                extent_priors.to(device).clamp(min=1e-10))
            masked_extent_logits = masked_extent_logits + extent_offset.unsqueeze(0)

        extent_ce = F.cross_entropy(masked_extent_logits, masked_extent_labels,
                                    label_smoothing=label_smoothing, reduction='none')

        if sample_weights is not None:
            masked_weights = sample_weights[extent_mask]
            extent_ce = extent_ce * masked_weights

        extent_ce = extent_ce.mean()
    else:
        # No damaged buildings in batch
        extent_ce = torch.tensor(0.0, device=device)

    type_loss = type_weight * type_ce
    extent_loss = extent_weight * extent_ce

    # Let NaN propagate — engine.py skips NaN batches and aborts the epoch
    # if the NaN fraction exceeds a threshold. Silent replacement used to
    # hide genuine gradient explosions.
    loss = seg_loss + type_loss + extent_loss
    return loss, seg_loss, type_loss, extent_loss


def calculate_gated_loss(seg_logits, gate_logits, leaf_logits, sem,
                          class_labels, device, class_weights=None,
                          label_smoothing=0.1, gate_weight=1.0, leaf_weight=2.0,
                          la_tau=0.0, gate_prior=None, leaf_priors=None,
                          gate_class_weights=None,
                          aux_severity_logits=None,
                          aux_severity_weight=0.0,
                          aux_severity_targets=None):
    """
    Gated damage loss (xView2 pattern): BCE gate + CE leaf (damaged only).

    Args:
        seg_logits: (B, 1, H, W) segmentation logits
        gate_logits: (N,) flat binary logits "is damaged"
        leaf_logits: (N, 4) flat 4-way logits over damaged leaves
        sem: (B, 1, H, W) seg targets
        class_labels: (N,) 5-class labels (0 = Undamaged, 1..4 = damaged leaves)
        class_weights: (5,) optional per-class weights
        label_smoothing: for leaf CE
        gate_weight, leaf_weight: loss component multipliers
        la_tau: logit-adjustment tau (Menon 2021). 0 = off.
        gate_prior: scalar P(damaged). Used only when la_tau > 0.
        leaf_priors: (4,) priors over damaged leaves, normalized.
    Returns:
        total_loss, seg_loss, gate_loss, leaf_loss
    """
    # === Segmentation Loss ===
    dl = dice_loss(seg_logits, sem).mean()
    bl = F.binary_cross_entropy_with_logits(seg_logits, sem)
    seg_loss = dl + bl

    gate_loss = torch.tensor(0.0, device=device, requires_grad=True)
    leaf_loss = torch.tensor(0.0, device=device, requires_grad=True)
    aux_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if gate_logits is None or gate_logits.numel() == 0 or class_labels.numel() == 0:
        return (seg_loss + gate_loss + leaf_loss,
                seg_loss, gate_loss, leaf_loss, aux_loss)

    valid_mask = class_labels != IGNORE_INDEX
    if not valid_mask.any():
        return (seg_loss + gate_loss + leaf_loss,
                seg_loss, gate_loss, leaf_loss, aux_loss)

    valid_gate = gate_logits[valid_mask]
    valid_leaf = leaf_logits[valid_mask]
    valid_labels = class_labels[valid_mask]

    # Binary gate target: "is damaged" (class > 0)
    gate_target = (valid_labels > 0).float()

    # Apply LA to gate logits (shift threshold by log-odds of damaged prior).
    if la_tau > 0.0 and gate_prior is not None:
        p_d = gate_prior.clamp(min=1e-10, max=1-1e-10)
        gate_offset = la_tau * (torch.log(p_d) - torch.log(1 - p_d))
        valid_gate_for_loss = valid_gate + gate_offset
    else:
        valid_gate_for_loss = valid_gate

    gate_bce = F.binary_cross_entropy_with_logits(
        valid_gate_for_loss, gate_target, reduction='none')
    if gate_class_weights is not None:
        # Boost gate BCE on rare-class instances (e.g. Total Roof) so the gate
        # learns to fire confidently on them, attacking the 30% gate-miss rate
        # observed for Total Roof in the v7 diagnostic.
        per_sample_gate_w = gate_class_weights.to(device).gather(0, valid_labels)
        gate_loss = (gate_bce * per_sample_gate_w).mean()
    else:
        gate_loss = gate_bce.mean()

    # Leaf loss: only on damaged samples, target = class - 1 in {0..3}
    damaged_mask = (valid_labels > 0)
    if damaged_mask.any():
        leaf_target = valid_labels[damaged_mask] - 1
        dmg_leaf_logits = valid_leaf[damaged_mask]

        # Apply LA to leaf logits
        if la_tau > 0.0 and leaf_priors is not None:
            leaf_offset = la_tau * torch.log(
                leaf_priors.to(device).clamp(min=1e-10))
            dmg_leaf_logits = dmg_leaf_logits + leaf_offset.unsqueeze(0)

        leaf_ce = F.cross_entropy(
            dmg_leaf_logits, leaf_target,
            label_smoothing=label_smoothing, reduction='none')

        # Optional per-sample class weights (index by original 5-class label)
        if class_weights is not None:
            sample_weights = class_weights.to(device).gather(
                0, valid_labels[damaged_mask])
            leaf_ce = leaf_ce * sample_weights

        leaf_loss = leaf_ce.mean()

    # Auxiliary severity regression
    if (aux_severity_logits is not None and aux_severity_targets is not None
            and aux_severity_weight > 0.0 and aux_severity_logits.numel() > 0):
        valid_aux = aux_severity_logits[valid_mask]
        # 5-class label → soft severity scalar via lookup table
        targets = aux_severity_targets.to(device).gather(0, valid_labels)
        # sigmoid-bound output to [0, 1] then smooth-L1 vs target
        aux_pred = torch.sigmoid(valid_aux)
        aux_loss = F.smooth_l1_loss(aux_pred, targets)

    total_loss = (seg_loss + gate_weight * gate_loss
                  + leaf_weight * leaf_loss
                  + aux_severity_weight * aux_loss)
    return total_loss, seg_loss, gate_loss, leaf_loss, aux_loss
