import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from src.training.loss import calculate_loss, calculate_hierarchical_loss, calculate_gated_loss
from src.model.gated_head import predict_5class_from_gate
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryAccuracy
from torchmetrics.classification import (
    MulticlassJaccardIndex, MulticlassF1Score, MulticlassAccuracy,
    MulticlassPrecision, MulticlassRecall,
)
from src.config import NUM_CLASSES, IGNORE_INDEX, CATEGORIES
from src.model.hierarchical_head import type_extent_to_class
from src.model.hierarchical_head import class_to_type_extent, type_extent_to_class
from src.data.preprocessing import mask_to_instances


def reduce_metric(tensor, world_size):
    """All-reduce a tensor across DDP ranks and return the mean.

    When ``world_size == 1`` (single-GPU / non-DDP) the tensor is returned
    unchanged so there is zero overhead in the non-distributed path.
    """
    if world_size <= 1:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    # Ensure the tensor is on the correct device before all_reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / world_size
    return tensor


def _pred_instances_from_seg(seg_probs_batch: torch.Tensor, min_area: int = 30):
    """
    Derive per-image instance masks from a batch of building-segmentation
    probability maps via connected components (min_area=30 matches the
    training preprocessing).

    Args:
        seg_probs_batch: (B, 1, H, W) sigmoid'd seg logits
    Returns:
        List[B] of (N_b, H, W) binary mask tensors on CPU. N_b may be 0.
    """
    out = []
    bin_batch = (seg_probs_batch > 0.5).squeeze(1).detach().cpu().numpy().astype("uint8")
    for b in range(bin_batch.shape[0]):
        insts = mask_to_instances(bin_batch[b], min_area=min_area)
        if not insts:
            out.append(torch.zeros(0, bin_batch.shape[1], bin_batch.shape[2]))
        else:
            # stack of (H, W) uint8 → (N, H, W)
            out.append(torch.from_numpy(np.stack(insts, axis=0)).float())
    return out


def _gt_pixel_class_map(inst_masks_gt, inst_sev_gt, H: int, W: int, device):
    """
    Build a per-pixel ground-truth damage class map from GT instance masks
    and severities. Unlabeled / background pixels = IGNORE_INDEX.
    """
    pix = torch.full((H, W), IGNORE_INDEX, dtype=torch.long, device=device)
    if inst_masks_gt is None or inst_masks_gt.numel() == 0:
        return pix
    for m, sev in zip(inst_masks_gt, inst_sev_gt):
        if sev.item() == IGNORE_INDEX:
            continue
        pix[m > 0] = int(sev.item())
    return pix


def _pred_instance_targets(pred_masks, gt_pixel_map):
    """
    For each predicted instance mask, assign the majority GT class under it.
    Predicted instances covering no GT building get IGNORE_INDEX (ignored
    in F1). This is the honest e2e target: the damage label the system
    would have to predict given its own detection.
    """
    if pred_masks.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=gt_pixel_map.device)
    targets = []
    for m in pred_masks:
        covered = gt_pixel_map[m > 0]
        valid = covered[covered != IGNORE_INDEX]
        if valid.numel() == 0:
            targets.append(IGNORE_INDEX)
        else:
            vals, counts = torch.unique(valid, return_counts=True)
            targets.append(int(vals[counts.argmax()].item()))
    return torch.tensor(targets, dtype=torch.long, device=gt_pixel_map.device)


def train_one_epoch(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    num_epochs: int,
                    class_weights: torch.Tensor = None,
                    use_focal: bool = True,
                    focal_gamma: float = 2.0,
                    label_smoothing: float = 0.1,
                    rank: int = 0,
                    world_size: int = 1) -> float:
    """
    Train for one epoch (standard 4-class mode).

    Args:
        model: The model to train
        loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        num_epochs: Total number of epochs
        class_weights: Optional class weights tensor
        use_focal: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        rank: DDP rank (0 = primary). Controls printing/tqdm.
        world_size: Total number of DDP processes. 1 = non-distributed.

    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    nan_batches = 0
    nan_abort_frac = 0.1  # abort epoch if >10% of batches NaN

    pbar = tqdm(loader, desc=f"Train {epoch+1}/{num_epochs}", disable=rank != 0)
    for batch in pbar:
        img = batch["image"].to(device)          # (B,3,H,W)
        sem = batch["sem_mask"].to(device)       # (B,1,H,W)

        # inst_masks is a list of tensors
        inst = [m.to(device) for m in batch["inst_masks"]]

        # inst_sev is a list of tensors
        sev_list = [s.to(device) for s in batch["inst_sev"]]

        sev_flat = torch.cat(sev_list, dim=0)  # Flatten targets

        seg_logits, sev_logits_list = model(img, inst_masks=inst)

        # sev_logits_list is [[t1, t2, ...]]
        # sev_logits is [t1, t2, ...]
        sev_logits = sev_logits_list[0]

        # Flatten logits
        if len(sev_logits) > 0 and any(s.numel() > 0 for s in sev_logits):
            sev_logits_flat = torch.cat(
                [s for s in sev_logits if s.numel() > 0], dim=0)
        else:
            sev_logits_flat = torch.zeros(0, NUM_CLASSES, device=device)

        # Compute loss
        loss, seg_loss, sev_loss = calculate_loss(
            seg_logits, sev_logits_flat, sem, sev_flat,
            class_weights, device,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing
        )

        # NaN loss: still run backward (required for DDP sync) but skip optimizer step
        optimizer.zero_grad(set_to_none=True)
        if torch.isnan(loss):
            nan_batches += 1
            if rank == 0:
                print(f"[WARN] NaN loss in batch, skipping "
                      f"(NaN count this epoch: {nan_batches})")
            if nan_batches >= 10 and nan_batches > nan_abort_frac * (num_batches + nan_batches):
                raise RuntimeError(
                    f"Aborting epoch {epoch+1}: {nan_batches} NaN batches "
                    f"(>{int(nan_abort_frac*100)}% of seen batches). "
                    "Fix loss/model instability before continuing.")
            # Use zero loss for backward to keep DDP gradient sync intact
            (loss * 0).backward()
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        # Only skip on NaN/Inf — clipping already bounds the update to
        # max_norm=1.0, so any finite pre-clip norm produces a safe step.
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            nan_batches += 1
            if rank == 0:
                print(
                    f"[WARN] NaN/Inf gradient norm, skipping step")
            continue
        # Observational warning — no step skip, just flag unusual norms.
        if rank == 0 and grad_norm > 1e5:
            print(f"[INFO] Large pre-clip grad_norm={grad_norm:.1f} "
                  f"(clipped to 1.0, step applied)")
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "seg": f"{seg_loss.item():.4f}",
            "sev": f"{sev_loss.item():.4f}",
            "grad": f"{grad_norm:.2f}"
        })

    if num_batches == 0:
        return float('nan')

    avg_loss = running_loss / num_batches
    avg_loss_t = reduce_metric(
        torch.tensor(avg_loss, device=device), world_size)
    return avg_loss_t.item() if isinstance(avg_loss_t, torch.Tensor) else avg_loss_t


def train_one_epoch_hierarchical(model: torch.nn.Module,
                                 loader: torch.utils.data.DataLoader,
                                 optimizer: torch.optim.Optimizer,
                                 device: torch.device,
                                 epoch: int,
                                 num_epochs: int,
                                 class_weights: torch.Tensor = None,
                                 use_focal: bool = True,
                                 focal_gamma: float = 2.0,
                                 label_smoothing: float = 0.1,
                                 type_weight: float = 5.0,
                                 extent_weight: float = 5.0,
                                 hard_class_boost: bool = True,
                                 rank: int = 0,
                                 world_size: int = 1,
                                 accumulation_steps: int = 1,
                                 la_tau: float = 0.0,
                                 type_priors=None,
                                 extent_priors=None,
                                 ema_model=None):
    """
    Train for one epoch with hierarchical Type+Extent classification.

    Args:
        model: Model with use_hierarchical=True
        loader: Training data loader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        num_epochs: Total epochs
        class_weights: Optional class weights tensor for cost-sensitive learning
        focal_gamma: Focal loss gamma
        label_smoothing: Label smoothing factor
        type_weight: Weight for type loss (default matches main.py CLI default)
        extent_weight: Weight for extent loss (default matches main.py CLI default)
        rank: DDP rank (0 = primary). Controls printing/tqdm.
        world_size: Total number of DDP processes. 1 = non-distributed.
        accumulation_steps: Gradient accumulation steps (effective batch =
            batch_size * world_size * accumulation_steps).

    Returns:
        dict with 'loss', 'seg_loss', 'type_loss', 'extent_loss', 'grad_norm'
        (epoch averages across all ranks).
    """
    model.train()
    running_loss = 0.0
    running_seg = 0.0
    running_type = 0.0
    running_extent = 0.0
    running_grad_norm = 0.0
    num_batches = 0
    num_grad_steps = 0  # for grad_norm averaging
    nan_batches = 0
    nan_abort_frac = 0.1  # abort epoch if >10% of batches NaN
    accum_counter = 0  # micro-batch counter for gradient accumulation
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch+1}/{num_epochs}", disable=rank != 0)
    for batch in pbar:
        img = batch["image"].to(device)
        sem = batch["sem_mask"].to(device)
        inst = [m.to(device) for m in batch["inst_masks"]]
        sev_list = [s.to(device) for s in batch["inst_sev"]]

        class_labels_flat = torch.cat(sev_list, dim=0)

        # Forward pass - returns (type_logits, extent_logits)
        seg_logits, damage_output = model(img, inst_masks=inst)
        type_logits_list, extent_logits_list = damage_output

        # Flatten logits
        if len(type_logits_list) > 0 and any(s.numel() > 0 for s in type_logits_list):
            type_logits_flat = torch.cat(
                [s for s in type_logits_list if s.numel() > 0], dim=0)
            extent_logits_flat = torch.cat(
                [s for s in extent_logits_list if s.numel() > 0], dim=0)
        else:
            type_logits_flat = torch.zeros(0, 2, device=device)
            extent_logits_flat = torch.zeros(0, 2, device=device)

        # Compute hierarchical loss
        loss, seg_loss, type_loss, extent_loss = calculate_hierarchical_loss(
            seg_logits, type_logits_flat, extent_logits_flat, sem,
            class_labels_flat, device,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            type_weight=type_weight,
            extent_weight=extent_weight,
            hard_class_boost=hard_class_boost,
            la_tau=la_tau,
            type_priors=type_priors,
            extent_priors=extent_priors,
        )

        # NaN loss: still run backward (required for DDP sync) but skip optimizer step
        if torch.isnan(loss):
            nan_batches += 1
            if rank == 0:
                print(f"[WARN] NaN loss in batch, skipping "
                      f"(NaN count this epoch: {nan_batches})")
            if nan_batches >= 10 and nan_batches > nan_abort_frac * (num_batches + nan_batches):
                raise RuntimeError(
                    f"Aborting epoch {epoch+1}: {nan_batches} NaN batches "
                    f"(>{int(nan_abort_frac*100)}% of seen batches). "
                    "Fix loss/model instability before continuing.")
            # Zero-loss backward to keep DDP grad sync alive; don't advance accum.
            (loss * 0).backward()
            continue

        # Scale loss for gradient accumulation (so effective gradient matches
        # that of a batch `accumulation_steps` times larger).
        (loss / accumulation_steps).backward()
        accum_counter += 1

        # Only step the optimizer once per accumulation window.
        if accum_counter >= accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            # Only skip on NaN/Inf — clipping already bounds the update to
            # max_norm=1.0, so any finite pre-clip norm produces a safe step.
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                nan_batches += 1
                if rank == 0:
                    print(
                        f"[WARN] NaN/Inf gradient norm, skipping step")
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                continue
            # Observational warning — no step skip.
            if rank == 0 and grad_norm > 1e5:
                print(f"[INFO] Large pre-clip grad_norm={grad_norm:.1f} "
                      f"(clipped to 1.0, step applied)")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # EMA update (step-by-step, after real optimizer step)
            if ema_model is not None:
                raw = model.module if hasattr(model, 'module') else model
                ema_model.update_parameters(raw)
            running_grad_norm += float(grad_norm)
            num_grad_steps += 1
            accum_counter = 0

        running_loss += loss.item()
        running_seg += seg_loss.item()
        running_type += type_loss.item()
        running_extent += extent_loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "seg": f"{seg_loss.item():.4f}",
            "type": f"{type_loss.item():.4f}",
            "ext": f"{extent_loss.item():.4f}",
        })

    # Flush any remaining accumulated gradients at end of epoch.
    if accum_counter > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            optimizer.step()
            if ema_model is not None:
                raw = model.module if hasattr(model, 'module') else model
                ema_model.update_parameters(raw)
            running_grad_norm += float(grad_norm)
            num_grad_steps += 1
        optimizer.zero_grad(set_to_none=True)

    if num_batches == 0:
        return {'loss': float('nan'), 'seg_loss': float('nan'),
                'type_loss': float('nan'), 'extent_loss': float('nan'),
                'grad_norm': float('nan')}

    avg = {
        'loss': running_loss / num_batches,
        'seg_loss': running_seg / num_batches,
        'type_loss': running_type / num_batches,
        'extent_loss': running_extent / num_batches,
        'grad_norm': running_grad_norm / max(num_grad_steps, 1),
    }
    # All-reduce across ranks so the returned values are global epoch averages.
    for k in list(avg.keys()):
        t = reduce_metric(torch.tensor(avg[k], device=device), world_size)
        avg[k] = t.item() if isinstance(t, torch.Tensor) else t
    return avg


def validate(model, loader, device, rank=0, world_size=1):
    """
    Validate the model (standard 4-class mode).

    Args:
        model: Model to validate
        loader: Validation data loader
        device: Device
        rank: DDP rank (0 = primary). Controls printing/tqdm.
        world_size: Total number of DDP processes. 1 = non-distributed.

    Returns:
        Dictionary of metrics
    """
    # Binary metrics for segmentation (Building vs Background)
    bin_jacc = BinaryJaccardIndex(threshold=0.5).to(device)
    bin_f1 = BinaryF1Score(threshold=0.5).to(device)
    bin_acc = BinaryAccuracy(threshold=0.5).to(device)

    # Multiclass metrics for damage assessment (Macro Average)
    mc_jacc = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)
    mc_f1 = MulticlassF1Score(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)
    mc_acc = MulticlassAccuracy(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)

    # Per-class F1
    mc_f1_per_class = MulticlassF1Score(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)

    # Confusion matrix tracking for analysis
    confusion_counts = torch.zeros(
        NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

    model.eval()
    bin_jacc.reset()
    bin_f1.reset()
    bin_acc.reset()
    mc_jacc.reset()
    mc_f1.reset()
    mc_acc.reset()
    mc_f1_per_class.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", disable=rank != 0):
            img = batch["image"].to(device)
            sem = batch["sem_mask"].to(device)

            inst = [m.to(device) for m in batch["inst_masks"]]
            sev_list = [s.to(device) for s in batch["inst_sev"]]
            sev_flat = torch.cat(sev_list, dim=0)

            # Pass inst_masks to get severity predictions (Oracle evaluation using GT masks)
            seg_logits, sev_logits_list = model(img, inst_masks=inst)

            # 1. Binary Segmentation Metrics
            p_seg = torch.sigmoid(seg_logits)
            bin_jacc.update(p_seg, sem)
            bin_f1.update(p_seg, sem)
            bin_acc.update(p_seg, sem)

            # 2. Multiclass Damage Metrics
            if sev_logits_list and len(sev_logits_list) > 0:
                sev_logits = sev_logits_list[0]

                if len(sev_logits) > 0 and any(s.numel() > 0 for s in sev_logits):
                    sev_logits_flat = torch.cat(
                        [s for s in sev_logits if s.numel() > 0], dim=0)

                    # Logits and targets are produced 1:1 from the same instance
                    # masks; if they're ever misaligned that's a model bug, not
                    # something to silently mask around.
                    assert sev_logits_flat.shape[0] == sev_flat.shape[0], (
                        f"sev_logits ({sev_logits_flat.shape[0]}) "
                        f"!= sev_targets ({sev_flat.shape[0]}); model output bug"
                    )

                    valid_mask = sev_flat != IGNORE_INDEX
                    if valid_mask.any():
                        valid_logits = sev_logits_flat[valid_mask]
                        valid_targets = sev_flat[valid_mask]

                        mc_jacc.update(valid_logits, valid_targets)
                        mc_f1.update(valid_logits, valid_targets)
                        mc_acc.update(valid_logits, valid_targets)
                        mc_f1_per_class.update(valid_logits, valid_targets)

                        # Update confusion matrix
                        preds = valid_logits.argmax(dim=-1)
                        for p, t in zip(preds, valid_targets):
                            if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                                confusion_counts[t, p] += 1

    # All-reduce confusion matrix across ranks before computing metrics
    if world_size > 1:
        dist.all_reduce(confusion_counts, op=dist.ReduceOp.SUM)

    # Flat path (non-hierarchical) leaves e2e metrics equal to oracle with a
    # warning — full e2e is implemented for the hierarchical default path.
    oracle_f1 = mc_f1.compute().item()
    metrics = {
        "Seg_IoU": bin_jacc.compute().item(),
        "Seg_F1": bin_f1.compute().item(),
        "Seg_Acc": bin_acc.compute().item(),
        "Damage_Macro_IoU": mc_jacc.compute().item(),
        "Damage_Macro_F1": oracle_f1,
        "Damage_Macro_F1_oracle": oracle_f1,
        "Damage_Macro_F1_e2e": oracle_f1,  # TODO: flat-path e2e not implemented
        "Damage_Macro_Acc": mc_acc.compute().item(),
    }

    # All-reduce scalar metrics across ranks
    if world_size > 1:
        for key in list(metrics.keys()):
            val_t = torch.tensor(metrics[key], device=device, dtype=torch.float32)
            dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
            metrics[key] = (val_t / world_size).item()

    metrics["_confusion_oracle"] = confusion_counts.detach().cpu()
    metrics["_confusion_e2e"] = confusion_counts.detach().cpu()

    # Add per-class F1 scores
    per_class_f1 = mc_f1_per_class.compute()
    for i, score in enumerate(per_class_f1):
        if i < len(CATEGORIES):
            key = f"F1_{CATEGORIES[i].replace(' ', '_')}"
            val = score.item()
            if world_size > 1:
                val_t = torch.tensor(val, device=device, dtype=torch.float32)
                dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
                val = (val_t / world_size).item()
            metrics[key] = val

    # Print confusion matrix summary
    if rank == 0:
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("  " + "  ".join([f"C{i}" for i in range(NUM_CLASSES)]))
        for i in range(NUM_CLASSES):
            row_sum = confusion_counts[i].sum().item()
            if row_sum > 0:
                row_str = "  ".join(
                    [f"{confusion_counts[i, j].item():3d}" for j in range(NUM_CLASSES)])
                print(f"C{i}: {row_str}")

    return metrics


def validate_hierarchical(model, loader, device, rank=0, world_size=1,
                          class_weights=None, focal_gamma=2.0,
                          label_smoothing=0.1, type_weight=1.0,
                          extent_weight=2.0):
    """
    Validate model with hierarchical Type+Extent classification.

    Combines type and extent predictions to get 5-class predictions:
    0=Undamaged, 1=Partial Roof, 2=Total Roof, 3=Partial Struct, 4=Total Struct

    Args:
        model: Model with use_hierarchical=True
        loader: Validation data loader
        device: Device
        rank: DDP rank (0 = primary). Controls printing/tqdm.
        world_size: Total number of DDP processes. 1 = non-distributed.
        class_weights: Optional class weights tensor for loss computation.
        focal_gamma: Focal loss gamma (for val loss computation).
        label_smoothing: Label smoothing factor (for val loss computation).
        type_weight: Weight for type loss component.
        extent_weight: Weight for extent loss component.

    Returns:
        Dictionary of metrics (includes Val_Loss if class_weights provided)
    """
    # Binary metrics for segmentation
    bin_jacc = BinaryJaccardIndex(threshold=0.5).to(device)
    bin_f1 = BinaryF1Score(threshold=0.5).to(device)

    # Oracle damage metrics (GT instance masks — upper bound)
    mc_f1 = MulticlassF1Score(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)
    mc_f1_per_class = MulticlassF1Score(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_prec_per_class = MulticlassPrecision(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_rec_per_class = MulticlassRecall(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)

    # End-to-end damage metrics (instance masks derived from seg head —
    # the real deployment number; a meaningfully lower ceiling than oracle).
    mc_f1_e2e = MulticlassF1Score(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)
    mc_f1_e2e_per_class = MulticlassF1Score(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_prec_e2e_per_class = MulticlassPrecision(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_rec_e2e_per_class = MulticlassRecall(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)

    # Sub-task F1 (type 3-way: None/Roof/Structural; extent 2-way: Partial/Total).
    type_f1_macro = MulticlassF1Score(
        num_classes=3, average='macro').to(device)
    extent_f1_macro = MulticlassF1Score(
        num_classes=2, average='macro').to(device)

    # Val support per class (instance counts) for interpreting F1 volatility.
    val_support = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)

    # Type/Extent accuracy (oracle)
    type_correct = 0
    extent_correct = 0
    total_valid = 0
    # Denominator for extent accuracy (only damaged buildings have extent)
    total_damaged = 0

    confusion_counts = torch.zeros(
        NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    confusion_counts_e2e = torch.zeros(
        NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

    model.eval()
    bin_jacc.reset()
    bin_f1.reset()
    mc_f1.reset()
    mc_f1_per_class.reset()
    mc_prec_per_class.reset()
    mc_rec_per_class.reset()
    mc_f1_e2e.reset()
    mc_f1_e2e_per_class.reset()
    mc_prec_e2e_per_class.reset()
    mc_rec_e2e_per_class.reset()
    type_f1_macro.reset()
    extent_f1_macro.reset()

    # Validation loss tracking (decomposed). Always compute — the loss
    # function handles class_weights=None gracefully, and val loss is a
    # core diagnostic regardless of weighting scheme.
    compute_loss = True
    val_loss_sum = 0.0
    val_seg_sum = 0.0
    val_type_sum = 0.0
    val_extent_sum = 0.0
    val_loss_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", disable=rank != 0):
            img = batch["image"].to(device)
            sem = batch["sem_mask"].to(device)
            inst = [m.to(device) for m in batch["inst_masks"]]
            sev_list = [s.to(device) for s in batch["inst_sev"]]
            class_labels_flat = torch.cat(sev_list, dim=0)

            seg_logits, damage_output = model(img, inst_masks=inst)

            # Segmentation metrics
            p_seg = torch.sigmoid(seg_logits)
            bin_jacc.update(p_seg, sem)
            bin_f1.update(p_seg, sem)

            if damage_output is None:
                # Compute seg-only loss when no instances
                if compute_loss:
                    seg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        seg_logits, sem.float())
                    val_loss_sum += seg_loss.item()
                    val_loss_count += 1
                continue

            type_logits_list, extent_logits_list = damage_output

            if len(type_logits_list) > 0 and any(s.numel() > 0 for s in type_logits_list):
                type_logits_flat = torch.cat(
                    [s for s in type_logits_list if s.numel() > 0], dim=0)
                extent_logits_flat = torch.cat(
                    [s for s in extent_logits_list if s.numel() > 0], dim=0)

                # Compute validation loss (decomposed)
                if compute_loss:
                    loss, seg_l, type_l, extent_l = calculate_hierarchical_loss(
                        seg_logits, type_logits_flat, extent_logits_flat, sem,
                        class_labels_flat, device,
                        class_weights=class_weights,
                        focal_gamma=focal_gamma,
                        label_smoothing=label_smoothing,
                        type_weight=type_weight,
                        extent_weight=extent_weight,
                    )
                    if not torch.isnan(loss):
                        val_loss_sum += loss.item()
                        val_seg_sum += seg_l.item()
                        val_type_sum += type_l.item()
                        val_extent_sum += extent_l.item()
                        val_loss_count += 1

                if type_logits_flat.numel() > 0 and class_labels_flat.numel() > 0:
                    # Logits and labels are emitted 1:1 from the same instance
                    # masks; mismatch means a model bug, not something to mask.
                    assert type_logits_flat.shape[0] == class_labels_flat.shape[0], (
                        f"type_logits ({type_logits_flat.shape[0]}) "
                        f"!= class_labels ({class_labels_flat.shape[0]}); model output bug"
                    )
                    assert extent_logits_flat.shape[0] == class_labels_flat.shape[0]

                    valid_mask = class_labels_flat != IGNORE_INDEX
                    if valid_mask.any():
                        valid_type = type_logits_flat[valid_mask]
                        valid_extent = extent_logits_flat[valid_mask]
                        valid_labels = class_labels_flat[valid_mask]

                        if valid_type.shape[0] > 0:
                            # Predictions
                            type_pred = valid_type.argmax(dim=1)
                            extent_pred = valid_extent.argmax(dim=1)

                            # Combine to 5-class using helper
                            # Combine to 5-class using correct hierarchical logic
                            class_pred = type_extent_to_class(type_pred, extent_pred)

                            # Ground truth decomposition
                            # Type: 0=Undamaged(0), 1=Roof(1,2), 2=Struct(3,4)
                            # Extent: 0=Partial(1,3), 1=Total(2,4)
                            # Undamaged(0) -> Type 0, Extent ? (0)
                            
                            type_gt = torch.zeros_like(valid_labels)
                            extent_gt = torch.zeros_like(valid_labels)
                            
                            # Roof (1,2) -> Type 1
                            type_gt[(valid_labels == 1) | (valid_labels == 2)] = 1
                            # Struct (3,4) -> Type 2
                            type_gt[(valid_labels == 3) | (valid_labels == 4)] = 2
                            
                            # Total (2,4) -> Extent 1
                            extent_gt[(valid_labels == 2) | (valid_labels == 4)] = 1
                            # Partial (1,3) -> Extent 0 (Already 0)
                            # Undamaged (0) -> Extent 0 (Already 0)

                            # Type Accuracy (All samples: Undamaged vs Roof vs Structure)
                            type_correct += (type_pred == type_gt).sum().item()
                            total_valid += valid_type.shape[0]

                            # Extent Accuracy (Only damaged samples: Partial vs Total)
                            # Undamaged (Type 0) doesn't have an extent, so we ignore it
                            damaged_mask = (type_gt > 0)
                            n_damaged = damaged_mask.sum().item()

                            if n_damaged > 0:
                                extent_correct += (extent_pred[damaged_mask]
                                                   == extent_gt[damaged_mask]).sum().item()
                                total_damaged += n_damaged

                            # Update 5-class metrics
                            mc_f1.update(class_pred, valid_labels)
                            mc_f1_per_class.update(class_pred, valid_labels)
                            mc_prec_per_class.update(class_pred, valid_labels)
                            mc_rec_per_class.update(class_pred, valid_labels)

                            # Sub-task F1 (type 3-way, extent 2-way)
                            type_f1_macro.update(type_pred, type_gt)
                            if damaged_mask.any():
                                extent_f1_macro.update(
                                    extent_pred[damaged_mask],
                                    extent_gt[damaged_mask])

                            # Support counts per class (for interpreting F1)
                            for t in valid_labels:
                                ti = int(t.item())
                                if 0 <= ti < NUM_CLASSES:
                                    val_support[ti] += 1

                            # Confusion matrix
                            for p, t in zip(class_pred, valid_labels):
                                if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                                    confusion_counts[t, p] += 1

            # ---- End-to-end pass ----------------------------------------
            # Re-classify using instance masks derived from the seg head
            # (not GT). This is the honest deployment metric; oracle F1
            # above assumes perfect segmentation.
            pred_masks_list = _pred_instances_from_seg(p_seg)
            pred_masks_list = [m.to(device) for m in pred_masks_list]
            # Skip the redundant second forward entirely if no predicted
            # instances in the whole batch.
            if any(m.numel() > 0 for m in pred_masks_list):
                _, damage_e2e = model(img, inst_masks=pred_masks_list)
                type_e2e_list, ext_e2e_list = damage_e2e

                H, W = img.shape[-2], img.shape[-1]
                for b in range(img.shape[0]):
                    tl = type_e2e_list[b]
                    el = ext_e2e_list[b]
                    if tl.numel() == 0:
                        continue
                    pred_cls = type_extent_to_class(
                        tl.argmax(dim=1), el.argmax(dim=1))
                    gt_pix = _gt_pixel_class_map(
                        inst[b], sev_list[b], H, W, device)
                    targets = _pred_instance_targets(
                        pred_masks_list[b], gt_pix)

                    assert pred_cls.shape[0] == targets.shape[0], (
                        f"e2e pred ({pred_cls.shape[0]}) vs target "
                        f"({targets.shape[0]}) mismatch"
                    )
                    valid = targets != IGNORE_INDEX
                    if not valid.any():
                        continue
                    vp, vt = pred_cls[valid], targets[valid]
                    mc_f1_e2e.update(vp, vt)
                    mc_f1_e2e_per_class.update(vp, vt)
                    mc_prec_e2e_per_class.update(vp, vt)
                    mc_rec_e2e_per_class.update(vp, vt)
                    for p, t in zip(vp, vt):
                        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                            confusion_counts_e2e[t, p] += 1

    # All-reduce confusion matrices and counters across ranks
    if world_size > 1:
        dist.all_reduce(confusion_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion_counts_e2e, op=dist.ReduceOp.SUM)
        # Reduce type/extent accuracy counters
        counters = torch.tensor(
            [type_correct, extent_correct, total_valid, total_damaged],
            dtype=torch.float64, device=device)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        type_correct = counters[0].item()
        extent_correct = counters[1].item()
        total_valid = counters[2].item()
        total_damaged = counters[3].item()

    # Reduce validation loss (and components) across ranks
    val_loss_components = {}
    if compute_loss and val_loss_count > 0:
        for name, running in (("Val_Loss", val_loss_sum),
                               ("Val_Seg_Loss", val_seg_sum),
                               ("Val_Type_Loss", val_type_sum),
                               ("Val_Extent_Loss", val_extent_sum)):
            avg = running / val_loss_count
            t = reduce_metric(torch.tensor(avg, device=device), world_size)
            val_loss_components[name] = t.item() if isinstance(t, torch.Tensor) else t
    avg_val_loss = val_loss_components.get("Val_Loss")

    # All-reduce support counts
    if world_size > 1:
        dist.all_reduce(val_support, op=dist.ReduceOp.SUM)

    metrics = {
        "Seg_IoU": bin_jacc.compute().item(),
        "Seg_F1": bin_f1.compute().item(),
        "Damage_Macro_F1_oracle": mc_f1.compute().item(),
        "Damage_Macro_F1_e2e": mc_f1_e2e.compute().item(),
        # Back-compat key — points to e2e so old callers (checkpointing,
        # logging) now track the deployment number, not the oracle ceiling.
        "Damage_Macro_F1": mc_f1_e2e.compute().item(),
        "Type_Acc": type_correct / max(total_valid, 1),
        "Extent_Acc": extent_correct / max(total_damaged, 1),
    }

    metrics.update(val_loss_components)

    # Sub-task F1 for hierarchical head (type 3-way / extent 2-way)
    try:
        metrics["Type_Macro_F1"] = type_f1_macro.compute().item()
    except Exception:
        pass
    try:
        metrics["Extent_Macro_F1"] = extent_f1_macro.compute().item()
    except Exception:
        pass

    # All-reduce scalar metrics across ranks
    if world_size > 1:
        for key in list(metrics.keys()):
            val_t = torch.tensor(metrics[key], device=device, dtype=torch.float32)
            dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
            metrics[key] = (val_t / world_size).item()

    # Per-class F1 / Precision / Recall (both oracle and e2e)
    def _reduce_scalar(x):
        if world_size > 1:
            t = torch.tensor(x, device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return (t / world_size).item()
        return x

    oracle_f1 = mc_f1_per_class.compute()
    oracle_prec = mc_prec_per_class.compute()
    oracle_rec = mc_rec_per_class.compute()
    e2e_f1 = mc_f1_e2e_per_class.compute()
    e2e_prec = mc_prec_e2e_per_class.compute()
    e2e_rec = mc_rec_e2e_per_class.compute()
    for i in range(len(CATEGORIES)):
        cat = CATEGORIES[i].replace(" ", "_")
        metrics[f"F1_{cat}_oracle"] = _reduce_scalar(oracle_f1[i].item())
        metrics[f"F1_{cat}_e2e"] = _reduce_scalar(e2e_f1[i].item())
        metrics[f"F1_{cat}"] = metrics[f"F1_{cat}_e2e"]  # back-compat
        metrics[f"Precision_{cat}_oracle"] = _reduce_scalar(oracle_prec[i].item())
        metrics[f"Precision_{cat}_e2e"] = _reduce_scalar(e2e_prec[i].item())
        metrics[f"Recall_{cat}_oracle"] = _reduce_scalar(oracle_rec[i].item())
        metrics[f"Recall_{cat}_e2e"] = _reduce_scalar(e2e_rec[i].item())
        # Val support (already all-reduced). Don't further divide.
        metrics[f"Support_{cat}"] = int(val_support[i].item())

    # Print confusion matrices
    if rank == 0:
        def _print_cm(name, cm):
            print(f"\n{name} Confusion Matrix (rows=true, cols=pred):")
            print("      " + "  ".join([f"C{i}" for i in range(NUM_CLASSES)]))
            for i in range(NUM_CLASSES):
                if cm[i].sum().item() > 0:
                    row_str = "  ".join(
                        [f"{cm[i, j].item():3d}" for j in range(NUM_CLASSES)])
                    print(f"  C{i}: {row_str}")
        _print_cm("Oracle", confusion_counts)
        _print_cm("End-to-end", confusion_counts_e2e)

        print(f"\nType Accuracy (oracle): {metrics['Type_Acc']:.4f}")
        print(f"Extent Accuracy (oracle): {metrics['Extent_Acc']:.4f} "
              f"(on {int(total_damaged)} damaged samples)")
        print(f"Damage Macro F1 — oracle: {metrics['Damage_Macro_F1_oracle']:.4f} | "
              f"e2e: {metrics['Damage_Macro_F1_e2e']:.4f}")

    # Stash confusion matrices for optional TensorBoard logging by caller.
    metrics["_confusion_oracle"] = confusion_counts.detach().cpu()
    metrics["_confusion_e2e"] = confusion_counts_e2e.detach().cpu()
    return metrics


def train_one_epoch_gated(model, loader, optimizer, device, epoch, num_epochs,
                           class_weights=None, label_smoothing=0.1,
                           gate_weight=1.0, leaf_weight=2.0,
                           rank=0, world_size=1, accumulation_steps=1,
                           la_tau=0.0, gate_prior=None, leaf_priors=None,
                           ema_model=None, gate_class_weights=None,
                           aux_severity_weight=0.0, aux_severity_targets=None):
    """Train one epoch with Damaged/Undamaged gate + conditional 4-way head."""
    model.train()
    running_loss = 0.0
    running_seg = 0.0
    running_gate = 0.0
    running_leaf = 0.0
    running_aux = 0.0
    running_grad_norm = 0.0
    num_batches = 0
    num_grad_steps = 0
    nan_batches = 0
    nan_abort_frac = 0.1
    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch+1}/{num_epochs}", disable=rank != 0)
    for batch in pbar:
        img = batch["image"].to(device)
        sem = batch["sem_mask"].to(device)
        inst = [m.to(device) for m in batch["inst_masks"]]
        sev_list = [s.to(device) for s in batch["inst_sev"]]
        class_labels_flat = torch.cat(sev_list, dim=0)

        seg_logits, damage_output = model(img, inst_masks=inst)
        if len(damage_output) == 3:
            gate_logits_list, leaf_logits_list, aux_logits_list = damage_output
        else:
            gate_logits_list, leaf_logits_list = damage_output
            aux_logits_list = None

        # Flatten
        if len(gate_logits_list) > 0 and any(s.numel() > 0 for s in gate_logits_list):
            gate_logits_flat = torch.cat(
                [s for s in gate_logits_list if s.numel() > 0], dim=0)
            leaf_logits_flat = torch.cat(
                [s for s in leaf_logits_list if s.numel() > 0], dim=0)
            if aux_logits_list is not None:
                aux_logits_flat = torch.cat(
                    [s for s in aux_logits_list if s.numel() > 0], dim=0)
            else:
                aux_logits_flat = None
        else:
            gate_logits_flat = torch.zeros(0, device=device)
            leaf_logits_flat = torch.zeros(0, 4, device=device)
            aux_logits_flat = None if aux_logits_list is None else torch.zeros(0, device=device)

        loss, seg_loss, gate_loss, leaf_loss, aux_loss = calculate_gated_loss(
            seg_logits, gate_logits_flat, leaf_logits_flat, sem,
            class_labels_flat, device,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            gate_weight=gate_weight,
            leaf_weight=leaf_weight,
            la_tau=la_tau,
            gate_prior=gate_prior,
            leaf_priors=leaf_priors,
            gate_class_weights=gate_class_weights,
            aux_severity_logits=aux_logits_flat,
            aux_severity_weight=aux_severity_weight,
            aux_severity_targets=aux_severity_targets,
        )

        if torch.isnan(loss):
            nan_batches += 1
            if rank == 0:
                print(f"[WARN] NaN loss in batch, skipping "
                      f"(NaN count this epoch: {nan_batches})")
            if nan_batches >= 10 and nan_batches > nan_abort_frac * (num_batches + nan_batches):
                raise RuntimeError(
                    f"Aborting epoch {epoch+1}: {nan_batches} NaN batches")
            (loss * 0).backward()
            continue

        (loss / accumulation_steps).backward()
        accum_counter += 1

        if accum_counter >= accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                nan_batches += 1
                if rank == 0:
                    print(f"[WARN] NaN/Inf gradient norm, skipping step")
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                continue
            if rank == 0 and grad_norm > 1e5:
                print(f"[INFO] Large pre-clip grad_norm={grad_norm:.1f}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                raw = model.module if hasattr(model, 'module') else model
                ema_model.update_parameters(raw)
            running_grad_norm += float(grad_norm)
            num_grad_steps += 1
            accum_counter = 0

        running_loss += loss.item()
        running_seg += seg_loss.item()
        running_gate += gate_loss.item()
        running_leaf += leaf_loss.item()
        running_aux += aux_loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "seg": f"{seg_loss.item():.4f}",
            "gate": f"{gate_loss.item():.4f}",
            "leaf": f"{leaf_loss.item():.4f}",
            "aux": f"{aux_loss.item():.4f}",
        })

    if accum_counter > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            optimizer.step()
            if ema_model is not None:
                raw = model.module if hasattr(model, 'module') else model
                ema_model.update_parameters(raw)
            running_grad_norm += float(grad_norm)
            num_grad_steps += 1
        optimizer.zero_grad(set_to_none=True)

    if num_batches == 0:
        return {'loss': float('nan'), 'seg_loss': float('nan'),
                'gate_loss': float('nan'), 'leaf_loss': float('nan'),
                'grad_norm': float('nan')}

    avg = {
        'loss': running_loss / num_batches,
        'seg_loss': running_seg / num_batches,
        'gate_loss': running_gate / num_batches,
        'leaf_loss': running_leaf / num_batches,
        'grad_norm': running_grad_norm / max(num_grad_steps, 1),
    }
    for k in list(avg.keys()):
        t = reduce_metric(torch.tensor(avg[k], device=device), world_size)
        avg[k] = t.item() if isinstance(t, torch.Tensor) else t
    return avg


def validate_gated(model, loader, device, rank=0, world_size=1,
                    class_weights=None, label_smoothing=0.1,
                    gate_weight=1.0, leaf_weight=2.0,
                    decoder_mode="joint_argmax"):
    """Validate model with gated head (Damaged gate + 4-way conditional)."""
    bin_jacc = BinaryJaccardIndex(threshold=0.5).to(device)
    bin_f1 = BinaryF1Score(threshold=0.5).to(device)

    mc_f1 = MulticlassF1Score(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)
    mc_f1_per_class = MulticlassF1Score(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_prec_per_class = MulticlassPrecision(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_rec_per_class = MulticlassRecall(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)

    mc_f1_e2e = MulticlassF1Score(
        num_classes=NUM_CLASSES, average='macro', ignore_index=IGNORE_INDEX).to(device)
    mc_f1_e2e_per_class = MulticlassF1Score(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_prec_e2e_per_class = MulticlassPrecision(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)
    mc_rec_e2e_per_class = MulticlassRecall(
        num_classes=NUM_CLASSES, average=None, ignore_index=IGNORE_INDEX).to(device)

    # Gate-specific metric: binary damaged-vs-undamaged accuracy
    gate_correct = 0
    gate_total = 0

    val_support = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    confusion_counts = torch.zeros(
        NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    confusion_counts_e2e = torch.zeros(
        NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

    model.eval()
    for m in [bin_jacc, bin_f1, mc_f1, mc_f1_per_class, mc_prec_per_class,
              mc_rec_per_class, mc_f1_e2e, mc_f1_e2e_per_class,
              mc_prec_e2e_per_class, mc_rec_e2e_per_class]:
        m.reset()

    val_loss_sum = 0.0
    val_seg_sum = 0.0
    val_gate_sum = 0.0
    val_leaf_sum = 0.0
    val_loss_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", disable=rank != 0):
            img = batch["image"].to(device)
            sem = batch["sem_mask"].to(device)
            inst = [m.to(device) for m in batch["inst_masks"]]
            sev_list = [s.to(device) for s in batch["inst_sev"]]
            class_labels_flat = torch.cat(sev_list, dim=0)

            seg_logits, damage_output = model(img, inst_masks=inst)
            p_seg = torch.sigmoid(seg_logits)
            bin_jacc.update(p_seg, sem)
            bin_f1.update(p_seg, sem)

            if damage_output is None:
                seg_l = torch.nn.functional.binary_cross_entropy_with_logits(
                    seg_logits, sem.float())
                val_loss_sum += seg_l.item()
                val_loss_count += 1
                continue

            if len(damage_output) == 3:
                gate_logits_list, leaf_logits_list, _ = damage_output
            else:
                gate_logits_list, leaf_logits_list = damage_output
            if len(gate_logits_list) > 0 and any(s.numel() > 0 for s in gate_logits_list):
                gate_logits_flat = torch.cat(
                    [s for s in gate_logits_list if s.numel() > 0], dim=0)
                leaf_logits_flat = torch.cat(
                    [s for s in leaf_logits_list if s.numel() > 0], dim=0)

                loss, seg_l, gate_l, leaf_l, _ = calculate_gated_loss(
                    seg_logits, gate_logits_flat, leaf_logits_flat, sem,
                    class_labels_flat, device,
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    gate_weight=gate_weight,
                    leaf_weight=leaf_weight,
                )
                if not torch.isnan(loss):
                    val_loss_sum += loss.item()
                    val_seg_sum += seg_l.item()
                    val_gate_sum += gate_l.item()
                    val_leaf_sum += leaf_l.item()
                    val_loss_count += 1

                if gate_logits_flat.numel() > 0 and class_labels_flat.numel() > 0:
                    valid_mask = class_labels_flat != IGNORE_INDEX
                    if valid_mask.any():
                        valid_gate = gate_logits_flat[valid_mask]
                        valid_leaf = leaf_logits_flat[valid_mask]
                        valid_labels = class_labels_flat[valid_mask]

                        # 5-class prediction via gate + leaf. Support both the
                        # legacy hard-threshold decoder and the true joint
                        # argmax so runs remain directly comparable.
                        if decoder_mode == "joint_argmax":
                            class_pred = predict_5class_from_gate(
                                valid_gate, valid_leaf)
                        elif decoder_mode == "threshold_0.5":
                            class_pred = predict_5class_from_gate(
                                valid_gate, valid_leaf, threshold=0.5)
                        else:
                            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

                        # Gate accuracy
                        gate_target = (valid_labels > 0).long()
                        gate_pred = (torch.sigmoid(valid_gate) > 0.5).long()
                        gate_correct += (gate_pred == gate_target).sum().item()
                        gate_total += valid_labels.shape[0]

                        mc_f1.update(class_pred, valid_labels)
                        mc_f1_per_class.update(class_pred, valid_labels)
                        mc_prec_per_class.update(class_pred, valid_labels)
                        mc_rec_per_class.update(class_pred, valid_labels)

                        for t in valid_labels:
                            ti = int(t.item())
                            if 0 <= ti < NUM_CLASSES:
                                val_support[ti] += 1

                        for p, t in zip(class_pred, valid_labels):
                            if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                                confusion_counts[t, p] += 1

            # E2E pass
            pred_masks_list = _pred_instances_from_seg(p_seg)
            pred_masks_list = [m.to(device) for m in pred_masks_list]
            if any(m.numel() > 0 for m in pred_masks_list):
                _, damage_e2e = model(img, inst_masks=pred_masks_list)
                if len(damage_e2e) == 3:
                    gate_e2e_list, leaf_e2e_list, _ = damage_e2e
                else:
                    gate_e2e_list, leaf_e2e_list = damage_e2e

                H, W = img.shape[-2], img.shape[-1]
                for b in range(img.shape[0]):
                    gl = gate_e2e_list[b]
                    ll = leaf_e2e_list[b]
                    if gl.numel() == 0:
                        continue
                    if decoder_mode == "joint_argmax":
                        pred_cls = predict_5class_from_gate(gl, ll)
                    elif decoder_mode == "threshold_0.5":
                        pred_cls = predict_5class_from_gate(
                            gl, ll, threshold=0.5)
                    else:
                        raise ValueError(f"Unknown decoder_mode: {decoder_mode}")
                    gt_pix = _gt_pixel_class_map(
                        inst[b], sev_list[b], H, W, device)
                    targets = _pred_instance_targets(
                        pred_masks_list[b], gt_pix)
                    assert pred_cls.shape[0] == targets.shape[0]
                    valid = targets != IGNORE_INDEX
                    if not valid.any():
                        continue
                    vp, vt = pred_cls[valid], targets[valid]
                    mc_f1_e2e.update(vp, vt)
                    mc_f1_e2e_per_class.update(vp, vt)
                    mc_prec_e2e_per_class.update(vp, vt)
                    mc_rec_e2e_per_class.update(vp, vt)
                    for p, t in zip(vp, vt):
                        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                            confusion_counts_e2e[t, p] += 1

    if world_size > 1:
        dist.all_reduce(confusion_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion_counts_e2e, op=dist.ReduceOp.SUM)
        counters = torch.tensor(
            [gate_correct, gate_total], dtype=torch.float64, device=device)
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        gate_correct = counters[0].item()
        gate_total = counters[1].item()
        dist.all_reduce(val_support, op=dist.ReduceOp.SUM)

    val_loss_components = {}
    if val_loss_count > 0:
        for name, running in (("Val_Loss", val_loss_sum),
                               ("Val_Seg_Loss", val_seg_sum),
                               ("Val_Gate_Loss", val_gate_sum),
                               ("Val_Leaf_Loss", val_leaf_sum)):
            avg = running / val_loss_count
            t = reduce_metric(torch.tensor(avg, device=device), world_size)
            val_loss_components[name] = t.item() if isinstance(t, torch.Tensor) else t

    metrics = {
        "Seg_IoU": bin_jacc.compute().item(),
        "Seg_F1": bin_f1.compute().item(),
        "Damage_Macro_F1_oracle": mc_f1.compute().item(),
        "Damage_Macro_F1_e2e": mc_f1_e2e.compute().item(),
        "Damage_Macro_F1": mc_f1_e2e.compute().item(),
        "Gate_Acc": gate_correct / max(gate_total, 1),
    }
    metrics.update(val_loss_components)

    if world_size > 1:
        for key in list(metrics.keys()):
            val_t = torch.tensor(metrics[key], device=device, dtype=torch.float32)
            dist.all_reduce(val_t, op=dist.ReduceOp.SUM)
            metrics[key] = (val_t / world_size).item()

    def _reduce_scalar(x):
        if world_size > 1:
            t = torch.tensor(x, device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return (t / world_size).item()
        return x

    oracle_f1 = mc_f1_per_class.compute()
    oracle_prec = mc_prec_per_class.compute()
    oracle_rec = mc_rec_per_class.compute()
    e2e_f1 = mc_f1_e2e_per_class.compute()
    e2e_prec = mc_prec_e2e_per_class.compute()
    e2e_rec = mc_rec_e2e_per_class.compute()
    for i in range(len(CATEGORIES)):
        cat = CATEGORIES[i].replace(" ", "_")
        metrics[f"F1_{cat}_oracle"] = _reduce_scalar(oracle_f1[i].item())
        metrics[f"F1_{cat}_e2e"] = _reduce_scalar(e2e_f1[i].item())
        metrics[f"F1_{cat}"] = metrics[f"F1_{cat}_e2e"]
        metrics[f"Precision_{cat}_oracle"] = _reduce_scalar(oracle_prec[i].item())
        metrics[f"Precision_{cat}_e2e"] = _reduce_scalar(e2e_prec[i].item())
        metrics[f"Recall_{cat}_oracle"] = _reduce_scalar(oracle_rec[i].item())
        metrics[f"Recall_{cat}_e2e"] = _reduce_scalar(e2e_rec[i].item())
        metrics[f"Support_{cat}"] = int(val_support[i].item())

    if rank == 0:
        print(f"\nGate Acc (Damaged vs Undamaged): {metrics['Gate_Acc']:.4f}")
        print(f"Damage Macro F1 — oracle: {metrics['Damage_Macro_F1_oracle']:.4f} | "
              f"e2e: {metrics['Damage_Macro_F1_e2e']:.4f}")

    metrics["_confusion_oracle"] = confusion_counts.detach().cpu()
    metrics["_confusion_e2e"] = confusion_counts_e2e.detach().cpu()
    return metrics
