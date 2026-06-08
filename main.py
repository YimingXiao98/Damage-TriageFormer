import time
_T_IMPORT_START = time.perf_counter()

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.config import (
    IMAGE_DIR, LABEL_DIR, MASK_DIR, DAMAGE_DIR,
    BATCH_SIZE, IM_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    DEVICE, NUM_CLASSES, CATEGORIES, USE_HIERARCHICAL,
    RUNS_ROOT,
)
from src.data.loader import get_dataloaders
from src.model.net import Model
from src.training.engine import (
    train_one_epoch, validate,
    train_one_epoch_hierarchical, validate_hierarchical,
    train_one_epoch_gated, validate_gated,
)
from src.utils.visualization import visualize, visualize_with_gt
from src.utils.common import seed_everything


def setup_distributed():
    """Initialize DDP if launched via torchrun, otherwise single-GPU defaults."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Destroy the process group if distributed is active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv3 Damage Assessment Training")

    # Basic training parameters
    parser.add_argument("--batch-size", type=int,
                        default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int,
                        default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float,
                        default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")

    # Model configuration
    parser.add_argument("--frozen-backbone", action="store_true",
                        help="Freeze the entire backbone (faster but less adaptive)")
    parser.add_argument("--unfreeze-blocks", type=int, default=1,
                        help="Number of transformer blocks to unfreeze from the end")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                        help="Learning rate multiplier for backbone parameters")

    # Loss configuration
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                        help="Use focal loss for severity classification")
    parser.add_argument("--no-focal-loss", dest="use_focal_loss",
                        action="store_false",
                        help="Disable focal loss (use vanilla CE)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--use-class-weights", action="store_true", default=True,
                        help="Use class weights in loss function")
    parser.add_argument("--no-class-weights", dest="use_class_weights",
                        action="store_false",
                        help="Disable class weights (uniform)")
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable DistributedWeightedSampler in DDP; use "
                             "plain DistributedSampler (instance-balanced).")
    parser.add_argument("--use-aux-severity", action="store_true",
                        help="Add aux severity-regression head on the gated "
                             "shared embedding. Predicts a [0,1] severity "
                             "scalar per instance; trained against soft "
                             "class-derived targets to pull damaged classes "
                             "apart along an interpretable axis.")
    parser.add_argument("--aux-severity-weight", type=float, default=0.5,
                        help="Loss multiplier for the aux severity head.")
    parser.add_argument("--aux-severity-targets", type=str,
                        default="0.0,0.3,0.7,0.5,1.0",
                        help="Comma-separated 5-vec of severity targets "
                             "indexed by 5-class label "
                             "(default: Und=0, PR=0.3, TR=0.7, PS=0.5, TS=1.0).")
    parser.add_argument("--gate-class-weights", type=str, default=None,
                        help="Comma-separated 5-vec of per-class weights for "
                             "the binary gate BCE (indexed by 5-class label, "
                             "e.g. '1.0,1.0,3.0,2.0,1.0' boosts Total Roof and "
                             "Partial Struct). Attacks the gate-miss problem "
                             "diagnosed for Total Roof.")
    parser.add_argument("--la-tau", type=float, default=0.0,
                        help="Logit Adjustment tau (Menon ICLR 2021). 0=off. "
                             "Applies tau*log(prior) offset to logits during "
                             "training on type and extent sub-heads.")
    parser.add_argument("--use-ema", action="store_true",
                        help="Maintain EMA copy of weights (decay 0.9995) and "
                             "use for validation + checkpointing.")
    parser.add_argument("--ema-decay", type=float, default=0.9995,
                        help="EMA decay rate (only used with --use-ema).")
    parser.add_argument("--drop-path", type=float, default=0.0,
                        help="Stochastic depth rate for DINOv3 backbone "
                             "(0 = off, 0.1-0.2 typical for ViT fine-tuning).")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--use-copy-paste", action="store_true",
                        help="Enable copy-paste augmentation to balance rare damage classes")
    parser.add_argument("--copy-paste-max-prob", type=float, default=0.3,
                        help="Peak paste probability after warmup (rescue recipe). "
                             "0.3 matches Ghiasi et al. default.")
    parser.add_argument("--copy-paste-warmup-epochs", type=int, default=2,
                        help="Epochs to linearly ramp paste_prob from 0 to "
                             "--copy-paste-max-prob. Keeps early training stable.")
    parser.add_argument("--copy-paste-max-pastes", type=int, default=2,
                        help="Hard cap on pastes per tile (prevents compounding "
                             "loss multipliers / gradient explosions).")
    parser.add_argument("--copy-paste-feather", type=int, default=3,
                        help="Gaussian-blur radius for soft alpha blending on "
                             "paste edges. 0 disables feathering.")

    # Balanced sampling
    parser.add_argument("--use-balanced-sampling", action="store_true",
                        help="Use balanced class sampling (limits overrepresented classes)")
    parser.add_argument("--samples-per-class", type=int, default=3000,
                        help="Target number of samples per class for balanced sampling")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the run (default: timestamp)")

    # Stratified splits (replaces the default 80/20 random split)
    parser.add_argument("--stratified-splits", type=str, default=None,
                        help="Path to stratified_splits.json (built by "
                             "tools/build_stratified_splits.py).  When set, "
                             "the random 80/20 split is replaced by the "
                             "splits in this file.")
    parser.add_argument("--val-split-key", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split key to use as the validation fold "
                             "during training (default: 'val').  Use 'test' "
                             "only for the final post-hoc evaluation.")

    # Architecture ablation flags (these were previously hard-coded to True)
    parser.add_argument("--no-multiscale", action="store_true",
                        help="Disable ASPP multi-scale feature aggregation in "
                             "the damage head pooling")
    parser.add_argument("--no-attention-pool", action="store_true",
                        help="Disable spatial attention pooling in the damage "
                             "head; fall back to plain mask average")
    parser.add_argument("--use-sfp", action="store_true",
                        help="Enable Simple Feature Pyramid: upsample ViT "
                             "features to 1/4 resolution before per-instance "
                             "pooling (much finer spatial detail)")
    parser.add_argument("--sfp-stride", type=int, default=4,
                        choices=[2, 4, 8],
                        help="SFP target stride: 4 -> 256x256 features (most "
                             "detail, most VRAM); 8 -> 128x128 (modest); 2 -> "
                             "512x512 (very expensive)")

    # Hierarchical and Tiling
    parser.add_argument("--use-hierarchical", action="store_true", default=USE_HIERARCHICAL,
                        help="Use hierarchical damage classification (Type×Extent)")
    parser.add_argument("--use-gated-head", action="store_true",
                        help="Use Damaged/Undamaged gate + conditional 4-way head "
                             "(xView2 pattern). Overrides --use-hierarchical.")
    parser.add_argument("--gate-weight", type=float, default=1.0,
                        help="Weight for gate (Damaged vs Undamaged) loss component")
    parser.add_argument("--leaf-weight", type=float, default=2.0,
                        help="Weight for leaf (4-way damaged classifier) loss component")
    parser.add_argument("--use-tiles", action="store_true", default=True,
                        help="Use tiled dataset (1024x1024)")
    parser.add_argument("--type-weight", type=float, default=5.0,
                        help="Weight for Type classification loss")
    parser.add_argument("--extent-weight", type=float, default=5.0,
                        help="Weight for Extent classification loss")
    parser.add_argument("--disable-hard-class-boost", action="store_true",
                        help="Disable the hardcoded 1.5x boost on class 2/3 "
                             "in the hierarchical loss. Class weights still apply. "
                             "Use this to run clean imbalance-lever experiments.")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch = "
                             "batch_size * world_size * accumulation_steps.")

    return parser.parse_args()


def load_or_compute_class_weights(train_loader, device):
    """
    Load precomputed class weights or compute from data.

    Uses inverse square root frequency for balanced weighting.
    """
    weights_file = "class_weights.json"

    if os.path.exists(weights_file):
        print(f"Loading class weights from {weights_file}")
        with open(weights_file, 'r') as f:
            weights = json.load(f)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        print("Computing class weights from training data...")
        from src.utils.weights import calculate_class_weights
        weights = calculate_class_weights(train_loader.dataset)
        weights = weights.to(device)

        # Save for future use
        with open(weights_file, 'w') as f:
            json.dump(weights.cpu().tolist(), f)
        print(f"Saved class weights to {weights_file}")

    print(f"Class weights: {weights}")
    return weights


def create_model(im_size, num_classes, frozen_backbone,
                 unfreeze_blocks, use_hierarchical, drop_path_rate=0.0,
                 use_gated_head=False, use_aux_severity=False,
                 use_attention=True, use_multiscale=True,
                 use_sfp=False, sfp_stride=4):
    """Build the post-event-only DINOv3 damage assessment model."""
    print("Creating Single-Image Model (post-disaster only)")
    return Model(
        im_size=im_size,
        num_classes=num_classes,
        frozen_backbone=frozen_backbone,
        unfreeze_blocks=unfreeze_blocks,
        severity_hidden_dim=512,
        use_attention=use_attention,
        use_multiscale=use_multiscale,
        use_hierarchical=use_hierarchical,
        drop_path_rate=drop_path_rate,
        use_gated_head=use_gated_head,
        use_aux_severity=use_aux_severity,
        use_sfp=use_sfp,
        sfp_stride=sfp_stride,
    )


def main():
    # Early diagnostic prints to identify hangs. Use flush=True to bypass any
    # buffering; print from every rank so we can see if one rank is stuck.
    _rank_env = os.environ.get("RANK", "0")
    print(f"[diag rank={_rank_env}] main() entered, imports done "
          f"(top-level took {time.perf_counter() - _T_IMPORT_START:.2f}s)",
          flush=True)
    t_main = time.perf_counter()
    args = parse_args()
    print(f"[diag rank={_rank_env}] parse_args done", flush=True)
    seed_everything(args.seed)
    print(f"[diag rank={_rank_env}] seed_everything done", flush=True)

    # --- Distributed setup (no-op when launched with plain `python`) ---
    print(f"[diag rank={_rank_env}] calling setup_distributed...", flush=True)
    rank, local_rank, world_size = setup_distributed()
    print(f"[diag rank={_rank_env}] setup_distributed done "
          f"(rank={rank}, local_rank={local_rank}, world_size={world_size})",
          flush=True)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[timing] module imports: {t_main - _T_IMPORT_START:.2f}s")
        if world_size > 1:
            print(f"Distributed training: world_size={world_size}")
        print(f"Using device: {device}")
        print(f"Configuration: {args}")

    # Setup Run Directory
    if args.name:
        run_name = args.name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(RUNS_ROOT, run_name)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run directory: {run_dir}")

        # Save configuration
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Ensure run_dir exists on all ranks before proceeding
    if world_size > 1:
        dist.barrier()

    # Setup TensorBoard (rank 0 only)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs")) if rank == 0 else None

    # 1. Data Loading
    if rank == 0:
        print("Loading data...")
    t_data = time.perf_counter()
    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(
        IMAGE_DIR, LABEL_DIR, MASK_DIR, DAMAGE_DIR,
        batch_size=args.batch_size,
        image_size=IM_SIZE,
        num_workers=args.num_workers,
        augment_train=not args.no_augment,
        use_tiles=args.use_tiles,
        use_balanced_sampling=args.use_balanced_sampling,
        samples_per_class=args.samples_per_class,
        use_copy_paste=args.use_copy_paste,
        copy_paste_initial_prob=0.0 if args.copy_paste_warmup_epochs > 0
                                else args.copy_paste_max_prob,
        copy_paste_max_pastes=args.copy_paste_max_pastes,
        copy_paste_feather=args.copy_paste_feather,
        rank=rank,
        world_size=world_size,
        no_weighted_sampler=args.no_weighted_sampler,
        stratified_splits_path=args.stratified_splits,
        val_split_key=args.val_split_key,
    )
    if rank == 0:
        print(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"[timing] data loading: {time.perf_counter() - t_data:.2f}s")

    # 2. Model Setup
    if rank == 0:
        print("Initializing model...")
    t_model = time.perf_counter()
    model = create_model(
        im_size=IM_SIZE,
        num_classes=NUM_CLASSES,
        frozen_backbone=args.frozen_backbone,
        unfreeze_blocks=args.unfreeze_blocks,
        use_hierarchical=args.use_hierarchical,
        drop_path_rate=args.drop_path,
        use_gated_head=args.use_gated_head,
        use_aux_severity=args.use_aux_severity,
        use_attention=not args.no_attention_pool,
        use_multiscale=not args.no_multiscale,
        use_sfp=args.use_sfp,
        sfp_stride=args.sfp_stride,
    ).to(device)
    if rank == 0:
        print(f"[timing] model init + .to(device): "
              f"{time.perf_counter() - t_model:.2f}s")

    # Count trainable parameters
    if rank == 0:
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Setup optimizer with different learning rates for backbone and heads
    # With DDP, get_param_groups must be called on the unwrapped model
    # (before wrapping), so we do it here.
    param_groups = model.get_param_groups(
        backbone_lr_mult=args.backbone_lr_mult)
    optimizer = optim.AdamW([
        {'params': param_groups[0]['params'], 'lr': args.lr},  # Heads
        {'params': param_groups[1]['params'],
            'lr': args.lr * args.backbone_lr_mult}  # Backbone
    ], weight_decay=args.weight_decay)

    # Learning Rate Scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 2
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint (load into unwrapped model before DDP wrap)
    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint (from last_checkpoint.pth)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    # LR override: preserve Adam momentum/variance but reset
                    # base LR to the new --lr value. Without this, the loaded
                    # optimizer state locks in whatever LR the checkpoint was
                    # saved with.
                    new_head_lr = args.lr
                    new_backbone_lr = args.lr * args.backbone_lr_mult
                    # Param groups: [0] = heads, [1] = backbone.
                    for pg, new_lr in zip(
                            optimizer.param_groups,
                            [new_head_lr, new_backbone_lr]):
                        pg['initial_lr'] = new_lr
                        pg['lr'] = new_lr
                    # Rebuild scheduler base_lrs to match the new initial_lr
                    # so LambdaLR multiplies against the new base.
                    if hasattr(scheduler, 'base_lrs'):
                        scheduler.base_lrs = [new_head_lr, new_backbone_lr]
                    # Apply current lambda to lr (so we're at the right
                    # schedule position for start_epoch, but with new base).
                    scheduler.step(start_epoch - 1)
                    if rank == 0:
                        print(f"LR override applied: head={new_head_lr:.2e}, "
                              f"backbone={new_backbone_lr:.2e} "
                              f"(Adam momentum preserved)")
                except Exception as _e:
                    if rank == 0:
                        print(f"Could not load optimizer state ({_e}), "
                              "starting fresh")
                best_f1 = checkpoint.get('best_f1', 0.0)
                if rank == 0:
                    print(
                        f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                # Raw state_dict (from best_model.pth)
                model.load_state_dict(checkpoint, strict=False)
                if rank == 0:
                    print(f"Loaded raw state_dict from '{args.resume}' (starting from epoch 0)")
        else:
            if rank == 0:
                print(f"No checkpoint found at '{args.resume}'")

    # Wrap with DDP after loading checkpoint, before training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Class weights for focal loss
    t_w = time.perf_counter()
    if args.use_class_weights:
        class_weights = load_or_compute_class_weights(train_loader, device)
    else:
        class_weights = None
        if rank == 0:
            print("Class weights disabled.")
    if rank == 0:
        print(f"[timing] class weights: {time.perf_counter() - t_w:.2f}s")

    # Compute class priors (always, for logit adjustment or diagnostics).
    # If class_weights were computed via inverse-sqrt, priors = (1/w^2) / sum.
    # Otherwise load them directly from class_weights.json if present.
    type_priors = None
    extent_priors = None
    gate_prior = None
    leaf_priors = None
    if args.la_tau > 0.0:
        # Always need priors if using LA — load/compute independent of
        # args.use_class_weights.
        weights_for_priors = class_weights
        if weights_for_priors is None:
            weights_for_priors = load_or_compute_class_weights(train_loader, device)
        w = weights_for_priors.to(device).float()
        class_counts = 1.0 / (w ** 2 + 1e-10)
        priors_5 = class_counts / class_counts.sum()
        # Type priors: 0 (None) = class 0, 1 (Roof) = classes 1+2, 2 (Struct) = classes 3+4
        type_priors = torch.tensor([
            priors_5[0].item(),
            priors_5[1].item() + priors_5[2].item(),
            priors_5[3].item() + priors_5[4].item(),
        ], device=device, dtype=torch.float32)
        # Extent priors (conditional on damaged): Partial = 1+3, Total = 2+4
        damaged_mass = 1.0 - priors_5[0].item()
        extent_priors = torch.tensor([
            (priors_5[1].item() + priors_5[3].item()) / damaged_mass,
            (priors_5[2].item() + priors_5[4].item()) / damaged_mass,
        ], device=device, dtype=torch.float32)
        # Gated-head priors: binary gate and 4-way leaf (conditional on damaged)
        gate_prior = torch.tensor(damaged_mass, device=device, dtype=torch.float32)
        leaf_priors = torch.tensor([
            priors_5[1].item() / damaged_mass,  # Partial Roof
            priors_5[2].item() / damaged_mass,  # Total Roof
            priors_5[3].item() / damaged_mass,  # Partial Structural
            priors_5[4].item() / damaged_mass,  # Total Structural
        ], device=device, dtype=torch.float32)
        if rank == 0:
            print(f"[LA] 5-class priors: {priors_5.cpu().tolist()}")
            print(f"[LA] type priors (None/Roof/Struct): {type_priors.cpu().tolist()}")
            print(f"[LA] extent priors (Partial/Total | Damaged): {extent_priors.cpu().tolist()}")
            print(f"[LA] gate prior (P damaged): {gate_prior.item():.4f}")
            print(f"[LA] leaf priors (PR/TR/PS/TS | Damaged): {leaf_priors.cpu().tolist()}")
            print(f"[LA] tau = {args.la_tau}")

    # Aux severity-regression targets (5-vec, indexed by 5-class label)
    aux_severity_targets_t = None
    if args.use_aux_severity:
        sev_targets = [float(x) for x in args.aux_severity_targets.split(",")]
        if len(sev_targets) != NUM_CLASSES:
            raise ValueError(
                f"--aux-severity-targets expects {NUM_CLASSES} comma-separated "
                f"values, got {len(sev_targets)}: {args.aux_severity_targets}")
        aux_severity_targets_t = torch.tensor(
            sev_targets, device=device, dtype=torch.float32)
        if rank == 0:
            print(f"[aux-severity] targets {dict(zip(CATEGORIES, sev_targets))}, "
                  f"weight={args.aux_severity_weight}")

    # Per-class gate-loss weights (boost rare-class gate BCE)
    gate_class_weights_t = None
    if args.gate_class_weights is not None:
        weights = [float(x) for x in args.gate_class_weights.split(",")]
        if len(weights) != NUM_CLASSES:
            raise ValueError(
                f"--gate-class-weights expects {NUM_CLASSES} comma-separated "
                f"values, got {len(weights)}: {args.gate_class_weights}")
        gate_class_weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
        if rank == 0:
            print(f"[gate-class-weights] {dict(zip(CATEGORIES, weights))}")

    # EMA model (step-by-step weight averaging for calibration / stability)
    ema_model = None
    if args.use_ema:
        from torch.optim.swa_utils import AveragedModel
        raw = model.module if hasattr(model, 'module') else model
        ema_decay = args.ema_decay
        ema_model = AveragedModel(
            raw,
            avg_fn=lambda avg, cur, _: ema_decay * avg + (1.0 - ema_decay) * cur,
        )
        ema_model = ema_model.to(device)
        if rank == 0:
            print(f"[EMA] enabled with decay={ema_decay}")

    # 3. Training Loop
    if rank == 0:
        print(f"[timing] total startup (imports -> start of training loop): "
              f"{time.perf_counter() - _T_IMPORT_START:.2f}s")
        print("Starting training...")
        print(f"  - Focal Loss: {args.use_focal_loss} (gamma={args.focal_gamma})")
        print(f"  - Label Smoothing: {args.label_smoothing}")
        print(f"  - Class Weights: {args.use_class_weights}")
        if args.use_class_weights and class_weights is not None:
            print(f"    Values: {class_weights.cpu().tolist()}")
        if args.use_hierarchical:
            print(f"  - Hierarchical Boost: Type={args.type_weight}, Extent={args.extent_weight}")
        print(f"  - Data Augmentation: {not args.no_augment}")
        if args.use_copy_paste:
            print(f"  - Copy-Paste Augmentation: enabled (rare-class balancing)")
        if args.use_balanced_sampling:
            print(f"  - Balanced Sampling: {args.samples_per_class} samples per class")

    nan_count = 0
    max_nan_epochs = 3  # Stop if NaN persists

    for epoch in range(start_epoch, args.epochs):
        # Set epoch on DistributedSampler so each epoch shuffles differently
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Ramp copy-paste probability: 0 → max over warmup_epochs (linear).
        # Shared mp.Value is read by forked DataLoader workers on each iter().
        if args.use_copy_paste:
            cp_aug = getattr(train_loader.dataset, "copy_paste", None)
            if cp_aug is not None:
                warmup = max(args.copy_paste_warmup_epochs, 0)
                if warmup == 0:
                    cp_prob = args.copy_paste_max_prob
                else:
                    cp_prob = args.copy_paste_max_prob * min(
                        (epoch + 1) / warmup, 1.0)
                cp_aug.set_paste_prob(cp_prob)
                if rank == 0:
                    print(f"[CopyPaste] epoch {epoch}: paste_prob={cp_prob:.3f}")

        # Train
        if args.use_gated_head:
            train_stats = train_one_epoch_gated(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=args.epochs,
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
                gate_weight=args.gate_weight,
                leaf_weight=args.leaf_weight,
                rank=rank,
                world_size=world_size,
                accumulation_steps=args.accumulation_steps,
                la_tau=args.la_tau,
                gate_prior=gate_prior,
                leaf_priors=leaf_priors,
                ema_model=ema_model,
                gate_class_weights=gate_class_weights_t,
                aux_severity_weight=args.aux_severity_weight if args.use_aux_severity else 0.0,
                aux_severity_targets=aux_severity_targets_t,
            )
        elif args.use_hierarchical:
            train_stats = train_one_epoch_hierarchical(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=args.epochs,
                class_weights=class_weights,
                use_focal=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                type_weight=args.type_weight,
                extent_weight=args.extent_weight,
                hard_class_boost=not args.disable_hard_class_boost,
                rank=rank,
                world_size=world_size,
                accumulation_steps=args.accumulation_steps,
                la_tau=args.la_tau,
                type_priors=type_priors,
                extent_priors=extent_priors,
                ema_model=ema_model,
            )
        else:
            train_loss_val = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=args.epochs,
                class_weights=class_weights,
                use_focal=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                rank=rank,
                world_size=world_size,
            )
            train_stats = {'loss': train_loss_val}

        train_loss = train_stats['loss']

        # Check for NaN loss
        if torch.isnan(torch.tensor(train_loss)):
            nan_count += 1
            if rank == 0:
                print(
                    f"[WARN] NaN loss detected (count: {nan_count}/{max_nan_epochs})")
            if nan_count >= max_nan_epochs:
                if rank == 0:
                    print("[ERROR] Too many NaN epochs, stopping training")
                break
            continue
        else:
            nan_count = 0

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
            writer.add_scalar("Loss/train", train_loss, epoch)
            # Decomposed train losses (hierarchical only)
            for k in ('seg_loss', 'type_loss', 'extent_loss'):
                if k in train_stats:
                    writer.add_scalar(f"Loss/train_{k}", train_stats[k], epoch)
            if 'grad_norm' in train_stats:
                writer.add_scalar("Train/grad_norm", train_stats['grad_norm'], epoch)

        # Validate — use EMA model if available (AveragedModel is a plain
        # nn.Module; each rank has identical EMA weights so no DDP wrap needed).
        eval_model = ema_model if ema_model is not None else model
        if args.use_gated_head:
            val_metrics = validate_gated(
                eval_model, val_loader, device, rank=rank, world_size=world_size,
                class_weights=class_weights,
                label_smoothing=args.label_smoothing,
                gate_weight=args.gate_weight, leaf_weight=args.leaf_weight)
        elif args.use_hierarchical:
            val_metrics = validate_hierarchical(
                eval_model, val_loader, device, rank=rank, world_size=world_size,
                class_weights=class_weights, focal_gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
                type_weight=args.type_weight, extent_weight=args.extent_weight)
        else:
            val_metrics = validate(eval_model, val_loader, device, rank=rank, world_size=world_size)

        if rank == 0:
            # Pop confusion matrices out before logging scalar metrics.
            cm_oracle = val_metrics.pop("_confusion_oracle", None)
            cm_e2e = val_metrics.pop("_confusion_e2e", None)

            scalar_metrics = {k: v for k, v in val_metrics.items()
                              if not k.startswith("_")}
            print(f"Epoch {epoch+1}/{args.epochs} - Val Metrics: {scalar_metrics}")

            for k, v in scalar_metrics.items():
                writer.add_scalar(f"Metrics/{k}", v, epoch)

            if "Val_Loss" in scalar_metrics:
                writer.add_scalar("Loss/val", scalar_metrics["Val_Loss"], epoch)
            # Decomposed val losses (hierarchical only).
            for k in ("Val_Seg_Loss", "Val_Type_Loss", "Val_Extent_Loss"):
                if k in scalar_metrics:
                    writer.add_scalar(
                        f"Loss/val_{k.replace('Val_', '').lower()}",
                        scalar_metrics[k], epoch)

            # Log confusion matrices as heatmap images (single image per epoch,
            # far more readable than per-cell scalars).
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt
                import numpy as _np
                import io as _io
                from PIL import Image as _Image
                for name, cm in (("oracle", cm_oracle), ("e2e", cm_e2e)):
                    if cm is None:
                        continue
                    cm_np = cm.cpu().numpy()
                    row_sums = cm_np.sum(axis=1, keepdims=True)
                    cm_norm = _np.divide(cm_np, row_sums,
                                         out=_np.zeros_like(cm_np, dtype=float),
                                         where=row_sums > 0)
                    fig, ax = _plt.subplots(figsize=(5, 4.5))
                    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                    ax.set_xticks(range(len(CATEGORIES)))
                    ax.set_yticks(range(len(CATEGORIES)))
                    labels = [c.replace(" ", "\n") for c in CATEGORIES]
                    ax.set_xticklabels(labels, fontsize=8)
                    ax.set_yticklabels(labels, fontsize=8)
                    ax.set_xlabel("Pred"); ax.set_ylabel("True")
                    ax.set_title(f"{name} CM (row-normalized) — ep{epoch+1}")
                    for i in range(cm_np.shape[0]):
                        for j in range(cm_np.shape[1]):
                            ax.text(j, i, f"{int(cm_np[i,j])}",
                                    ha="center", va="center", fontsize=7,
                                    color="white" if cm_norm[i, j] > 0.5 else "black")
                    fig.colorbar(im, ax=ax, fraction=0.046)
                    fig.tight_layout()
                    buf = _io.BytesIO()
                    fig.savefig(buf, format="png", dpi=110)
                    _plt.close(fig)
                    buf.seek(0)
                    arr = _np.array(_Image.open(buf).convert("RGB"))
                    writer.add_image(f"Confusion/{name}", arr, epoch,
                                     dataformats="HWC")
            except Exception as _cm_err:
                # Fall back to per-cell scalars if matplotlib isn't available
                for name, cm in (("oracle", cm_oracle), ("e2e", cm_e2e)):
                    if cm is None:
                        continue
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            writer.add_scalar(
                                f"Confusion_{name}/true{i}_pred{j}",
                                cm[i, j].item(), epoch)

            # Log per-class F1 specifically (show e2e as primary)
            for cat in CATEGORIES:
                key = f"F1_{cat.replace(' ', '_')}"
                if key in val_metrics:
                    print(f"  {cat}: F1={val_metrics[key]:.4f}")

        # Step the scheduler
        scheduler.step()
        if rank == 0:
            head_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR/head", head_lr, epoch)
            print(f"  LR (head): {head_lr:.2e}")
            if len(optimizer.param_groups) > 1:
                backbone_lr = optimizer.param_groups[-1]["lr"]
                writer.add_scalar("LR/backbone", backbone_lr, epoch)
                print(f"  LR (backbone): {backbone_lr:.2e}")

        # Save checkpoints (rank 0 only)
        if rank == 0:
            # Unwrap DDP for state_dict
            raw_model = model.module if hasattr(model, 'module') else model
            ckpt = {
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': vars(args),
            }
            # Also persist EMA state in last_checkpoint so we can resume it.
            if ema_model is not None:
                # AveragedModel wraps the module under .module attribute.
                ema_raw = ema_model.module
                ckpt['ema_state_dict'] = ema_raw.state_dict()
            torch.save(ckpt, os.path.join(run_dir, "last_checkpoint.pth"))

            # Save best model -- selection on the end-to-end F1 so we don't
            # save a checkpoint whose "best" depends on GT segmentation masks.
            # If EMA is enabled, best_model.pth is the EMA weights (which is
            # what we evaluated on). Otherwise, the regular model weights.
            current_f1 = val_metrics.get(
                "Damage_Macro_F1_e2e", val_metrics.get("Damage_Macro_F1", 0.0))
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_state_dict = (ema_model.module.state_dict()
                                   if ema_model is not None
                                   else raw_model.state_dict())
                torch.save(best_state_dict, os.path.join(
                    run_dir, "best_model.pth"))
                tag = "(EMA)" if ema_model is not None else ""
                print(f"  ★ New best model! {tag} (e2e Macro F1: {best_f1:.4f})")

        # Visualize validation samples with GT comparison (class-balanced)
        if rank == 0 and epoch % 5 == 0:  # Visualize every 5 epochs
            vis_model = model.module if hasattr(model, 'module') else model
            vis_model.eval()
            vis_dir = os.path.join(run_dir, "vis", f"epoch_{epoch}")

            # Collect samples for balanced visualization (5 per class = 20 total)
            collected_batches = []
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            target_per_class = 5
            max_batches = 50  # Limit search to avoid long loops

            val_iter = iter(val_loader)
            for _ in range(max_batches):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break

                # Check which classes are in this batch
                for sev_tensor in batch["inst_sev"]:
                    for sev in sev_tensor:
                        s = int(sev.item()) if hasattr(sev, 'item') else int(sev)
                        if 0 <= s <= 3 and class_counts[s] < target_per_class:
                            class_counts[s] += 1
                            collected_batches.append(batch)
                            break
                    else:
                        continue
                    break

                # Check if we have enough
                if all(c >= target_per_class for c in class_counts.values()):
                    break

            # Merge batches for visualization
            if collected_batches:
                merged_batch = {
                    "image": torch.cat([b["image"] for b in collected_batches], dim=0),
                    "inst_masks": sum([b["inst_masks"] for b in collected_batches], []),
                    "inst_sev": sum([b["inst_sev"] for b in collected_batches], []),
                    "image_path": sum([b["image_path"] for b in collected_batches], [])
                }
                visualize_with_gt(
                    merged_batch, vis_model, device, vis_dir,
                    num_samples=20,
                    use_hierarchical=USE_HIERARCHICAL and not args.use_gated_head,
                    use_gated_head=args.use_gated_head,
                )

    if rank == 0:
        print("Training complete.")
        print(f"Best Macro F1: {best_f1:.4f}")
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
