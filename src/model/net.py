import torch.nn as nn
from src.config import BACKBONE, IM_SIZE, NUM_CLASSES
from src.model.backbone import DinoV3Backbone
from src.model.heads import SegHead, SeverityHead
from src.model.hierarchical_head import HierarchicalDamageHead
from src.model.gated_head import GatedDamageHead
from src.model.sfp import SimpleFeaturePyramid


class Model(nn.Module):
    """
    DINOv3-based Building Damage Assessment Model.

    Architecture:
    - DINOv3 ViT-L backbone (optionally partially fine-tuned)
    - Segmentation head for building detection
    - Damage classification head (standard 4-class or hierarchical type+extent)

    Args:
        im_size: Input image size
        num_classes: Number of damage severity classes
        frozen_backbone: Whether to freeze the backbone entirely
        unfreeze_blocks: Number of transformer blocks to unfreeze (if not frozen)
        severity_hidden_dim: Hidden dimension for severity head
        use_attention: Whether to use attention pooling in severity head
        use_multiscale: Whether to use multi-scale features in severity head
        use_hierarchical: Whether to use hierarchical Type+Extent classification
    """

    def __init__(self, im_size=IM_SIZE, num_classes=NUM_CLASSES,
                 frozen_backbone=False, unfreeze_blocks=4,
                 severity_hidden_dim=512, use_attention=True, use_multiscale=True,
                 use_hierarchical=False, drop_path_rate=0.0,
                 use_gated_head=False, use_aux_severity=False,
                 use_sfp=False, sfp_stride=4):
        super().__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical
        self.use_gated_head = use_gated_head
        self.use_sfp = use_sfp

        # Load DINOv3 backbone with partial fine-tuning
        self.backbone = DinoV3Backbone(
            BACKBONE,
            frozen=frozen_backbone,
            unfreeze_blocks=unfreeze_blocks,
            unfreeze_norm=True,
            drop_path_rate=drop_path_rate,
        )

        # Channel dim of backbone features (DINOv3 ViT-L = 1024) — taken
        # from the model config to avoid a CPU dummy forward at init.
        c = self.backbone.vit.config.hidden_size

        # Segmentation head with batch norm for stability
        self.seg_head = SegHead(c, 256)

        # Optional Simple Feature Pyramid for the damage head.  Upsamples the
        # 1/16-stride ViT feature map to a finer stride (1/4 or 1/8).  The
        # seg head keeps using the original 1/16 features (its decoder does
        # its own upsampling).
        if use_sfp:
            self.sfp = SimpleFeaturePyramid(
                in_channels=c,
                target_stride=sfp_stride,
                source_stride=16,
            )
        else:
            self.sfp = None

        # Damage classification head — three mutually-exclusive paths.
        if use_gated_head:
            # Damaged/Undamaged gate + conditional 4-way (xView2 pattern).
            self.damage_head = GatedDamageHead(
                in_ch=c,
                hidden_dim=severity_hidden_dim,
                num_layers=2,
                dropout=0.3,
                use_attention=use_attention,
                use_multiscale=use_multiscale,
                use_aux_severity=use_aux_severity,
            )
        elif use_hierarchical:
            # Hierarchical: separate Type (Roof/Structural) and Extent (Partial/Total)
            self.damage_head = HierarchicalDamageHead(
                in_ch=c,
                hidden_dim=severity_hidden_dim,
                num_layers=2,
                dropout=0.3,
                use_attention=use_attention,
                use_multiscale=use_multiscale
            )
        else:
            # Standard 4-class classification
            self.damage_head = SeverityHead(
                in_ch=c,
                num_classes=self.num_classes,
                hidden_dim=severity_hidden_dim,
                num_layers=3,
                dropout=0.3,
                use_attention=use_attention,
                use_multiscale=use_multiscale
            )

    def forward(self, x, inst_masks=None):
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)
            inst_masks: Optional list of instance masks for damage prediction
        Returns:
            seg_logits: Segmentation logits (B, 1, H, W)
            damage_logits: 
                If hierarchical: (type_logits_list, extent_logits_list)
                If standard: [sev_logits_list]
        """
        # Extract backbone features
        f = self.backbone(x)  # (B, C, H, W) at stride 16

        # Segmentation prediction uses the original 1/16-stride features —
        # the seg decoder does its own upsampling internally.
        seg_logits = self.seg_head(f, out_hw=x.shape[-2:])

        # For per-instance damage features, optionally apply SFP to get a
        # higher-resolution feature map.  Mask downsampling inside each
        # damage head will adapt to whatever spatial resolution we pass.
        f_dmg = self.sfp(f) if self.sfp is not None else f

        # Damage prediction (only if instance masks provided)
        if inst_masks is not None:
            if self.use_gated_head:
                damage_out = self.damage_head(f_dmg, inst_masks)
                # damage_out is 2- or 3-tuple depending on use_aux_severity
                return seg_logits, damage_out
            elif self.use_hierarchical:
                type_logits, extent_logits = self.damage_head(f_dmg, inst_masks)
                return seg_logits, (type_logits, extent_logits)
            else:
                sev_logits = self.damage_head(f_dmg, inst_masks)
                return seg_logits, [sev_logits]
        else:
            return seg_logits, None

    def get_param_groups(self, backbone_lr_mult=0.1):
        """
        Get parameter groups with different learning rates.

        Args:
            backbone_lr_mult: Multiplier for backbone learning rate
        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {'params': head_params, 'lr_mult': 1.0},
            {'params': backbone_params, 'lr_mult': backbone_lr_mult}
        ]


class ModelFrozen(Model):
    """
    Model variant with completely frozen backbone.
    Faster training but may not adapt as well to specific damage patterns.
    """

    def __init__(self, im_size=IM_SIZE, num_classes=NUM_CLASSES,
                 severity_hidden_dim=512, use_attention=True, use_multiscale=True,
                 use_hierarchical=False):
        # Call parent with frozen backbone
        super().__init__(
            im_size=im_size,
            num_classes=num_classes,
            frozen_backbone=True,
            unfreeze_blocks=0,
            severity_hidden_dim=severity_hidden_dim,
            use_attention=use_attention,
            use_multiscale=use_multiscale,
            use_hierarchical=use_hierarchical
        )
