import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from einops import rearrange
from src.config import BACKBONE, PATCH_SIZE


class DinoV3Backbone(nn.Module):
    """
    DINOv3 Vision Transformer backbone with optional partial fine-tuning.

    Args:
        name_or_model: HuggingFace model name or pre-loaded model
        frozen: If True, freeze all backbone parameters
        unfreeze_blocks: Number of transformer blocks to unfreeze from the end
                        (only used if frozen=False). Set to -1 to unfreeze all.
        unfreeze_norm: Whether to unfreeze all LayerNorm layers (helps adaptation)
        drop_path_rate: Stochastic depth rate. 0 = no drop path. 0.1-0.2 is a
                        common regularizer for ViT fine-tuning on small data.
    """

    def __init__(self, name_or_model=BACKBONE, frozen=True,
                 unfreeze_blocks=4, unfreeze_norm=True, drop_path_rate=0.0):
        super().__init__()

        if isinstance(name_or_model, str):
            if drop_path_rate > 0.0:
                config = AutoConfig.from_pretrained(name_or_model)
                if hasattr(config, 'drop_path_rate'):
                    config.drop_path_rate = drop_path_rate
                    print(f"[Backbone] Setting drop_path_rate={drop_path_rate}")
                else:
                    print(f"[Backbone] Warning: config has no drop_path_rate field; ignoring")
                self.vit = AutoModel.from_pretrained(name_or_model, config=config)
            else:
                self.vit = AutoModel.from_pretrained(name_or_model)
        else:
            self.vit = name_or_model
        
        # Get the number of transformer blocks
        # DINOv3 uses model.layer directly (not model.encoder.layer)
        if hasattr(self.vit, 'encoder') and hasattr(self.vit.encoder, 'layer'):
            self.transformer_layers = self.vit.encoder.layer
        elif hasattr(self.vit, 'layer'):
            self.transformer_layers = self.vit.layer
        else:
            raise ValueError("Could not find transformer layers in model")
        
        self.num_blocks = len(self.transformer_layers)
        
        if frozen:
            # Freeze all parameters
            for p in self.vit.parameters():
                p.requires_grad_(False)
            print(f"[Backbone] All {self.num_blocks} blocks frozen")
        else:
            # Partial fine-tuning strategy
            self._setup_partial_finetuning(unfreeze_blocks, unfreeze_norm)
    
    def _setup_partial_finetuning(self, unfreeze_blocks, unfreeze_norm):
        """
        Set up partial fine-tuning by unfreezing specific layers.
        
        Strategy:
        1. Always freeze patch embedding (low-level features)
        2. Freeze early transformer blocks (generic features)
        3. Unfreeze last N blocks (task-specific adaptation)
        4. Optionally unfreeze all LayerNorm layers (helps domain adaptation)
        """
        # First, freeze everything
        for p in self.vit.parameters():
            p.requires_grad_(False)
        
        # Unfreeze the last N transformer blocks
        if unfreeze_blocks == -1:
            unfreeze_blocks = self.num_blocks
        
        unfreeze_from = max(0, self.num_blocks - unfreeze_blocks)
        
        for i, block in enumerate(self.transformer_layers):
            if i >= unfreeze_from:
                for p in block.parameters():
                    p.requires_grad_(True)
        
        # Unfreeze final layer norm if it exists
        for norm_name in ['layernorm', 'norm', 'ln']:
            if hasattr(self.vit, norm_name):
                norm_layer = getattr(self.vit, norm_name)
                for p in norm_layer.parameters():
                    p.requires_grad_(True)
                break
        
        # Optionally unfreeze all LayerNorm layers for better adaptation
        if unfreeze_norm:
            for name, module in self.vit.named_modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad_(True)
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.vit.parameters())
        
        print(f"[Backbone] Unfreezing last {unfreeze_blocks} of {self.num_blocks} blocks")
        print(f"[Backbone] Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    def forward(self, x):
        """
        Forward pass through the backbone.
        
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            Spatial feature maps (B, D, H/patch_size, W/patch_size)
        """
        out = self.vit(pixel_values=x)
        hs = out.last_hidden_state  # (B, N_extra + N_patches, D)

        # Get patch grid shape
        gh = x.shape[-2] // PATCH_SIZE
        gw = x.shape[-1] // PATCH_SIZE
        expected_tokens = gh * gw

        # Keep only spatial tokens (remove CLS token and any register tokens)
        hs = hs[:, -expected_tokens:, :]  # (B, num_patches, D)

        # Rearrange to (B, D, H, W)
        feat = rearrange(hs, "b (h w) d -> b d h w", h=gh, w=gw)
        return feat  # (B, D, H/16, W/16)
