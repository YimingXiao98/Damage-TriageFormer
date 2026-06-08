import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegHead(nn.Module):
    """Segmentation head for building detection."""
    
    def __init__(self, in_ch, mid=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, f, out_hw):
        x = F.relu(self.bn1(self.conv1(f)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.out(x)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return x


class ResidualMLP(nn.Module):
    """MLP block with residual connection for better gradient flow."""
    
    def __init__(self, dim, expansion=2, dropout=0.3):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class SpatialAttentionPooling(nn.Module):
    """
    Attention-weighted spatial pooling for instance features.
    Instead of simple masked averaging, learns to focus on discriminative regions.
    """
    
    def __init__(self, in_ch):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, 1),
        )
    
    def forward(self, features, mask):
        """
        Args:
            features: (C, H, W) single image features
            mask: (N, H, W) instance masks
        Returns:
            pooled: (N, C) pooled features per instance
        """
        C, _H, _W = features.shape
        N = mask.shape[0]

        if N == 0:
            return torch.zeros(0, C, device=features.device)

        # Compute attention weights
        attn = self.attention(features.unsqueeze(0))        # (1, 1, H, W)
        attn = torch.sigmoid(attn).squeeze(0)               # (1, H, W)

        # Vectorized weighted-average pool over all N instances in one go
        # (was a Python for-loop — one CUDA kernel per building, ~5-10x slower).
        combined = mask * attn                              # (N, H, W)
        weight_sum = combined.sum(dim=(1, 2)) + 1e-6        # (N,)
        # features: (1, C, H, W); combined: (N, 1, H, W)   -> broadcast (N, C, H, W)
        weighted = features.unsqueeze(0) * combined.unsqueeze(1)
        pooled = weighted.sum(dim=(2, 3)) / weight_sum.unsqueeze(1)
        return pooled                                       # (N, C)


class MultiScaleROIPooling(nn.Module):
    """
    Multi-scale ROI pooling to capture both local details and global context.
    Uses multiple dilation rates to aggregate features at different scales.
    """
    
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        
        # Multi-scale feature extraction
        self.conv1x1 = nn.Conv2d(in_ch, in_ch // 4, 1)
        self.conv3x3 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=1)
        self.conv3x3_d2 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=2, dilation=2)
        self.conv3x3_d4 = nn.Conv2d(in_ch, in_ch // 4, 3, padding=4, dilation=4)
        
        self.fuse = nn.Conv2d(in_ch, in_ch, 1)
        self.norm = nn.BatchNorm2d(in_ch)
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W)
        Returns:
            multi_scale_features: (B, C, H, W)
        """
        f1 = self.conv1x1(features)
        f2 = self.conv3x3(features)
        f3 = self.conv3x3_d2(features)
        f4 = self.conv3x3_d4(features)
        
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.fuse(fused)
        fused = self.norm(fused)
        
        return F.relu(fused)


class SeverityHead(nn.Module):
    """
    Enhanced severity classification head with:
    - Multi-scale feature extraction
    - Attention-weighted ROI pooling
    - Deep residual MLP classifier
    - Proper normalization for stability
    """
    
    def __init__(self, in_ch, num_classes, hidden_dim=512, num_layers=3, dropout=0.3, 
                 use_attention=True, use_multiscale=True):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        
        # Multi-scale feature extraction
        if use_multiscale:
            self.multiscale = MultiScaleROIPooling(in_ch)
        
        # Attention pooling
        if use_attention:
            self.attn_pool = SpatialAttentionPooling(in_ch)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual MLP blocks
        self.res_blocks = nn.ModuleList([
            ResidualMLP(hidden_dim, expansion=2, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _pool_instance_features(self, features, mask):
        """
        Pool features for a single instance using mask.
        
        Args:
            features: (C, h, w) feature map
            mask: (h, w) instance mask
        Returns:
            pooled: (C,) pooled features
        """
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
            List of (N_i, num_classes) logits per image
        """
        B, C, H, W = f.shape
        
        # Apply multi-scale feature extraction
        if self.use_multiscale:
            f = self.multiscale(f)
        
        outs = []
        for b in range(B):
            masks_b = inst_masks[b]  # (N, H_orig, W_orig)
            
            if masks_b.numel() == 0:
                outs.append(torch.zeros(0, self.num_classes, device=f.device))
                continue
            
            # Resize masks to feature map size
            N = masks_b.shape[0]
            masks_ds = F.interpolate(
                masks_b.unsqueeze(1).float(),  # (N, 1, H_orig, W_orig)
                size=(H, W),
                mode="nearest"
            ).squeeze(1)  # (N, H, W)
            
            # Pool features per instance
            if self.use_attention:
                # Use attention-weighted pooling
                feats = self.attn_pool(f[b], masks_ds)  # (N, C)
            else:
                # Simple masked average pooling
                feats = []
                for i in range(N):
                    pooled = self._pool_instance_features(f[b], masks_ds[i])
                    feats.append(pooled)
                feats = torch.stack(feats, dim=0) if feats else torch.zeros(0, C, device=f.device)
            
            if feats.shape[0] == 0:
                outs.append(torch.zeros(0, self.num_classes, device=f.device))
                continue
            
            # Project and process through residual blocks
            x = self.input_proj(feats)
            for block in self.res_blocks:
                x = block(x)
            
            # Classify
            logits = self.classifier(x)
            outs.append(logits)
        
        return outs


class SeverityHeadSimple(nn.Module):
    """
    Simpler but still improved severity head.
    Use this if the full SeverityHead is too slow or overfits.
    """
    
    def __init__(self, in_ch, num_classes, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, f, inst_masks):
        B, C, h, w = f.shape
        outs = []
        
        for b in range(B):
            if inst_masks[b].numel() == 0:
                outs.append(torch.zeros(0, self.num_classes, device=f.device))
                continue
            
            m = inst_masks[b].unsqueeze(1)  # (N, 1, H, W)
            m_ds = F.interpolate(m, size=(h, w), mode="nearest")  # (N, 1, h, w)
            wsum = m_ds.sum(dim=(2, 3)).clamp(min=1e-6)  # (N, 1)
            feat = (m_ds * f[b].unsqueeze(0)).sum(dim=(2, 3)) / wsum  # (N, C)
            outs.append(self.mlp(feat))
        
        return outs
