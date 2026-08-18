"""
NFD Shallow U-Net + FiLM conditioning
Xue et al., CoRL 2023  —  extended with Feature-wise Linear Modulation.

FiLM conditions every encoder and decoder conv block on a vector of
object/material properties (e.g. friction coefficient, object size, pusher width).

    FiLM(x | z) = γ(z) ⊙ x + β(z)

where γ, β ∈ R^C are produced by a small MLP from the property vector z,
and applied channel-wise after each conv+ReLU block.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# FiLM generator  (one per block — each block has its own γ/β head)
# ─────────────────────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Maps a property vector z ∈ R^{cond_dim} to per-channel scale γ and shift β.

    Args:
        cond_dim : dimensionality of the conditioning vector z
        num_ch   : number of feature-map channels C to modulate
        hidden   : hidden size of the 1-layer MLP
    """
    def __init__(self, cond_dim: int, num_ch: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_ch * 2),   # outputs [γ | β]
        )
        # Init γ ≈ 1, β ≈ 0  so FiLM is identity at the start of training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias[:num_ch],  1.0)   # γ = 1
        nn.init.constant_(self.mlp[-1].bias[num_ch:],  0.0)   # β = 0

    def forward(self, z: torch.Tensor):
        """
        Args:
            z : (B, cond_dim)
        Returns:
            gamma : (B, C, 1, 1)
            beta  : (B, C, 1, 1)
        """
        out   = self.mlp(z)                           # (B, 2C)
        C     = out.shape[1] // 2
        gamma = out[:, :C].unsqueeze(-1).unsqueeze(-1)
        beta  = out[:, C:].unsqueeze(-1).unsqueeze(-1)
        return gamma, beta


# ─────────────────────────────────────────────────────────────────────────────
# Conv block  +  FiLM modulation
# ─────────────────────────────────────────────────────────────────────────────

class FiLMConvBlock(nn.Module):
    """
    3×3 Conv → ReLU → FiLM(γ, β)

    FiLM is applied after ReLU so the modulation acts on the activated features.
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, hidden: int = 64):
        super().__init__()
        self.conv      = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv2      = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu      = nn.ReLU(inplace=True)
        self.film_gen  = FiLMGenerator(cond_dim, out_ch, hidden)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_ch, H, W)  feature map
            z : (B, cond_dim)     conditioning vector
        Returns:
            out : (B, out_ch, H, W)  modulated feature map
        """
        x             = self.relu(self.conv(x))
        x             = self.relu(self.conv2(x))
        gamma, beta   = self.film_gen(z)
        return gamma * x + beta


# ─────────────────────────────────────────────────────────────────────────────
# NFD U-Net  with FiLM at every encoder + decoder block
# ─────────────────────────────────────────────────────────────────────────────

class NFDUNetFiLM(nn.Module):
    """
    NFD shallow U-Net dynamics model conditioned on material properties via FiLM.

    Args:
        in_channels : input channels  (default 3: state + 2 rendered actions)
        out_channels: output channels (default 1: predicted next state)
        cond_dim    : dimensionality of the property/conditioning vector z
        base        : base channel width (default 4, matching Figure 7)
        film_hidden : hidden size of each FiLM MLP (default 64)
    """

    def __init__(
        self,
        in_channels:  int = 2,
        out_channels: int = 1,
        cond_dim:     int = 3,    # e.g. [friction, obj_size, pusher_width, density]
        film_hidden:  int = 64,
        residual_channel: int = 0,
    ):
        super().__init__()
        self.residual_channel = residual_channel

        kw = dict(cond_dim=cond_dim, hidden=film_hidden)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = FiLMConvBlock(in_channels, 4,    **kw)
        self.enc2 = FiLMConvBlock(4,           8,  **kw)
        self.enc3 = FiLMConvBlock(8,         16,  **kw)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = FiLMConvBlock(16, 32, **kw)
        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = FiLMConvBlock(16 + 32, 16, **kw)  

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = FiLMConvBlock(8 + 16, 8,   **kw)   

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = FiLMConvBlock(4 + 8,      4, **kw) 

        # ── Output head (plain 1×1 conv, no FiLM) ───────────────────────────
        self.head = nn.Conv2d(4, out_channels, kernel_size=1)

    def forward(
        self,
        x:  torch.Tensor,
        props:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x      : (B, in_channels, H, W)  input tensor
            props  : (B, cond_dim)        material/object property vector z
                                          e.g. [friction, obj_size,
                                                pusher_width, density]
        Returns:
            pred   : (B, 1, H, W)        predicted next state ŝ_{t+1}
        """

        # Encoder
        s1 = self.enc1(x,             props) 
        s2 = self.enc2(self.pool(s1), props) 
        s3 = self.enc3(self.pool(s2), props) 

        # Bottleneck
        b_feat = self.bottleneck(self.pool(s3), props)  # (B, b*4, H/8, W/8)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b_feat), s3], dim=1), props)
        d2 = self.dec2(torch.cat([self.up2(d3),     s2], dim=1), props)
        d1 = self.dec1(torch.cat([self.up1(d2),     s1], dim=1), props)

        residual = self.head(d1)
        current_state = x[:, self.residual_channel:self.residual_channel + 1, :, :]
        return torch.sigmoid(current_state + residual)     # (B, 1, H, W)


class NFDUNetFiLMShallow(nn.Module):
    """
    Less-deep FiLM U-Net variant.

    Keeps the same input/output resolution and residual current-state skip, but
    removes the deepest encoder/decoder stage. This gives two pooling levels
    instead of three.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        cond_dim: int = 3,
        film_hidden: int = 64,
        residual_channel: int = 0,
    ):
        super().__init__()
        self.residual_channel = residual_channel
        kw = dict(cond_dim=cond_dim, hidden=film_hidden)

        self.enc1 = FiLMConvBlock(in_channels, 4, **kw)
        self.enc2 = FiLMConvBlock(4, 8, **kw)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = FiLMConvBlock(8, 16, **kw)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = FiLMConvBlock(8 + 16, 8, **kw)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = FiLMConvBlock(4 + 8, 4, **kw)

        self.head = nn.Conv2d(4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x, props)
        s2 = self.enc2(self.pool(s1), props)

        b_feat = self.bottleneck(self.pool(s2), props)

        d2 = self.dec2(torch.cat([self.up2(b_feat), s2], dim=1), props)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1), props)

        residual = self.head(d1)
        current_state = x[:, self.residual_channel:self.residual_channel + 1, :, :]
        return torch.clamp(current_state + residual, 0.0, 1.0)
