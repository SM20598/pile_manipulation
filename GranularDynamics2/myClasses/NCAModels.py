import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# NCA single-step module
# -------------------------
class NCAUpdate(nn.Module):
    """
    One local NCA update step.
    Input: grid (B, C_in, H, W) where C_in includes current state channel(s) + action channels.
    Output: delta to be added to the state channel(s) (same spatial size).
    Implementation: small conv net with 3x3 perception and 1x1 output.
    """
    def __init__(self, in_ch, hidden_ch=8, out_ch=1, dt=0.5):
        super().__init__()
        self.out_ch = out_ch
        self.dt = dt  # step size for residual update

        # perception conv: shareable 3x3 that extracts local neighborhood info
        self.perception = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # compute delta for state channels
        self.delta_layer = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)

        # optional gating for stability
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
            nn.Tanh()  # gate in [-1,1]
        )

    def forward(self, grid):
        """
        grid: (B, C_in, H, W)
        returns: delta (B, state_channels, H, W)
        """
        feat = self.perception(grid)
        raw_delta = self.delta_layer(feat)
        gate = self.gate(feat)  # optional multiplicative gating
        delta = self.dt * raw_delta * (1.0 + gate)  # scale and gate
        return delta


# -------------------------
# Stack K NCA steps
# -------------------------
class NCAStack(nn.Module):
    """
    Apply the same NCAUpdate K times.
    Optionally mask updates (stochastic or deterministic).
    """
    def __init__(self, in_ch, out_ch=1, hidden_ch=64, steps=8, dt=0.5):
        super().__init__()
        self.steps = steps
        self.nca = NCAUpdate(in_ch, hidden_ch=hidden_ch, out_ch=out_ch, dt=dt)

    def forward(self, input, aux_channels=None):
        """
        state: (B, state_ch, H, W)
        action_grid: (B, A, H, W)  -- tool geometry / action encodings
        aux_channels: optional (B, M, H, W) extra channels (swept mask, distance, etc.)
        returns: cumulative_delta (B, state_ch, H, W), state_after_nca (state + cumulative_delta)
        """
        state = input[:,0:1,:,:]  # current state channel
        action_grid = input[:,1:,:,:] if input.shape[1]>1 else None
        B, sc, H, W = state.shape
        cum_delta = torch.zeros_like(state)

        # build static part of grid that doesn't change across steps (actions, aux)
        static_parts = [action_grid] if action_grid is not None else []
        if aux_channels is not None:
            static_parts.append(aux_channels)

        # At each step we provide the current state plus static channels
        cur_state = state
        for t in range(self.steps):
            grid_inputs = [cur_state] + static_parts  # current dynamic state + static fields
            grid = torch.cat(grid_inputs, dim=1)  # (B, C_in, H, W)
            delta = self.nca(grid)  # (B, state_ch, H, W)
            cur_state = cur_state + delta
            cum_delta = cum_delta + delta

        return cum_delta, cur_state


# -------------------------
# Small residual UNet correction
# -------------------------
class SmallUNetRes(nn.Module):
    """
    Small UNet-like residual correction head.
    Input: state_after_nca concatenated with action + aux -> (B, C, H, W)
    Output: delta_correction (B, state_ch, H, W)
    """
    def __init__(self, in_ch, out_ch=1, features=[8, 16, 32]):
        super().__init__()
        # encoder
        self.inc = nn.Sequential(
            nn.Conv2d(in_ch, features[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(features[0], features[1], 3, padding=1),
                                   nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(features[1], features[2], 3, padding=1),
                                   nn.ReLU(inplace=True))
        # bottleneck
        self.bot = nn.Sequential(
            nn.Conv2d(features[2], features[2], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # decoder (upsample + conv)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(features[2], features[1], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(features[1], features[0], 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, state_after, action_grid, aux_channels=None):
        # build input
        parts = [state_after, action_grid] if action_grid is not None else [state_after]
        if aux_channels is not None:
            parts.append(aux_channels)
        x = torch.cat(parts, dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        b = self.bot(x3)
        u2 = self.up2(b)
        u2 = u2 + x2  # skip add
        u1 = self.up1(u2)
        u1 = u1 + x1
        delta_corr = self.outc(u1)
        return delta_corr


# -------------------------
# Full Model: NCA + UNet Correction
# -------------------------
class NCAPlusUNet(nn.Module):
    """
    state: (B, state_ch, H, W)
    action_grid: (B, A, H, W)  -- tool rasterization channels, etc.
    aux_channels: (B, M, H, W) optional
    """
    def __init__(self, in_ch=1, out_ch=1,
                 nca_hidden=8, nca_steps=8, unet_features=[8, 16, 32]):
        super().__init__()
        self.in_ch = in_ch

        self.nca_stack = NCAStack(in_ch=in_ch, out_ch=out_ch,
                                  hidden_ch=nca_hidden, steps=nca_steps, dt=0.5)

        # UNet correction input channels = state_after + action + aux
        self.corr_unet = SmallUNetRes(in_ch=in_ch, out_ch=out_ch, features=unet_features)

    def forward(self, input):
        """
        returns: next_state_pred, dict with internals
        """
        # NCA local iterative updates
        cum_delta, state_after = self.nca_stack(input)
        action = input[:,1:,:,:] 
        # UNet residual correction (one global pass)
        delta_corr = self.corr_unet(state_after, action)

        # Final prediction (residual)
        next_state = state_after + delta_corr  # equivalently: state + cum_delta + delta_corr

        return next_state#, {"delta_nca": cum_delta, "delta_corr": delta_corr}


# -------------------------
# Utilities / quick test
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W = 2, 128, 128
    state_ch = 1
    action_ch = 2
    aux_ch = 1  # e.g., swept_mask channel

    model = NCAPlusUNet(state_ch=state_ch, action_ch=action_ch, aux_ch=aux_ch,
                        nca_hidden=48, nca_steps=12, unet_feats=[32, 64, 128]).to(device)

    state = torch.rand(B, state_ch, H, W, device=device)
    action = torch.rand(B, action_ch, H, W, device=device)
    aux = torch.rand(B, aux_ch, H, W, device=device)

    with torch.no_grad():
        pred_next, info = model(state, action, aux)
    print("pred_next.shape:", pred_next.shape)
    print("delta_nca.shape:", info["delta_nca"].shape)
    print("delta_corr.shape:", info["delta_corr"].shape)
    print("num params:", sum(p.numel() for p in model.parameters()))
