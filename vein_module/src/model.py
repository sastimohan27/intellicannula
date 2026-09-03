"""Attention U-Net for vein segmentation (Oktay et al., 2018).

The attention gate is the reason to prefer this over a plain U-Net here. Veins
occupy roughly a few percent of a forearm frame, so encoder skip connections
carry mostly irrelevant skin. The gate lets the coarser decoder stage suppress
those regions before concatenation, which matters more the more imbalanced the
foreground is.

Kept under 15M parameters deliberately - this has to run on a bedside device,
not a training rig.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018, eq. 1-2).

    The gating signal `g` comes from the coarser decoder stage and carries
    "what the network currently believes is vein"; `x` is the higher-resolution
    encoder skip. Both are projected to a shared space, summed, passed through
    ReLU then a 1x1 convolution to a single channel, and squashed with a
    sigmoid to give a per-pixel attention coefficient in [0, 1]. That map
    rescales `x` before it is concatenated into the decoder.

    Note this is soft, not hard, gating: nothing is discarded, low-attention
    regions are attenuated. That matters clinically - a faint vein the network
    is unsure about is dimmed, not deleted.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        theta = self.theta(skip)
        phi = self.phi(gate)

        # The gating signal is one stage coarser, so align it to the skip's
        # resolution. Interpolation rather than a learned upsample keeps the
        # gate cheap - it is applied at every skip connection.
        if phi.shape[-2:] != theta.shape[-2:]:
            phi = F.interpolate(
                phi, size=theta.shape[-2:], mode="bilinear", align_corners=False
            )

        attention = torch.sigmoid(self.norm(self.psi(F.relu(theta + phi))))
        return skip * attention


class AttentionUNet(nn.Module):
    """4-stage encoder/decoder with attention-gated skip connections.

    Parameters
    ----------
    in_channels :
        3 by default, matching `preprocessing.build_model_input`:
        [preprocessed R, preprocessed G, Frangi vesselness].
    base_channels :
        32, doubling per stage (32/64/128/256, bottleneck 512).

    Output is a single-channel **logit** map at input resolution. No sigmoid is
    applied - `BCEWithLogitsLoss` needs raw logits for numerical stability, and
    `predict()` is provided for the places that want probabilities.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32
    ):
        super().__init__()
        c = base_channels
        widths = [c, c * 2, c * 4, c * 8]

        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(widths[3], widths[3] * 2)

        self.up4 = nn.ConvTranspose2d(widths[3] * 2, widths[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(widths[3], widths[3], widths[3] // 2)
        self.dec4 = ConvBlock(widths[3] * 2, widths[3])

        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(widths[2], widths[2], widths[2] // 2)
        self.dec3 = ConvBlock(widths[2] * 2, widths[2])

        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(widths[1], widths[1], widths[1] // 2)
        self.dec2 = ConvBlock(widths[1] * 2, widths[1])

        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(widths[0], widths[0], widths[0] // 2)
        self.dec1 = ConvBlock(widths[0] * 2, widths[0])

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([self.att4(e4, d4), d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(e3, d3), d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(e2, d2), d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.att1(e1, d1), d1], dim=1))

        return self.head(d1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Probability map in [0, 1]. Use this at inference, not `forward`."""
        return torch.sigmoid(self(x))


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count, for the deployability budget."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


if __name__ == "__main__":
    model = AttentionUNet()
    n = count_parameters(model)
    print(f"AttentionUNet parameters: {n:,}")
    print(f"under 15M budget: {n < 15_000_000}")
    dummy = torch.randn(1, 3, 512, 512)
    print(f"input {tuple(dummy.shape)} -> output {tuple(model(dummy).shape)}")
