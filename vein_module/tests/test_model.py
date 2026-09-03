"""Tests for the Attention U-Net."""

import torch

from vein_module.src.model import AttentionGate, AttentionUNet, count_parameters


class TestAttentionUNet:
    def test_output_matches_input_resolution(self):
        model = AttentionUNet(base_channels=8)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 1, 64, 64)

    def test_stays_under_the_deployability_budget(self):
        """15M cap - this has to run on a bedside device."""
        assert count_parameters(AttentionUNet()) < 15_000_000

    def test_forward_returns_logits_not_probabilities(self):
        """BCEWithLogitsLoss needs raw logits; a sigmoid here would double-apply it."""
        torch.manual_seed(0)
        out = AttentionUNet(base_channels=8)(torch.randn(1, 3, 64, 64) * 5)
        assert out.min() < 0.0 or out.max() > 1.0

    def test_predict_returns_probabilities(self):
        out = AttentionUNet(base_channels=8).predict(torch.randn(1, 3, 64, 64))
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_gradients_reach_the_encoder(self):
        """Guards against a detached skip connection silently freezing the encoder."""
        model = AttentionUNet(base_channels=8)
        model(torch.randn(1, 3, 64, 64)).mean().backward()
        grad = model.enc1.block[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0

    def test_accepts_single_channel_input_when_configured(self):
        """NIR deployment path."""
        out = AttentionUNet(in_channels=1, base_channels=8)(torch.randn(1, 1, 64, 64))
        assert out.shape == (1, 1, 64, 64)


class TestAttentionGate:
    def test_preserves_skip_shape(self):
        gate = AttentionGate(gate_channels=16, skip_channels=16, inter_channels=8)
        skip = torch.randn(1, 16, 32, 32)
        assert gate(skip, torch.randn(1, 16, 32, 32)).shape == skip.shape

    def test_resamples_a_coarser_gating_signal(self):
        """The gate arrives one stage down; it must be aligned, not rejected."""
        gate = AttentionGate(gate_channels=16, skip_channels=16, inter_channels=8)
        skip = torch.randn(1, 16, 32, 32)
        assert gate(skip, torch.randn(1, 16, 16, 16)).shape == skip.shape

    def test_attenuates_rather_than_deletes(self):
        """Soft gating: a faint vein should be dimmed, never zeroed."""
        torch.manual_seed(0)
        gate = AttentionGate(16, 16, 8)
        skip = torch.ones(1, 16, 32, 32)
        out = gate(skip, torch.randn(1, 16, 32, 32))
        assert (out.abs() > 0).all()
        assert (out.abs() <= skip.abs() + 1e-6).all()
