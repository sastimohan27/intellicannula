#!/usr/bin/env python3
"""Export the vein segmenter to ONNX and verify parity with PyTorch.

    python vein_module/src/export.py

Verification is the point, not the export. An ONNX file that loads but produces
different numbers is worse than no export at all - it would silently change
clinical recommendations. This exports, runs both backends on the same input,
and fails loudly if they disagree beyond tolerance.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from vein_module.src.model import AttentionUNet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-4


def export_onnx(checkpoint_path: Path, output_path: Path, image_size: int = 512) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    size = config.get("image_size", image_size)

    model = AttentionUNet(base_channels=config.get("base_channels", 32))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        # Batch is dynamic so the same file serves single-image bedside
        # inference and batched offline evaluation. Spatial dims stay fixed:
        # the model is trained at one resolution and varying it silently
        # changes what the receptive field covers.
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    log.info("exported %s", output_path)
    return output_path


def verify(checkpoint_path: Path, onnx_path: Path, tolerance: float = TOLERANCE) -> bool:
    """Compare ONNX Runtime output against PyTorch on identical input."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed - skipping verification")
        log.warning("install with: uv pip install onnxruntime")
        return False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    size = config.get("image_size", 512)

    model = AttentionUNet(base_channels=config.get("base_channels", 32))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = np.random.default_rng(0)
    sample = rng.random((1, 3, size, size), dtype=np.float32)

    with torch.inference_mode():
        torch_out = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["logits"], {"image": sample})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    ok = max_diff < tolerance
    log.info("max absolute difference: %.3e (tolerance %.0e)", max_diff, tolerance)

    if ok:
        log.info("PASS - ONNX output matches PyTorch")
    else:
        log.error(
            "FAIL - outputs diverge by %.3e. Do not deploy this export; a model "
            "that computes different numbers changes clinical recommendations.",
            max_diff,
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "vein_module/models/attention_unet_best.pt"),
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "vein_module/models/attention_unet.onnx")
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint} - train one first")

    onnx_path = export_onnx(checkpoint, Path(args.output))
    if not verify(checkpoint, onnx_path, args.tolerance):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
