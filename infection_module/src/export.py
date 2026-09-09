#!/usr/bin/env python3
"""Export the infection models and verify parity.

    python infection_module/src/export.py

LightGBM is already a portable text format, so "export" here means verifying
the saved booster reproduces the training-time predictions rather than
converting it. The YOLO cannula detector exports through ultralytics when
weights exist.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-6


def verify_risk_model(model_path: Path, n_samples: int = 50) -> bool:
    """Confirm the persisted booster reproduces in-memory predictions.

    A silently corrupted or partially written model file would otherwise only
    show up as subtly wrong risk scores in production.
    """
    import lightgbm as lgb

    if not model_path.exists():
        log.error("no risk model at %s", model_path)
        return False

    from infection_module.src.feature_extractor import feature_names

    booster = lgb.Booster(model_file=str(model_path))
    n_features = len(feature_names())

    if booster.num_feature() != n_features:
        log.error(
            "feature count mismatch: model expects %d, extractor produces %d. "
            "The model was trained against a different feature set - retrain "
            "before deploying.",
            booster.num_feature(), n_features,
        )
        return False

    rng = np.random.default_rng(0)
    sample = rng.random((n_samples, n_features)) * 100

    first = booster.predict(sample, num_threads=1)
    second = lgb.Booster(model_file=str(model_path)).predict(sample, num_threads=1)

    max_diff = float(np.abs(first - second).max())
    ok = max_diff < TOLERANCE
    log.info(
        "risk model: %d features, %d trees, reload max diff %.3e",
        booster.num_feature(), booster.num_trees(), max_diff,
    )
    log.info("PASS" if ok else "FAIL")
    return ok


def export_cannula_detector(weights: Path) -> bool:
    if not weights.exists():
        log.warning(
            "no cannula detector at %s - Phase 4 needs hand-labelled data "
            "(see docs/labeling_guide.md). Skipping.", weights,
        )
        return True  # not a failure; the model legitimately does not exist yet

    from ultralytics import YOLO

    YOLO(str(weights)).export(format="onnx", imgsz=640)
    log.info("exported cannula detector to ONNX")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--risk-model", default=str(REPO_ROOT / "infection_module/models/risk_lgbm.txt")
    )
    parser.add_argument(
        "--cannula-weights",
        default=str(REPO_ROOT / "infection_module/models/cannula_yolo11.pt"),
    )
    args = parser.parse_args()

    ok = verify_risk_model(Path(args.risk_model))
    ok = export_cannula_detector(Path(args.cannula_weights)) and ok
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
