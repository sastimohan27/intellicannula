"""LightGBM risk model over colour + texture features.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS MODULE
-----------------------------------------------------
The target is a PROXY. No public dataset labels a wound as infected, so
`is_infected` is built as:

    1  AZH wound tissue   {venous, diabetic, pressure, surgical}
    0  AZH normal skin    {normal}

What the model learns is therefore **wound tissue vs intact skin**, not
**infected vs uninfected**. Those are different questions. This is a defensible
first target for IntelliCannula - an IV site developing phlebitis moves from
intact skin toward inflamed tissue, which is the same direction - but the
reported precision/recall are proxy-task numbers and must always be labelled as
such. A real infection classifier needs labelled infected IV sites, which means
an IRB-approved prospective collection.

Second caveat: AZH images are chronic and surgical wounds photographed at close
range, not cannula sites. The transfer from "open wound" to "early phlebitis"
is an assumption, not a measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from infection_module.src.data_loader import WOUND_CLASSES, wound_group_id
from infection_module.src.feature_extractor import extract_all_features, feature_names

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# Small values chosen deliberately: ~800 training images and 30 features. A
# deeper forest memorises the AZH capture setup rather than learning erythema.
LGBM_PARAMS = {
    "objective": "binary",
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


def _center_roi(image: np.ndarray, frac: float = 0.5) -> list[float]:
    """Central ROI box for a pre-cropped wound image.

    AZH images are already tightly cropped on the wound, so there is no cannula
    hub to anchor to. The central region stands in for the ROI and the
    surrounding band supplies the healthy-skin reference - the same geometry
    the deployed pipeline produces via `crop_roi`, so the feature distributions
    stay comparable.
    """
    height, width = image.shape[:2]
    half = frac / 2.0
    return [
        width * (0.5 - half), height * (0.5 - half),
        width * (0.5 + half), height * (0.5 + half),
    ]


def build_training_table(processed_dir: Path, split: str = "train") -> pd.DataFrame:
    """Run feature extraction over a processed split.

    Returns a DataFrame of features plus `is_infected` (proxy), `wound_type`,
    and `source_file`. Images too small to yield a healthy-skin patch are
    skipped and counted, not silently dropped.
    """
    root = Path(processed_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found - run `python infection_module/src/data_loader.py --prepare`"
        )

    rows, skipped = [], 0
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = 1 if class_dir.name in WOUND_CLASSES else 0
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                skipped += 1
                continue
            try:
                features = extract_all_features(image, _center_roi(image))
            except ValueError:
                skipped += 1
                continue
            rows.append(
                {
                    **features,
                    "is_infected": label,
                    "wound_type": class_dir.name,
                    "source_file": path.name,
                    # Carried through so CV can hold every crop of one wound
                    # out together - see cross_validate.
                    "wound_group": wound_group_id(class_dir.name, path.name),
                }
            )

    if skipped:
        log.warning("skipped %d images (unreadable or too small for a patch)", skipped)
    table = pd.DataFrame(rows)
    log.info(
        "%s: %d rows, %d positive (%.1f%%)",
        split, len(table), int(table["is_infected"].sum()),
        100 * table["is_infected"].mean(),
    )
    return table


def cross_validate(table: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified, **group-aware** 5-fold CV.

    Grouping is not optional here. AZH ships several ROI crops of one wound as
    `<id>_0.jpg`, `<id>_1.jpg`; an ungrouped StratifiedKFold puts crops of the
    same wound in both the training and validation folds. Measured directly:
    ungrouped CV reported F1 0.990 / AUC 0.995, which was memorisation, not
    generalisation.

    Reports **average precision** alongside AUC. With a ~7:1 class imbalance,
    ROC-AUC flatters a model - average precision reflects what a clinician
    experiences, being dominated by how many flagged sites are truly abnormal.
    """
    columns = feature_names()
    X = table[columns].to_numpy()
    y = table["is_infected"].to_numpy()
    groups = table["wound_group"].to_numpy()

    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {k: [] for k in ("precision", "recall", "f1", "auc", "avg_precision")}

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y, groups), start=1):
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        assert not overlap, f"wound leaked across fold {fold}: {sorted(overlap)[:3]}"

        model = lgb.LGBMClassifier(**LGBM_PARAMS, class_weight="balanced", random_state=seed)
        model.fit(X[train_idx], y[train_idx])

        probs = model.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        truth = y[val_idx]

        scores["precision"].append(precision_score(truth, preds, zero_division=0))
        scores["recall"].append(recall_score(truth, preds, zero_division=0))
        scores["f1"].append(f1_score(truth, preds, zero_division=0))
        scores["auc"].append(roc_auc_score(truth, probs))
        scores["avg_precision"].append(average_precision_score(truth, probs))
        log.info("fold %d: F1 %.3f AUC %.3f", fold, scores["f1"][-1], scores["auc"][-1])

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()}


# Feature groups for the ablation. The question this answers: is the model
# using the erythema signal that early phlebitis actually presents as, or is it
# using wound "messiness" that intact skin will never show?
FEATURE_GROUPS = {
    "erythema_only": [
        "color_lab_distance", "color_a_delta", "color_b_delta", "color_l_delta",
        "color_a_mean", "color_b_mean", "color_hue_mean", "color_sat_mean",
    ],
    "heterogeneity_only": [
        "color_a_std", "color_b_std", "color_hue_std", "color_sat_std",
    ],
    "texture_only": [n for n in feature_names() if n.startswith("texture_")],
    "all": feature_names(),
}


def ablation(table: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> dict:
    """Cross-validate each feature group separately.

    Motivation. The full model scores F1 ~0.99, but its top features by gain are
    all *standard deviations* of colour channels, not the erythema distances the
    module was designed around. That points at the model separating classes by
    colour heterogeneity - an open wound contains red granulation, yellow slough
    and black eschar at once, while intact skin is uniform - rather than by
    redness.

    That distinction decides whether this model is useful. Early phlebitis
    presents as *uniformly slightly red intact skin*: high erythema, low
    heterogeneity. A model leaning on heterogeneity would miss it entirely while
    scoring beautifully on this proxy task.
    """
    y = table["is_infected"].to_numpy()
    groups = table["wound_group"].to_numpy()
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = {}
    for group_name, columns in FEATURE_GROUPS.items():
        X = table[columns].to_numpy()
        f1s, aps = [], []
        for train_idx, val_idx in folds.split(X, y, groups):
            model = lgb.LGBMClassifier(
                **LGBM_PARAMS, class_weight="balanced", random_state=seed
            )
            model.fit(X[train_idx], y[train_idx])
            probs = model.predict_proba(X[val_idx])[:, 1]
            f1s.append(f1_score(y[val_idx], (probs >= 0.5).astype(int), zero_division=0))
            aps.append(average_precision_score(y[val_idx], probs))
        results[group_name] = {
            "n_features": len(columns),
            "f1": float(np.mean(f1s)),
            "avg_precision": float(np.mean(aps)),
        }
        log.info("ablation %-20s F1 %.3f  AP %.3f", group_name, results[group_name]["f1"],
                 results[group_name]["avg_precision"])
    return results


def train_final(table: pd.DataFrame, model_path: Path, seed: int = 42) -> lgb.LGBMClassifier:
    """Fit on the full table and persist.

    `class_weight="balanced"` because the proxy target is ~7:1 imbalanced;
    without it the model maximises accuracy by rarely predicting the negative
    class, which is precisely the wrong failure direction for a screening tool.
    """
    columns = feature_names()
    model = lgb.LGBMClassifier(**LGBM_PARAMS, class_weight="balanced", random_state=seed)
    model.fit(table[columns].to_numpy(), table["is_infected"].to_numpy())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    (model_path.parent / "risk_model_features.json").write_text(json.dumps(columns, indent=2))
    log.info("model -> %s", model_path)
    return model


_booster_cache: dict[str, lgb.Booster] = {}


def predict_risk(features: dict[str, float], model_path: Path | None = None) -> float:
    """Risk probability in [0, 1] for one feature dict.

    Features are ordered by `feature_names()` rather than by dict order, so a
    caller building the dict differently cannot silently shuffle the vector.
    """
    path = model_path or (REPO_ROOT / "infection_module/models/risk_lgbm.txt")
    key = str(path)
    if key not in _booster_cache:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"no risk model at {path} - train one with "
                "`python infection_module/src/risk_model.py --train`"
            )
        _booster_cache[key] = lgb.Booster(model_file=str(path))

    vector = np.array([[features[name] for name in feature_names()]])
    return float(_booster_cache[key].predict(vector)[0])


def _write_report(cv: dict, table: pd.DataFrame, importances, path: Path,
                  groups: dict | None = None) -> None:
    lines = [
        "# Infection Risk Model — Report",
        "",
        "Generated by `infection_module/src/risk_model.py`. Do not edit by hand.",
        "",
        "## ⚠ Proxy labels — read first",
        "",
        "No public dataset labels a wound as infected. The target is:",
        "",
        "| Label | Source |",
        "|---|---|",
        "| `1` | AZH wound tissue — venous, diabetic, pressure, surgical |",
        "| `0` | AZH normal skin |",
        "",
        "So this model discriminates **wound tissue from intact skin**, not "
        "**infected from uninfected**. Every number below is proxy-task "
        "performance. A real infection classifier requires labelled infected IV "
        "sites from an IRB-approved prospective collection.",
        "",
        "AZH images are also chronic and surgical wounds at close range, not "
        "cannula sites — the transfer to early phlebitis is an assumption, not a "
        "measurement.",
        "",
        "## Data",
        "",
        f"- Rows: **{len(table)}**",
        f"- Positive (wound): **{int(table['is_infected'].sum())}** "
        f"({100 * table['is_infected'].mean():.1f}%)",
        f"- Features: **{len(feature_names())}** (colour + texture)",
        "",
        "## 5-fold cross-validated performance",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for name in ("precision", "recall", "f1", "auc", "avg_precision"):
        lines.append(f"| {name} | {cv[name]['mean']:.4f} | ±{cv[name]['std']:.4f} |")

    lines += [
        "",
        "Average precision is reported alongside AUC deliberately. At roughly "
        "7:1 imbalance, ROC-AUC flatters the model; average precision better "
        "reflects what a clinician experiences, being dominated by how many "
        "flagged sites are genuinely abnormal.",
        "",
        "## Top features by gain",
        "",
        "| Feature | Gain |",
        "|---|---:|",
    ]
    lines += [f"| `{name}` | {value:.1f} |" for name, value in importances]
    if groups:
        lines += [
            "",
            "## Feature-group ablation — what is the model actually using?",
            "",
            "| Feature group | n | F1 | Avg precision |",
            "|---|---:|---:|---:|",
        ]
        for name, stats in groups.items():
            lines.append(
                f"| {name} | {stats['n_features']} | {stats['f1']:.4f} | "
                f"{stats['avg_precision']:.4f} |"
            )
        lines += [
            "",
            "**This table matters more than the headline score.** The top "
            "features by gain are all standard deviations of colour channels, "
            "not the erythema distances this module was built around. That "
            "means the model separates wound from skin by *colour "
            "heterogeneity* — an open wound shows red granulation, yellow "
            "slough and black eschar at once, while intact skin is uniform.",
            "",
            "Early phlebitis at an IV site looks like the opposite: **uniformly "
            "slightly red intact skin** — high erythema, low heterogeneity. A "
            "model leaning on heterogeneity would miss exactly the case "
            "IntelliCannula exists to catch, while still scoring near-perfectly "
            "on this proxy task. Compare the `erythema_only` row against "
            "`heterogeneity_only` to see how much of the score survives when "
            "the messiness cue is removed.",
        ]

    lines += [
        "",
        "## Limitations",
        "",
        "1. **Proxy target** — wound vs skin, not infected vs uninfected.",
        "2. **Grouped by wound, not by patient** — CV holds every ROI crop of "
        "one wound out together, but AZH publishes no patient IDs, so two "
        "wounds on the same patient may still straddle folds.",
        "3. **Single source** — all training data is one centre, two cameras. "
        "The Medetec external set is the transfer check.",
        "4. **No longitudinal data** — the temporal tracker's alert thresholds "
        "are calibrated only against synthetic progressions.",
        "",
    ]
    path.write_text("\n".join(lines))
    log.info("report -> %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="build features, CV, and fit")
    parser.add_argument(
        "--processed-dir", default=str(REPO_ROOT / "data/processed/infection")
    )
    parser.add_argument(
        "--model-path", default=str(REPO_ROOT / "infection_module/models/risk_lgbm.txt")
    )
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "docs/infection_risk_model_report.md")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train:
        parser.error("nothing to do - pass --train")

    table = build_training_table(Path(args.processed_dir), "train")
    cv = cross_validate(table, seed=args.seed)
    groups = ablation(table, seed=args.seed)
    model = train_final(table, Path(args.model_path), seed=args.seed)

    importances = sorted(
        zip(feature_names(), model.booster_.feature_importance("gain")),
        key=lambda kv: -kv[1],
    )[:12]

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    _write_report(cv, table, importances, Path(args.report), groups)

    for name, stats in cv.items():
        print(f"{name:>15}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
