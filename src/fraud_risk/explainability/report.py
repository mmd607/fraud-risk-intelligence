"""Explanation report for a saved model bundle.

The validation partition is re-derived from the split parameters recorded in the bundle manifest
and checked against the recorded index fingerprint, so explanations use exactly the data the
model was validated on. The test partition is not used. Local explanations are produced for the
highest-scored validation rows and identified by ``row_id`` only; no feature values are written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_risk.data.splitting import (
    SplitConfig,
    SplitStrategy,
    split_dataset,
    split_features_target,
)
from fraud_risk.explainability.explainers import (
    CAUSALITY_CAVEAT,
    ExplainabilityError,
    LocalMethod,
    ModelInfo,
    explain_local,
    global_permutation_importance,
    resolve_method,
)
from fraud_risk.training.artifacts import ModelBundle


@dataclass(frozen=True, slots=True)
class ExplanationReport:
    """Global importance plus a few local explanations for one model."""

    path: Path
    payload: dict[str, Any]


def model_info(bundle: ModelBundle) -> ModelInfo:
    """Model identity from a bundle manifest."""
    manifest = bundle.manifest
    return ModelInfo(manifest.model_id, manifest.model_name, manifest.model_sha256)


def split_config_from_manifest(split: dict[str, Any]) -> SplitConfig:
    """Rebuild the split configuration recorded in a manifest."""
    try:
        return SplitConfig(
            test_size=float(split["test_size"]),
            validation_size=float(split["validation_size"]),
            seed=int(split["seed"]),
            strategy=SplitStrategy(split["strategy"]),
            group_duplicates=bool(split["group_duplicates"]),
        )
    except (KeyError, ValueError) as error:
        msg = "Bundle manifest lacks usable split metadata."
        raise ExplainabilityError(msg) from error


def build_explanation_report(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    reports_dir: Path,
    n_local: int = 5,
    top_k: int = 8,
    n_repeats: int = 3,
    background_size: int = 2_000,
) -> ExplanationReport:
    """Compute global importance on validation and local explanations for top-scored rows."""
    manifest = bundle.manifest
    config = split_config_from_manifest(manifest.split)
    splits = split_dataset(frame, config)
    expected = manifest.split.get("validation", {}).get("index_fingerprint")
    if expected != splits.metadata.validation.index_fingerprint:
        msg = "Validation partition does not match the bundle manifest (different data or split)."
        raise ExplainabilityError(msg)

    info = model_info(bundle)
    x_validation, y_validation = split_features_target(splits.validation)
    global_explanation = global_permutation_importance(
        bundle.model,
        x_validation,
        y_validation,
        model_info=info,
        n_repeats=n_repeats,
        seed=config.seed,
    )
    x_train, _ = split_features_target(splits.train)
    background = x_train.sample(n=min(background_size, len(x_train)), random_state=config.seed)
    scores = pd.Series(bundle.model.predict_proba(x_validation)[:, 1], index=x_validation.index)
    top_rows = scores.sort_values(ascending=False, kind="stable").index[:n_local]
    method = resolve_method(bundle.model, LocalMethod.AUTO)
    local = [
        {
            "row_id": int(row_id),
            "label": int(y_validation.loc[row_id]),
            **explain_local(
                bundle.model,
                x_validation.loc[[row_id]],
                model_info=info,
                method=method,
                background=background,
            ).to_dict(top_k=top_k),
        }
        for row_id in top_rows
    ]
    payload = {
        "model": {"model_id": info.model_id, "model_name": info.model_name},
        "partition": "validation",
        "global": global_explanation.to_dict(),
        "local_method": method.value,
        "local_examples": local,
        "caveat": CAUSALITY_CAVEAT,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"explanations_{info.model_id}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ExplanationReport(path=path, payload=payload)
