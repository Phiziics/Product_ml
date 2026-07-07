"""Score customers with the final Logistic Regression T-Learner.

Workflow
--------
1. Load saved treatment and control pipelines.
2. Validate required pre-treatment features.
3. Predict conversion probability under treatment.
4. Predict conversion probability under control.
5. Calculate predicted uplift.
6. Rank customers.
7. Mark the top targeting fraction for treatment.
8. Save a business-ready CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_TARGET_FRACTION = 0.20


def load_metadata(
    metadata_path: Path,
) -> dict:
    """Load model metadata saved by train.py."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    return metadata


def validate_scoring_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Check that scoring data contains all required features."""

    missing_columns = sorted(
        set(feature_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Scoring data is missing required columns: "
            + ", ".join(missing_columns)
        )


def score_customers(
    input_path: Path,
    model_dir: Path,
    output_path: Path,
    target_fraction: float,
) -> pd.DataFrame:
    """Score, rank, and export customer recommendations."""

    if not 0 < target_fraction <= 1:
        raise ValueError(
            "target_fraction must be greater than 0 "
            "and less than or equal to 1."
        )

    treatment_model_path = (
        model_dir
        / "treatment_model.joblib"
    )

    control_model_path = (
        model_dir
        / "control_model.joblib"
    )

    metadata_path = (
        model_dir
        / "model_metadata.json"
    )

    required_artifacts = [
        treatment_model_path,
        control_model_path,
        metadata_path,
    ]

    for path in required_artifacts:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: "
                f"{path}"
            )

    treatment_model = joblib.load(
        treatment_model_path
    )

    control_model = joblib.load(
        control_model_path
    )

    metadata = load_metadata(
        metadata_path
    )

    feature_columns = (
        metadata[
            "feature_columns"
        ]
    )

    df = pd.read_csv(
        input_path
    )

    if "zip_code" in df.columns:
        df["zip_code"] = (
            df["zip_code"]
            .replace(
                {
                    "Surburban": "Suburban"
                }
            )
        )

    validate_scoring_columns(
        df=df,
        feature_columns=feature_columns,
    )

    X = df[
        feature_columns
    ]

    p_treatment = (
        treatment_model
        .predict_proba(X)[:, 1]
    )

    p_control = (
        control_model
        .predict_proba(X)[:, 1]
    )

    predicted_uplift = (
        p_treatment
        - p_control
    )

    results = df.copy()

    results.insert(
        0,
        "customer_row_id",
        np.arange(
            len(results)
        ),
    )

    results[
        "p_treatment"
    ] = p_treatment

    results[
        "p_control"
    ] = p_control

    results[
        "predicted_uplift"
    ] = predicted_uplift

    results[
        "uplift_rank"
    ] = (
        results[
            "predicted_uplift"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    n_target = max(
        1,
        int(
            np.floor(
                len(results)
                * target_fraction
            )
        ),
    )

    results[
        "recommendation"
    ] = np.where(
        results[
            "uplift_rank"
        ] <= n_target,
        "Target",
        "Do Not Target",
    )

    results = (
        results
        .sort_values(
            "uplift_rank"
        )
        .reset_index(
            drop=True
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    target_count = (
        results["recommendation"]
        .eq("Target")
        .sum()
    )

    print(
        "Scoring complete."
    )

    print(
        f"Customers scored: "
        f"{len(results):,}"
    )

    print(
        f"Customers recommended for targeting: "
        f"{target_count:,}"
    )

    print(
        f"Output saved to: "
        f"{output_path}"
    )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Score customers with the final "
            "Logistic Regression T-Learner."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(
            "data/processed/"
            "hillstrom_cleaned.csv"
        ),
        help=(
            "CSV containing customers to score."
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "models"
        ),
        help=(
            "Directory containing saved "
            "model artifacts."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/"
            "customer_targeting_recommendations.csv"
        ),
        help=(
            "Path for the scored recommendation CSV."
        ),
    )

    parser.add_argument(
        "--target-fraction",
        type=float,
        default=DEFAULT_TARGET_FRACTION,
        help=(
            "Fraction of highest-uplift customers "
            "to recommend for treatment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run batch customer scoring."""

    args = parse_args()

    score_customers(
        input_path=args.input_path,
        model_dir=args.model_dir,
        output_path=args.output_path,
        target_fraction=args.target_fraction,
    )


if __name__ == "__main__":
    main()