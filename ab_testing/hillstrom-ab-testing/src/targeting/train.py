"""Train the final Logistic Regression T-Learner.

This script productionizes the model selected in Notebook 03.

Workflow
--------
1. Load the cleaned Hillstrom dataset.
2. Keep Mens E-Mail treatment and No E-Mail control rows.
3. Use pre-treatment customer features only.
4. Fit one treatment-response model.
5. Fit one control-response model.
6. Save both fitted pipelines and training metadata.

Important
---------
Notebook 03 already handled model comparison and validation.

Because Logistic Regression was selected as the final model,
this script refits the final treatment and control pipelines
on all eligible rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TREATMENT_LABEL = "Mens E-Mail"
CONTROL_LABEL = "No E-Mail"

TARGET_COLUMN = "conversion"
GROUP_COLUMN = "segment"

NUMERIC_FEATURES = [
    "recency",
    "history",
]

CATEGORICAL_FEATURES = [
    "history_segment",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


def validate_columns(
    df: pd.DataFrame,
) -> None:
    """Check that all required columns exist."""

    required_columns = set(
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            GROUP_COLUMN,
        ]
    )

    missing_columns = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input data is missing required columns: "
            + ", ".join(missing_columns)
        )


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def build_model_pipeline() -> Pipeline:
    """Create one Logistic Regression response model."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_final_models(
    data_path: Path,
    model_dir: Path,
) -> None:
    """Train and save final treatment and control models."""

    df = pd.read_csv(
        data_path
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

    validate_columns(
        df
    )

    modeling_df = df.loc[
        df[GROUP_COLUMN].isin(
            [
                TREATMENT_LABEL,
                CONTROL_LABEL,
            ]
        )
    ].copy()

    if modeling_df.empty:
        raise ValueError(
            "No eligible treatment/control rows were found."
        )

    treatment_df = modeling_df.loc[
        modeling_df[GROUP_COLUMN]
        == TREATMENT_LABEL
    ].copy()

    control_df = modeling_df.loc[
        modeling_df[GROUP_COLUMN]
        == CONTROL_LABEL
    ].copy()

    if (
        treatment_df.empty
        or control_df.empty
    ):
        raise ValueError(
            "Both treatment and control groups "
            "must contain rows."
        )

    X_treatment = (
        treatment_df[
            FEATURE_COLUMNS
        ]
    )

    y_treatment = (
        treatment_df[
            TARGET_COLUMN
        ]
    )

    X_control = (
        control_df[
            FEATURE_COLUMNS
        ]
    )

    y_control = (
        control_df[
            TARGET_COLUMN
        ]
    )

    treatment_model = (
        build_model_pipeline()
    )

    control_model = (
        build_model_pipeline()
    )

    treatment_model.fit(
        X_treatment,
        y_treatment,
    )

    control_model.fit(
        X_control,
        y_control,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    joblib.dump(
        treatment_model,
        treatment_model_path,
    )

    joblib.dump(
        control_model,
        control_model_path,
    )

    metadata = {
        "model_type": (
            "Logistic Regression T-Learner"
        ),
        "treatment_label": (
            TREATMENT_LABEL
        ),
        "control_label": (
            CONTROL_LABEL
        ),
        "target_column": (
            TARGET_COLUMN
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "treatment_training_rows": int(
            len(treatment_df)
        ),
        "control_training_rows": int(
            len(control_df)
        ),
        "treatment_conversion_rate": float(
            y_treatment.mean()
        ),
        "control_conversion_rate": float(
            y_control.mean()
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Training complete."
    )

    print(
        f"Treatment model saved to: "
        f"{treatment_model_path}"
    )

    print(
        f"Control model saved to: "
        f"{control_model_path}"
    )

    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the final Logistic Regression "
            "T-Learner."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "data/processed/"
            "hillstrom_cleaned.csv"
        ),
        help=(
            "Path to the cleaned Hillstrom CSV."
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "models"
        ),
        help=(
            "Directory where trained models "
            "are saved."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run final model training."""

    args = parse_args()

    train_final_models(
        data_path=args.data_path,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()