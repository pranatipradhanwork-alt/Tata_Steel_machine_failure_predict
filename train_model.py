"""Train the demo Random Forest and save it for the Streamlit app.

Usage:
    python train_model.py path/to/train.csv

Writes models/machine_failure_rf.joblib. The app loads that file at startup,
so visitors can score readings without uploading the training CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Machine failure"
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
]
FEATURES = ["Type", *NUMERIC_FEATURES]
REQUIRED_COLUMNS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", TARGET,
]
MODEL_PATH = Path(__file__).parent / "models" / "machine_failure_rf.joblib"


def train_model(data: pd.DataFrame) -> dict:
    """Fit the notebook's pipeline and return the model with training stats."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError("Training CSV is missing required columns: " + ", ".join(missing))
    if data[TARGET].nunique(dropna=True) != 2:
        raise ValueError("Training CSV must contain both failure classes (0 and 1).")

    y = data[TARGET].astype(int)
    X = data[[
        "Type", "Air temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]",
    ]].copy()
    X["Temperature difference [K]"] = (
        data["Process temperature [K]"] - data["Air temperature [K]"]
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, ["Type"]),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    return {
        "model": model,
        "row_count": len(data),
        "training_failures": int(y_train.sum()),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python train_model.py path/to/train.csv")
    bundle = train_model(pd.read_csv(sys.argv[1]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1_000_000
    print(f"Saved {MODEL_PATH} ({size_mb:.1f} MB)")
