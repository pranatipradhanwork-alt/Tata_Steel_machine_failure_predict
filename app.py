from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Machine failure"
THRESHOLD = 0.2480
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

st.set_page_config(page_title="Machine Failure Risk Demo", page_icon="🏭", layout="centered")
st.title("Machine Failure Risk Demo")
st.caption("Predictive maintenance prototype • synthetic training dataset")
st.warning(
    "Educational demonstration only. This model was developed with synthetic data "
    "and is not validated for real equipment or safety-critical decisions."
)
st.markdown(
    "Upload the project's training CSV to fit the Random Forest workflow used in "
    "the notebook. Then enter one machine's readings to get a failure-risk score. "
    "The app uses the notebook's validation-selected probability threshold of 0.248."
)

@st.cache_resource(show_spinner="Training the model from the uploaded data…")
def train_model(csv_bytes: bytes):
    data = pd.read_csv(io.BytesIO(csv_bytes))
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
    return model, len(data), int(y_train.sum())

with st.sidebar:
    st.header("1. Load training data")
    uploaded_file = st.file_uploader(
        "Choose the project's train.csv",
        type=["csv"],
        help="The test.csv file has no labels and cannot train this demo.",
    )
    st.caption(
        "The app needs labeled data to fit the model. The synthetic train.csv is "
        "not included in the public repository."
    )

if uploaded_file is None:
    st.info("Upload train.csv in the sidebar to activate the prediction form.")
    st.stop()

try:
    model, row_count, training_failures = train_model(uploaded_file.getvalue())
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.success(
    f"Model ready: trained on {row_count:,} rows; "
    f"{training_failures:,} failures in its training split."
)
st.header("2. Enter machine readings")
col_a, col_b = st.columns(2)
with col_a:
    machine_type = st.selectbox("Machine type", ["L", "M", "H"])
    air_temp = st.number_input(
        "Air temperature (K)", min_value=280.0, max_value=320.0, value=300.0, step=0.1
    )
    process_temp = st.number_input(
        "Process temperature (K)", min_value=280.0, max_value=330.0, value=310.0, step=0.1
    )
with col_b:
    speed = st.number_input(
        "Rotational speed (rpm)", min_value=0, max_value=5000, value=1500, step=10
    )
    torque = st.number_input(
        "Torque (Nm)", min_value=0.0, max_value=150.0, value=40.0, step=0.1
    )
    tool_wear = st.number_input(
        "Tool wear (min)", min_value=0, max_value=500, value=100, step=1
    )

if st.button("Estimate failure risk", type="primary", use_container_width=True):
    input_row = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature difference [K]": process_temp - air_temp,
    }], columns=FEATURES)
    probability = float(model.predict_proba(input_row)[0, 1])
    prediction = int(probability >= THRESHOLD)

    st.subheader("Prediction")
    st.metric("Estimated failure probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if prediction:
        st.error(f"Above the {THRESHOLD:.1%} demo threshold: consider an engineering review.")
    else:
        st.success(
            f"Below the {THRESHOLD:.1%} demo threshold. "
            "This is not a guarantee of safe operation."
        )

    result = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Estimated failure probability": probability,
        "Predicted Machine failure": prediction,
        "Demo threshold": THRESHOLD,
    }])
    with st.expander("View scored input"):
        st.dataframe(result, use_container_width=True)
    st.download_button(
        "Download this prediction as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="machine_failure_risk_prediction.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Prototype limitations: synthetic data, one stratified holdout split, and no "
    "labeled external test set. Do not use this score as an autonomous shutdown "
    "or maintenance instruction."
)
