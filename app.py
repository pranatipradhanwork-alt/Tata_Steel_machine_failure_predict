from __future__ import annotations

import io

import joblib
import pandas as pd
import streamlit as st

from train_model import FEATURES, MODEL_PATH, train_model

THRESHOLD = 0.2480

st.set_page_config(page_title="Machine Failure Risk Demo", page_icon="🏭", layout="centered")
st.title("Machine Failure Risk Demo")
st.caption("Predictive maintenance prototype • synthetic training dataset")
st.warning(
    "Educational demonstration only. This model was developed with synthetic data "
    "and is not validated for real equipment or safety-critical decisions."
)
st.markdown(
    "Enter one machine's readings to get a failure-risk score from the Random "
    "Forest trained in the notebook. The app uses the notebook's "
    "validation-selected probability threshold of 0.248."
)

@st.cache_resource(show_spinner="Loading the trained model…")
def load_bundled_model() -> dict:
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Training the model from the uploaded data…")
def train_uploaded_model(csv_bytes: bytes) -> dict:
    return train_model(pd.read_csv(io.BytesIO(csv_bytes)))

with st.sidebar:
    st.header("Model")
    st.caption(
        "A pre-trained model is included, so no upload is needed. Optionally, "
        "upload a labeled training CSV with the same columns to retrain it."
    )
    uploaded_file = st.file_uploader(
        "Retrain with your own train.csv (optional)",
        type=["csv"],
        help="The test.csv file has no labels and cannot train this demo.",
    )

try:
    if uploaded_file is None:
        bundle = load_bundled_model()
        source = "pre-trained model"
    else:
        bundle = train_uploaded_model(uploaded_file.getvalue())
        source = "model retrained on your upload"
except Exception as exc:
    st.error(str(exc))
    st.stop()

model = bundle["model"]
st.success(
    f"Using the {source}: trained on {bundle['row_count']:,} rows; "
    f"{bundle['training_failures']:,} failures in its training split."
)
st.header("Enter machine readings")
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
