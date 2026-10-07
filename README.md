# TATA Steel Machine Failure Prediction

A predictive maintenance classification project that estimates whether a machine is likely to fail from operating measurements. The goal is to help maintenance teams prioritize inspections while making the tradeoffs caused by rare failures visible.

> **Learning prototype:** The supplied dataset is synthetic. The model is not validated for real equipment and must not be used for safety-critical or autonomous maintenance decisions.

## Project results

- The training data contains 136,429 machine records; 2,148 (1.57%) are labeled as failures.
- Compared a majority-class baseline, class-weighted logistic regression, and a class-weighted Random Forest.
- The Random Forest achieved average precision 0.415 on a stratified validation split, compared with 0.0158 for the majority baseline.
- At a validation-selected threshold of 0.248, recall was 61.9%, precision was 41.6%, and F1 was 49.8%. The threshold trades more false alarms for catching more failures.
- Rotational speed, torque, temperature difference, and tool wear ranked among the strongest permutation-importance signals. Importance indicates predictive usefulness, not causation.

## Workflow

1. Inspect data quality, feature distributions, and failure prevalence.
2. Engineer process-minus-air temperature difference and remove identifiers and failure-mode columns that could leak the outcome.
3. Split with stratification and compare models using average precision and threshold-aware precision/recall metrics.
4. Select an operating threshold on validation data, review feature importance, and expose an interactive Streamlit demo.

## Repository contents

- [Colab notebook](notebooks/tata_steel_machine_failure_prediction.ipynb): EDA, preparation, model comparison, evaluation, and feature importance.
- [Streamlit application](app.py): fits the model from an uploaded labeled training CSV and estimates risk for entered readings.
- [Deployment instructions](DEPLOYMENT.md): local run and Streamlit Community Cloud setup.

## Run locally

Requires Python 3.12. Install dependencies and start the app from the repository root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

In the app, upload the project labeled `train.csv` file. The file is intentionally not stored in this public repository. The unlabeled `test.csv` cannot be used to train the demo.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.

## Limitations

This project uses synthetic data, a single stratified validation split, and no labeled external test set. Validation results may not generalize to real production equipment. The demo score is a prioritization example for learning; it is not a maintenance instruction.
