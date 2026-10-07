# Streamlit demo deployment

The repository contains a small interactive risk-scoring demo in app.py. It does not bundle the dataset. A user uploads the project's synthetic train.csv in the sidebar; the app fits the same preprocessing pipeline and Random Forest settings used in the notebook, then accepts one set of machine readings and returns a probability and threshold-based review flag.

## Run locally

Install the dependencies from the repository root and start the app:

    pip install -r requirements.txt
    streamlit run app.py

Upload train.csv when prompted. The app does not use test.csv because that file has no target labels.

## Deploy on Streamlit Community Cloud

1. Merge this deployment feature branch into main.
2. Sign in at share.streamlit.io with the GitHub account that owns the repository.
3. Create an app from the Tata Steel machine failure prediction repository, select main as the branch, and set app.py as the entrypoint.
4. Use Python 3.12 in Advanced settings, then deploy.
5. Open the app, upload train.csv, and try the prediction form. Add the resulting public app URL to this section after deployment.

The app uses the notebook's validation-selected threshold of 0.248. The score is an educational prototype trained on synthetic data, not a production maintenance recommendation. The test file is unlabeled, so its prediction rate is not a model performance metric.
