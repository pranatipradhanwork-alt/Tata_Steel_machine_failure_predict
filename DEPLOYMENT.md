# Streamlit demo deployment

The repository contains a small interactive risk-scoring demo in app.py. It does not bundle the dataset. Instead it ships a pre-trained model (models/machine_failure_rf.joblib) built by train_model.py with the same preprocessing pipeline and Random Forest settings used in the notebook. The app accepts one set of machine readings and returns a probability and threshold-based review flag. Uploading train.csv in the sidebar is optional and retrains the model.

To rebuild the bundled model after changing the training code:

    python train_model.py path/to/train.csv

## Run locally

Install the dependencies from the repository root and start the app:

    pip install -r requirements.txt
    streamlit run app.py

No upload is needed. The app does not use test.csv because that file has no target labels.

## Deploy on Streamlit Community Cloud

1. Merge this deployment feature branch into main.
2. Sign in at share.streamlit.io with the GitHub account that owns the repository.
3. Create an app from the Tata Steel machine failure prediction repository, select main as the branch, and set app.py as the entrypoint.
4. Use Python 3.12 in Advanced settings, then deploy.
5. Open the app and try the prediction form. Add the resulting public app URL to this section after deployment.

The app uses the notebook's validation-selected threshold of 0.248. The score is an educational prototype trained on synthetic data, not a production maintenance recommendation. The test file is unlabeled, so its prediction rate is not a model performance metric.
