# Depressive Symptom Trajectory Predictor

Streamlit research application for predicting four 2011–2020 depressive-symptom trajectory groups among older Chinese adults.

## Current release

- Final model: XGBoost multiclass classifier
- Development/test split: 1,946 / 835 participants
- Predictors: 18 baseline variables selected by LASSO
- Encoded features: 33
- Trees: 800 (200 boosting rounds × 4 classes)
- Test Hand–Till AUC: 0.698
- Test macro one-vs-rest AUC: 0.740
- Test MCC: 0.290

ADL impairment is defined as difficulty in at least one of six ADL activities. Baseline IADL impairment is defined as difficulty in at least one of five activities (housework, cooking, shopping, taking medication, and managing money); telephone use is excluded because it was unavailable at the 2011 baseline.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Re-export the model

`export_model.R` reads the final fitted R objects and creates `artifacts/model.json.gz`. It does not refit the model and does not export participant IDs or individual records. The four required source files are documented at the top of the script.

```r
source("export_model.R")
```

Run `python verify_model.py` after export. The application and verification script use a dependency-free Python implementation of XGBoost tree traversal, checked against 54 synthetic profiles from the original R model.

## Disclaimer

This model was developed and internally validated for research. It does not establish a clinical diagnosis and should not replace professional assessment.
