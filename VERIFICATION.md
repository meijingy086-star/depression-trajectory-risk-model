# Model verification, 17 November 2026

- Artifact SHA-256: `84853acdc870849bd5845aa7446fc9d486361a317fa6023441a4e01df5393718`.
- Model: final XGBoost multiclass classifier from the updated 36-candidate-variable analysis.
- Raw predictors: 18; encoded features: 33.
- Trees: 800 (200 boosting rounds × 4 classes).
- Synthetic raw-input profiles checked against R: 54.
- Maximum portable probability error versus the original R model: `1.282928e-07`.
- Development rows used to verify feature-column identity: 1,946.
- Python `verify_model.py` loads the packaged artifact and rechecks every R parity profile.
- Unit tests cover strict XGBoost split semantics, encoding, softmax probabilities, malformed trees, invalid inputs, display labels, artifact fingerprinting, and deployment assets.

The export is inference-only: it does not refit the model and does not include participant IDs or records. ADL and IADL definitions match the final analysis, and episodic memory and executive-function variables are not website inputs.
