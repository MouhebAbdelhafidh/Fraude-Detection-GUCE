#!/usr/bin/env python
"""
score_with_champion.py

Standalone scoring entrypoint -- NOT an Airflow task. Loads whichever
model version currently holds the 'champion' alias in the MLflow Model
Registry and scores a batch of files with it.

Run manually, from a scheduled non-Airflow job, or wrap `score_files()`
in a small API for on-demand scoring. Requires:
  - MLFLOW_TRACKING_URI pointing at the same store the pipeline writes to
  - scaled_features already computed the same way tabular_data_prep.py
    does (same 18-dim StandardScaler output)
"""

import os
import mlflow
import pandas as pd

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "file:///home/mouhe/MLopsPipeline/mlruns"
)
MODEL_NAME = "siat_fraud_fusion_model"
ALIAS = "champion"


def load_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")


def score_files(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    input_df must have:
      - 'scaled_features': list[float] per row (18-dim, same pipeline as training)
      - 'file_id': optional, enables the GNN structural component; rows
        without it (or with an unseen file_id) get a neutral 0.5 GNN prior.
    Returns input_df's columns plus ensemble_risk_score, gnn_risk_score,
    final_fraud_score, final_verdict.
    """
    model = load_champion_model()
    return model.predict(input_df)


if __name__ == "__main__":
    # Minimal smoke test -- replace with a real batch pulled from Postgres
    # / the tabular_data_prep output before using this for anything real.
    import numpy as np
    dummy = pd.DataFrame({
        "file_id": [101, 102],
        "scaled_features": [np.random.randn(18).tolist(), np.random.randn(18).tolist()],
    })
    result = score_files(dummy)
    print(result)