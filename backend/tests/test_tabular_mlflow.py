import numpy as np
import pandas as pd
from mlflow_utils import start_pipeline_run
from tabular_models import (
    train_isolation_forest,
    train_hbos,
    train_sgd_one_class_svm,
    train_pytorch_autoencoder,
    train_unsupervised_xgboost,
    load_saved_models,
    evaluate_ensemble_anomalies,
)

root_run_id = start_pipeline_run(dag_run_id="tabular_smoke_test")
print("root run_id:", root_run_id)

# >= 20000 rows needed: train_unsupervised_xgboost samples 40000 rows from
# a real+synthetic stack that's 2x this size, without replacement.
N = 25000
X = np.random.randn(N, 10)
df = pd.DataFrame({"file_id": range(N)})

df = train_isolation_forest(X, df, n_trials=2, parent_run_id=root_run_id)
df = train_hbos(X, df, n_trials=2, parent_run_id=root_run_id)
df = train_sgd_one_class_svm(X, df, n_trials=2, parent_run_id=root_run_id)
df = train_pytorch_autoencoder(X, df, n_trials=2, parent_run_id=root_run_id)
df = train_unsupervised_xgboost(X, df, n_trials=2, parent_run_id=root_run_id)

print("\nAnomaly counts per model:")
print(df.filter(like="is_anomaly").sum())

models = load_saved_models(input_dim=10)
df = evaluate_ensemble_anomalies(models, X, df, parent_run_id=root_run_id)
print("\nEnsemble columns:", [c for c in df.columns if "ensemble" in c])

print(f"\ndone — open http://localhost:5000/#/experiments/1/runs/{root_run_id}")
