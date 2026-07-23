"""
model_registry_utils.py

Registers one versioned MLflow pyfunc model per pipeline run — a snapshot
of the tabular ensemble (OCSVM/IForest/Autoencoder/HBOS/XGB) + the GNN's
per-file risk scores, fused with fixed weights, exactly like
task_final_fraud_fusion() does at inference time.

"Best performing" is judged without fraud labels, using a composite of
the proxy metrics each training task already logs to MLflow:
  - OCSVM / IForest / HBOS -> best_silhouette_score      (want higher)
  - Autoencoder            -> best_val_loss               (want lower)
  - XGBoost (synthetic)    -> best_val_auc                (want higher)
  - GNN link predictor     -> final_train_loss, mean_link_prob

Each is mapped to a 0..1 "goodness" value, then combined into one number.
This is a proxy signal, not ground-truth fraud precision — treat it as a
guardrail against clearly-regressed runs, not a substitute for periodic
manual/labeled evaluation.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "siat_fraud_fusion_model"
ALIAS_CHAMPION = "champion"

# Same defaults mlflow_utils.py uses -- kept in sync so this module can
# resolve the experiment correctly even when called with no active run
# (e.g. from task_register_champion_model before it opens its own run).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "file:///home/mouhe/MLopsPipeline/mlruns"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "fraud_detection_pipeline"
)

# Relative contribution of each component to the composite score.
# Missing components (e.g. a task didn't run) are dropped and the rest
# are renormalized, so this still works on partial DAG runs.
DEFAULT_WEIGHTS = {
    "ocsvm": 1.0,
    "iforest": 1.0,
    "autoencoder": 1.0,
    "hbos": 1.0,
    "xgboost": 1.0,
    "gnn": 1.5,  # weighted a bit higher since it's the structural signal
}


def _goodness_silhouette(metrics: dict) -> float | None:
    s = metrics.get("best_silhouette_score")
    if s is None:
        return None
    return (s + 1.0) / 2.0  # silhouette in [-1, 1] -> [0, 1]


def _goodness_autoencoder(metrics: dict) -> float | None:
    v = metrics.get("best_val_loss")
    if v is None:
        return None
    return 1.0 / (1.0 + v)  # lower loss -> closer to 1


def _goodness_xgb(metrics: dict) -> float | None:
    auc = metrics.get("best_val_auc")
    if auc is None:
        return None
    return max(0.0, min(1.0, auc))


def _goodness_gnn(metrics: dict) -> float | None:
    loss = metrics.get("final_train_loss")
    prob = metrics.get("mean_link_prob")
    if loss is None and prob is None:
        return None
    parts = []
    if loss is not None:
        parts.append(1.0 / (1.0 + loss))
    if prob is not None:
        parts.append(max(0.0, min(1.0, prob)))
    return sum(parts) / len(parts)


_GOODNESS_FNS = {
    "ocsvm": _goodness_silhouette,
    "iforest": _goodness_silhouette,
    "hbos": _goodness_silhouette,
    "autoencoder": _goodness_autoencoder,
    "xgboost": _goodness_xgb,
    "gnn": _goodness_gnn,
}


def compute_composite_score(metrics_by_component: dict, weights: dict = None) -> float:
    """
    metrics_by_component: {"ocsvm": {...mlflow metrics dict...}, "gnn": {...}, ...}
    Returns a single float in roughly [0, 1]. Higher = better.
    """
    weights = weights or DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    for component, metrics in metrics_by_component.items():
        fn = _GOODNESS_FNS.get(component)
        if fn is None:
            continue
        goodness = fn(metrics or {})
        if goodness is None:
            continue  # component's task didn't run / metric missing -> skip, don't zero it out
        w = weights.get(component, 1.0)
        weighted_sum += w * goodness
        total_weight += w

    if total_weight == 0.0:
        raise ValueError("No usable metrics found to compute a composite score from.")
    return weighted_sum / total_weight


def _find_artifact_file(root_dir: str, candidates) -> str:
    """
    Locates a file logged under artifacts={key: local_path} inside a
    downloaded model bundle. MLflow's storage layout for these entries
    varies by version:
      - some versions nest each entry under a folder named after its
        dict key (e.g. .../artifacts/gnn_encoder/hetero_gnn_encoder.pt)
      - others flatten everything into one 'artifacts' folder keyed by
        each file's original basename (e.g. .../artifacts/hetero_gnn_encoder.pt)
    `candidates` accepts either the dict key or a list of acceptable
    exact basenames, and this checks both layouts rather than assuming one.
    """
    if isinstance(candidates, str):
        candidates = [candidates]

    # Flattened layout: a file whose basename exactly matches a candidate.
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname in candidates:
                return os.path.join(dirpath, fname)

    # Nested layout: a directory named after one of the candidates.
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) in candidates:
            real_files = [f for f in filenames if not f.startswith(".")]
            if real_files:
                return os.path.join(dirpath, real_files[0])

    raise FileNotFoundError(f"Could not find any of {candidates} anywhere under {root_dir}.")


def resolve_champion_gnn_paths(local_cache_dir: str = None) -> dict:
    """
    Looks up the model version currently aliased ALIAS_CHAMPION for
    MODEL_NAME in the MLflow Model Registry, and downloads its GNN encoder
    / predictor weight artifacts to a local cache directory so callers
    (e.g. RealTimeCustomsTriageEngine) can torch.load() them exactly like
    the old hardcoded local .pt files.

    Returns:
        {
          "encoder_path": <local .pt path>,
          "predictor_path": <local .pt path>,
          "version": <registry model version, str>,
          "run_id": <mlflow run id>,
        }

    Raises RuntimeError if no champion version is registered yet, or if
    that version's bundle doesn't contain gnn_encoder/gnn_predictor
    artifacts (e.g. it predates this change) -- callers decide whether to
    fall back to local files or fail hard.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
    except Exception as e:
        raise RuntimeError(
            f"No '{ALIAS_CHAMPION}' version registered for model '{MODEL_NAME}' yet: {e}"
        )

    dest_dir = local_cache_dir or os.path.join(
        "/tmp", "mlflow_champion_cache", MODEL_NAME, f"v{mv.version}"
    )
    os.makedirs(dest_dir, exist_ok=True)

    # Downloading the whole model bundle via the models:/name@alias URI
    # (instead of hand-building a run-relative artifact path) works
    # regardless of whether this MLflow version stores the pyfunc
    # artifacts under the run's artifact folder or under its own Logged
    # Model entity -- mlflow resolves that internally.
    try:
        model_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}@{ALIAS_CHAMPION}",
            dst_path=dest_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download champion model bundle for '{MODEL_NAME}' "
            f"v{mv.version} (run {mv.run_id}): {e}"
        )

    try:
        encoder_path = _find_artifact_file(model_dir, ["hetero_gnn_encoder.pt", "gnn_encoder"])
        predictor_path = _find_artifact_file(model_dir, ["link_predictor.pt", "gnn_predictor"])
    except Exception as e:
        raise RuntimeError(
            f"Champion model '{MODEL_NAME}' v{mv.version} (run {mv.run_id}) has no "
            f"usable gnn_encoder/gnn_predictor artifacts: {e}"
        )

    return {
        "encoder_path": encoder_path,
        "predictor_path": predictor_path,
        "version": mv.version,
        "run_id": mv.run_id,
    }


def get_child_run_metrics(parent_run_id: str, run_name: str) -> dict:
    """
    Pulls the metrics dict logged by a specific child run (e.g. 'hbos',
    'gnn_hetero_link_predictor') nested under this pipeline's root run.
    Returns {} if that child run isn't found (e.g. task was skipped).

    Resolves the experiment explicitly by name via MlflowClient rather
    than relying on an active mlflow run / fluent state -- this function
    is called from task_register_champion_model BEFORE it opens its own
    run, so there is no "current" run/experiment in the fluent API's
    sense yet. Passing None as an experiment name/id to the tracking
    server is a hard 400 (INVALID_PARAMETER_VALUE), not a soft fallback.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        print(f"⚠️ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found -- skipping metrics for '{run_name}'.")
        return {}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.mlflow.parentRunId = '{parent_run_id}' "
            f"and tags.mlflow.runName = '{run_name}'"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"⚠️ No child run named '{run_name}' found under parent {parent_run_id} -- skipping its metrics.")
        return {}
    return dict(runs[0].data.metrics)


class FraudFusionPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    Bundles the tabular ensemble + a frozen GNN risk lookup into one
    servable/versioned unit, replicating task_final_fraud_fusion()'s logic.

    Expected model_input: a pandas DataFrame with:
      - "scaled_features": list[float] per row (the same 18-dim vector
        produced by tabular_data_prep.py's StandardScaler stage)
      - "file_id": optional; enables joining the GNN structural score.
        Rows without a file_id, or whose file_id isn't in the graph
        snapshot this model was registered with, get a neutral 0.5 GNN
        prior -- same fallback behavior as the DAG's fusion task.

    Known limitation: the GNN component is a frozen snapshot (file_id ->
    gnn_risk_score) taken at registration time, not live graph inference.
    A genuinely new file_id (created after the last full pipeline run)
    will fall back to the neutral prior until the next retrain.
    """

    def load_context(self, context):
        import json
        import joblib
        import torch
        import pandas as pd
        from tabular_models import OptunaAutoencoder

        self.ocsvm = joblib.load(context.artifacts["ocsvm"])
        self.iforest = joblib.load(context.artifacts["iforest"])
        self.hbos = joblib.load(context.artifacts["hbos"])
        self.xgb = joblib.load(context.artifacts["xgb"])

        with open(context.artifacts["ae_arch"]) as f:
            ae_arch = json.load(f)
        self.autoencoder = OptunaAutoencoder(**ae_arch)
        self.autoencoder.load_state_dict(
            torch.load(context.artifacts["autoencoder"], map_location="cpu")
        )
        self.autoencoder.eval()

        self.gnn_lookup = pd.read_parquet(context.artifacts["gnn_lookup"])

        with open(context.artifacts["fusion_config"]) as f:
            self.config = json.load(f)

    def _tabular_ensemble_score(self, feature_matrix):
        import numpy as np
        import torch

        preds = [
            np.where(self.ocsvm.predict(feature_matrix) == -1, 1, 0),
            np.where(self.iforest.predict(feature_matrix) == -1, 1, 0),
        ]

        X_t = torch.FloatTensor(feature_matrix)
        with torch.no_grad():
            recon = self.autoencoder(X_t)
            errors = torch.mean((X_t - recon) ** 2, dim=1).numpy()
        # Frozen threshold from the full training set at registration time
        # (not a percentile of this inference batch) -- a batch of 1-2
        # rows can't meaningfully compute its own percentile.
        preds.append((errors > self.config["ae_error_threshold"]).astype(int))

        preds.append(self.hbos.predict(feature_matrix))

        xgb_scores = self.xgb.predict_proba(feature_matrix)[:, 1]
        preds.append(np.where(xgb_scores >= self.config["xgb_score_threshold"], 1, 0))

        stacked = np.column_stack(preds)
        return stacked.mean(axis=1)

    def predict(self, context, model_input):
        import numpy as np
        import pandas as pd

        feature_matrix = np.array(model_input["scaled_features"].tolist())
        ensemble_risk = self._tabular_ensemble_score(feature_matrix)

        out = pd.DataFrame(index=model_input.index)
        if "file_id" in model_input.columns:
            out["file_id"] = model_input["file_id"].values
        out["ensemble_risk_score"] = ensemble_risk

        if "file_id" in out.columns:
            out = out.merge(self.gnn_lookup[["file_id", "gnn_risk_score"]], on="file_id", how="left")
        if "gnn_risk_score" not in out.columns:
            out["gnn_risk_score"] = np.nan
        out["gnn_risk_score"] = out["gnn_risk_score"].fillna(0.5)

        w_tab = self.config["w_tabular"]
        w_gnn = self.config["w_gnn"]
        threshold = self.config["risk_threshold"]

        out["final_fraud_score"] = out["ensemble_risk_score"] * w_tab + out["gnn_risk_score"] * w_gnn
        out["final_verdict"] = np.where(
            out["final_fraud_score"] >= threshold, "HIGH_RISK", "PASS"
        )
        return out