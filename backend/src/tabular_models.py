# tabular_models.py
import os
import json
import joblib
import optuna
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

# Scikit-Learn imports
from sklearn.linear_model import SGDOneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split

# PyOD import
from pyod.models.hbos import HBOS

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mlflow_utils import child_run

# Suppress Optuna verbosity for clean console tracking
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# PATH ANCHORING: same reasoning as pipeline_tasks.py's PROJECT_DIR --
# Airflow's Task SDK runs each PythonOperator in its own subprocess and
# there's no guarantee os.getcwd() matches PROJECT_DIR across tasks. A
# bare relative string like "models/autoencoder/x.pth" can silently land
# in a different physical directory than the one task_register_champion_model
# reads from (which IS correctly anchored to PROJECT_DIR). This caused
# FileNotFoundError on best_autoencoder_arch.json in production -- every
# save/load path below is now built from MODELS_DIR, never a bare string.
# ------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


# =====================================================================
# 1. SGD ONE-CLASS SVM
# =====================================================================
def train_sgd_one_class_svm(feature_matrix: np.ndarray, local_df: pd.DataFrame, n_trials: int = 15,
                             parent_run_id: str = None) -> pd.DataFrame:
    print("\n[MODEL 1/5] Initiating SGD One-Class SVM Pipeline...")
    model_dir = os.path.join(MODELS_DIR, "one_class_svm")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    sample_size = min(10000, feature_matrix.shape[0])
    sample_indices = np.random.choice(feature_matrix.shape[0], sample_size, replace=False)
    X_sample = feature_matrix[sample_indices]

    def objective_ocsvm(trial):
        learning_rate = trial.suggest_categorical("learning_rate", ["optimal", "constant", "invscaling", "adaptive"])
        tol = trial.suggest_float("tol", 1e-4, 1e-2, log=True)

        eta0 = 0.01
        if learning_rate in ["constant", "invscaling", "adaptive"]:
            eta0 = trial.suggest_float("eta0", 1e-4, 1e-1, log=True)

        power_t = 0.5
        if learning_rate == "invscaling":
            power_t = trial.suggest_float("power_t", 0.1, 0.9)

        model = SGDOneClassSVM(
            nu=0.01,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            tol=tol,
            random_state=42
        )
        model.fit(X_sample)
        preds = model.predict(X_sample)

        if len(np.unique(preds)) == 1:
            return -1.0

        return silhouette_score(X_sample, preds)

    with child_run("sgd_one_class_svm", parent_run_id):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("nu", 0.01)
        mlflow.log_param("optuna_sample_size", X_sample.shape[0])

        print("Optimizing Hyperparameters via Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_ocsvm, n_trials=n_trials)

        print("Best OCSVM Hyperparameters:", study.best_params)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_silhouette_score", study.best_value)

        print("Training Final Model across all records...")
        best_ocsvm = SGDOneClassSVM(**study.best_params, nu=0.01, random_state=42)
        best_ocsvm.fit(feature_matrix)

        local_df["is_anomaly_ocsvm"] = np.where(best_ocsvm.predict(feature_matrix) == -1, 1, 0)

        svm_path = os.path.join(model_dir, "best_ocsvm_model.joblib")
        joblib.dump(best_ocsvm, svm_path)
        print(f"Model artifact saved to: {svm_path}")

        n_anomalies = int(local_df["is_anomaly_ocsvm"].sum())
        print(f"Total Anomalies Tagged: {n_anomalies}")
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", n_anomalies / len(local_df))
        mlflow.log_artifact(svm_path)

    return local_df


# =====================================================================
# 2. ISOLATION FOREST
# =====================================================================
def train_isolation_forest(feature_matrix: np.ndarray, local_df: pd.DataFrame, n_trials: int = 20,
                            parent_run_id: str = None) -> pd.DataFrame:
    print("\n[MODEL 2/5] Initiating Isolation Forest Pipeline...")
    model_dir = os.path.join(MODELS_DIR, "isolation_forest")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    sample_size = min(10000, feature_matrix.shape[0])
    sample_indices = np.random.choice(feature_matrix.shape[0], sample_size, replace=False)
    X_sample = feature_matrix[sample_indices]

    def objective_iforest(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        max_samples = trial.suggest_categorical("max_samples", [512, 1024, 2048, 4096])
        max_features = trial.suggest_float("max_features", 0.5, 1.0)

        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            contamination=0.01,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_sample)
        labels = model.predict(X_sample)

        if len(np.unique(labels)) == 1:
            return -1.0

        return silhouette_score(X_sample, labels)

    with child_run("isolation_forest", parent_run_id):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("contamination", 0.01)
        mlflow.log_param("optuna_sample_size", X_sample.shape[0])

        print("Optimizing Hyperparameters via Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_iforest, n_trials=n_trials)

        print("Best IForest Hyperparameters:", study.best_params)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_silhouette_score", study.best_value)

        print("Training Final Model across all records...")
        best_iforest = IsolationForest(**study.best_params, contamination=0.01, n_jobs=-1, random_state=42)
        predictions = best_iforest.fit_predict(feature_matrix)

        local_df["iforest_score"] = best_iforest.decision_function(feature_matrix)
        local_df["is_anomaly_iforest"] = np.where(predictions == -1, 1, 0)

        iforest_path = os.path.join(model_dir, "best_iforest_model.joblib")
        joblib.dump(best_iforest, iforest_path)
        print(f"Model artifact saved to: {iforest_path}")

        n_anomalies = int(local_df["is_anomaly_iforest"].sum())
        print(f"Total Anomalies Tagged: {n_anomalies}")
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", n_anomalies / len(local_df))
        mlflow.log_metric("mean_iforest_score", float(local_df["iforest_score"].mean()))
        mlflow.log_artifact(iforest_path)

    return local_df


# =====================================================================
# 3. PYTORCH DEEP AUTOENCODER
# =====================================================================
class OptunaAutoencoder(nn.Module):
    def __init__(self, input_dim, layer_1_size, bottleneck_size, dropout_rate):
        super(OptunaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer_1_size),
            nn.BatchNorm1d(layer_1_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(layer_1_size, bottleneck_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, layer_1_size),
            nn.BatchNorm1d(layer_1_size),
            nn.LeakyReLU(0.1),
            nn.Linear(layer_1_size, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_pytorch_autoencoder(feature_matrix: np.ndarray, local_df: pd.DataFrame, n_trials: int = 15,
                               parent_run_id: str = None) -> pd.DataFrame:
    print("\n[MODEL 3/5] Initiating PyTorch Neural Autoencoder Pipeline...")
    model_dir = os.path.join(MODELS_DIR, "autoencoder")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cpu")
    X_tensor = torch.FloatTensor(feature_matrix).to(device)

    dataset_size = len(X_tensor)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(TensorDataset(X_tensor, X_tensor), [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    input_dim = feature_matrix.shape[1]

    def objective_ae(trial):
        layer_1_out = trial.suggest_int("layer_1_size", 16, 64, step=16)
        bottleneck = trial.suggest_int("bottleneck_size", 4, 12, step=4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        model = OptunaAutoencoder(input_dim, layer_1_out, bottleneck, dropout_rate).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(5):
            model.train()
            for batch_features, _ in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_features), batch_features)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, _ in val_loader:
                loss = criterion(model(batch_features), batch_features)
                val_loss += loss.item() * batch_features.size(0)

        return val_loss / val_size

    with child_run("pytorch_autoencoder", parent_run_id):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)

        print("Optimizing Hyperparameters via Optuna...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_ae, n_trials=n_trials)

        print("Best Autoencoder Hyperparameters:", study.best_params)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)

        print("Training Final Network Profile...")
        best_model = OptunaAutoencoder(
            input_dim=input_dim,
            layer_1_size=study.best_params["layer_1_size"],
            bottleneck_size=study.best_params["bottleneck_size"],
            dropout_rate=study.best_params["dropout_rate"]
        ).to(device)

        best_optimizer = optim.AdamW(
            best_model.parameters(),
            lr=study.best_params["lr"],
            weight_decay=study.best_params["weight_decay"]
        )
        criterion = nn.MSELoss(reduction='none')

        full_train_loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=1024, shuffle=True)

        final_epochs = 15
        best_model.train()
        for epoch in range(final_epochs):
            epoch_loss_total = 0.0
            for batch_features, _ in full_train_loader:
                best_optimizer.zero_grad()
                reconstructed = best_model(batch_features)
                loss = criterion(reconstructed, batch_features).mean()
                loss.backward()
                best_optimizer.step()
                epoch_loss_total += loss.item() * batch_features.size(0)
            mlflow.log_metric("final_train_epoch_loss", epoch_loss_total / dataset_size, step=epoch)

        best_model.eval()
        with torch.no_grad():
            reconstructed_data = best_model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed_data) ** 2, dim=1).numpy()

        threshold = np.percentile(reconstruction_errors, 99)
        local_df["ae_reconstruction_error"] = reconstruction_errors
        local_df["is_anomaly_autoencoder"] = (reconstruction_errors > threshold).astype(int)

        ae_path = os.path.join(model_dir, "best_autoencoder_weights.pth")
        torch.save(best_model.state_dict(), ae_path)
        print(f"State Dict Weights successfully saved to: {ae_path}")

        # Architecture snapshot -- required by load_saved_models() and by
        # model_registry_utils.FraudFusionPyfuncModel to re-instantiate the
        # exact network shape Optuna picked this run (layer sizes vary
        # trial to trial, so a fixed constructor call can't be relied on).
        ae_arch_path = os.path.join(model_dir, "best_autoencoder_arch.json")
        with open(ae_arch_path, "w") as f:
            json.dump({
                "input_dim": input_dim,
                "layer_1_size": study.best_params["layer_1_size"],
                "bottleneck_size": study.best_params["bottleneck_size"],
                "dropout_rate": study.best_params["dropout_rate"],
            }, f)
        print(f"Architecture saved to: {ae_arch_path}")

        n_anomalies = int(local_df["is_anomaly_autoencoder"].sum())
        print(f"Total Anomalies Tagged: {n_anomalies}")
        mlflow.log_metric("reconstruction_error_threshold_p99", float(threshold))
        mlflow.log_metric("mean_reconstruction_error", float(reconstruction_errors.mean()))
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", n_anomalies / len(local_df))
        mlflow.log_artifact(ae_path)
        mlflow.log_artifact(ae_arch_path)

    return local_df


# =====================================================================
# 4. HISTOGRAM-BASED OUTLIER SCORE (HBOS)
# =====================================================================
def train_hbos(feature_matrix: np.ndarray, local_df: pd.DataFrame, n_trials: int = 15,
                parent_run_id: str = None) -> pd.DataFrame:
    print("\n[MODEL 4/5] Initiating PyOD Histogram-Based Outlier Score Pipeline...")
    model_dir = os.path.join(MODELS_DIR, "hbos")
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(42)
    sample_size = min(10000, feature_matrix.shape[0])
    sample_indices = np.random.choice(feature_matrix.shape[0], sample_size, replace=False)
    X_sample = feature_matrix[sample_indices]

    def objective_hbos(trial):
        n_bins = trial.suggest_int("n_bins", 10, 150, step=10)
        alpha = trial.suggest_float("alpha", 0.1, 0.5)

        model = HBOS(contamination=0.01, n_bins=n_bins, alpha=alpha)
        model.fit(X_sample)
        preds = model.labels_

        if len(np.unique(preds)) <= 1:
            return -1.0

        return silhouette_score(X_sample, preds)

    with child_run("hbos", parent_run_id):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("contamination", 0.01)
        mlflow.log_param("optuna_sample_size", X_sample.shape[0])

        print("Optimizing Hyperparameters via Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_hbos, n_trials=n_trials)

        print("Best HBOS Hyperparameters:", study.best_params)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_silhouette_score", study.best_value)

        print("Training Final Model across all records...")
        best_hbos = HBOS(contamination=0.01, **study.best_params)
        best_hbos.fit(feature_matrix)

        local_df["is_anomaly_hbos"] = best_hbos.labels_

        hbos_path = os.path.join(model_dir, "best_hbos_model.joblib")
        joblib.dump(best_hbos, hbos_path)
        print(f"Model artifact saved to: {hbos_path}")

        n_anomalies = int(local_df["is_anomaly_hbos"].sum())
        print(f"Total Anomalies Tagged: {n_anomalies}")
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", n_anomalies / len(local_df))
        mlflow.log_artifact(hbos_path)

    return local_df


# =====================================================================
# 5. UNSUPERVISED XGBOOST (SYNTHETIC SHUFFLING)
# =====================================================================
def train_unsupervised_xgboost(feature_matrix: np.ndarray, local_df: pd.DataFrame, n_trials: int = 10,
                                parent_run_id: str = None) -> pd.DataFrame:
    print("\n[MODEL 5/5] Initiating Unsupervised Synthetic Shuffling XGBoost Pipeline...")
    model_dir = os.path.join(MODELS_DIR, "xgboost_unsupervised")
    os.makedirs(model_dir, exist_ok=True)

    X_real = feature_matrix.copy()
    y_real = np.zeros(X_real.shape[0])

    X_synthetic = np.zeros_like(X_real)
    for col_idx in range(X_real.shape[1]):
        X_synthetic[:, col_idx] = np.random.permutation(X_real[:, col_idx])
    y_synthetic = np.ones(X_synthetic.shape[0])

    X_combined = np.vstack([X_real, X_synthetic])
    y_combined = np.concatenate([y_real, y_synthetic])

    np.random.seed(42)
    optuna_indices = np.random.choice(X_combined.shape[0], 40000, replace=False)
    X_optuna_sub, y_optuna_sub = X_combined[optuna_indices], y_combined[optuna_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_optuna_sub, y_optuna_sub, test_size=0.3, random_state=42, stratify=y_optuna_sub
    )

    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    with child_run("unsupervised_xgboost", parent_run_id):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optuna_sample_size", X_optuna_sub.shape[0])

        print("Optimizing Hyperparameters via Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_xgb, n_trials=n_trials)

        print("Best XGBoost Hyperparameters:", study.best_params)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_auc", study.best_value)

        print("Training Final Classifier over joint synthetic matrices...")
        best_xgb = xgb.XGBClassifier(**study.best_params, random_state=42, n_jobs=-1, eval_metric="logloss")
        best_xgb.fit(X_combined, y_combined)

        anomaly_scores_xgb = best_xgb.predict_proba(X_real)[:, 1]
        threshold_xgb = np.percentile(anomaly_scores_xgb, 99)
        local_df["is_anomaly_xgb"] = np.where(anomaly_scores_xgb >= threshold_xgb, 1, 0)

        xgb_path = os.path.join(model_dir, "best_xgb_anomaly_model.joblib")
        joblib.dump(best_xgb, xgb_path)
        print(f"Model artifact saved to: {xgb_path}")

        n_anomalies = int(local_df["is_anomaly_xgb"].sum())
        print(f"Total Anomalies Tagged: {n_anomalies}")
        mlflow.log_metric("anomaly_score_threshold_p99", float(threshold_xgb))
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", n_anomalies / len(local_df))
        mlflow.log_artifact(xgb_path)

    return local_df


# =====================================================================
# LOAD AND EVALUATE
# =====================================================================

def load_saved_models(input_dim: int):
    print("\nLoading serialized model artifacts from disk...")
    models = {}

    # 1. Load SGD One-Class SVM
    svm_path = os.path.join(MODELS_DIR, "one_class_svm", "best_ocsvm_model.joblib")
    if os.path.exists(svm_path):
        models["ocsvm"] = joblib.load(svm_path)
        print(f"Loaded: {svm_path}")

    # 2. Load Isolation Forest
    iforest_path = os.path.join(MODELS_DIR, "isolation_forest", "best_iforest_model.joblib")
    if os.path.exists(iforest_path):
        models["iforest"] = joblib.load(iforest_path)
        print(f"Loaded: {iforest_path}")

    # 3. Load PyTorch Autoencoder (requires re-instantiating architecture first)
    ae_path = os.path.join(MODELS_DIR, "autoencoder", "best_autoencoder_weights.pth")
    ae_arch_path = os.path.join(MODELS_DIR, "autoencoder", "best_autoencoder_arch.json")
    if os.path.exists(ae_path):
        try:
            if os.path.exists(ae_arch_path):
                with open(ae_arch_path) as f:
                    ae_arch = json.load(f)
                if ae_arch.get("input_dim") != input_dim:
                    print(
                        f"⚠️ Saved autoencoder was trained with input_dim="
                        f"{ae_arch.get('input_dim')}, but current data has "
                        f"input_dim={input_dim}. Skipping load -- retrain needed."
                    )
                    raise ValueError("input_dim mismatch")
                model = OptunaAutoencoder(
                    input_dim=ae_arch["input_dim"],
                    layer_1_size=ae_arch["layer_1_size"],
                    bottleneck_size=ae_arch["bottleneck_size"],
                    dropout_rate=ae_arch["dropout_rate"],
                )
            else:
                print(
                    f"⚠️ {ae_arch_path} not found (older run predates architecture "
                    "logging) -- falling back to placeholder dims 48/12/0.0. "
                    "Retrain to get an accurate architecture file."
                )
                model = OptunaAutoencoder(
                    input_dim=input_dim, layer_1_size=48, bottleneck_size=12, dropout_rate=0.0
                )

            model.load_state_dict(torch.load(ae_path, map_location=torch.device('cpu')))
            model.eval()
            models["autoencoder"] = model
            print(f"Loaded: {ae_path}")
        except Exception as e:
            print(f"Note: Autoencoder structural reload failed. Skipping deep evaluation step. Error: {e}")

    # 4. Load HBOS
    hbos_path = os.path.join(MODELS_DIR, "hbos", "best_hbos_model.joblib")
    if os.path.exists(hbos_path):
        models["hbos"] = joblib.load(hbos_path)
        print(f"Loaded: {hbos_path}")

    # 5. Load XGBoost Unsupervised
    xgb_path = os.path.join(MODELS_DIR, "xgboost_unsupervised", "best_xgb_anomaly_model.joblib")
    if os.path.exists(xgb_path):
        models["xgboost"] = joblib.load(xgb_path)
        print(f"Loaded: {xgb_path}")

    return models


def evaluate_ensemble_anomalies(models: dict, feature_matrix: np.ndarray, local_df: pd.DataFrame,
                                 parent_run_id: str = None) -> pd.DataFrame:
    print("\nRunning batch validation and ensemble risk profiling...")

    with child_run("ensemble_risk_evaluation", parent_run_id):
        mlflow.log_param("n_models_loaded", len(models))
        mlflow.log_param("models_loaded", ",".join(sorted(models.keys())))

        preds_matrix = []

        if "ocsvm" in models:
            svm_preds = np.where(models["ocsvm"].predict(feature_matrix) == -1, 1, 0)
            local_df["eval_anomaly_ocsvm"] = svm_preds
            preds_matrix.append(svm_preds)
            mlflow.log_metric("eval_n_anomalies_ocsvm", int(svm_preds.sum()))

        if "iforest" in models:
            iforest_preds = np.where(models["iforest"].predict(feature_matrix) == -1, 1, 0)
            local_df["eval_anomaly_iforest"] = iforest_preds
            preds_matrix.append(iforest_preds)
            mlflow.log_metric("eval_n_anomalies_iforest", int(iforest_preds.sum()))

        if "autoencoder" in models:
            X_tensor = torch.FloatTensor(feature_matrix)
            with torch.no_grad():
                reconstructed = models["autoencoder"](X_tensor)
                errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            ae_thresh = np.percentile(errors, 99)
            ae_preds = (errors > ae_thresh).astype(int)
            local_df["eval_anomaly_autoencoder"] = ae_preds
            preds_matrix.append(ae_preds)
            mlflow.log_metric("eval_n_anomalies_autoencoder", int(ae_preds.sum()))

        if "hbos" in models:
            hbos_preds = models["hbos"].predict(feature_matrix)
            local_df["eval_anomaly_hbos"] = hbos_preds
            preds_matrix.append(hbos_preds)
            mlflow.log_metric("eval_n_anomalies_hbos", int(np.sum(hbos_preds)))

        if "xgboost" in models:
            xgb_scores = models["xgboost"].predict_proba(feature_matrix)[:, 1]
            xgb_thresh = np.percentile(xgb_scores, 99)
            xgb_preds = np.where(xgb_scores >= xgb_thresh, 1, 0)
            local_df["eval_anomaly_xgb"] = xgb_preds
            preds_matrix.append(xgb_preds)
            mlflow.log_metric("eval_n_anomalies_xgb", int(xgb_preds.sum()))

        if len(preds_matrix) > 0:
            stacked_preds = np.column_stack(preds_matrix)

            local_df["ensemble_risk_score"] = np.mean(stacked_preds, axis=1)

            majority_threshold = max(2, int(len(preds_matrix) / 2) + 1)
            local_df["ensemble_consensus_verdict"] = np.where(np.sum(stacked_preds, axis=1) >= majority_threshold, 1, 0)

            n_consensus = int(local_df["ensemble_consensus_verdict"].sum())
            mean_risk = float(local_df["ensemble_risk_score"].mean())

            print("\nEvaluation Summary:")
            print(f"Total evaluated records: {len(local_df)}")
            print(f"Consensus-voted anomalies (triggered by >= {majority_threshold} models): {n_consensus}")
            print(f"Average structural ensemble risk level: {mean_risk:.4f}")

            mlflow.log_metric("majority_threshold", majority_threshold)
            mlflow.log_metric("n_consensus_anomalies", n_consensus)
            mlflow.log_metric("consensus_rate", n_consensus / len(local_df))
            mlflow.log_metric("mean_ensemble_risk_score", mean_risk)
        else:
            print("Error: No loaded model pipelines available for inference processing.")
            mlflow.set_tag("evaluation_status", "no_models_loaded")

    return local_df