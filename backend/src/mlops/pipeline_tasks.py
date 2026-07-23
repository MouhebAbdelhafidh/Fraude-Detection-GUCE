# pipeline_tasks.py
import os
import sys

# ------------------------------------------------------------------
# PATH INJECTION
# ------------------------------------------------------------------
PHYSICAL_FILE = os.path.realpath(__file__)
MLOPS_DIR = os.path.dirname(PHYSICAL_FILE)
SRC_DIR = os.path.dirname(MLOPS_DIR)
BACKEND_DIR = os.path.dirname(SRC_DIR)

for path in [MLOPS_DIR, SRC_DIR, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pandas as pd
import mlflow

from gnn_models import train_hetero_gnn
from gnn_inference import compute_gnn_risk_scores
from mlflow_utils import start_pipeline_run, child_run, log_to_run
from adversarial.defenses.tabular_defense import TabularAdversarialShield

# ------------------------------------------------------------------
# PATH ANCHORING & DIRECTORY DEFINITIONS
# ------------------------------------------------------------------
PROJECT_DIR = BACKEND_DIR
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODELS_GNN_DIR = os.path.join(BACKEND_DIR, "models", "gnn")

# Subdirectories inside /backend/data
TMP_DIR = os.path.join(DATA_DIR, "tmp")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GNN_OUTPUT_DIR = os.path.join(DATA_DIR, "gnn_output")

# Reports and monitoring directory
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
MONITORING_DIR = os.path.join(REPORTS_DIR, "monitoring")

# File paths
EXTRACTED_FEATURES_PATH = os.path.join(TMP_DIR, "extracted_features.pkl")
ENSEMBLE_CSV_PATH = os.path.join(DATA_DIR, "ensemble_evaluation_output.csv")
FINAL_PREDICTIONS_PATH = os.path.join(DATA_DIR, "final_fraud_predictions.csv")
GNN_PREDICTIONS_PATH = os.path.join(GNN_OUTPUT_DIR, "gnn_predictions.parquet")

# Ensure ALL required directories exist (TMP_DIR MUST be defined above this!)
for path in [TMP_DIR, PROCESSED_DIR, GNN_OUTPUT_DIR, MODELS_GNN_DIR, MONITORING_DIR]:
    os.makedirs(path, exist_ok=True)

PARENT_RUN_TASK_ID = "pyspark_data_preparation"

def _get_parent_run_id(context) -> str:
    """
    Pulls the MLflow parent run_id pushed by task_prepare_data's return
    value. Returns None (falls back to un-nested runs) if it's missing --
    e.g. when a task is run standalone outside Airflow, or when
    task_prepare_data itself failed and this task is being retried/cleared
    without a fresh run upstream.
    """
    ti = context.get("ti")
    if ti is None:
        return None
    return ti.xcom_pull(task_ids=PARENT_RUN_TASK_ID)


# ------------------------------------------------------------------
# CRITICAL SYSTEM ALIGNMENT: Force Stable Java 17 to bypass JVM Crashes
# ------------------------------------------------------------------
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from tabular_data_prep import prepare_tabular_data
from gnn_data_prep import cross_link_graph_spaces, generate_pyg_tensors
from tabular_models import (
    train_sgd_one_class_svm,
    train_isolation_forest,
    train_pytorch_autoencoder,
    train_hbos,
    train_unsupervised_xgboost,
    load_saved_models,
    evaluate_ensemble_anomalies
)

# Shared configuration credentials
DB_PROPERTIES = {
    "user": "postgres",
    "password": "nobodycan@2H",
    "driver": "org.postgresql.Driver"
}

# Dynamically fetch the Windows host IP address from WSL
windows_host_ip = os.popen("ip route | grep default | awk '{print $3}'").read().strip()
DB_URL = f"jdbc:postgresql://{windows_host_ip}:5432/guce_siat_cc"

def get_spark_session():
    # Force Java 17 check again for worker sub-shells
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

    return SparkSession.builder \
        .appName("FraudPipelineDataPrep") \
        .config("spark.jars", "/home/mouhe/MLopsPipeline/postgresql-42.7.2.jar") \
        .getOrCreate()

def task_prepare_data(**context):
    """
    Also opens the MLflow parent run for the whole DAG run. Returning the
    run_id here makes Airflow auto-push it to XCom under this task's own
    task_id (PARENT_RUN_TASK_ID above), which is how every other task in
    this module finds it again -- there is no other channel that survives
    across Airflow subprocess boundaries.
    """
    print(f"Executing Airflow Task: Ingesting from Postgres via {DB_URL}...")

    dag_run = context.get("dag_run")
    dag_run_id = dag_run.run_id if dag_run else "standalone_run"
    parent_run_id = start_pipeline_run(dag_run_id=dag_run_id)

    spark = get_spark_session()
    try:
        with child_run("pyspark_data_preparation", parent_run_id):
            df_scaled, fitted_transformer = prepare_tabular_data(spark, DB_URL, DB_PROPERTIES, export_parquet=True)

            print("Extracting features into temporary Pandas buffer for model consumption...")
            # file_id (the raw DB primary key) is carried through alongside
            # numero_dossier because it's the reliable join key back to the GNN
            # branch's output later -- numero_dossier and file_id live in
            # different identifier spaces and only file_id is guaranteed to
            # match 1:1 between the two branches (see gnn_data_prep.py).
            local_df = df_scaled.select("file_id", "numero_dossier", "code", "valeur_fob", "ip_address", "scaled_features").toPandas()

            os.makedirs(TMP_DIR, exist_ok=True)
            local_df.to_pickle(EXTRACTED_FEATURES_PATH)
            print(f"Data Preparation complete. Saved temporary state buffer to: {EXTRACTED_FEATURES_PATH}")

            mlflow.log_param("db_url_host", windows_host_ip)
            mlflow.log_metric("n_records_extracted", len(local_df))
    finally:
        spark.stop()

    # Returned value is auto-pushed to XCom by PythonOperator under this
    # task's task_id -- this IS the hand-off point for every downstream task.
    return parent_run_id

def task_preprocess_graph_nodes():
    print("Executing Airflow Task: Extracting and Aligning Heterogeneous Graph Entities...")
    cross_link_graph_spaces()

def task_generate_graph_tensors():
    print("Executing Airflow Task: Serializing PyTorch Geometric Graph Structures...")
    generate_pyg_tensors()

def task_train_models(**context):
    print("Executing Airflow Task: Unsupervised Optuna Optimization & Model Training...")
    if not os.path.exists(EXTRACTED_FEATURES_PATH):
        raise FileNotFoundError(f"Missing temporary feature buffer at {EXTRACTED_FEATURES_PATH}. Ensure data prep step ran successfully.")

    parent_run_id = _get_parent_run_id(context)

    local_df = pd.read_pickle(EXTRACTED_FEATURES_PATH)
    feature_matrix = np.array([v.toArray() for v in local_df["scaled_features"]])

    local_df = train_sgd_one_class_svm(feature_matrix, local_df, n_trials=15, parent_run_id=parent_run_id)
    local_df = train_isolation_forest(feature_matrix, local_df, n_trials=20, parent_run_id=parent_run_id)
    local_df = train_pytorch_autoencoder(feature_matrix, local_df, n_trials=15, parent_run_id=parent_run_id)
    local_df = train_hbos(feature_matrix, local_df, n_trials=15, parent_run_id=parent_run_id)
    local_df = train_unsupervised_xgboost(feature_matrix, local_df, n_trials=10, parent_run_id=parent_run_id)

    print("All models successfully trained and serialized to disk.")

def task_train_gnn_model(**context):
    print("Executing Airflow Task: Training Heterogeneous Graph Neural Network...")

    parent_run_id = _get_parent_run_id(context)

    # ------------------------------------------------------------------
    # STRIPPED: Bypasses extracted_features.pkl buffer to run purely on graph data
    # ------------------------------------------------------------------
    file_nodes_path = os.path.join(PROCESSED_DIR, "final_graph_file_nodes")

    if not os.path.exists(file_nodes_path):
        raise FileNotFoundError(
            f"Missing processed graph files at {file_nodes_path}. Ensure graph prep tasks ran successfully."
        )

    print("Loading file node structural tracking keys directly from graph storage...")
    # Load file nodes dataframe to map results natively to 'file_gnn_idx'
    graph_files_df = pd.read_parquet(file_nodes_path).sort_values("file_gnn_idx")

    # Execute the SageGNN architecture using pure graph layouts. train_hetero_gnn
    # opens its own "gnn_hetero_link_predictor" child run nested under
    # parent_run_id (see gnn_models.py) -- this task runs in-process via
    # PythonOperator, so parent_run_id is threaded straight through as an
    # argument rather than the MLFLOW_PARENT_RUN_ID env var, which is only
    # needed for the standalone `python gnn_models.py` / BashOperator path
    # (see gnn_tasks.py / gnn_models.py's __main__ block).
    updated_graph_df = train_hetero_gnn(local_df=graph_files_df, n_epochs=40, parent_run_id=parent_run_id)

    # Save predictions independently to prevent state interference
    os.makedirs(GNN_OUTPUT_DIR, exist_ok=True)

    # Persist the output vectors. file_id is used here (not numero_dossier) --
    # the source graph node table (see gnn_data_prep.py / the original notebook's
    # file-node prep step) never carried numero_dossier; file_id is the real
    # DB primary key and the correct join column back to the tabular branch's
    # output in task_final_fraud_fusion.
    updated_graph_df[["file_id", "file_gnn_idx", "is_anomaly_gnn"]].to_parquet(GNN_PREDICTIONS_PATH, index=False)
    print(f"GNN Modeling complete. Independent predictions saved to: {GNN_PREDICTIONS_PATH}")

def task_evaluate_ensemble(noise_std: float = 0.15, **context):
    """
    Evaluates tabular features through the Tabular Adversarial Shield first
    for feature sanitization/attack detection, then profiles ensemble risk.
    """
    print("Executing Airflow Task: Defended Ensemble Loading & Performance Evaluation...")
    if not os.path.exists(EXTRACTED_FEATURES_PATH):
        raise FileNotFoundError(
            f"Missing {EXTRACTED_FEATURES_PATH}. This is written by "
            "task_prepare_data (task_id='pyspark_data_preparation') and consumed here."
        )

    parent_run_id = _get_parent_run_id(context)

    local_df = pd.read_pickle(EXTRACTED_FEATURES_PATH)
    feature_matrix = np.array(
        [v.toArray() if hasattr(v, "toArray") else v for v in local_df["scaled_features"]], 
        dtype=np.float32
    )
    input_dim = feature_matrix.shape[1]

    # 1. Load tabular anomaly models
    loaded_models_dict = load_saved_models(input_dim=input_dim)

    # 2. Initialize Adversarial Shield
    print("🛡️ Passing feature vectors through Tabular Adversarial Shield...")
    shield = TabularAdversarialShield(
        normal_reference_matrix=feature_matrix, 
        noise_std=noise_std
    )

    # 3. Defend & Sanitize incoming features
    defense_results = shield.defend_and_evaluate(loaded_models_dict, feature_matrix)

    # Safely unpack output regardless of exact dictionary key name
    if isinstance(defense_results, dict):
        sanitized_features = defense_results.get("sanitized_features", feature_matrix)
        adversarial_flags = defense_results.get(
            "is_adversarial_flag", 
            defense_results.get("is_adversarial", defense_results.get("adversarial_flags", np.zeros(len(feature_matrix), dtype=int)))
        )
    else:
        # Fallback if shield returns array directly
        sanitized_features = defense_results
        adversarial_flags = np.zeros(len(feature_matrix), dtype=int)

    # Convert to numpy array to ensure downstream slicing/summing works
    adversarial_flags = np.array(adversarial_flags, dtype=int)

    # 4. Score sanitized feature matrix through ensemble models
    evaluated_df = evaluate_ensemble_anomalies(
        loaded_models_dict, 
        sanitized_features, 
        local_df, 
        parent_run_id=parent_run_id
    )

    # 5. Append Adversarial Defense Metadata
    evaluated_df["shield_adversarial_flag"] = adversarial_flags
    n_adversarial_detected = int(np.sum(adversarial_flags))

    # Output CSV path setup
    os.makedirs(DATA_DIR, exist_ok=True)
    target_columns = [
        "file_id", 
        "numero_dossier", 
        "code", 
        "valeur_fob", 
        "ensemble_risk_score", 
        "ensemble_consensus_verdict",
        "shield_adversarial_flag"
    ]
    
    # Save decorated CSV containing shield flags
    evaluated_df[target_columns].to_csv(ENSEMBLE_CSV_PATH, index=False)
    print(f"Tabular ensemble metrics with Adversarial Shield flags saved to: {ENSEMBLE_CSV_PATH}")

    # Log defense metrics to MLflow child run
    if parent_run_id:
        with child_run("tabular_adversarial_shield", parent_run_id):
            mlflow.log_param("noise_std", noise_std)
            mlflow.log_metric("n_adversarial_attacks_flagged", n_adversarial_detected)
            mlflow.log_metric("adversarial_flag_rate", n_adversarial_detected / max(1, len(evaluated_df)))

def task_cleanup_temp_buffer(**context):
    """
    Removes intermediate scratch files once the full pipeline (both branches)
    has finished, regardless of individual task outcomes. Wired in the DAG
    with trigger_rule='all_done' so it always runs last and never blocks or
    is blocked by task failures upstream.

    Also the natural (and only reliable) place to record the pipeline's
    final MLflow status: it's the one task guaranteed to run exactly once,
    last, regardless of which upstream branch failed. Note this logs a
    "pipeline_final_status" param onto the already-closed root run rather
    than reopening/re-terminating it with a native run status -- our
    mlflow_utils.py intentionally closes the root run immediately inside
    start_pipeline_run() and only exposes log_to_run() for appending after
    the fact, not a way to flip the run's native status a second time.
    """
    print("Executing Airflow Task: Cleaning up temporary pipeline buffers...")
    if os.path.exists(EXTRACTED_FEATURES_PATH):
        os.remove(EXTRACTED_FEATURES_PATH)
        print(f"Removed {EXTRACTED_FEATURES_PATH}")
    else:
        print(f"{EXTRACTED_FEATURES_PATH} already absent -- nothing to clean up.")

    parent_run_id = _get_parent_run_id(context)
    if parent_run_id:
        dag_run = context.get("dag_run")
        overall_failed = False
        ti = context.get("ti")
        if ti is not None and dag_run is not None:
            task_instances = dag_run.get_task_instances()
            overall_failed = any(t.state == "failed" for t in task_instances)
        final_status = "FAILED" if overall_failed else "FINISHED"
        log_to_run(parent_run_id, params={"pipeline_final_status": final_status})
        print(f"MLflow parent run {parent_run_id} tagged with final status: {final_status}")

def task_run_data_drift_audit(**context):
    """
    Compares an early vs. late slice of the GNN's file-node feature space
    to catch input drift before it silently degrades the trained GNN's
    scoring. This is an observability signal (HTML report + MLflow
    metrics) -- it is intentionally NOT fed into task_final_fraud_fusion
    as a scoring input.
    """
    print("Executing Airflow Task: Data Drift Audit (Evidently)...")

    from evidently import Report
    from evidently.presets import DataDriftPreset

    parent_run_id = _get_parent_run_id(context)

    file_nodes_path = os.path.join(PROCESSED_DIR, "final_graph_file_nodes")
    if not os.path.exists(file_nodes_path):
        raise FileNotFoundError(
            f"Missing processed graph files at {file_nodes_path}. Ensure graph prep tasks ran successfully."
        )

    df_nodes = pd.read_parquet(file_nodes_path)

    if "file_features" in df_nodes.columns:
        features_df = pd.DataFrame(df_nodes["file_features"].tolist()).add_prefix("feature_")
    else:
        features_df = df_nodes.select_dtypes(include=["float64", "int64"])

    if len(features_df) < 20:
        print(f"⚠️ Only {len(features_df)} rows available -- too few for a meaningful drift split. Skipping.")
        return

    # Zero-variance columns (e.g. padding dims in file_features) produce
    # "invalid value encountered in divide" warnings from the underlying
    # stat tests and carry no drift signal anyway -- drop them up front.
    constant_cols = [c for c in features_df.columns if features_df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        preview = constant_cols[:10]
        suffix = "..." if len(constant_cols) > 10 else ""
        print(f"Dropping {len(constant_cols)} constant column(s) before drift comparison (no variance): {preview}{suffix}")
        features_df = features_df.drop(columns=constant_cols)

    if features_df.shape[1] == 0:
        print("⚠️ No non-constant columns left after filtering -- nothing to compare. Skipping.")
        return

    split_idx = int(len(features_df) * 0.7)
    reference_data = features_df.iloc[:split_idx]
    current_data = features_df.iloc[split_idx:]

    report = Report([DataDriftPreset(method="wasserstein")])
    my_eval = report.run(current_data=current_data, reference_data=reference_data)

    # ------------------------------------------------------------------
    # SAVE TO backend/data/reports/monitoring
    # ------------------------------------------------------------------
    report_dir = os.path.join(DATA_DIR, "reports", "monitoring")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "data_drift_latest.html")
    my_eval.save_html(report_path)
    print(f"✅ Drift report saved to: {report_path}")

    result_dict = {}
    for method_name in ("dict", "as_dict", "model_dump"):
        method = getattr(my_eval, method_name, None)
        if callable(method):
            try:
                result_dict = method()
                break
            except Exception:
                continue

    import json
    debug_json_path = os.path.join(report_dir, "data_drift_latest_debug.json")
    try:
        with open(debug_json_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"📝 Full drift result dict saved for inspection: {debug_json_path}")
    except Exception as e:
        print(f"⚠️ Could not serialize result dict to JSON for debugging: {e}")

    def _find_key(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for v in node.values():
                found = _find_key(v, key)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find_key(item, key)
                if found is not None:
                    return found
        return None

    dataset_drift = None
    for key in ("dataset_drift", "drift_detected", "dataset_drift_detected"):
        dataset_drift = _find_key(result_dict, key)
        if dataset_drift is not None:
            break

    share_drifted = None
    for key in ("share_of_drifted_columns", "drift_share", "share_drifted_columns"):
        share_drifted = _find_key(result_dict, key)
        if share_drifted is not None:
            break

    if dataset_drift is not None:
        print(f"Dataset drift detected: {dataset_drift} (share of drifted columns: {share_drifted})")
    else:
        top_level = list(result_dict.keys()) if isinstance(result_dict, dict) else type(result_dict).__name__
        print("⚠️ Could not locate a dataset-drift summary key under any of the tried names. "
              f"Top-level keys in the exported dict: {top_level}. "
              f"Inspect {debug_json_path} to find the actual field name, then it can be added to the search list above.")

    with child_run("data_drift_audit", parent_run_id):
        mlflow.log_param("split_method", "positional_70_30")
        mlflow.log_param("drift_method", "wasserstein")
        mlflow.log_metric("n_reference_rows", len(reference_data))
        mlflow.log_metric("n_current_rows", len(current_data))
        if dataset_drift is not None:
            mlflow.log_metric("dataset_drift", int(bool(dataset_drift)))
        if share_drifted is not None:
            mlflow.log_metric("share_of_drifted_columns", float(share_drifted))
        mlflow.log_artifact(report_path)

def task_final_fraud_fusion(w_tabular: float = 0.60, w_gnn: float = 0.40, risk_threshold: float = 0.5, **context):
    """
    Combines the tabular ensemble's weighted-vote score with the GNN's
    structural risk score into one final fraud prediction.

    Requires BOTH branches to have already completed:
      - task_evaluate_ensemble          -> data/ensemble_evaluation_output.csv
      - task_train_gnn_model            -> trained weights in models/gnn/

    This is the one place the two branches are intentionally allowed to
    converge, since fusion is the explicit point of this task.
    """
    print("Executing Airflow Task: Fusing Tabular Ensemble Vote with GNN Structural Risk...")

    parent_run_id = _get_parent_run_id(context)

    if not os.path.exists(ENSEMBLE_CSV_PATH):
        raise FileNotFoundError(
            f"Missing {ENSEMBLE_CSV_PATH}. Ensure task_evaluate_ensemble ran successfully."
        )
    df_tabular = pd.read_csv(ENSEMBLE_CSV_PATH)

    print("Running single full-graph GNN inference pass for structural risk scores...")
    df_gnn = compute_gnn_risk_scores(path_prefix=PROCESSED_DIR, model_dir=MODELS_GNN_DIR)
    gnn_lookup_path = os.path.join(GNN_OUTPUT_DIR, "gnn_risk_lookup.parquet")
    df_gnn[["file_id", "gnn_risk_score"]].to_parquet(gnn_lookup_path, index=False)
    print(f"GNN risk lookup snapshot saved for model registration: {gnn_lookup_path}")

    if "file_id" not in df_tabular.columns:
        raise ValueError(
            f"{ENSEMBLE_CSV_PATH} is missing file_id -- ensure task_prepare_data selects it "
            "and task_evaluate_ensemble includes it in target_columns."
        )
    if "file_id" not in df_gnn.columns:
        raise ValueError(
            "gnn_inference.compute_gnn_risk_scores() did not return file_id -- "
            "check final_graph_file_nodes carries file_id (see gnn_data_prep.py)."
        )

    # ------------------------------------------------------------------
    # file_id is the join key: it's the one identifier both branches derive
    # from the same `files` table primary key. numero_dossier is a
    # business-formatted code (e.g. "DV000002") that the graph branch does
    # not reliably have access to -- do not join on it (see gnn_data_prep.py
    # and gnn_inference.py for why an earlier attempt to substitute file_id
    # for numero_dossier produced a key that could never match).
    # ------------------------------------------------------------------
    df_tabular["file_id"] = pd.to_numeric(df_tabular["file_id"], errors="coerce").astype("Int64")
    df_gnn["file_id"] = pd.to_numeric(df_gnn["file_id"], errors="coerce").astype("Int64")

    overlap = set(df_tabular["file_id"].dropna()) & set(df_gnn["file_id"].dropna())
    print(f"file_id overlap between tabular and GNN outputs: {len(overlap)} / {df_tabular['file_id'].nunique()} tabular records")
    if len(overlap) == 0:
        print("⚠️ Zero overlap on file_id -- check that both branches are reading from the "
              "same underlying files table / DB, and that neither side is silently "
              "substituting a different id column. Sample values -- "
              f"tabular: {df_tabular['file_id'].head(3).tolist()}, "
              f"gnn: {df_gnn['file_id'].head(3).tolist()}")

    merge_cols = ["file_id", "gnn_risk_score"]
    df_final = df_tabular.merge(df_gnn[merge_cols], on="file_id", how="left")

    # Files with no scoreable file/user edge get a neutral 0.5 prior rather
    # than silently being treated as low-risk (0.0) or dropped.
    n_missing_gnn = df_final["gnn_risk_score"].isna().sum()
    if n_missing_gnn:
        print(f"⚠️ {n_missing_gnn} file(s) had no GNN edge to score -- defaulting to neutral risk 0.5.")
    df_final["gnn_risk_score"] = df_final["gnn_risk_score"].fillna(0.5)

    df_final["final_fraud_score"] = (
        df_final["ensemble_risk_score"] * w_tabular + df_final["gnn_risk_score"] * w_gnn
    )
    df_final["final_verdict"] = np.where(
        df_final["final_fraud_score"] >= risk_threshold,
        "🚨 HIGH RISK - AUDIT",
        "✅ PASS (BENIGN)",
    )

    df_final.sort_values("final_fraud_score", ascending=False).to_csv(FINAL_PREDICTIONS_PATH, index=False)
    n_flagged = int((df_final["final_verdict"] == "🚨 HIGH RISK - AUDIT").sum())
    print(f"Final fused fraud predictions saved to: {FINAL_PREDICTIONS_PATH}")
    print(f"Flagged for audit: {n_flagged} / {len(df_final)}")

    with child_run("final_fraud_fusion", parent_run_id):
        mlflow.log_param("w_tabular", w_tabular)
        mlflow.log_param("w_gnn", w_gnn)
        mlflow.log_param("risk_threshold", risk_threshold)
        mlflow.log_metric("file_id_overlap", len(overlap))
        mlflow.log_metric("n_missing_gnn_score", int(n_missing_gnn))
        mlflow.log_metric("n_flagged_for_audit", n_flagged)
        mlflow.log_metric("flag_rate", n_flagged / len(df_final))
        mlflow.log_metric("mean_final_fraud_score", float(df_final["final_fraud_score"].mean()))
        mlflow.log_artifact(FINAL_PREDICTIONS_PATH)


def task_register_champion_model(w_tabular: float = 0.60, w_gnn: float = 0.40,
                                  risk_threshold: float = 0.5, **context):
    """
    Packages this run's trained artifacts into one versioned MLflow pyfunc
    model, scores it with a composite of already-logged proxy metrics, and
    promotes it to the 'champion' alias only if it beats the current
    champion's stored score. Must run before task_cleanup_temp_buffer,
    since it needs EXTRACTED_FEATURES_PATH to freeze the AE/XGB thresholds.
    """
    import json
    import joblib
    import numpy as np
    import torch
    import mlflow
    from mlflow.tracking import MlflowClient
    from tabular_models import OptunaAutoencoder
    from model_registry_utils import (
        compute_composite_score, get_child_run_metrics,
        MODEL_NAME, ALIAS_CHAMPION, FraudFusionPyfuncModel,
    )

    parent_run_id = _get_parent_run_id(context)
    if parent_run_id is None:
        print("⚠️ No parent_run_id -- skipping model registration.")
        return

    if not os.path.exists(EXTRACTED_FEATURES_PATH):
        print(f"⚠️ {EXTRACTED_FEATURES_PATH} missing -- cannot freeze thresholds. Skipping registration.")
        return

    gnn_lookup_path = os.path.join(GNN_OUTPUT_DIR, "gnn_risk_lookup.parquet")
    if not os.path.exists(gnn_lookup_path):
        print(f"⚠️ {gnn_lookup_path} missing -- ensure task_final_fraud_fusion ran first. Skipping registration.")
        return

    local_df = pd.read_pickle(EXTRACTED_FEATURES_PATH)
    feature_matrix = np.array([v.toArray() for v in local_df["scaled_features"]])

    ae_arch_path = os.path.join(PROJECT_DIR, "models", "autoencoder", "best_autoencoder_arch.json")
    ae_weights_path = os.path.join(PROJECT_DIR, "models", "autoencoder", "best_autoencoder_weights.pth")
    with open(ae_arch_path) as f:
        ae_arch = json.load(f)
    autoencoder = OptunaAutoencoder(**ae_arch)
    autoencoder.load_state_dict(torch.load(ae_weights_path, map_location="cpu"))
    autoencoder.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(feature_matrix)
        errors = torch.mean((X_t - autoencoder(X_t)) ** 2, dim=1).numpy()
    ae_error_threshold = float(np.percentile(errors, 99))

    xgb_path = os.path.join(PROJECT_DIR, "models", "xgboost_unsupervised", "best_xgb_anomaly_model.joblib")
    xgb_model = joblib.load(xgb_path)
    xgb_scores = xgb_model.predict_proba(feature_matrix)[:, 1]
    xgb_score_threshold = float(np.percentile(xgb_scores, 99))

    # GNN encoder/predictor weights -- attached to the registered model so
    # the triage engine can resolve them straight from the registry
    # (models:/MODEL_NAME@champion) instead of a hardcoded local path.
    gnn_encoder_path = os.path.join(PROJECT_DIR, "models", "gnn", "hetero_gnn_encoder.pt")
    gnn_predictor_path = os.path.join(PROJECT_DIR, "models", "gnn", "link_predictor.pt")
    if not os.path.exists(gnn_encoder_path) or not os.path.exists(gnn_predictor_path):
        print(f"⚠️ GNN weights missing at {gnn_encoder_path} / {gnn_predictor_path} -- skipping registration.")
        return

    fusion_config = {
        "w_tabular": w_tabular, "w_gnn": w_gnn, "risk_threshold": risk_threshold,
        "ae_error_threshold": ae_error_threshold, "xgb_score_threshold": xgb_score_threshold,
    }
    fusion_config_path = os.path.join(TMP_DIR, "fusion_config.json")
    with open(fusion_config_path, "w") as f:
        json.dump(fusion_config, f, indent=2)

    metrics_by_component = {
        "ocsvm": get_child_run_metrics(parent_run_id, "sgd_one_class_svm"),
        "iforest": get_child_run_metrics(parent_run_id, "isolation_forest"),
        "autoencoder": get_child_run_metrics(parent_run_id, "pytorch_autoencoder"),
        "hbos": get_child_run_metrics(parent_run_id, "hbos"),
        "xgboost": get_child_run_metrics(parent_run_id, "unsupervised_xgboost"),
        "gnn": get_child_run_metrics(parent_run_id, "gnn_hetero_link_predictor"),
    }
    composite_score = compute_composite_score(metrics_by_component)
    print(f"Composite quality score: {composite_score:.4f}")

    artifacts = {
        "ocsvm": os.path.join(PROJECT_DIR, "models", "one_class_svm", "best_ocsvm_model.joblib"),
        "iforest": os.path.join(PROJECT_DIR, "models", "isolation_forest", "best_iforest_model.joblib"),
        "autoencoder": ae_weights_path,
        "ae_arch": ae_arch_path,
        "hbos": os.path.join(PROJECT_DIR, "models", "hbos", "best_hbos_model.joblib"),
        "xgb": xgb_path,
        "gnn_lookup": gnn_lookup_path,
        "fusion_config": fusion_config_path,
        "gnn_encoder": gnn_encoder_path,
        "gnn_predictor": gnn_predictor_path,
    }

    client = MlflowClient()
    with mlflow.start_run(run_name="fusion_model_registration",
                           tags={"mlflow.parentRunId": parent_run_id}) as run:
        mlflow.log_metric("composite_quality_score", composite_score)
        mlflow.log_dict(metrics_by_component, "component_metrics.json")
        mlflow.pyfunc.log_model(
            artifact_path="fusion_model",
            python_model=FraudFusionPyfuncModel(),
            artifacts=artifacts,
            pip_requirements=["torch", "scikit-learn", "xgboost", "pyod", "pandas", "numpy"],
        )
        mv = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/fusion_model", name=MODEL_NAME)

    dag_run = context.get("dag_run")
    client.set_model_version_tag(MODEL_NAME, mv.version, "composite_quality_score", str(composite_score))
    client.set_model_version_tag(MODEL_NAME, mv.version, "dag_run_id",
                                  dag_run.run_id if dag_run else "standalone")

    champion_score = None
    try:
        champion_mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
        champion_score = float(champion_mv.tags.get("composite_quality_score", "-inf"))
    except Exception:
        pass  # no champion registered yet

    if champion_score is None or composite_score > champion_score:
        client.set_registered_model_alias(MODEL_NAME, ALIAS_CHAMPION, mv.version)
        print(f"🏆 v{mv.version} promoted to '{ALIAS_CHAMPION}' ({composite_score:.4f} > {champion_score}).")
    else:
        print(f"v{mv.version} registered, NOT promoted ({composite_score:.4f} <= champion {champion_score:.4f}).")