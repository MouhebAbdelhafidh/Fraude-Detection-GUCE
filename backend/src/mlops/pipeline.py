import os
import sys
from pathlib import Path
import numpy as np
from pyspark.sql import SparkSession

# --- 1. Dynamic Path Setup ---
# Current file: /backend/src/mlops/script.py
# BACKEND_SRC_DIR: /backend/src
BACKEND_SRC_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC_DIR))

# --- 2. Java & Environment Safety Setup ---
os.environ["JAVA_HOME"] = "/usr/lib/jvm/default-java"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")
os.environ["JDK_JAVA_OPTIONS"] = (
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED"
)

# --- 3. Imports from /backend/src ---
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

def run_mlops_pipeline():
    windows_host_ip = os.popen("ip route | grep default | awk '{print $3}'").read().strip()
    url = f"jdbc:postgresql://{windows_host_ip}:5432/guce_siat_cc"
    properties = {
        "user": "postgres",
        "password": "nobodycan@2H",
        "driver": "org.postgresql.Driver"
    }

    # Path to postgresql jar located at root '/'
    postgres_jar_path = "/postgresql-42.7.2.jar"

    spark = SparkSession.builder \
        .appName("PostgresMLOpsDataPrepPipeline") \
        .config("spark.jars", postgres_jar_path) \
        .config("spark.driver.extraJavaOptions", os.environ["JDK_JAVA_OPTIONS"]) \
        .config("spark.executor.extraJavaOptions", os.environ["JDK_JAVA_OPTIONS"]) \
        .getOrCreate()

    try:
        # 1. Run Tabular Prep Engine
        df_scaled, fitted_transformer = prepare_tabular_data(spark, url, properties, export_parquet=True)
        
        # 2. Run Graph Structural Generation Sequences
        print("\n🌐 Forking execution into structural Graph synthesis blocks...")
        cross_link_graph_spaces()
        generate_pyg_tensors()

        # 3. Handle Local tabular downstream extraction matrices
        print("\nExtracting Spark geometric feature profiles into NumPy space...")
        local_df = df_scaled.select("numero_dossier", "code", "valeur_fob", "ip_address", "scaled_features").toPandas()
        feature_matrix = np.array([v.toArray() for v in local_df["scaled_features"]])
        input_dim = feature_matrix.shape[1]
        
        # Load and verify tabular models
        loaded_models_dict = load_saved_models(input_dim=input_dim)
        evaluated_df = evaluate_ensemble_anomalies(loaded_models_dict, feature_matrix, local_df)
        
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "ensemble_evaluation_output.csv")
        evaluated_df[["numero_dossier", "code", "valeur_fob", "ensemble_risk_score", "ensemble_consensus_verdict"]].to_csv(output_csv_path, index=False)
        print(f"Ensemble batch evaluation scores saved safely to: {output_csv_path}")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    run_mlops_pipeline()