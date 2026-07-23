#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIAT Fraud Detection Project - Part 2: Graph Synthesis & PyTorch Geometric Tensor Pipeline
Assembles decoupled node spaces and edge arrays into an undirected HeteroData object,
and outputs verified Train/Val/Test data tensors.
"""

import os
import sys
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# CRITICAL SYSTEM ALIGNMENT: Force Working Java 17 Runtime Path
# ------------------------------------------------------------------
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.functions import vector_to_array

# PyTorch and PyG validation imports
try:
    import torch
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# ------------------------------------------------------------------
# PATH RESOLUTION: Locate project root and backend paths dynamically
# ------------------------------------------------------------------
# Script location: /backend/src/gnn_data_prep.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))               # /backend/src
BACKEND_DIR = os.path.dirname(CURRENT_DIR)                              # /backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                             # Project Root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Shared configuration dictionaries
DB_PROPERTIES = {
    "user": "postgres",
    "password": "nobodycan@2H",
    "driver": "org.postgresql.Driver"
}

# Dynamically fetch the Windows host IP address from WSL
windows_host_ip = os.popen("ip route | grep default | awk '{print $3}'").read().strip()

# Set the URL using the dynamic IP
DB_URL = f"jdbc:postgresql://{windows_host_ip}:5432/guce_siat_cc"

def get_spark_session(app_name="FraudPipelineDataPrep"):
    """Instantiates an optimized localized SparkSession with explicit environment safety."""
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ.get("PATH", "")

    jar_path = os.path.join(PROJECT_ROOT, "postgresql-42.7.2.jar")

    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.jars", jar_path) \
        .getOrCreate()

def cross_link_graph_spaces():
    print("🏗️ Synthesizing node spaces and multi-relational edges...")
    spark = get_spark_session()
    
    path_prefix = os.path.join(BACKEND_DIR, "data", "processed")
    
    # ------------------------------------------------------------------
    # 1. LOAD STAGED FEATURE ARRAYS FROM PARQUET LAYER
    # ------------------------------------------------------------------
    df_users_src = spark.read.parquet(os.path.join(path_prefix, "gnn_users.parquet"))
    df_companies_src = spark.read.parquet(os.path.join(path_prefix, "gnn_companies.parquet"))
    df_files_src = spark.read.parquet(os.path.join(path_prefix, "files_graph_nodes.parquet"))

    # ------------------------------------------------------------------
    # 2. ISOLATE AND INDEX UNIQUE ENTITY SPACES
    # ------------------------------------------------------------------
    # A. Clean User Nodes
    df_u_unique = df_users_src.select(
        F.col("id").alias("user_id"), "role", "user_features"
    ).dropDuplicates(["user_id"])

    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_gnn_idx", handleInvalid="keep")
    df_users_final = user_indexer.fit(df_u_unique).transform(df_u_unique) \
        .withColumn("user_gnn_idx", F.col("user_gnn_idx").cast("long")) \
        .withColumn("user_id", F.col("user_id").cast("long"))

    if "vector" in str(df_users_final.schema["user_features"].dataType).lower():
        df_users_final = df_users_final.withColumn("user_features", vector_to_array("user_features"))

    # B. Clean Company Nodes
    df_c_unique = df_companies_src.select(
        F.col("id").alias("company_id"), "company_name", "company_features"
    ).dropDuplicates(["company_id"])

    company_indexer = StringIndexer(inputCol="company_id", outputCol="company_gnn_idx", handleInvalid="keep")
    df_companies_final = company_indexer.fit(df_c_unique).transform(df_c_unique) \
        .withColumn("company_gnn_idx", F.col("company_gnn_idx").cast("long")) \
        .withColumn("company_id", F.col("company_id").cast("long"))

    if "vector" in str(df_companies_final.schema["company_features"].dataType).lower():
        df_companies_final = df_companies_final.withColumn("company_features", vector_to_array("company_features"))

    # C. Clean File Nodes
    file_feat_col = "node_features" if "node_features" in df_files_src.columns else "file_features"

    if "numero_dossier" in df_files_src.columns:
        select_cols = [
            F.col("file_gnn_idx").cast("long"),
            F.col("file_id").cast("long"),
            F.col("numero_dossier").cast("string").alias("numero_dossier"),
            F.col("file_type_id").cast("long"),
            F.coalesce(F.col("assigned_user_id").cast("long"), F.lit(-1)).alias("assigned_user_id"),
            F.col(file_feat_col).alias("file_features"),
        ]
    else:
        print("ℹ️ 'numero_dossier' not found in files_graph_nodes.parquet source -- "
              "graph output will carry file_id only; fusion will join on file_id.")
        select_cols = [
            F.col("file_gnn_idx").cast("long"),
            F.col("file_id").cast("long"),
            F.col("file_type_id").cast("long"),
            F.coalesce(F.col("assigned_user_id").cast("long"), F.lit(-1)).alias("assigned_user_id"),
            F.col(file_feat_col).alias("file_features"),
        ]

    df_files_final = df_files_src.select(*select_cols)

    if "vector" in str(df_files_final.schema["file_features"].dataType).lower():
        df_files_final = df_files_final.withColumn("file_features", vector_to_array("file_features"))

    # ------------------------------------------------------------------
    # 3. EXTRACT INTERLINKED GRAPH RELATIONAL EDGE MAPS
    # ------------------------------------------------------------------
    df_edge_file_to_user = df_files_final.filter(F.col("assigned_user_id") != -1) \
        .join(df_users_final.select("user_id", "user_gnn_idx"), 
              df_files_final.assigned_user_id == df_users_final.user_id, how="inner") \
        .select(F.col("file_gnn_idx").alias("source_file_idx"), F.col("user_gnn_idx").alias("target_user_idx"))

    df_edge_file_to_company = df_files_final \
        .join(df_companies_final.select("company_id", "company_gnn_idx"), 
              df_files_final.file_type_id == df_companies_final.company_id, how="inner") \
        .select(F.col("file_gnn_idx").alias("source_file_idx"), F.col("company_gnn_idx").alias("target_company_idx"))

    # ------------------------------------------------------------------
    # 4. SAVE FINALIZED ALIGNED LAYER CONTEXT TO DISK
    # ------------------------------------------------------------------
    print("💾 Saving structural graph tensor elements back to disk...")
    df_users_final.write.mode("overwrite").parquet(os.path.join(path_prefix, "final_graph_user_nodes"))
    df_companies_final.write.mode("overwrite").parquet(os.path.join(path_prefix, "final_graph_company_nodes"))
    df_files_final.write.mode("overwrite").parquet(os.path.join(path_prefix, "final_graph_file_nodes"))

    df_edge_file_to_user.write.mode("overwrite").parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
    df_edge_file_to_company.write.mode("overwrite").parquet(os.path.join(path_prefix, "final_edge_file_to_company"))

    print("=" * 75)
    print("🚀 STRUCTURAL INTERLINKING COMPLETE!")
    print(f"• User Nodes Aligned:    {df_users_final.count()}")
    print(f"• Company Nodes Aligned: {df_companies_final.count()}")
    print(f"• File Nodes Aligned:    {df_files_final.count()}")
    print(f"• Assignment Links Generated (User):    {df_edge_file_to_user.count()}")
    print(f"• Operational Hub Links Generated (Comp): {df_edge_file_to_company.count()}")
    print("=" * 75)
    
    spark.stop()

def generate_pyg_tensors():
    """Loads vectorized frames into a single PyG HeteroData object and generates link prediction splits."""
    if not PYG_AVAILABLE:
        print("\n⚠️ PyTorch or PyTorch Geometric is missing. Skipping PyG tensor compilation stage.")
        print("Install them using your environment block instructions to finalize serialization.")
        return

    print("\n📦 Loading processed matrices into memory for tensor processing...")
    path_prefix = os.path.join(BACKEND_DIR, "data", "processed")

    # Load and force precise contiguous sort alignment
    df_u = pd.read_parquet(os.path.join(path_prefix, "final_graph_user_nodes")).sort_values("user_gnn_idx")
    df_c = pd.read_parquet(os.path.join(path_prefix, "final_graph_company_nodes")).sort_values("company_gnn_idx")
    df_f = pd.read_parquet(os.path.join(path_prefix, "final_graph_file_nodes")).sort_values("file_gnn_idx")

    edges_user = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
    edges_comp = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_company"))

    # Form PyG object
    data = HeteroData()

    # Load node representations
    data['user'].x = torch.tensor(df_u['user_features'].tolist(), dtype=torch.float)
    data['company'].x = torch.tensor(df_c['company_features'].tolist(), dtype=torch.float)
    data['file'].x = torch.tensor(df_f['file_features'].tolist(), dtype=torch.float)

    # Compile target graph matrices [2, Edge_Count]
    data['file', 'assigned_to', 'user'].edge_index = torch.tensor(
        edges_user[['source_file_idx', 'target_user_idx']].values, dtype=torch.long
    ).t().contiguous()

    data['file', 'linked_to', 'company'].edge_index = torch.tensor(
        edges_comp[['source_file_idx', 'target_company_idx']].values, dtype=torch.long
    ).t().contiguous()

    # Apply message-passing transformations across boundaries
    data = T.ToUndirected()(data)
    print("✅ HeteroData graph structure successfully instantiated.")

    # Execute Heterogeneous Random Link Splits
    print("\n✂️ Slicing graph connections into structural Train / Val / Test target subsets...")
    link_splitter = T.RandomLinkSplit(
        num_val=0.10,
        num_test=0.10,
        neg_sampling_ratio=1.0,
        edge_types=('file', 'assigned_to', 'user'),
        add_negative_train_samples=True,
    )

    train_data, val_data, test_data = link_splitter(data)

    print("=" * 70)
    print("🎉 PYG TRAINING TENSORS READY FOR FRAUD MODELING!")
    print("=" * 70)
    print(f"• Training Set -> Edge Index Matrix Shape: {train_data['file', 'assigned_to', 'user'].edge_index.shape}")
    print(f"• Training Set -> Edge Label Tensors Shape: {train_data['file', 'assigned_to', 'user'].edge_label.shape}")
    print(f"• Validation Set -> Edge Index Matrix Shape: {val_data['file', 'assigned_to', 'user'].edge_index.shape}")
    print(f"• Test Set -> Edge Index Matrix Shape:       {test_data['file', 'assigned_to', 'user'].edge_index.shape}")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--preprocess":
            cross_link_graph_spaces()
        elif mode == "--synthesis":
            generate_pyg_tensors()
        else:
            print(f"⚠️ Unknown routing argument '{mode}'. Executing full fall-back sequencing.")
            cross_link_graph_spaces()
            generate_pyg_tensors()
    else:
        cross_link_graph_spaces()
        generate_pyg_tensors()