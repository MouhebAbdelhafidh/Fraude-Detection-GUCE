
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mlflow
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero, LayerNorm
from SAGEGNN import HeteroGNNEncoder, LinkPredictor

from mlops.mlflow_utils import child_run

# ------------------------------------------------------------------
# PATH ANCHORING: same reasoning as pipeline_tasks.py / tabular_models.py --
# Airflow's Task SDK (and the standalone `python gnn_models.py` entrypoint,
# possibly run from a different venv/cwd via gnn_tasks.py's BashOperator)
# give no guarantee about os.getcwd(). Every read/write path below is
# built from PROJECT_DIR, never a bare relative string -- a bare
# "models/gnn/..." string caused the same class of FileNotFoundError as
# the autoencoder's arch.json bug in tabular_models.py.
# ------------------------------------------------------------------
import os

# Get /backend directory (stepping 1 level up from /backend/src)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define subdirectories relative to /backend
MODELS_GNN_DIR = os.path.join(BACKEND_DIR, "src", "models", "gnn") # Adjust "src" if models/ is directly in backend/
DATA_PROCESSED_DIR = os.path.join(BACKEND_DIR, "data", "processed")
GNN_OUTPUT_DIR = os.path.join(BACKEND_DIR, "data", "gnn_output")

# Ensure target output directories exist
os.makedirs(MODELS_GNN_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(GNN_OUTPUT_DIR, exist_ok=True)


# ======================================================================
# 2. ISOLATED PIPELINE ENTRYPOINT
# ======================================================================
def train_hetero_gnn(local_df: pd.DataFrame, n_epochs: int = 40, parent_run_id: str = None) -> pd.DataFrame:
    """
    Trains the Heterogeneous Graph Neural Network using pure graph storage files.
    Appends anomaly scoring columns directly onto the mapped node space.
    """
    print("\n[MODEL 6/6] Initiating Heterogeneous Graph Neural Network Pipeline...")
    os.makedirs(MODELS_GNN_DIR, exist_ok=True)

    device = torch.device('cpu')
    path_prefix = DATA_PROCESSED_DIR

    print("📦 Constructing HeteroData graph structures from Parquet node layers...")
    try:
        df_u = pd.read_parquet(os.path.join(path_prefix, "final_graph_user_nodes")).sort_values("user_gnn_idx")
        df_c = pd.read_parquet(os.path.join(path_prefix, "final_graph_company_nodes")).sort_values("company_gnn_idx")
        df_f = pd.read_parquet(os.path.join(path_prefix, "final_graph_file_nodes")).sort_values("file_gnn_idx")
        edges_user = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
        edges_comp = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_company"))

        data = HeteroData()
        data['user'].x = torch.tensor(np.stack(df_u['user_features'].values), dtype=torch.float)
        data['company'].x = torch.tensor(np.stack(df_c['company_features'].values), dtype=torch.float)
        data['file'].x = torch.tensor(np.stack(df_f['file_features'].values), dtype=torch.float)

        data['file', 'assigned_to', 'user'].edge_index = torch.tensor(
            edges_user[['source_file_idx', 'target_user_idx']].values, dtype=torch.long
        ).t().contiguous()

        data['file', 'linked_to', 'company'].edge_index = torch.tensor(
            edges_comp[['source_file_idx', 'target_company_idx']].values, dtype=torch.long
        ).t().contiguous()

        # --------------------------------------------------------------
        # FIX: Convert to undirected ONCE, before splitting. This makes
        # the ('user', 'rev_assigned_to', 'file') relation actually exist
        # ahead of time (so RandomLinkSplit's rev_edge_types argument is
        # meaningful), and guarantees that every downstream view of this
        # graph (train_data, val_data, test_data, and the full graph used
        # for inference) shares the exact same set/order of edge types.
        # --------------------------------------------------------------
        data = T.ToUndirected()(data)

    except Exception as e:
        print(f"⚠️ Graph instantiation failed: {e}. Falling back to clean zero-state predictions.")
        local_df["is_anomaly_gnn"] = 0
        return local_df

    # Slicing relationships into Train / Val / Test subsets
    link_splitter = T.RandomLinkSplit(
        num_val=0.10, num_test=0.10, neg_sampling_ratio=1.0,
        edge_types=('file', 'assigned_to', 'user'), add_negative_train_samples=True,
        rev_edge_types=('user', 'rev_assigned_to', 'file')
    )
    train_data, val_data, test_data = link_splitter(data)

    # Initialize Architecture
    hidden_dim = 64
    base_encoder = HeteroGNNEncoder(hidden_channels=hidden_dim)
    encoder = to_hetero(base_encoder, train_data.metadata(), aggr='sum').to(device)
    predictor = LinkPredictor().to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.001)
    criterion = torch.nn.BCELoss()

    with child_run("gnn_hetero_link_predictor", parent_run_id):
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("lr", 0.001)
        mlflow.log_param("aggr", "sum")
        mlflow.log_param("num_val", 0.10)
        mlflow.log_param("num_test", 0.10)

        # Optimization Loop
        print(f"🏋️ Optimizing Graph Embeddings across {n_epochs} epochs...")
        for epoch in range(1, n_epochs + 1):
            encoder.train()
            predictor.train()
            optimizer.zero_grad()

            batch_data = train_data.to(device)
            node_embeddings = encoder(batch_data.x_dict, batch_data.edge_index_dict)
            predictions = predictor(node_embeddings['file'], node_embeddings['user'], batch_data['file', 'assigned_to', 'user'].edge_label_index)
            ground_truth = batch_data['file', 'assigned_to', 'user'].edge_label.float()

            loss = criterion(predictions, ground_truth)
            loss.backward()
            optimizer.step()

            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # Serialize trained weights -- anchored to PROJECT_DIR/models/gnn,
        # never a bare "models/gnn/..." string (see module-level comment).
        encoder_path = os.path.join(MODELS_GNN_DIR, "hetero_gnn_encoder.pt")
        predictor_path = os.path.join(MODELS_GNN_DIR, "link_predictor.pt")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(predictor.state_dict(), predictor_path)
        print(f"✅ GNN model artifacts successfully saved to: {MODELS_GNN_DIR}")

        # Inference Loop
        encoder.eval()
        predictor.eval()
        with torch.no_grad():
            full_data = data.clone().to(device)

            assert full_data.metadata() == train_data.metadata(), (
                "full_data and train_data metadata mismatch: "
                f"{full_data.metadata()} vs {train_data.metadata()}"
            )

            embeddings = encoder(full_data.x_dict, full_data.edge_index_dict)

            edges_to_evaluate = full_data['file', 'assigned_to', 'user'].edge_index
            src_files = edges_to_evaluate[0].cpu().numpy()

            probs = predictor(embeddings['file'], embeddings['user'], edges_to_evaluate).cpu().numpy()

            gnn_anomalies = np.zeros(full_data['file'].x.size(0))
            for idx, f_idx in enumerate(src_files):
                if f_idx < len(gnn_anomalies) and probs[idx] < 0.15:
                    gnn_anomalies[f_idx] = 1

            if len(local_df) == len(gnn_anomalies):
                local_df["is_anomaly_gnn"] = gnn_anomalies.astype(int)
            else:
                local_df["is_anomaly_gnn"] = np.resize(gnn_anomalies, len(local_df)).astype(int)

        n_anomalies = int(local_df['is_anomaly_gnn'].sum())
        print(f"Total Structural Anomalies Tagged by GNN: {n_anomalies}")

        mlflow.log_metric("final_train_loss", loss.item())
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", float(local_df['is_anomaly_gnn'].mean()))
        mlflow.log_metric("mean_link_prob", float(probs.mean()))
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(predictor_path)

    return local_df


if __name__ == "__main__":
    # Allows `python gnn_models.py` to be run standalone (matches the
    # command built in gnn_tasks.py's get_gnn_training_cmd()).
    path_prefix = DATA_PROCESSED_DIR

    file_nodes_path = os.path.join(path_prefix, "final_graph_file_nodes")
    if not os.path.exists(file_nodes_path):
        raise FileNotFoundError(
            f"Missing processed graph files at {file_nodes_path}. "
            "Run gnn_data_prep.py --preprocess / --synthesis first."
        )

    graph_files_df = pd.read_parquet(file_nodes_path).sort_values("file_gnn_idx")

    # When launched by gnn_tasks.py's get_gnn_training_cmd() (i.e. as an
    # Airflow BashOperator subprocess in the GNN venv), MLFLOW_PARENT_RUN_ID
    # is exported so this run nests under the pipeline-root run created by
    # task_prepare_data -- XCom itself isn't reachable from this subprocess,
    # so the env var is the hand-off point. Falls back to an un-nested run
    # (parent_run_id=None) when run standalone, e.g. `python gnn_models.py`.
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID") or None
    updated_graph_df = train_hetero_gnn(local_df=graph_files_df, n_epochs=40, parent_run_id=parent_run_id)

    os.makedirs(GNN_OUTPUT_DIR, exist_ok=True)
    gnn_output_path = os.path.join(GNN_OUTPUT_DIR, "gnn_predictions.parquet")
    updated_graph_df[["numero_dossier", "file_gnn_idx", "is_anomaly_gnn"]].to_parquet(gnn_output_path, index=False)
    print(f"GNN Modeling complete. Independent predictions saved to: {gnn_output_path}")