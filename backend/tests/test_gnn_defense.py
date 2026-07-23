import os
import sys

# ======================================================================
# 1. DYNAMIC SYSTEM PATH SETUP (MUST RUN BEFORE LOCAL IMPORTS)
# ======================================================================
# Resolves root directory (/home/mouhe/MLopsPipeline)
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ======================================================================
# 2. STANDARD & THIRD-PARTY IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

# ======================================================================
# 3. PROJECT-SPECIFIC IMPORTS
# ======================================================================
from SAGEGNN import HeteroGNNEncoder, LinkPredictor
from adversarial.attacks.gnn_attack import GNNAdversarialAttack
from adversarial.defenses.gnn_defense import GNNAdversarialShield

# ======================================================================
# 4. ABSOLUTE PATH CONSTANTS
# ======================================================================
MODELS_GNN_DIR = os.path.join(PROJECT_DIR, "models", "gnn")
DATA_PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")


def test_gnn_pipeline():
    print("\n🛡️ Testing GNN Attack and Defense Pipeline...")
    device = torch.device("cpu")

    # 1. Load Processed HeteroData Graph
    print("📦 Reconstructing HeteroData graph structure...")
    df_u = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "final_graph_user_nodes")).sort_values("user_gnn_idx")
    df_c = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "final_graph_company_nodes")).sort_values("company_gnn_idx")
    df_f = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "final_graph_file_nodes")).sort_values("file_gnn_idx")
    edges_user = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "final_edge_file_to_user"))
    edges_comp = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, "final_edge_file_to_company"))

    data = HeteroData()
    data["user"].x = torch.tensor(np.stack(df_u["user_features"].values), dtype=torch.float)
    data["company"].x = torch.tensor(np.stack(df_c["company_features"].values), dtype=torch.float)
    data["file"].x = torch.tensor(np.stack(df_f["file_features"].values), dtype=torch.float)

    data["file", "assigned_to", "user"].edge_index = torch.tensor(
        edges_user[["source_file_idx", "target_user_idx"]].values, dtype=torch.long
    ).t().contiguous()

    data["file", "linked_to", "company"].edge_index = torch.tensor(
        edges_comp[["source_file_idx", "target_company_idx"]].values, dtype=torch.long
    ).t().contiguous()

    data = T.ToUndirected()(data)

    # 2. Load Trained GNN Artifacts
    encoder_path = os.path.join(MODELS_GNN_DIR, "hetero_gnn_encoder.pt")
    predictor_path = os.path.join(MODELS_GNN_DIR, "link_predictor.pt")

    if not os.path.exists(encoder_path) or not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Missing GNN artifacts at {MODELS_GNN_DIR}. Ensure model weights exist.")

    hidden_dim = 64
    base_encoder = HeteroGNNEncoder(hidden_channels=hidden_dim)
    encoder = to_hetero(base_encoder, data.metadata(), aggr="sum").to(device)
    predictor = LinkPredictor().to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    encoder.eval()
    predictor.eval()

    # 3. Clean Graph Baseline
    with torch.no_grad():
        clean_emb = encoder(data.x_dict, data.edge_index_dict)
        eval_edges = data["file", "assigned_to", "user"].edge_index
        clean_probs = predictor(clean_emb["file"], clean_emb["user"], eval_edges).cpu().numpy()
        clean_anomalies = int((clean_probs < 0.15).sum())

    print(f"✅ Clean Baseline Anomalies: {clean_anomalies}")

    # 4. Execute Attack
    attacker = GNNAdversarialAttack(epsilon=0.25, drop_rate=0.10)
    attacked_data = attacker.execute_full_attack(data, target_node_type="file")

    with torch.no_grad():
        attacked_emb = encoder(attacked_data.x_dict, attacked_data.edge_index_dict)
        attacked_probs = predictor(attacked_emb["file"], attacked_emb["user"], eval_edges).cpu().numpy()
        attacked_anomalies = int((attacked_probs < 0.15).sum())

    print(f"⚠️ Under Adversarial Attack -> Anomalies Detected: {attacked_anomalies} (Shift: {attacked_anomalies - clean_anomalies})")

    # 5. Execute Defense Shield
    shield = GNNAdversarialShield(noise_std=0.01, feature_squeeze_bits=6)    
    defended_data, adv_flags = shield.defend_and_evaluate(attacked_data)

    with torch.no_grad():
        defended_emb = encoder(defended_data.x_dict, defended_data.edge_index_dict)
        defended_probs = predictor(defended_emb["file"], defended_emb["user"], eval_edges).cpu().numpy()
        defended_anomalies = int((defended_probs < 0.15).sum())

    flagged_file_nodes = int(adv_flags["file"].sum().item())
    print(f"🛡️ Post Defense Shield -> Anomalies Recovered: {defended_anomalies}")
    print(f"🚨 Shield Flagged {flagged_file_nodes} node vectors as suspicious/adversarial.")


if __name__ == "__main__":
    test_gnn_pipeline()