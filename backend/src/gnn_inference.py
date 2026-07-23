"""
gnn_inference.py

Computes structural fraud-risk scores from the trained Hetero-GNN for every
file in the graph, in a single full-graph forward pass.

This intentionally replaces an earlier approach that:
  - capped inference at a hardcoded FILE_CAPACITY, silently dropping files
    past that index
  - rebuilt a fresh single-edge subgraph per file and re-ran the encoder for
    every row, which both defeats the purpose of neighborhood aggregation
    (SAGEConv on a 1-edge graph has almost nothing to aggregate) and is far
    slower than necessary
  - swallowed inference errors into a neutral 0.5 score, hiding failures

Reuses HeteroGNNEncoder / LinkPredictor from gnn_models.py rather than a
separate SAGEGNN.py copy, so architecture changes only need to happen in
one place.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from gnn_models import HeteroGNNEncoder, LinkPredictor

# Anchor defaults to this script's own directory rather than the caller's
# cwd -- see the note in pipeline_tasks.py's PROJECT_DIR block for why bare
# relative paths are unsafe across Airflow task subprocesses.
# 1. Step up 1 level from /backend/src to reach /backend
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Define data and models paths relative to /backend
_DEFAULT_PATH_PREFIX = os.path.join(BACKEND_DIR, "data", "processed")
_DEFAULT_MODEL_DIR = os.path.join(BACKEND_DIR, "models", "gnn")


def compute_gnn_risk_scores(
    path_prefix: str = None,
    model_dir: str = None,
    hidden_channels: int = 64,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        file_id, numero_dossier, gnn_risk_score  (0-1, higher = riskier)

    gnn_risk_score is derived from the trained link predictor's confidence
    that a file's assigned_to-user edge is genuine: low predicted affinity
    is treated as structurally anomalous (1 - probability), then min-max
    normalized across the current batch.
    """
    path_prefix = path_prefix or _DEFAULT_PATH_PREFIX
    model_dir = model_dir or _DEFAULT_MODEL_DIR

    df_u = pd.read_parquet(os.path.join(path_prefix, "final_graph_user_nodes")).sort_values("user_gnn_idx")
    df_c = pd.read_parquet(os.path.join(path_prefix, "final_graph_company_nodes")).sort_values("company_gnn_idx")
    df_f = pd.read_parquet(os.path.join(path_prefix, "final_graph_file_nodes")).sort_values("file_gnn_idx")
    edges_user = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
    edges_comp = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_company"))

    if edges_user.empty:
        raise ValueError(
            "final_edge_file_to_user is empty -- no file/user assignment edges to score. "
            "Check that gnn_data_prep.cross_link_graph_spaces() produced valid edges."
        )

    # --------------------------------------------------------------
    # Rebuild the SAME graph construction used in training (gnn_models.py):
    # ToUndirected() applied once, before any splitting/inference, so the
    # metadata this graph produces matches what the saved encoder expects.
    # --------------------------------------------------------------
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

    data = T.ToUndirected()(data)

    # --------------------------------------------------------------
    # Rebuild architecture and load trained weights
    # --------------------------------------------------------------
    base_encoder = HeteroGNNEncoder(hidden_channels=hidden_channels)
    encoder = to_hetero(base_encoder, data.metadata(), aggr='sum')
    predictor = LinkPredictor()

    encoder.load_state_dict(torch.load(os.path.join(model_dir, "hetero_gnn_encoder.pt"), map_location='cpu'))
    predictor.load_state_dict(torch.load(os.path.join(model_dir, "link_predictor.pt"), map_location='cpu'))
    encoder.eval()
    predictor.eval()

    # --------------------------------------------------------------
    # Single full-graph forward pass -- every file/user edge scored at once
    # --------------------------------------------------------------
    with torch.no_grad():
        embeddings = encoder(data.x_dict, data.edge_index_dict)
        edge_index = data['file', 'assigned_to', 'user'].edge_index
        probs = predictor(embeddings['file'], embeddings['user'], edge_index).cpu().numpy()

    src_files = edge_index[0].cpu().numpy()
    gnn_raw_risk = 1.0 - probs  # low predicted affinity -> high structural risk

    df_scores = pd.DataFrame({'file_gnn_idx': src_files, 'gnn_raw_risk': gnn_raw_risk})
    # A file can have more than one assignment edge historically; keep the
    # single riskiest (max) score rather than averaging it away.
    df_scores = df_scores.groupby('file_gnn_idx', as_index=False)['gnn_raw_risk'].max()

    id_col = 'file_id' if 'file_id' in df_f.columns else 'id'
    join_cols = ['file_gnn_idx', id_col]
    has_numero_dossier = 'numero_dossier' in df_f.columns
    if has_numero_dossier:
        join_cols.append('numero_dossier')

    df_scores = df_scores.merge(df_f[join_cols], on='file_gnn_idx', how='left')
    if id_col != 'file_id':
        df_scores.rename(columns={id_col: 'file_id'}, inplace=True)

    # Min-max normalize to [0, 1]
    lo, hi = df_scores['gnn_raw_risk'].min(), df_scores['gnn_raw_risk'].max()
    if hi > lo:
        df_scores['gnn_risk_score'] = (df_scores['gnn_raw_risk'] - lo) / (hi - lo)
    else:
        # All scores identical (e.g. tiny graph) -- no signal to normalize
        df_scores['gnn_risk_score'] = 0.0

    # file_id is the reliable join key back to the tabular branch (both
    # branches derive it from the same `files` table). numero_dossier is
    # included when available for readability but should not be relied on
    # for joining -- see the note in gnn_data_prep.py.
    keep_cols = ['file_id', 'gnn_risk_score']
    if has_numero_dossier:
        keep_cols.insert(1, 'numero_dossier')

    return df_scores[keep_cols]


if __name__ == "__main__":
    scores = compute_gnn_risk_scores()
    print(f"Computed GNN risk scores for {len(scores)} files.")
    print(scores.sort_values('gnn_risk_score', ascending=False).head(10).to_string(index=False))