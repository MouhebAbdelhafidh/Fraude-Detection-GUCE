import pytest
import torch
import io
import pandas as pd
import numpy as np
from unittest.mock import patch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero

from SAGEGNN import HeteroGNNEncoder, LinkPredictor
from gnn_models import train_hetero_gnn

# =====================================================================
# 📊 FIXTURES: SCHEMA-MATCHING GENERATORS
# =====================================================================

@pytest.fixture
def sample_hetero_graph_data():
    """Generates a stable, unified undirected HeteroData object for functional compilation."""
    torch.manual_seed(42)
    data = HeteroData()
    
    num_users = 15
    num_companies = 5
    num_files = 30
    feature_dim = 64

    data['user'].x = torch.rand((num_users, feature_dim))
    data['company'].x = torch.rand((num_companies, feature_dim))
    data['file'].x = torch.rand((num_files, feature_dim))

    src_u = torch.randint(0, num_files, (45,))
    dst_u = torch.randint(0, num_users, (45,))
    data['file', 'assigned_to', 'user'].edge_index = torch.stack([src_u, dst_u], dim=0)

    src_c = torch.randint(0, num_files, (30,))
    dst_c = torch.randint(0, num_companies, (30,))
    data['file', 'linked_to', 'company'].edge_index = torch.stack([src_c, dst_c], dim=0)

    return T.ToUndirected()(data)


# =====================================================================
# 🔬 MODEL ARCHITECTURE VALIDATION DECK
# =====================================================================

def test_hetero_encoder_shape_matching(sample_hetero_graph_data):
    """Verifies that the compiled model accepts input vectors and retains space footprints."""
    hidden_dim = 64
    data = sample_hetero_graph_data
    
    base_encoder = HeteroGNNEncoder(hidden_channels=hidden_dim)
    hetero_encoder = to_hetero(base_encoder, data.metadata(), aggr='sum')
    hetero_encoder.eval()
    
    with torch.no_grad():
        embeddings = hetero_encoder(data.x_dict, data.edge_index_dict)
        
    assert 'user' in embeddings
    assert 'file' in embeddings
    assert 'company' in embeddings
    assert embeddings['file'].shape == (data['file'].num_nodes, hidden_dim)


def test_link_predictor_probabilities(sample_hetero_graph_data):
    """Ensures math layers tightly contract predictive confidences within [0.0, 1.0]."""
    data = sample_hetero_graph_data
    num_eval_edges = 10
    
    mock_file_embeddings = torch.randn((data['file'].num_nodes, 64))
    mock_user_embeddings = torch.randn((data['user'].num_nodes, 64))
    eval_edge_index = torch.randint(0, 5, (2, num_eval_edges), dtype=torch.long)
    
    predictor = LinkPredictor()
    predictor.eval()
    
    with torch.no_grad():
        probs = predictor(mock_file_embeddings, mock_user_embeddings, eval_edge_index)
        
    assert probs.shape == (num_eval_edges,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_model_device_agnosticism(sample_hetero_graph_data):
    """Ensures features and layers migrate seamlessly if target execution points change (CPU vs CUDA)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = sample_hetero_graph_data.to(device)
    
    base_encoder = HeteroGNNEncoder(hidden_channels=64).to(device)
    hetero_encoder = to_hetero(base_encoder, data.metadata(), aggr='sum')
    hetero_encoder.eval()
    
    with torch.no_grad():
        embeddings = hetero_encoder(data.x_dict, data.edge_index_dict)
    
    assert embeddings['file'].device.type == device.type


def test_model_serialization_integrity(sample_hetero_graph_data):
    """Production Save/Load Check: Ensures tracking artifacts save and restore weights identically."""
    data = sample_hetero_graph_data
    base_encoder = HeteroGNNEncoder(hidden_channels=64)
    hetero_encoder = to_hetero(base_encoder, data.metadata(), aggr='sum')
    hetero_encoder.eval()

    # Pass data once to initialize lazy layers fully before serialization
    with torch.no_grad():
        original_outputs = hetero_encoder(data.x_dict, data.edge_index_dict)

    # Save to buffer, reload into an identical blueprint architecture
    buffer = io.BytesIO()
    torch.save(hetero_encoder.state_dict(), buffer)
    buffer.seek(0)

    fresh_base_encoder = HeteroGNNEncoder(hidden_channels=64)
    fresh_hetero_encoder = to_hetero(fresh_base_encoder, data.metadata(), aggr='sum')
    
    # Initialize fresh weights before mapping state
    with torch.no_grad():
        _ = fresh_hetero_encoder(data.x_dict, data.edge_index_dict)
        
    fresh_hetero_encoder.load_state_dict(torch.load(buffer, weights_only=True))
    fresh_hetero_encoder.eval()

    with torch.no_grad():
        restored_outputs = fresh_hetero_encoder(data.x_dict, data.edge_index_dict)

    assert torch.allclose(original_outputs['file'], restored_outputs['file']), \
        "Weights or processing layers drifted during saving/loading cycle."


def test_train_hetero_gnn_pipeline_fallback():
    """Verifies task fallback processing logic with insulated infrastructure mocking."""
    mock_local_df = pd.DataFrame({
        "numero_dossier": ["DV0001", "DV0002", "DV0003"],
        "file_gnn_idx": [0, 1, 2]
    })
    
    with patch('gnn_models.mlflow'), patch('gnn_models.child_run'):
        processed_df = train_hetero_gnn(local_df=mock_local_df, n_epochs=1, parent_run_id=None)
    
    assert "is_anomaly_gnn" in processed_df.columns
    assert (processed_df["is_anomaly_gnn"] == 0).all()