import pytest
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData

# =====================================================================
# 📊 SECTION 1: PRODUCTION-MATCHING MOCK FIXTURES (HeteroData)
# =====================================================================

@pytest.fixture
def mock_tabular_dataframe():
    """Generates a mock production transaction dataframe for testing."""
    np.random.seed(42)
    rows = 100
    data = {
        "transaction_id": [f"tx_{i}" for i in range(rows)],
        "user_id": [f"usr_{np.random.randint(1, 20)}" for _ in range(rows)],
        "amount": np.random.exponential(scale=50.0, size=rows) + 1.0,
        "account_age_days": np.random.randint(0, 365, size=rows),
        "device_score": np.random.uniform(0.0, 1.0, size=rows),
        "is_fraud": np.random.choice([0, 1], size=rows, p=[0.95, 0.05])
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_hetero_graph_data():
    """
    Generates a mock PyTorch Geometric HeteroData object perfectly 
    matching the production configuration dimensions.
    """
    torch.manual_seed(42)
    data = HeteroData()
    
    num_users = 20
    num_companies = 10
    num_files = 50
    feature_dim = 64  # Production standard uniform dimension block

    # Node Features matching production input spaces
    data['user'].x = torch.rand((num_users, feature_dim))
    data['company'].x = torch.rand((num_companies, feature_dim))
    data['file'].x = torch.rand((num_files, feature_dim))

    # Edges 1: file -> assigned_to -> user
    src_files = torch.randint(0, num_files, (60,))
    dst_users = torch.randint(0, num_users, (60,))
    data['file', 'assigned_to', 'user'].edge_index = torch.stack([src_files, dst_users], dim=0)

    # Edges 2: file -> linked_to -> company
    src_files_comp = torch.randint(0, num_files, (40,))
    dst_companies = torch.randint(0, num_companies, (40,))
    data['file', 'linked_to', 'company'].edge_index = torch.stack([src_files_comp, dst_companies], dim=0)

    # Synthetic Ground-Truth targets
    data['file'].y = (torch.rand(num_files) > 0.9).long()

    return data


# =====================================================================
# 🔬 SECTION 2: PRODUCTION HETEROGENEOUS DATA TESTING
# =====================================================================

def test_graph_data_shapes(mock_hetero_graph_data):
    """Validates structural dimensions and enforces strict NaN/Inf walls."""
    graph = mock_hetero_graph_data
    
    assert graph['user'].num_nodes == 20
    assert graph['file'].num_nodes == 50
    
    assert graph['file', 'assigned_to', 'user'].edge_index.shape[0] == 2
    assert graph['file', 'linked_to', 'company'].edge_index.shape[0] == 2
    
    # Production Data Protection Walls
    for node_type in graph.node_types:
        assert not torch.isnan(graph[node_type].x).any(), f"NaN values encountered in {node_type} feature space."
        assert not torch.isinf(graph[node_type].x).any(), f"Inf tensors found in {node_type} feature space."


def test_tabular_physical_constraints(mock_tabular_dataframe):
    """Ensures incoming financial and security signals respect logical distributions."""
    df = mock_tabular_dataframe
    assert (df["amount"] > 0).all(), "Negative transaction values detected."
    assert (df["account_age_days"] >= 0).all(), "Negative temporal metric detected."
    assert df["device_score"].min() >= 0.0 and df["device_score"].max() <= 1.0


def test_graph_structural_anomalies(mock_hetero_graph_data):
    """
    Production Leak Guard: Validates that source and target node assignments 
    fall within the true matrix dimension spaces.
    """
    graph = mock_hetero_graph_data
    
    for edge_type, edge_store in graph.edge_index_dict.items():
        src_node_type, _, dst_node_type = edge_type
        
        # Upper Bound Index Protections
        assert torch.all(edge_store[0] < graph[src_node_type].num_nodes), f"Index overflow in source keys of {edge_type}"
        assert torch.all(edge_store[1] < graph[dst_node_type].num_nodes), f"Index overflow in destination keys of {edge_type}"
        
        # Lower Bound Index Protections
        assert torch.all(edge_store >= 0), f"Negative node tracking pointer found within relation: {edge_type}"


def test_label_distribution_imbalance(mock_tabular_dataframe):
    """Verifies baseline label representation across downstream dataset blocks."""
    df = mock_tabular_dataframe
    fraud_ratio = df["is_fraud"].mean()
    assert 0.0 < fraud_ratio < 0.5, f"Imbalanced operational distribution bounds breached: {fraud_ratio:.2%}"