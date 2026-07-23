import torch
import numpy as np
from torch_geometric.data import HeteroData


class GNNAdversarialAttack:
    """
    Generates targeted adversarial perturbations against Heterogeneous Graph Neural Networks.
    Provides methods for feature vector manipulation and topological edge removal.
    """
    def __init__(self, epsilon: float = 0.25, drop_rate: float = 0.15):
        self.epsilon = epsilon
        self.drop_rate = drop_rate

    def perturb_features(self, data: HeteroData, node_type: str = "file") -> HeteroData:
        """
        Injects Gaussian feature noise into target node representations.
        """
        data_attacked = data.clone()
        x = data_attacked[node_type].x
        noise = torch.randn_like(x) * self.epsilon
        data_attacked[node_type].x = x + noise
        return data_attacked

    def perturb_edges(self, data: HeteroData, edge_type: tuple) -> HeteroData:
        """
        Simulates topological evasion attacks by selectively dropping graph edges.
        """
        data_attacked = data.clone()
        edge_index = data_attacked[edge_type].edge_index
        num_edges = edge_index.size(1)

        mask = torch.rand(num_edges) > self.drop_rate
        data_attacked[edge_type].edge_index = edge_index[:, mask]

        if hasattr(data_attacked[edge_type], "edge_label_index"):
            data_attacked[edge_type].edge_label_index = data_attacked[edge_type].edge_label_index[:, mask]

        return data_attacked

    def execute_full_attack(self, data: HeteroData, target_node_type: str = "file") -> HeteroData:
        """
        Executes a joint feature perturbation and topological disturbance attack.
        """
        attacked_data = self.perturb_features(data, node_type=target_node_type)
        
        # Apply edge perturbations across available relational edges
        for edge_type in attacked_data.edge_types:
            attacked_data = self.perturb_edges(attacked_data, edge_type=edge_type)

        return attacked_data