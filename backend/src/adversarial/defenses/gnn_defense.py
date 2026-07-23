import torch
import numpy as np
from torch_geometric.data import HeteroData


class GNNAdversarialShield:
    """
    Defends Heterogeneous Graph Neural Networks by sanitizing node vectors and
    flagging potential adversarial inputs via feature-distortion thresholds.
    """
    def __init__(self, noise_std: float = 0.05, feature_squeeze_bits: int = 4):
        self.noise_std = noise_std
        self.squeeze_steps = 2 ** feature_squeeze_bits - 1

    def sanitize_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies bit-squeezing quantization followed by Gaussian smoothing.
        """
        min_val, max_val = x.min(), x.max()
        if max_val > min_val:
            x_norm = (x - min_val) / (max_val - min_val)
            x_squeezed = torch.round(x_norm * self.squeeze_steps) / self.squeeze_steps
            x = x_squeezed * (max_val - min_val) + min_val

        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def defend_and_evaluate(self, data: HeteroData) -> tuple[HeteroData, dict[str, torch.Tensor]]:
        """
        Sanitizes all node types inside HeteroData and detects distorted inputs.
        """
        defended_data = data.clone()
        adversarial_flags = {}

        for node_type in defended_data.node_types:
            original_x = defended_data[node_type].x
            sanitized_x = self.sanitize_features(original_x)

            # Compute relative distortion variance to detect perturbed node vectors
            distortion = torch.norm(original_x - sanitized_x, dim=1)
            threshold = torch.mean(distortion) + 2.5 * torch.std(distortion)
            is_adv = (distortion > threshold).int()

            defended_data[node_type].x = sanitized_x
            adversarial_flags[node_type] = is_adv

        return defended_data, adversarial_flags