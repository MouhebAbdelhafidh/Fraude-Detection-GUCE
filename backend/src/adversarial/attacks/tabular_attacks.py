# adversarial/attacks/tabular_attacks.py
import numpy as np
import torch
import torch.nn as nn

class TabularAutoencoderPGD:
    """
    Projected Gradient Descent (PGD) Evasion Attack against PyTorch Autoencoder.
    
    Goal: Perturb anomalous input features x_anom to minimize the reconstruction error 
    ||x_adv - AE(x_adv)||^2 below the 99th percentile threshold, evading detection.
    """
    def __init__(self, autoencoder_model: nn.Module, epsilon: float = 0.15, alpha: float = 0.02, steps: int = 20):
        self.model = autoencoder_model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.model.eval()

    def generate(self, X_anom: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X_anom, dtype=torch.float32, requires_grad=True)
        X_orig = X_tensor.clone().detach()

        criterion = nn.MSELoss(reduction='none')

        for _ in range(self.steps):
            if X_tensor.grad is not None:
                X_tensor.grad.zero_()

            reconstructed = self.model(X_tensor)
            # Loss is reconstruction error per sample; adversary seeks to MINIMIZE this error
            loss = criterion(reconstructed, X_tensor).mean(dim=1).sum()
            loss.backward()

            with torch.no_grad():
                # Step OPPOSITE to gradient direction to REDUCE reconstruction error
                grad_sign = X_tensor.grad.sign()
                X_tensor = X_tensor - self.alpha * grad_sign
                
                # Projection back within L-infinity epsilon ball
                eta = torch.clamp(X_tensor - X_orig, min=-self.epsilon, max=self.epsilon)
                X_tensor = (X_orig + eta).detach().requires_grad_(True)

        return X_tensor.detach().cpu().numpy()


class TabularMimicryAttacker:
    """
    Mimicry / Boundary Evasion Attack targeting tree-based and distance models (IForest, HBOS, OCSVM, XGBoost).
    
    Goal: Blend anomalous feature vectors into normal feature distributions by scaling 
    sensitive features toward normal population centroids.
    """
    def __init__(self, normal_reference_matrix: np.ndarray, perturbation_budget: float = 0.20):
        self.normal_medians = np.median(normal_reference_matrix, axis=0)
        self.budget = perturbation_budget

    def generate(self, X_anom: np.ndarray) -> np.ndarray:
        X_adv = X_anom.copy()
        
        # Shift features toward benign population medians within the budget limit
        direction = self.normal_medians - X_adv
        X_adv = X_adv + self.budget * direction
        return X_adv


class UnifiedTabularAdversary:
    """
    Unified Adversarial Generator applying hybrid evasion across all 5 tabular models.
    """
    def __init__(self, autoencoder_model: nn.Module = None, normal_ref: np.ndarray = None, epsilon: float = 0.15):
        self.ae_attacker = TabularAutoencoderPGD(autoencoder_model, epsilon=epsilon) if autoencoder_model else None
        self.mimicry_attacker = TabularMimicryAttacker(normal_ref, perturbation_budget=epsilon) if normal_ref is not None else None

    def attack(self, X_anom: np.ndarray) -> np.ndarray:
        X_adv = X_anom.copy()
        
        # 1. Apply mimicry perturbation
        if self.mimicry_attacker:
            X_adv = self.mimicry_attacker.generate(X_adv)
            
        # 2. Refine using gradient-based optimization if PyTorch AE is available
        if self.ae_attacker:
            X_adv = self.ae_attacker.generate(X_adv)
            
        return X_adv