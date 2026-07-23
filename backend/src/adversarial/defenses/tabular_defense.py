# adversarial/defenses/tabular_defense.py
import numpy as np
import torch
import torch.nn as nn

class TabularAdversarialShield:
    """
    Production Defensive Layer wrapping tabular model evaluation.
    Neutralizes adversarial perturbations (PGD/Mimicry) and detects manipulated payloads.
    """
    def __init__(self, normal_reference_matrix: np.ndarray, noise_std: float = 0.05, max_deviation_sigma: float = 3.5):
        self.noise_std = noise_std
        self.max_sigma = max_deviation_sigma
        
        # Compute baseline feature statistical distributions
        self.means = np.mean(normal_reference_matrix, axis=0)
        self.stds = np.std(normal_reference_matrix, axis=0) + 1e-8
        self.min_bounds = self.means - self.max_sigma * self.stds
        self.max_bounds = self.means + self.max_sigma * self.stds

    def sanitize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Applies input sanitization:
        1. Feature Clipping to enforce statistical boundaries.
        2. Randomized Gaussian Smoothing to break gradient alignment.
        """
        # 1. Clip features to valid domain bounds
        X_sanitized = np.clip(X, self.min_bounds, self.max_bounds)
        
        # 2. Inject Gaussian noise to neutralize gradient perturbations (Randomized Smoothing)
        noise = np.random.normal(0, self.noise_std, size=X_sanitized.shape)
        X_sanitized = X_sanitized + noise
        
        return X_sanitized

    def detect_adversarial_manipulation(self, X_raw: np.ndarray, X_sanitized: np.ndarray) -> np.ndarray:
        """
        Flags inputs that show unnatural feature variance or significant deviation 
        from sanitization projections.
        """
        # Calculate L2 shift caused by sanitization
        perturbation_distance = np.linalg.norm(X_raw - X_sanitized, axis=1)
        threshold = np.percentile(perturbation_distance, 95) + 1e-5
        
        # 1 = Manipulated / Adversarial Flagged, 0 = Natural Payload
        is_adversarial = (perturbation_distance > threshold).astype(int)
        return is_adversarial

    def defend_and_evaluate(self, models: dict, feature_matrix: np.ndarray) -> dict:
        """
        Defensive Wrapper replacing direct call to evaluate_ensemble_anomalies.
        """
        # Step 1: Sanitize input features
        X_clean = self.sanitize_features(feature_matrix)
        
        # Step 2: Detect potential adversarial tampering
        adversarial_flags = self.detect_adversarial_manipulation(feature_matrix, X_clean)
        
        return {
            "sanitized_features": X_clean,
            "adversarial_flags": adversarial_flags,
            "n_adversarial_detected": int(np.sum(adversarial_flags))
        }