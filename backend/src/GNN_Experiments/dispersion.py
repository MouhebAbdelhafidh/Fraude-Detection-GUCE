# run_axis2_dispersion.py
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =====================================================================
# 1. MATHEMATICAL DISPERSION CORE
# =====================================================================
def compute_pairwise_dispersion(embeddings, max_samples=2000):
    """
    Computes Mean Pairwise Cosine Feature Dispersion.
    Includes memory-safe subsampling to prevent OOM errors on large GUCE tables.
    """
    N = embeddings.size(0)
    if N <= 1:
        return 0.0
        
    # Memory safety: randomly subsample if node count exceeds max_samples
    if N > max_samples:
        indices = torch.randperm(N)[:max_samples]
        X = embeddings[indices]
        N = max_samples
    else:
        X = embeddings

    # L2 Normalize for Cosine Distance
    X_norm = F.normalize(X, p=2, dim=-1)
    
    # Compute the NxN cosine similarity matrix
    similarity_matrix = torch.mm(X_norm, X_norm.t())
    
    # Convert to Cosine Distance: D = 1 - S
    distance_matrix = 1.0 - similarity_matrix
    
    # Extract strictly the upper triangle (exclude self-to-self distance of 0)
    triu_indices = torch.triu_indices(N, N, offset=1)
    pairwise_distances = distance_matrix[triu_indices[0], triu_indices[1]]
    
    return pairwise_distances.mean().item()

# =====================================================================
# 2. EXPERIMENTAL TRACE HOOKS (SIMULATED DATA LOADER FOR PIPELINE)
# =====================================================================
# Note: In your actual environment, replace this mock function with your 
# actual model forward passes exactly as you did in Step 4 for Axis 1.
def extract_layer_embeddings(model_name, layer_depth):
    """
    Placeholder representing the embeddings extracted from your models.
    Returns simulated tensor shapes mimicking the GUCE dataset behavior.
    """
    # Base configuration: 5000 files, 64 hidden dimensions
    N, D = 5000, 64 
    
    if layer_depth == 0:
        # Layer 0: Raw padded inputs (High variance, low semantic alignment)
        return torch.randn(N, D) * 5.0 
        
    if model_name in ["SAGE", "RGCN"]:
        # Spatial/Relational: Healthy dispersion expansion
        base = torch.randn(N, D)
        noise = torch.randn(N, D) * (0.2 if layer_depth == 1 else 0.4)
        return base + noise
        
    elif model_name == "GATv2":
        if layer_depth == 1:
            return torch.randn(N, D)
        else:
            # Layer 2 Explosion: GATv2 bipartite shattering
            shattered = torch.randn(N, D) * 1000.0 
            # Force vectors to extreme opposite poles
            shattered[::2] = shattered[::2] * -1 
            return shattered
            
    elif model_name == "HAN":
        # Baseline static due to meta-path configuration constraints
        return torch.randn(N, D) * 5.0 

# =====================================================================
# 3. EXECUTE GRAPH SURGERY
# =====================================================================
print("\n🔬 STEP 4: Processing Axis 2 (Feature Dispersion) Traces...\n")

models = ["SAGE", "GATv2", "RGCN", "HAN"]
layers = [0, 1, 2]
results = []

print("=================================================================")
print("Layer Depth  | SAGE       | GATv2      | RGCN       | HAN")
print("=================================================================")

for l in layers:
    row_str = f"Layer {l}      |"
    for m in models:
        # 1. Extract the latent space for this model at this depth
        emb = extract_layer_embeddings(m, l)
        
        # 2. Compute Global Dispersion mathematically
        dispersion_val = compute_pairwise_dispersion(emb)
        
        # 3. Log results
        results.append({'Model': m, 'Layer': f"Layer {l}", 'Dispersion': dispersion_val})
        row_str += f" {dispersion_val:10.4f} |"
    print(row_str)

print("=================================================================\n")

# =====================================================================
# 4. RENDER THESIS VISUALIZATION
# =====================================================================
print("🎨 STEP 5: Rendering Axis 2 Dispersion Plot...")

df = pd.DataFrame(results)

# Setup professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Define an elegant color palette
colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]

sns.barplot(
    data=df, 
    x='Layer', 
    y='Dispersion', 
    hue='Model', 
    palette=colors,
    edgecolor='black',
    linewidth=1.2,
    alpha=0.9
)

# Professional Formatting
plt.title("Axis 2: Mean Pairwise Feature Dispersion ($\mathcal{D}_{Cos}$) Across Layers", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Global Cosine Dispersion", fontsize=12, fontweight='bold')
plt.xlabel("Message Passing Depth", fontsize=12, fontweight='bold')
plt.ylim(0, 2.1) # Cosine distance max bounds

# Add a reference line for theoretically perfect orthogonal scattering
plt.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Orthogonal Baseline (1.0)')

# Refine legend
plt.legend(title='Architecture', title_fontsize='11', fontsize='10', loc='upper right')

# Save high-resolution image for LaTeX compilation
output_path = 'all_models_dispersion_axis2.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"🏆 SUCCESS: Benchmark chart successfully saved as '{output_path}'!")