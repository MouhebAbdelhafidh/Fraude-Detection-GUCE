import os
import sys
import pandas as pd
import numpy as np
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# ⚙️ GLOBAL CONFIGURATION & PATHS
# ======================================================================
MAX_SPECTRAL_NODES = 1200  # Strict cap to prevent cubic OOM memory crashes
REWIRING_PROBABILITY = 0.02 # Add 2% strategic shortcut links to break bottlenecks

# ======================================================================
# 🔌 ELECTRICAL METRIC ENGINE (AXIS 3)
# ======================================================================
def compute_effective_resistance_matrix(G):
    """
    Computes the full exact pairwise Effective Resistance matrix 
    using the Moore-Penrose pseudoinverse of the Graph Laplacian.
    """
    # 1. Extract the standard Graph Laplacian Matrix (L = D - A)
    nodes = list(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
    
    # 2. Compute the Moore-Penrose Pseudoinverse (L_dagger)
    # Using la.pinv is stable for singular matrices like the Laplacian
    L_dagger = la.pinv(L)
    
    # 3. Vectorized implementation of: R_eff(i,j) = L_ii* + L_jj* - 2*L_ij*
    diag = np.diag(L_dagger)
    R_eff = diag[:, None] + diag[None, :] - 2.0 * L_dagger
    
    # Clip tiny negative numerical noise values to absolute 0
    R_eff = np.clip(R_eff, a_min=0.0, a_max=None)
    return R_eff, nodes

# ======================================================================
# 🏗️ DATA LOADING & STRUCTURAL SURGERY PACK
# ======================================================================
def load_guce_homogeneous_projection():
    print("🔄 STEP 1: Reading real GUCE Parquet records from disk...")
    path_prefix = os.path.join("notebooks", "siat_cc", "data", "processed")
    
    if not os.path.exists(path_prefix):
        print(f"❌ ERROR: Processed data directory missing at: {path_prefix}")
        sys.exit(1)

    try:
        edges_user = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
        edges_comp = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_company"))
    except Exception as e:
        print(f"❌ ERROR: Failed to read parquet files. Details: {e}")
        sys.exit(1)

    print("⚡ STEP 2: Projecting Heterogeneous Topology to Undirected Homogeneous Graph...")
    G_base = nx.Graph()

    # Map nodes with string prefixing to maintain type integrity inside a single index
    for _, row in edges_user.iterrows():
        f_node = f"file_{int(row['source_file_idx'])}"
        u_node = f"user_{int(row['target_user_idx'])}"
        G_base.add_edge(f_node, u_node)

    for _, row in edges_comp.iterrows():
        f_node = f"file_{int(row['source_file_idx'])}"
        c_node = f"company_{int(row['target_company_idx'])}"
        G_base.add_edge(f_node, c_node)

    # Memory Safety Guard: Extract a highly connected sub-component for matrix computation
    print(f"🛡️ Total raw graph size: {G_base.number_of_nodes()} nodes. Applying spectral cap...")
    largest_cc = max(nx.connected_components(G_base), key=len)
    G_sub = G_base.subgraph(largest_cc).copy()
    
    if G_sub.number_of_nodes() > MAX_SPECTRAL_NODES:
        # Intelligently sample a dense local neighborhood using Breadth-First Search
        seed_node = list(G_sub.nodes())[0]
        bfs_nodes = list(nx.bfs_tree(G_sub, source=seed_node).nodes())[:MAX_SPECTRAL_NODES]
        G_final = G_sub.subgraph(bfs_nodes).copy()
    else:
        G_final = G_sub

    # Ensure final component is perfectly connected (required for valid electrical paths)
    main_component = max(nx.connected_components(G_final), key=len)
    return G_final.subgraph(main_component).copy()


def perform_surgical_rewiring(G_baseline):
    """
    Applies an aggressive topological intervention:
    Injects virtual homogeneous shortcut links between File nodes that share 
    mutual target dependencies, bypasses bipartite routing bottlenecks.
    """
    print("✂️ STEP 3: Executing Structural Surgical Rewiring Interventions...")
    G_rewired = G_baseline.copy()
    
    file_nodes = [n for n in G_baseline.nodes() if n.startswith("file_")]
    num_shortcuts = int(len(file_nodes) * REWIRING_PROBABILITY)
    
    added = 0
    attempts = 0
    while added < num_shortcuts and attempts < 5000:
        attempts += 1
        u, v = np.random.choice(file_nodes, size=2, replace=False)
        if not G_rewired.has_edge(u, v):
            # Check if they share a common operational hub (User/Broker) to make the link realistic
            u_neighbors = set(G_baseline.neighbors(u))
            v_neighbors = set(G_baseline.neighbors(v))
            if not u_neighbors.isdisjoint(v_neighbors):
                G_rewired.add_edge(u, v)
                added += 1

    # Fallback to random bridging if strict conditional shortcuts are exhausted
    while added < num_shortcuts:
        u, v = np.random.choice(file_nodes, size=2, replace=False)
        if not G_rewired.has_edge(u, v):
            G_rewired.add_edge(u, v)
            added += 1
            
    print(f"  ✅ Injected {added} virtual File-to-File shortcut bonds into the circuit.")
    return G_rewired

# ======================================================================
# 🚀 EXPERIMENT RUNNER
# ======================================================================
def run_axis3_experiment():
    G_base = load_guce_homogeneous_projection()
    N_nodes = G_base.number_of_nodes()
    print(f"  ✅ Active Spectral Core Isolated: {N_nodes} nodes | {G_base.number_of_edges()} edges.")

    G_rewired = perform_surgical_rewiring(G_base)

    print("\n🔬 STEP 4: Solving Electrical Laplacian Matrices (Moore-Penrose Grid)...")
    R_base, _ = compute_effective_resistance_matrix(G_base)
    R_rewired, _ = compute_effective_resistance_matrix(G_rewired)

    # Extract Key Analytical Metric Diameters
    r_max_base = np.max(R_base)
    r_max_rewired = np.max(R_rewired)
    r_mean_base = np.mean(R_base[R_base > 0])
    r_mean_rewired = np.mean(R_rewired[R_rewired > 0])

    print("\n" + "="*70)
    print(f" STRUCTURAL INDICATOR           | BASELINE GRAPH | SURGICALLY REWIRED")
    print("="*70)
    print(f" Global Resistance Diameter (Max) | {r_max_base:<14.4f} | {r_max_rewired:.4f}")
    print(f" Mean Pairwise Resistance        | {r_mean_base:<14.4f} | {r_mean_rewired:.4f}")
    print(f" Total Graph Edge Densities       | {G_base.number_of_edges():<14} | {G_rewired.number_of_edges()}")
    print("="*70 + "\n")

    print("🎨 STEP 5: Rendering Axis 3 Resistance Profile Plots...")
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Full Distribution Densities
    sns.kdeplot(R_base.flatten(), fill=True, color="#d62728", label=f"Baseline (Max: {r_max_base:.2f})", ax=ax1, linewidth=2)
    sns.kdeplot(R_rewired.flatten(), fill=True, color="#2ca02c", label=f"Surgically Rewired (Max: {r_max_rewired:.2f})", ax=ax1, linewidth=2)
    ax1.set_title("Pairwise Effective Resistance Density ($\mathcal{R}_{Eff}$)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Effective Resistance Value (Ohms $\Omega$)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(shadow=True, frameon=True)

    # Plot 2: Direct Critical Path Summary Comparison
    bars = ax2.bar(["Baseline Customs Graph", "Surgically Rewired Graph"], [r_max_base, r_max_rewired], color=["#d62728", "#2ca02c"], width=0.5, edgecolor='black', linewidth=1.2)
    ax2.set_title("Axis 3 Metric: Network Resistance Diameter ($\mathcal{R}_{Max}$)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Maximum Effective Resistance $\mathcal{R}_{Max}$", fontsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (r_max_base*0.02), f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=11)

    output_filename = "axis3_effective_resistance_benchmark.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"🏆 SUCCESS: Axis 3 structural diagnostic chart saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_axis3_experiment()