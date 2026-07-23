import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero

# ======================================================================
# ⚙️ SYSTEM PATH & DYNAMIC MODULE IMPORTS
# ======================================================================
MODULE_DIR = os.path.join(os.getcwd(), "notebooks", "siat_cc")
sys.path.append(MODULE_DIR)

# Importing your specific architecture classes
from SAGEGNN import HeteroGNNEncoder
from GAT import HeteroGATv2Encoder
from HAN import HeteroHANEncoder
from RGCN import HeteroRGCNEncoder

# 🛠️ MODEL HYPERPARAMETERS (MATCH THESE TO YOUR TRAINING CONFIGURATION)
HIDDEN_CHANNELS = 64  # Set this to 32, 64, 128 etc., depending on your trained models!
SAMPLE_SIZE = 100     # Number of files to trace back (prevents RAM explosion)
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

CHECKPOINTS = {
    'GraphSAGE': os.path.join(MODULE_DIR, "models", "sagegnn", "hetero_gnn_encoder.pt"),
    'GATv2':     os.path.join(MODULE_DIR, "models", "gat", "hetero_gatv2_encoder.pt"),
    'RGCN':      os.path.join(MODULE_DIR, "models", "rgcn", "hetero_rgcn_encoder.pt"),
    'HAN':       os.path.join(MODULE_DIR, "models", "han", "hetero_han_encoder.pt")
}

# ======================================================================
# 🏗️ DATA ENGINE: REAL FEATURES & TOPOLOGY
# ======================================================================
def load_real_guce_hetero_graph():
    print("🔄 STEP 1: Loading Real GUCE Features and Topology from Parquet...")
    path_prefix = os.path.join(MODULE_DIR, "data", "processed")
    
    # Load raw feature matrices
    df_u = pd.read_parquet(os.path.join(path_prefix, "final_graph_user_nodes")).sort_values("user_gnn_idx")
    df_c = pd.read_parquet(os.path.join(path_prefix, "final_graph_company_nodes")).sort_values("company_gnn_idx")
    df_f = pd.read_parquet(os.path.join(path_prefix, "final_graph_file_nodes")).sort_values("file_gnn_idx")
    
    # Load edge connections
    edges_user = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
    edges_comp = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_company"))

    data = HeteroData()
    
    # Inject real features as tensors
    data['user'].x = torch.tensor(np.stack(df_u['user_features'].values), dtype=torch.float)
    data['company'].x = torch.tensor(np.stack(df_c['company_features'].values), dtype=torch.float)
    data['file'].x = torch.tensor(np.stack(df_f['file_features'].values), dtype=torch.float)

    # Inject real edges
    data['file', 'assigned_to', 'user'].edge_index = torch.tensor(
        edges_user[['source_file_idx', 'target_user_idx']].values, dtype=torch.long).t().contiguous()
    data['file', 'linked_to', 'company'].edge_index = torch.tensor(
        edges_comp[['source_file_idx', 'target_company_idx']].values, dtype=torch.long).t().contiguous()

    # Create reverse edges for backward gradient flow
    data = T.ToUndirected()(data)
    
    return data

# ======================================================================
# 🔬 PRE-TRAINED JACOBIAN BENCHMARK SUITE
# ======================================================================
def run_pretrained_benchmark():
    base_data = load_real_guce_hetero_graph()
    num_files = base_data['file'].num_nodes
    target_nodes = np.random.choice(num_files, min(SAMPLE_SIZE, num_files), replace=False)

    models_classes = {
        'GraphSAGE': HeteroGNNEncoder,
        'GATv2': HeteroGATv2Encoder,
        'RGCN': HeteroRGCNEncoder,
        'HAN': HeteroHANEncoder
    }

    all_results = []

    print("\n⚡ STEP 2: Executing Autograd Tracing on Trained Weights...")
    for model_name, model_class in models_classes.items():
        ckpt_path = CHECKPOINTS[model_name]
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skipping {model_name}: Checkpoint not found at {ckpt_path}")
            continue

        print(f"\n🚀 Evaluating [{model_name}]...")
        
        # Clone data and enable gradient tracking for inputs
        data = base_data.clone()
        data['file'].x.requires_grad_(True)
        data['user'].x.requires_grad_(True)
        data['company'].x.requires_grad_(True)

        # Initialize model with its required architectural signature
        try:
            if model_name in ['GraphSAGE', 'GATv2']:
                # 1. Instantiate the base homogeneous model
                base_model = model_class(hidden_channels=HIDDEN_CHANNELS)
                # 2. Wrap it dynamically into a heterogeneous model to match the dictionaries
                model = to_hetero(base_model, metadata=data.metadata(), aggr='sum')
                
            elif model_name in ['RGCN', 'HAN']:
                # RGCN and HAN are natively heterogeneous, so they take metadata directly
                model = model_class(hidden_channels=HIDDEN_CHANNELS, metadata=data.metadata())
            
            # Load your pre-trained weights safely onto CPU
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=False)
            model.eval()
            
        except Exception as e:
            print(f"❌ Failed to instantiate/load {model_name}. Error: {e}")
            continue

        # Forward Pass through trained weights
        out_dict = model(data.x_dict, data.edge_index_dict)
        file_embeddings = out_dict['file']

        for node_idx in target_nodes:
            # Clear gradients safely
            if data['user'].x.grad is not None: data['user'].x.grad.zero_()
            if data['company'].x.grad is not None: data['company'].x.grad.zero_()

            # Autograd trigger
            loss = torch.norm(file_embeddings[node_idx])
            loss.backward(retain_graph=True)

            u_norm = data['user'].x.grad.norm().item() if data['user'].x.grad is not None else 0.0
            c_norm = data['company'].x.grad.norm().item() if data['company'].x.grad is not None else 0.0

            all_results.append({'Model': model_name, 'Relation': 'File -> User (Broker)', 'Jacobian Norm': u_norm})
            all_results.append({'Model': model_name, 'Relation': 'File -> Company', 'Jacobian Norm': c_norm})

    if not all_results:
        print("\n🛑 No models successfully processed. Exiting.")
        return

    # ======================================================================
    # 📊 AGGREGATION & REPORT RENDERING
    # ======================================================================
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*75)
    print(" PRE-TRAINED AXIS 4 BENCHMARKING SUMMARY REPORT")
    print("="*75)
    summary = df.groupby(['Model', 'Relation'])['Jacobian Norm'].mean().unstack()
    summary['Imbalance Ratio (User/Comp)'] = summary['File -> User (Broker)'] / summary['File -> Company']
    print(summary.to_string())
    print("="*75)

    # Render Plot
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    ax = sns.boxplot(x='Model', y='Jacobian Norm', hue='Relation', data=df, palette=["#1f77b4", "#ff7f0e"], linewidth=1.5)
    
    plt.title("Axis 4 Pre-Trained Drift: Learned Sensitivity & Over-squashing Diagnosis", fontsize=15, fontweight='bold')
    plt.ylabel("Learned Gradient Sensitivity ($\| \partial h^{(L)} / \partial x^{(0)} \|$)", fontsize=12)
    plt.xlabel("Trained Heterogeneous GNN Architectures", fontsize=12)
    plt.yscale("log")
    
    plt.tight_layout()
    output_filename = "axis4_real_trained_benchmark.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n🏆 SUCCESS: Final thesis chart saved as '{output_filename}'!")
    plt.show()

if __name__ == "__main__":
    run_pretrained_benchmark()