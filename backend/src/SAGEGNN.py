# import sys
# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import HeteroData
# import torch_geometric.transforms as T
# from torch_geometric.nn import SAGEConv, to_hetero, LayerNorm
# from sklearn.metrics import roc_auc_score

# # ======================================================================
# # CONFIGURATION SWITCHES
# # ======================================================================
# FORCE_CPU = True  # Keeps execution stable on CPU to bypass version warnings

# # ======================================================================
# # 1. DEFINE THE NEURAL NETWORK ARCHITECTURE (STABILIZED)
# # ======================================================================
# class HeteroGNNEncoder(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.conv1 = SAGEConv((-1, -1), hidden_channels)
#         self.ln1 = LayerNorm(hidden_channels) # Prevents feature magnitude explosion
        
#         self.conv2 = SAGEConv((-1, -1), hidden_channels)
#         self.ln2 = LayerNorm(hidden_channels)

#     def forward(self, x, edge_index):
#         # Layer 1
#         x = self.conv1(x, edge_index)
#         x = self.ln1(x)
#         x = F.relu(x)
        
#         # Layer 2
#         x = self.conv2(x, edge_index)
#         x = self.ln2(x)
#         return x

# class LinkPredictor(torch.nn.Module):
#     def forward(self, x_file, x_user, edge_label_index):
#         src, dst = edge_label_index[0], edge_label_index[1]
        
#         # Normalize vectors to compute Cosine Similarity instead of raw dot-product.
#         # This keeps raw affinity logits bounded strictly between -1.0 and 1.0.
#         f_vector = F.normalize(x_file[src], p=2, dim=-1)
#         u_vector = F.normalize(x_user[dst], p=2, dim=-1)
        
#         scores = (f_vector * u_vector).sum(dim=-1) 
        
#         # Optional scaling factor to allow tighter sigmoid adjustments
#         return torch.sigmoid(scores * 5.0)

# # ======================================================================
# # PROTOCOL BARRIER: MODULE IMPORT PROTECTION SCRIPT GUARD
# # ======================================================================
# if __name__ == "__main__":
#     """
#     Any code running inside this block executes ONLY when running SAGEGNN.py directly.
#     When imported as a module by deployment/triage_engine.py, this entire block 
#     is skipped, protecting FastAPI from path errors and training loop collisions.
#     """
    
#     # Hardware Configuration
#     device = torch.device('cpu')
#     print("💻 Target Device: Explicitly falling back to CPU for architecture stability.")

#     # Verify and Load Data
#     print("\n🔎 Verifying path structures...")
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     path_prefix = os.path.join(current_dir, "data", "processed")

#     if not os.path.exists(path_prefix):
#         print(f"❌ ERROR: Processed data directory missing at: {path_prefix}")
#         sys.exit(1)

#     print("📦 Loading node and edge matrices from storage...")
#     try:
#         df_u = pd.read_parquet(os.path.join(path_prefix, "final_graph_user_nodes")).sort_values("user_gnn_idx")
#         df_c = pd.read_parquet(os.path.join(path_prefix, "final_graph_company_nodes")).sort_values("company_gnn_idx")
#         df_f = pd.read_parquet(os.path.join(path_prefix, "final_graph_file_nodes")).sort_values("file_gnn_idx")
#         edges_user = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_user"))
#         edges_comp = pd.read_parquet(os.path.join(path_prefix, "final_edge_file_to_company"))

#         data = HeteroData()
#         data['user'].x = torch.tensor(np.stack(df_u['user_features'].values), dtype=torch.float)
#         data['company'].x = torch.tensor(np.stack(df_c['company_features'].values), dtype=torch.float)
#         data['file'].x = torch.tensor(np.stack(df_f['file_features'].values), dtype=torch.float)

#         data['file', 'assigned_to', 'user'].edge_index = torch.tensor(
#             edges_user[['source_file_idx', 'target_user_idx']].values, dtype=torch.long
#         ).t().contiguous()

#         data['file', 'linked_to', 'company'].edge_index = torch.tensor(
#             edges_comp[['source_file_idx', 'target_company_idx']].values, dtype=torch.long
#         ).t().contiguous()

#         print("✅ Directed HeteroData graph base structure successfully instantiated!")

#     except Exception as e:
#         print(f"❌ Python Exception Caught During Data Stage: {str(e)}")
#         sys.exit(1)

#     # Run Heterogeneous Random Link Split
#     print("\n✂️ Slicing structural relationships into Train / Val / Test subsets...")
#     link_splitter = T.RandomLinkSplit(
#         num_val=0.10,                      
#         num_test=0.10,                     
#         neg_sampling_ratio=1.0,            
#         edge_types=('file', 'assigned_to', 'user'), 
#         add_negative_train_samples=True,   
#         rev_edge_types=('user', 'rev_assigned_to', 'file')
#     )

#     train_data, val_data, test_data = link_splitter(data)

#     train_data = T.ToUndirected()(train_data)
#     val_data = T.ToUndirected()(val_data)
#     test_data = T.ToUndirected()(test_data)

#     print("✅ Graph split partitions securely isolated and transformed into undirected views.")

#     # Initialize Architecture & Map Heterogeneous Metadata
#     hidden_dim = 64
#     base_encoder = HeteroGNNEncoder(hidden_channels=hidden_dim)

#     encoder = to_hetero(base_encoder, train_data.metadata(), aggr='sum')
#     predictor = LinkPredictor()

#     encoder = encoder.to(device)
#     predictor = predictor.to(device)

#     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.001)
#     criterion = torch.nn.BCELoss()

#     # Define Hardware Execution Loops
#     def train_epoch():
#         encoder.train()
#         predictor.train()
#         optimizer.zero_grad()
        
#         batch_data = train_data.to(device)
#         node_embeddings = encoder(batch_data.x_dict, batch_data.edge_index_dict)
        
#         file_emb = node_embeddings['file']
#         user_emb = node_embeddings['user']
#         target_edges = batch_data['file', 'assigned_to', 'user'].edge_label_index
        
#         predictions = predictor(file_emb, user_emb, target_edges)
#         ground_truth = batch_data['file', 'assigned_to', 'user'].edge_label.float()
        
#         loss = criterion(predictions, ground_truth)
#         loss.backward()
#         optimizer.step()
#         return loss.item()

#     @torch.no_grad()
#     def evaluate_split(split_data):
#         encoder.eval()
#         predictor.eval()
        
#         batch_data = split_data.to(device)
#         node_embeddings = encoder(batch_data.x_dict, batch_data.edge_index_dict)
        
#         target_edges = batch_data['file', 'assigned_to', 'user'].edge_label_index
#         predictions = predictor(node_embeddings['file'], node_embeddings['user'], target_edges)
        
#         y_true = batch_data['file', 'assigned_to', 'user'].edge_label.cpu().numpy()
#         y_pred = predictions.cpu().numpy()
        
#         return roc_auc_score(y_true, y_pred)

#     # Run Optimization Pipeline
#     print("\n🏋️ Commencing model training routine over 40 epochs...")
#     print("=" * 65)

#     for epoch in range(1, 41):
#         loss_val = train_epoch()
        
#         if epoch % 5 == 0 or epoch == 1:
#             val_auc = evaluate_split(val_data)
#             print(f"🔥 Epoch {epoch:02d}/40 | Optimization Loss: {loss_val:.4f} | Val ROC-AUC: {val_auc:.4f}")

#     test_auc = evaluate_split(test_data)
#     print("=" * 65)
#     print(f"🎯 FINAL PERFORMANCE BENCHMARK ON HOLDOUT TEST DECK: {test_auc:.4f}")
#     print("=" * 65)

#     # Save Model Weights
#     print("\n💾 Saving trained model checkpoints to disk...")
#     torch.save(encoder.state_dict(), "hetero_gnn_encoder.pt")
#     torch.save(predictor.state_dict(), "link_predictor.pt")
#     print("✅ Checkpoints saved as 'hetero_gnn_encoder.pt' and 'link_predictor.pt'")

#     @torch.no_grad()
#     def calculate_link_probability(file_idx, user_idx):
#         encoder.eval()
#         predictor.eval()
        
#         test_view = test_data.to(device)
#         node_embeddings = encoder(test_view.x_dict, test_view.edge_index_dict)
        
#         query_edge = torch.tensor([[file_idx], [user_idx]], dtype=torch.long).to(device)
#         probability = predictor(node_embeddings['file'], node_embeddings['user'], query_edge)
        
#         return probability.item()

#     print("\n🎯 Entering Active Inference Testing Mode...")
#     print("-" * 65)

#     sample_file = 0
#     sample_user = 0

#     try:
#         score = calculate_link_probability(sample_file, sample_user)
#         print(f"🔮 Query Link Assessment:")
#         print(f"  • File GNN Index : {sample_file}")
#         print(f"  • User GNN Index : {sample_user}")
#         print(f"  • Match Confidence Probability: {score * 100:.2f}%")

#         if score < 0.15:
#             print("🚨 ALERT: Extremely low structural affinity. Anomaly/High Risk configuration!")
#         elif score > 0.85:
#             print("🟢 SAFE: Highly typical behavioral connection pattern.")
#         else:
#             print("🟡 CAUTION: Ambiguous/Unusual structural relationship.")
#     except Exception as e:
#         print(f"⚠️ Inference skipping: Index bounds vary based on dataset layout limits ({str(e)}).")
#     print("-" * 65)

#     # Automated Anomaly Audit
#     print("🕵️ Analyzing existing links to extract hidden high-risk anomalies...")

#     encoder.eval()
#     predictor.eval()
#     with torch.no_grad():
#         val_view = val_data.to(device)
#         val_embeddings = encoder(val_view.x_dict, val_view.edge_index_dict)
        
#         real_edges = val_view['file', 'assigned_to', 'user'].edge_label_index
#         real_labels = val_view['file', 'assigned_to', 'user'].edge_label
        
#         true_positive_mask = (real_labels == 1)
#         real_edges = real_edges[:, true_positive_mask]
        
#         src, dst = real_edges[0], real_edges[1]
#         file_emb = val_embeddings['file'][src]
#         user_emb = val_embeddings['user'][dst]
        
#         raw_logits = (file_emb * user_emb).sum(dim=-1).cpu().numpy()
#         probabilities = torch.sigmoid(torch.tensor(raw_logits)).numpy()

#     max_render = min(5, len(raw_logits))
#     if max_render > 0:
#         suspicious_indices = np.argsort(raw_logits)[:max_render]

#         print(f"\n🚨 TOP {max_render} MOST SUSPICIOUS SYSTEM INTERACTIONS DETECTED:")
#         print("=================================================================")
#         for rank, idx in enumerate(suspicious_indices, 1):
#             f_idx = real_edges[0][idx].item()
#             u_idx = real_edges[1][idx].item()
#             logit_val = raw_logits[idx]
#             model_prob = probabilities[idx]
            
#             print(f" Rank {rank} | File #{f_idx:5d} linked to User #{u_idx:2d} | Raw Affinity Logit: {logit_val:.4f} (Prob: {model_prob * 100:.2f}%)")
#         print("=================================================================")
#     else:
#         print("⚠️ No valid structural evaluation edges found to execute anomaly log comparisons.")

# SAGEGNN.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm

class HeteroGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.ln1 = LayerNorm(hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.ln2 = LayerNorm(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        return x

class LinkPredictor(torch.nn.Module):
    def forward(self, x_file, x_user, edge_label_index):
        src, dst = edge_label_index[0], edge_label_index[1]
        f_vector = F.normalize(x_file[src], p=2, dim=-1)
        u_vector = F.normalize(x_user[dst], p=2, dim=-1)
        scores = (f_vector * u_vector).sum(dim=-1)
        return torch.sigmoid(scores * 5.0)