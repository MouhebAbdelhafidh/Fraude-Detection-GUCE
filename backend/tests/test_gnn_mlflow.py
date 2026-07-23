import pandas as pd
from mlflow_utils import start_pipeline_run
from gnn_models import train_hetero_gnn

root_run_id = start_pipeline_run(dag_run_id="gnn_smoke_test")
print("root run_id:", root_run_id)

graph_files_df = pd.read_parquet("data/processed/final_graph_file_nodes").sort_values("file_gnn_idx")
print(f"Loaded {len(graph_files_df)} real file nodes")

updated_df = train_hetero_gnn(local_df=graph_files_df, n_epochs=3, parent_run_id=root_run_id)
print(updated_df[["file_id", "is_anomaly_gnn"]].head())

print(f"\ndone — open http://localhost:5000/#/experiments/1/runs/{root_run_id}")
