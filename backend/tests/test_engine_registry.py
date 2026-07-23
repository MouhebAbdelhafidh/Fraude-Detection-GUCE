# test_engine_registry.py
import os
import sys

# Ensure the project root is in the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from deployment.triage_engine import RealTimeCustomsTriageEngine

def main():
    print("=== STARTING REGISTRY INTEGRATION TEST ===")
    
    # Define paths pointing to your local test data
    master_ledger = os.path.join(PROJECT_ROOT, "data", "processed", "scaled_siat_fraud_dataset.parquet")
    graph_nodes = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # 1. Initialize the engine (Triggers MLflow download)
    try:
        engine = RealTimeCustomsTriageEngine(
            main_data_path=master_ledger,
            processed_dir_path=graph_nodes
        )
        print("\n✅ SUCCESS: Engine initialized successfully from MLflow registry!")
        
        # 2. Verify config mapping
        print(f"📊 Active Fusion Weights -> Tabular: {engine.fusion_config.get('w_tabular')}, GNN: {engine.fusion_config.get('w_gnn')}")
        
        # 3. Quick structural sanity check on GNN layers
        assert hasattr(engine, 'encoder'), "Encoder layer missing!"
        assert hasattr(engine, 'predictor'), "Predictor layer missing!"
        print("🧠 GNN Neural layers safely loaded in evaluation state.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()