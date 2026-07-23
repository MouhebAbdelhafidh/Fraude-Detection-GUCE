import time
import random
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Import the metrics exporter functions we wrote earlier
from monitoring.metrics_exporter import start_monitoring_server, INFERENCE_LATENCY, PREDICTION_COUNT, AGENT_FALLBACK_COUNT, AGENT_CONFIDENCE, FEATURE_DRIFT_SHARE

# 1. Create baseline references for simulation
def generate_live_mock_data():
    """Simulates real-time incoming production adjustments."""
    data = HeteroData()
    num_files = random.randint(25, 40)
    num_users = 15
    feature_dim = 64
    
    data['user'].x = torch.rand((num_users, feature_dim))
    data['file'].x = torch.rand((num_files, feature_dim))
    
    # Introduce random fluctuations to edge sizes (Structural Drift)
    num_edges = random.randint(30, 70)
    src_u = torch.randint(0, num_files, (num_edges,))
    dst_u = torch.randint(0, num_users, (num_edges,))
    data['file', 'assigned_to', 'user'].edge_index = torch.stack([src_u, dst_u], dim=0)
    
    return data

if __name__ == "__main__":
    print("🚀 Starting Prometheus Metrics Exporter on port 8000...")
    start_monitoring_server(port=8000)
    print("📈 Simulation loop active. Sending metrics stream every 3 seconds. Press Ctrl+C to stop.")
    
    while True:
        # Simulate processing latency variance (e.g., 20ms to 120ms)
        simulated_latency = random.uniform(0.020, 0.120)
        
        # 1. Track simulated GNN execution time
        with INFERENCE_LATENCY.time():
            time.sleep(simulated_latency) 
            
        # 2. Simulate raw prediction counts
        fraud_events = random.choices([0, 1], weights=[0.92, 0.08], k=random.randint(5, 15))
        frauds = sum(fraud_events)
        legit = len(fraud_events) - frauds
        
        PREDICTION_COUNT.labels(class_type='fraud').inc(frauds)
        PREDICTION_COUNT.labels(class_type='legitimate').inc(legit)
        
        # 3. Simulate drift increments
        simulated_drift = random.uniform(0.05, 0.45)
        FEATURE_DRIFT_SHARE.set(simulated_drift)
        
        # 4. Simulate Agent behavior metrics
        agent_confidence = random.uniform(0.75, 0.99) if simulated_drift < 0.35 else random.uniform(0.50, 0.75)
        AGENT_CONFIDENCE.set(agent_confidence)
        
        if agent_confidence < 0.65:
            print("⚠️ Agent confidence dropped below safety threshold! Triggering mock fallback system...")
            AGENT_FALLBACK_COUNT.inc()
            
        time.sleep(3)