import time
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# 1. System & GNN Performance Metrics
INFERENCE_LATENCY = Histogram('gnn_inference_latency_seconds', 'Time spent running GNN forward pass')
PREDICTION_COUNT = Counter('gnn_predictions_total', 'Total fraud predictions made', ['class_type'])
FEATURE_DRIFT_SHARE = Gauge('gnn_feature_drift_share', 'Percentage of node features showing statistically significant drift')

# 2. Agent Operation Metrics
AGENT_FALLBACK_COUNT = Counter('agent_fallback_total', 'Number of times agent fell back to static rules')
AGENT_CONFIDENCE = Gauge('agent_decision_confidence', 'Confidence level of the active orchestrator agent')

def start_monitoring_server(port=8001):
    """Starts a background server scraping endpoint for Prometheus."""
    start_http_server(port)

# Example Usage inside your production inference loop:
def run_monitored_inference(agent, hetero_encoder, predictor, current_graph_data):
    start_time = time.perf_counter()
    
    with INFERENCE_LATENCY.time():
        with torch.no_grad():
            # Run Model
            x_dict = current_graph_data.x_dict
            edge_dict = current_graph_data.edge_index_dict
            embeddings = hetero_encoder(x_dict, edge_dict)
            
            # Let's assume predicting link existence on 'file' to 'user'
            edge_index = current_graph_data['file', 'assigned_to', 'user'].edge_index
            probs = predictor(embeddings['file'], embeddings['user'], edge_index)
            
    # Track output metrics
    fraud_alerts = (probs > 0.85).sum().item()
    safe_transactions = probs.shape[0] - fraud_alerts
    PREDICTION_COUNT.labels(class_type='fraud').inc(fraud_alerts)
    PREDICTION_COUNT.labels(class_type='legitimate').inc(safe_transactions)
    
    # Track Agent Health
    if agent.failed_external_api:
        AGENT_FALLBACK_COUNT.inc()
    AGENT_CONFIDENCE.set(agent.last_action_confidence)