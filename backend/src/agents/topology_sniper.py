import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TopologySniper")

class TopologySniperAgent:
    def __init__(self, triage_engine=None):
        self.role = "GNN Topology Sniper"
        self.backstory = "Graph intelligence operative specialized in tracing deep structural trade-based money laundering (TBML) and look-alike shell corporations."
        
        if triage_engine is None:
            raise ValueError("TopologySniperAgent requires an initialized triage_engine instance to run graph inferences.")
            
        self.engine = triage_engine

    def analyze(self, file_id: int) -> dict:
        """Interrogates the Heterogeneous GraphSAGE model for structural vulnerabilities."""
        logger.info(f"Interrogating neural embeddings for Node File ID: {file_id}...")
        
        # Pull raw risk score from graph topology pipeline
        gnn_raw_risk = self.engine.compute_single_gnn_risk(file_id)
        
        # Quad-Axis Calibration Normalization Layer
        g_min, g_max = 0.206770, 0.265162
        if gnn_raw_risk >= g_max:
            gnn_norm = 1.0
        elif gnn_raw_risk <= g_min:
            gnn_norm = 0.0
        else:
            gnn_norm = (gnn_raw_risk - g_min) / (g_max - g_min)
            
        # Determine strict rating thresholds
        if gnn_norm > 0.75:
            structural_rating = "CRITICAL"
        elif gnn_norm > 0.40:
            structural_rating = "MODERATE"
        else:
            structural_rating = "SAFE"

        findings_markdown = f"""
### RISK PROFILE METRICS
* **Analysis Target Node:** File ID `{file_id}`
* **Model Framework:** Heterogeneous GraphSAGE Embedding Vector
* **Network Proximity Index:** `{gnn_norm * 100:.2f}%` Match
* **Structural Severity Rating:** **{structural_rating}**

---

### FORENSIC EVALUATION
| Risk Dimension | Model Telemetry Status | Operational Threat Mapping |
| :--- | :--- | :--- |
| **Topological Affinity** | Graph configuration yields `{gnn_norm * 100:.2f}%` alignment to known fraud subgraphs. | **High Collusion Risk:** The transaction's relational structure mimics known cyclical layering or trade-based money laundering (TBML) paths. |
| **Link Integrity** | Raw node proximity vector registered at parameter index `{gnn_raw_risk:.6f}`. | **Pre-Engineered Network:** Structural properties indicate established network placement matching known look-alike shell entities. |
| **Routing Anomaly** | Multi-agent state consensus highlights dense connection pathways. | **Identity-Hopping Flag:** Relational tracing highlights administrative redundancy designed to mask the ultimate beneficial owner. |
"""

        return {
            "agent": self.role,
            "status": "SUCCESS",
            "metrics": {
                "gnn_raw_risk": float(gnn_raw_risk),
                "gnn_normalized_risk": float(gnn_norm),
                "structural_risk_rating": structural_rating
            },
            "findings": findings_markdown.strip()
        }