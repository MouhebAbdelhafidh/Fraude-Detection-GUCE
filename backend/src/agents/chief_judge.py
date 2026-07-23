import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChiefComplianceJudge")

class ChiefComplianceJudgeAgent:
    def __init__(self):
        self.role = "Chief Customs Compliance Judge"
        self.w_tabular = 0.60
        self.w_gnn = 0.40

    def evaluate_consensus(self, tabular_report: dict, topology_report: dict, metadata: dict) -> dict:
        """Aggregates sub-agent reports and enforces the multi-modal routing matrix."""
        logger.info("Initiating executive consensus deliberation loop...")
        
        t_score = tabular_report["metrics"]["tabular_anomaly_score"]
        g_risk = topology_report["metrics"]["gnn_normalized_risk"]
        
        # Apply Multi-Modal Fusion Score Formula
        ensemble_score = (t_score * self.w_tabular) + (g_risk * self.w_gnn)
        
        # Evaluate Triage Matrix
        if ensemble_score >= 0.85:
            tier = "🔴 TIER 1: RED ALERT (IMMEDIATE SEIZURE & AUDIT)"
            reasoning = f"CRITICAL SYSTEM DISCREPANCY: High behavioral alerts ({t_score}) verified by high-risk network topology tracking ({g_risk * 100:.1f}%). Conclusive match for trade fraud."
            action = "Halt customs release immediately. Dispatch physical container inspection unit."
        elif t_score == 0.0 and g_risk > 0.70:
            tier = "🟡 TIER 2: TOPOLOGICAL ANOMALY (MOCK/SHELL NETWORK)"
            reasoning = f"STRUCTURAL NETWORK EXCEPTION: Transaction parameters appear clean locally (0.0), but GNN highlights extreme hidden corporate cluster proximity ({g_risk * 100:.1f}%). High probability of evasive shell operations."
            action = "Pass cargo release but flag declaration for corporate tax background audit."
        elif t_score == 1.0 and g_risk < 0.10:
            tier = "🔵 TIER 3: LOCALIZED ANOMALY (USER ACCESS EXCEPTION)"
            reasoning = f"OPERATIONAL ACCESS EXCEPTION: Local transactional metadata flagged an anomaly ({t_score}), but network graph remains structurally pristine. Likely an isolated data entry error or shared user credential event."
            action = "Approve entry. Log IP anomaly as a potential agent credential sharing incident."
        else:
            tier = "🟢 TIER 4: COMPLIANT STANDARD PASS"
            reasoning = "STANDARD ALIGNMENT CONSENSUS: Both local metadata and global network topology confirm compliance parameters."
            action = "Automated green channel release via GUCE system."

        return {
            "evaluator": self.role,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_file_id": metadata.get("file_id"),
            "dossier_summary": {
                "numero_dossier": metadata.get("numero_dossier", "UNKNOWN"),
                "exporter": metadata.get("emetteur", "UNKNOWN"),
                "valeur_fob": metadata.get("valeur_fob", 0.0)
            },
            "deliberation_logs": [tabular_report, topology_report],
            "verdict": {
                "triage_classification": tier,
                "multi_modal_score": ensemble_score,
                "executive_reasoning_summary": reasoning,
                "operational_directive": action
            }
        }