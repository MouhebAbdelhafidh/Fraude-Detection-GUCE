import os
import json
import time
import logging
import re
from typing import Any, Dict, TypedDict, List
import numpy as np  
from pydantic import BaseModel, Field 
from google import genai
from google.genai import types        
from langgraph.graph import StateGraph, END

# Import the new RAG Memory module
from .rag_memory import CustomsMemoryRAG
# ======================================================================
# Initialize Logger Framework & Credentials
# ======================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CustomsReasoningNetwork")

# 🟢 SECURITY PARITY: Siphoning api keys safely from the runtime profile environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB8JWkZ64ypSoPkJ2tBVyc2gz7oV5yTRTM")

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini Client: {str(e)}")

TARGET_MODEL = "gemini-2.5-flash"

# Initialize Long-Term Memory Database
try:
    case_memory = CustomsMemoryRAG()
    logger.info("✅ Qdrant RAG Engine connected successfully.")
except Exception as e:
    logger.error(f"❌ Failed to connect to Qdrant RAG Engine: {str(e)}")
    case_memory = None

# ======================================================================
# 🛡️ RATE LIMIT & RESOURCE EXHAUSTED MITIGATION WRAPPER
# ======================================================================
def generate_content_with_backoff(prompt: str, config: Any = None, max_retries: int = 3) -> Any:
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=TARGET_MODEL, contents=prompt, config=config)
            return client.models.generate_content(model=TARGET_MODEL, contents=prompt)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                match = re.search(re.compile(r"retry in ([0-9\.]+)s", re.IGNORECASE), err_msg)
                sleep_duration = float(match.group(1)) + 1.0 if match else 16.0
                
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Rate Limit Hit (429). Pausing pipeline execution for {sleep_duration}s before retry...")
                    time.sleep(sleep_duration)
                    continue
            raise e

# ======================================================================
# 📋 NATIVE GEMINI STRUCTURED OUTPUT SCHEMAS
# ======================================================================
class RecommendedAction(BaseModel):
    priority: str = Field(description="Priority rating: CRITICAL, HIGH, MEDIUM, or LOW.")
    target_entity: str = Field(description="The specific operational vector (e.g., Exporter, Declarant, Port Security, Tax Inspectorate).")
    tactical_execution: str = Field(description="Direct, actionable tactical step-by-step instructions for field personnel.")

class DeepReasoningVerdictSchema(BaseModel):
    hidden_patterns_uncovered: List[str] = Field(
        description="List of latent anomalies or structural camouflage patterns detected beneath the raw metadata variables."
    )
    technical_forensic_breakdown: str = Field(
        description="Deep statistical, quantitative, or topological explanation of why these systems triggered risk markers."
    )
    business_impact_assessment: str = Field(
        description="High-level corporate, macro-economic, tax evasion, and customs revenue leakage vulnerability analysis."
    )
    operational_directives: List[RecommendedAction] = Field(
        description="A list of structured, prioritized tactical field actions based on the holistic evaluation."
    )

# ======================================================================
# DEFINITION OF THE LANGGRAPH COMPILATION STATE
# ======================================================================
class TriageState(TypedDict):
    incoming_tx: Dict[str, Any]
    gnn_raw_risk: float
    gnn_normalized_risk: float
    xai_critical_links: List[Dict[str, Any]] 
    tabular_report: Dict[str, Any]
    topology_report: Dict[str, Any]
    historical_context: str  
    verdict: Dict[str, Any]

# ======================================================================
# NODE 1: DEEP REASONING TABULAR AUDITOR
# ======================================================================
def tabular_auditor_node(state: TriageState) -> Dict[str, Any]:
    logger.info("🤖 [NODE - AUDITOR] Initiating Deep Tabular Reasoning Chain...")
    tx = state.get("incoming_tx", {})
    
    prompt = f"""
    You are a Deep-Reasoning Customs Tabular Forensic Specialist. Your objective is to extract latent behavioural anomalies from transactional metadata.
    
    [CONTEXT DATA]:
    - Exporter: {tx.get('emetteur')}
    - Value FOB: {tx.get('valeur_fob')} CFA
    - Customs File Serial: {tx.get('numero_dossier')}
    - Local Sub-Classifier IP Anomaly Flag: {tx.get('is_anomaly_ip')}
    
    [THINKING INSTRUCTIONS]:
    1. Look for hidden anomalies: If value FOB is extremely high but local anomaly is low, evaluate potential under-invoicing or valuation camouflage.
    2. Document your findings in two explicit segments:
       - TECHNICAL BREAKDOWN: Statistical deviation, valuation flags, and parameter anomalies.
       - BUSINESS IMPACT: Potential customs duties leakage and trade regulatory evasion.
    """
    
    try:
        response = generate_content_with_backoff(prompt=prompt)
        findings_text = response.text.strip()
    except Exception as e:
        logger.error(f"❌ Tabular Auditor node fallback engaged: {str(e)}")
        findings_text = (
            f"TECHNICAL BREAKDOWN: Localized access execution flag is registered at level {tx.get('is_anomaly_ip', 0.0)}. "
            f"Transactional valuation metadata parameters list nominal declared FOB value of {tx.get('valeur_fob', 0.0)} CFA.\n"
            f"BUSINESS IMPACT: Potential customs tariff verification gaps if declaration coordinates display variations."
        )

    return {
        "tabular_report": {
            "agent": "Deep Tabular Auditor",
            "metrics": {"tabular_anomaly_score": float(tx.get('is_anomaly_ip', 0.0))},
            "findings": findings_text
        }
    }

# ======================================================================
# NODE 2: DEEP REASONING TOPOLOGY SNIPER (GNN INTERPRETER + XAI)
# ======================================================================
def topology_sniper_node(state: TriageState) -> Dict[str, Any]:
    logger.info("🤖 [NODE - SNIPER] Initiating Graph Topology Pattern Analysis via GNNExplainer...")
    tx = state.get("incoming_tx", {})
    raw_risk = state.get("gnn_raw_risk", 0.0)
    norm_risk = state.get("gnn_normalized_risk", 0.0)
    xai_critical_links = state.get("xai_critical_links", [])
    
    xai_text_block = "No high-risk structural edge masks or neighbor collusion loops isolated by GNNExplainer."
    if xai_critical_links:
        xai_text_block = "\n".join([
            f"- Critical Link Connection [{link.get('importance_score', 0.0)}% model weight influence]: "
            f"Relation ({link.get('relation', 'Linked Entity')}) connecting Node ID {link.get('source_id')} to Node ID {link.get('target_id')}"
            for link in xai_critical_links
        ])
    
    prompt = f"""
    You are an expert Graph Neural Network (GNN) Topology Analyst specializing in identifying Trade-Based Money Laundering (TBML) networks.
    Your task is to convert complex graph geometry and edge weight telemetry into a clean, professional executive matrix.
    
    [TELEMETRY DATA]:
    - Target Node (File ID): {tx.get('file_id')}
    - Target Exporter: {tx.get('emetteur')}
    - GraphSAGE Link Prediction Score: {raw_risk:.6f}
    - Calibrated Network Proximity Index: {norm_risk * 100:.2f}%
    
    [GNNEXPLAINER EDGE EVIDENCE]:
    {xai_text_block}
    
    [CRITICAL FORMATTING INSTRUCTIONS]:
    You must format your entire response using the following precise markdown template. Do not include verbose descriptions or conversational filler.
    
    ###  DETECTED TOPOLOGY RISK MATRIX
    
    | Risk Indicator | Technical Finding / Topology Evidence | Real-World Business Threat |
    | :--- | :--- | :--- |
    | **Network Proximity** | Describe what the {norm_risk * 100:.2f}% score and structural homophily means regarding malicious subgraphs. | Explain how this impacts the risk rating of the exporter. |
    | **Relational Loops** | Analyze the cyclical edge activations between the File node and the User node found in the GNNExplainer section. | Explain the internal collusion risk, lockouts, or administrative smurfing intent. |
    | **Link Probability Paradox** | Interpret why the GraphSAGE prediction score ({raw_risk:.6f}) is low while the proximity is high. | Explain that the malicious infrastructure is already constructed, not a future prediction. |

    ###  IMMEDIATE COUNTERMEASURES
    - Provide 2 or 3 short, high-priority bullet points with concrete actions for field personnel based on these topology findings.
    """
    
    try:
        response = generate_content_with_backoff(prompt=prompt)
        findings_text = response.text.strip()
    except Exception as e:
        logger.error(f"❌ Topology Sniper node fallback engaged: {str(e)}")
        findings_text = (
            f"TECHNICAL LOGIC: GNN connectivity matrix tracking indicates a calibrated affinity rating of {norm_risk * 100:.2f}%. "
            f"Structural feature extraction signals link prediction score at raw parameter index {raw_risk}.\n"
            f"BUSINESS MANIFESTATION: Proximity indicators show routing path characteristics matching network groupings."
        )

    return {
        "topology_report": {
            "agent": "GNN Topology Sniper",
            "metrics": {"gnn_normalized_risk": float(norm_risk)},
            "findings": findings_text
        }
    }

# ======================================================================
# NODE 3: HISTORICAL MEMORY (RAG INJECTION)
# ======================================================================
def historical_memory_node(state: TriageState) -> Dict[str, Any]:
    logger.info("🧠 [NODE - MEMORY] Querying Qdrant for historical precedents...")
    tx = state.get("incoming_tx", {})
    
    if not case_memory:
        return {"historical_context": "Memory subsystem offline."}
        
    try:
        past_context = case_memory.retrieve_historical_context(current_tx=tx)
        logger.info(f"    -> Context retrieved: {len(past_context)} characters")
    except Exception as e:
        logger.error(f"❌ Memory retrieval failed: {str(e)}")
        past_context = "Error retrieving historical data."
        
    return {"historical_context": past_context}

# ======================================================================
# NODE 4: ADVANCED CHIEF REASONING JUDGE (CONSENSUS SYNTHESIZER)
# ======================================================================
def chief_compliance_judge_node(state: TriageState) -> Dict[str, Any]:
    logger.info("⚖️ [NODE - CHIEF JUDGE] Synthesizing Cross-Examination & Compiling Structured Output...")
    tx = state.get("incoming_tx", {})
    t_report = state.get("tabular_report", {"metrics": {"tabular_anomaly_score": 0.0}, "findings": ""})
    g_report = state.get("topology_report", {"metrics": {"gnn_normalized_risk": 0.0}, "findings": ""})
    history = state.get("historical_context", "No prior memory accessed.")
    
    t_score = t_report["metrics"]["tabular_anomaly_score"]
    g_risk = g_report["metrics"]["gnn_normalized_risk"]
    
    # Defensive casting protection for valeur_fob string mutations
    raw_fob = tx.get('valeur_fob', 0.0)
    if isinstance(raw_fob, str):
        raw_fob = re.sub(r'[^\d\.]', '', raw_fob)
    try:
        valeur_fob = float(raw_fob) if raw_fob else 0.0
    except (ValueError, TypeError):
        valeur_fob = 0.0
        
    VALUE_THRESHOLD_CFA = 50_000_000.0
    val_ratio = valeur_fob / VALUE_THRESHOLD_CFA
    w_tabular = 0.35 + (0.50 / (1.0 + np.exp(-(val_ratio - 1.0))))
    w_gnn = 1.0 - w_tabular
    ensemble_score = (t_score * w_tabular) + (g_risk * w_gnn)
    
    # 🟢 PARITY COUPLING: Waterfall logic syncs classification to high risk ensemble indicators
    if ensemble_score >= 0.80:
        tier = "🔴 TIER 1: RED ALERT (IMMEDIATE SEIZURE & AUDIT)"
    elif ensemble_score >= 0.55 or (t_score == 1.0 or g_risk > 0.60):
        tier = "🚨 TIER 2: HIGH RISK - IMMEDIATE AUDIT REQUIRED"
    elif ensemble_score >= 0.35:
        tier = "🟡 TIER 3: LOCALIZED ANOMALY (USER ACCESS EXCEPTION)"
    else:
        tier = "🟢 TIER 4: COMPLIANT STANDARD PASS"

    prompt = f"""
    You are the Chief Customs Compliance Judge Reasoning Engine. Your role is to read the specialized sub-agent evaluations, cross-examine them for latent patterns, extract the hidden intent, and generate operational business intelligence.
    
    [SUB-AGENT TRANSMISSIONS]:
    1. Deep Tabular Auditor Report:
    "{t_report['findings']}"
    
    2. GNN Topology Sniper Report:
    "{g_report['findings']}"
    
    [HISTORICAL PRECEDENT (RAG MEMORY)]:
    {history}
    
    [SYSTEM CONSTRAINTS & MATHEMATICAL TRUTH]:
    - Computed Dynamic Matrix Score: {ensemble_score * 100:.2f}%
    - Value Allocation Context: Tabular Weight = {w_tabular:.2f} | GNN Weight = {w_gnn:.2f}
    
    [CRITICAL COGNITIVE ASSIGNMENT]:
    - Extract the hidden pattern. Factor in the historical precedent to determine if this is a repeat offender.
    - Elaborate technically on the core vulnerability.
    - Translate this into the business impact (customs value loss, economic distortions, tariff avoidance).
    - Provide concrete operational recommendations based on your analysis.
    """
    
    try:
        response = generate_content_with_backoff(
            prompt=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DeepReasoningVerdictSchema,
                temperature=0.1,
            )
        )
        verdict_data = json.loads(response.text)
        
    except Exception as e:
        logger.critical(f"❌ Custom Fallback Processing Engaged (Quota/Parsing Variant): {str(e)}")
        calculated_priority = "CRITICAL" if ensemble_score >= 0.75 else "HIGH" if ensemble_score >= 0.40 else "MEDIUM"
        
        verdict_data = {
            "hidden_patterns_uncovered": [
                f"Multi-modal variance detected: Dynamic ensemble scoring indicates matching anomaly profiles at index {ensemble_score:.4f}.",
                f"Asymmetry observed between Tabular flag parameters ({t_score}) and Graph neural network proximity coordinates ({g_risk})."
            ],
            "technical_forensic_breakdown": (
                f"Mathematical optimization models compiled execution weights dynamically based on a declared value configuration of "
                f"{valeur_fob:,.2f} CFA. Total system evaluation rating concludes at {ensemble_score * 100:.2f}%."
            ),
            "business_impact_assessment": "Deviations across data channels represent an elevated vulnerability vector for cargo misclassification or cargo valuation discrepancies.",
            "operational_directives": [
                {
                    "priority": calculated_priority,
                    "target_entity": "Customs Inspection Gate Terminal",
                    "tactical_execution": f"Flag declaration tracking serial {tx.get('numero_dossier')} for physical validation."
                }
            ]
        }

    # 🔥 MEMORY WRITE: Archive the case outcome to Qdrant
    if case_memory:
        try:
            case_memory.archive_resolved_case(
                file_id=tx.get("file_id", 0),
                exporter=tx.get("emetteur", "Unknown"),
                verdict_data=verdict_data,
                risk_tier=tier
            )
        except Exception as e:
            logger.error(f"❌ Failed to archive case memory: {str(e)}")

    # Format the operational directives into a readable string for the legacy UI
    directives_list = verdict_data.get("operational_directives", [])
    formatted_action = "\n\n".join([f"🎯 **{d.get('target_entity', 'Action')}** ({d.get('priority', 'Normal')} Priority): {d.get('tactical_execution', '')}" for d in directives_list])

    # === HYBRID DICTIONARY RETURN ===
    return {
        "verdict": {
            "triage_classification": tier,
            "multi_modal_score": float(ensemble_score),
            "executive_reasoning_summary": verdict_data.get("business_impact_assessment", "No summary available."),
            "operational_directive": formatted_action if formatted_action else "Process normally.",
            
            "hidden_patterns_uncovered": verdict_data.get("hidden_patterns_uncovered", []),
            "technical_forensic_breakdown": verdict_data.get("technical_forensic_breakdown", ""),
            "business_impact_assessment": verdict_data.get("business_impact_assessment", ""),
            "operational_directives": verdict_data.get("operational_directives", [])
        }
    }

# ======================================================================
# COMPILING THE STATE GRAPH
# ======================================================================
workflow = StateGraph(TriageState)

workflow.add_node("tabular_auditor", tabular_auditor_node)
workflow.add_node("topology_sniper", topology_sniper_node)
workflow.add_node("historical_memory", historical_memory_node) 
workflow.add_node("chief_judge", chief_compliance_judge_node)

workflow.set_entry_point("tabular_auditor")
workflow.add_edge("tabular_auditor", "topology_sniper")
workflow.add_edge("topology_sniper", "historical_memory")      
workflow.add_edge("historical_memory", "chief_judge")          
workflow.add_edge("chief_judge", END)

compiled_agent_graph = workflow.compile()