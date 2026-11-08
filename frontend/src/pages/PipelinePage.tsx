import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Play, RotateCcw, GitBranch,
  Clock, Terminal, Cpu, Database, Filter, Zap, Layers,
  FileCode, ArrowRight, Loader2
} from "lucide-react";

// Fallback if @/data isn't providing these
const AIRFLOW_DAG_ID = "mlops_fraud_pipeline_v1"; 
const AIRFLOW_DAG_ID_DEFAULT_RUN = ""; 

// --- Types & Interfaces ---
type NodeStatus = "completed" | "running" | "failed" | "pending" | "queued";

interface TaskInstance {
  task_id: string;
  state: string;
  try_number?: int;
}

interface DagNode {
  id: string;
  label: string;
  type: "source" | "transform" | "enrich" | "score" | "decide" | "sink";
  status: NodeStatus;
  x: number;
  y: number;
  desc: string;
}

interface Edge {
  from: string;
  to: string;
}

// --- Static DAG Configuration (Based on Airflow UI) ---
const STATIC_NODES: Omit<DagNode, "status">[] = [
  { id: "preprocess_graph_nodes", label: "Preprocess Nodes", type: "transform", x: 40, y: 80, desc: "Clean and format graph node data" },
  { id: "pyspark_data_preparation", label: "PySpark Prep", type: "source", x: 40, y: 200, desc: "Prepare raw data using PySpark" },
  
  { id: "generate_graph_tensors", label: "Graph Tensors", type: "enrich", x: 260, y: 80, desc: "Generate tensor representations for graph" },
  { id: "unsupervised_models_training", label: "Train Unsupervised", type: "score", x: 260, y: 200, desc: "Train unsupervised models" },
  
  { id: "gnn_model_training", label: "GNN Training", type: "score", x: 480, y: 20, desc: "Train Graph Neural Network" },
  { id: "evidently_data_drift_audit", label: "Data Drift Audit", type: "transform", x: 480, y: 140, desc: "Audit data drift using Evidently" },
  { id: "ensemble_risk_evaluation", label: "Ensemble Eval", type: "decide", x: 480, y: 260, desc: "Evaluate risk using model ensemble" },
  
  { id: "final_fraud_fusion", label: "Fraud Fusion", type: "decide", x: 700, y: 140, desc: "Final fusion of fraud indicators" },
  
  { id: "register_champion_model", label: "Register Model", type: "sink", x: 920, y: 140, desc: "Register champion model to artifact store" },
];

const EDGES: Edge[] = [
  { from: "preprocess_graph_nodes", to: "generate_graph_tensors" },
  { from: "pyspark_data_preparation", to: "unsupervised_models_training" },
  { from: "generate_graph_tensors", to: "gnn_model_training" },
  { from: "generate_graph_tensors", to: "evidently_data_drift_audit" },
  { from: "unsupervised_models_training", to: "ensemble_risk_evaluation" },
  { from: "gnn_model_training", to: "final_fraud_fusion" },
  { from: "evidently_data_drift_audit", to: "final_fraud_fusion" },
  { from: "ensemble_risk_evaluation", to: "final_fraud_fusion" },
  { from: "final_fraud_fusion", to: "register_champion_model" },
];

// --- Helpers & UI Mappings ---
const nodeTypeIcon: Record<string, React.ComponentType<any>> = {
  source: Database, transform: Filter, enrich: Layers,
  score: Cpu, decide: Zap, sink: FileCode,
};

const statusColor: Record<NodeStatus, string> = {
  completed: "#10b981", running: "#0ea5e9", failed: "#f43f5e",
  pending: "#94a3b8", queued: "#f59e0b",
};

const mapAirflowState = (state: string | undefined): NodeStatus => {
  if (!state) return "pending";
  const s = state.toLowerCase();
  if (["success", "completed"].includes(s)) return "completed";
  if (["running", "in_progress"].includes(s)) return "running";
  if (["failed", "error", "up_for_retry"].includes(s)) return "failed";
  if (["queued", "scheduled"].includes(s)) return "queued";
  return "pending";
};

// Inline sub-components
const StatusPill = ({ status }: { status: string }) => (
  <div className="px-2 py-0.5 rounded-full bg-slate-100 text-[11px] font-bold text-slate-600 border border-slate-200">
    {status}
  </div>
);

const Metric = ({ label, value }: { label: string; value: string }) => (
  <div className="bg-slate-50 border border-slate-100 rounded-lg p-2 flex flex-col">
    <span className="text-[10px] uppercase font-bold text-slate-400">{label}</span>
    <span className="text-[12px] font-medium text-slate-700">{value}</span>
  </div>
);

// --- Main Page Component ---
export function PipelinePage() {
  const [dagRunId, setDagRunId] = useState(AIRFLOW_DAG_ID_DEFAULT_RUN);
  const [tasks, setTasks] = useState<TaskInstance[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string>("preprocess_graph_nodes");
  
  const [taskLogs, setTaskLogs] = useState<string | null>(null);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [triggering, setTriggering] = useState(false);
  
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [taskLogs]);

  // Fetch Pipeline Status
  const fetchPipelineStatus = useCallback(async (runId: string) => {
    if (!runId?.trim()) return;
    try {
      const res = await fetch(`http://localhost:8000/api/v1/pipeline/status?dag_run_id=${encodeURIComponent(runId)}`);
      if (!res.ok) throw new Error("Failed to fetch status");
      const data = await res.json();
      setTasks(data.tasks || data.task_instances || []);
    } catch (err) {
      console.error("Pipeline fetch status error:", err);
    }
  }, []);

  // Fetch Exact Airflow Logs from FastAPI
  // Fetch Exact Airflow Logs from FastAPI
  const fetchTaskLogs = useCallback(async (taskId: string, runId: string) => {
    if (!taskId || !runId) {
      setTaskLogs("Waiting for a valid DAG Run ID to fetch logs...");
      return;
    }
    setLoadingLogs(true);
    
    // Find if the task has a specific try_number from the status payload
    const taskInfo = tasks.find(t => t.task_id === taskId);
    const tryNumber = taskInfo?.try_number || 1;

    try {
      const res = await fetch(
        `http://localhost:8000/api/v1/pipeline/logs?dag_run_id=${encodeURIComponent(runId)}&task_id=${encodeURIComponent(taskId)}&try_number=${tryNumber}`
      );
      
      if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
      
      const data = await res.json();
      let logContent = data.content; 
      
      if (!logContent) {
        setTaskLogs("Log file is empty or hasn't been generated yet.");
        return;
      }

      // Airflow sometimes returns the JSON array as a giant string. 
      // Let's attempt to parse it if it looks like JSON.
      if (typeof logContent === "string" && logContent.trim().startsWith("[")) {
        try {
          logContent = JSON.parse(logContent);
        } catch (e) {
          // If it fails to parse, we leave it as a string
        }
      }

      // If it is an array of structured Airflow JSON logs, format it beautifully
      if (Array.isArray(logContent)) {
        logContent = logContent.map((log: any) => {
          if (typeof log === "string") return log;
          
          // Format timestamp: "2026-07-25T16:49:09.035102Z" -> "2026-07-25 16:49:09"
          let timestamp = "";
          if (log.timestamp) {
             const cleanTime = log.timestamp.replace("T", " ").substring(0, 19);
             timestamp = `[${cleanTime}] `;
          }

          // Format level: "info" -> "INFO -"
          const level = log.level ? `${log.level.toUpperCase()} - ` : "";
          
          // Extract the actual log message
          const event = log.event || "";

          // Ignore empty structural group tags if you want cleaner logs
          if (event === "::endgroup::" || event.startsWith("::group::")) {
              return `--- ${event.replace(/::/g, '')} ---`; 
          }

          return `${timestamp}${level}${event}`;
        }).join("\n");
      } else if (typeof logContent === "object") {
        // Fallback for any other random JSON objects
        logContent = JSON.stringify(logContent, null, 2);
      }
      
      setTaskLogs(logContent);
    } catch (err: any) {
      setTaskLogs(`Error connecting to Gateway API: ${err.message}`);
    } finally {
      setLoadingLogs(false);
    }
  }, [tasks]);

  // Sync initial load & setup 10-second polling
  useEffect(() => {
    fetchPipelineStatus(dagRunId);
    const interval = setInterval(() => fetchPipelineStatus(dagRunId), 10000);
    return () => clearInterval(interval);
  }, [dagRunId, fetchPipelineStatus]);

  // Fetch logs on task selection
  useEffect(() => {
    fetchTaskLogs(selectedTaskId, dagRunId);
  }, [selectedTaskId, dagRunId, fetchTaskLogs]);

  // Trigger Retrain
  async function trigger() {
    setTriggering(true);
    try {
      const res = await fetch("http://localhost:8000/api/v1/pipeline/trigger", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dag_id: AIRFLOW_DAG_ID }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to trigger pipeline");
      
      const newRunId = data.dag_run_id || data.run_id;
      if (newRunId) {
        setDagRunId(newRunId);
        fetchPipelineStatus(newRunId);
      }
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Could not reach API Gateway");
    } finally {
      setTriggering(false);
    }
  }

  // Map dynamic states into the static node layout
  const nodes: DagNode[] = STATIC_NODES.map((n) => {
    const taskInfo = tasks.find((t) => t.task_id === n.id);
    return { ...n, status: mapAirflowState(taskInfo?.state) };
  });

  const selectedNode = nodes.find((n) => n.id === selectedTaskId) || nodes[0];
  const isRunning = nodes.some(n => n.status === "running");

  return (
    <div className="space-y-5 max-w-[1400px] mx-auto pb-10">
      
      {/* Run Controls Header */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200">
        <div className="flex items-center justify-between px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-orange-600 flex items-center justify-center shadow-inner">
              <GitBranch size={18} className="text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[15px] font-bold text-slate-800">{AIRFLOW_DAG_ID}</span>
                <StatusPill status={isRunning ? "Running" : "Idle"} />
              </div>
              <div className="text-[12px] text-slate-500 mt-0.5 flex items-center gap-2">
                <span className="font-mono bg-slate-100 px-1.5 rounded">{dagRunId || "No Active Run"}</span>
                <span>·</span>
                <span className="flex items-center text-orange-500">
                  <Clock size={12} className="mr-1 animate-pulse" /> Auto-refreshing (10s)
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={trigger}
              disabled={triggering}
              className={`flex items-center gap-1.5 text-[13px] font-semibold px-4 py-2.5 rounded-lg text-white transition-all shadow-sm ${
                triggering ? "bg-slate-400 cursor-not-allowed" : "bg-orange-500 hover:bg-orange-600 hover:-translate-y-0.5"
              }`}
            >
              {triggering ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
              {triggering ? "Dispatching..." : "Trigger Retrain"}
            </button>
            <button 
              onClick={() => { fetchPipelineStatus(dagRunId); fetchTaskLogs(selectedTaskId, dagRunId); }}
              className="flex items-center gap-1.5 text-[13px] font-semibold px-4 py-2.5 rounded-lg text-slate-600 border border-slate-200 hover:bg-slate-50 transition-colors"
            >
              <RotateCcw size={15} /> Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid: Canvas + Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        
        {/* SVG Pipeline Canvas */}
        <div className="bg-white rounded-xl lg:col-span-2 shadow-sm border border-slate-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-100 flex justify-between items-center">
            <div>
              <h2 className="text-sm font-bold text-slate-800">Execution DAG</h2>
              <p className="text-[12px] text-slate-500">Click a node to inspect logs</p>
            </div>
            <div className="flex items-center gap-3 text-[11px]">
              {(["completed", "running", "failed", "queued", "pending"] as NodeStatus[]).map((s) => (
                <span key={s} className="flex items-center gap-1.5 text-slate-500 capitalize">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: statusColor[s] }} />
                  {s}
                </span>
              ))}
            </div>
          </div>

          <div className="relative bg-[radial-gradient(circle,#cbd5e1_1px,transparent_1px)] bg-[size:24px_24px] overflow-x-auto" style={{ minHeight: 340 }}>
            <div className="min-w-[1100px] h-full absolute inset-0">
              <DagCanvas nodes={nodes} edges={EDGES} selected={selectedTaskId} onSelect={setSelectedTaskId} />
            </div>
          </div>
        </div>

        {/* Node Inspector */}
        <div className="bg-white shadow-sm border border-slate-200 rounded-xl flex flex-col">
          <div className="px-5 py-4 border-b border-slate-100">
            <h2 className="text-sm font-bold text-slate-800">Node Inspector</h2>
            <p className="text-[12px] text-slate-500">{selectedNode?.label || "Select a node"}</p>
          </div>
          
          <div className="p-5 flex-1 space-y-5">
            {selectedNode ? (
              <>
                <div className="flex items-center gap-4">
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center shadow-sm"
                    style={{ background: `${statusColor[selectedNode.status]}15`, border: `1px solid ${statusColor[selectedNode.status]}30` }}
                  >
                    {(() => {
                      const Icon = nodeTypeIcon[selectedNode.type];
                      return <Icon size={22} style={{ color: statusColor[selectedNode.status] }} />;
                    })()}
                  </div>
                  <div className="flex-1">
                    <div className="text-[14px] font-bold text-slate-800">{selectedNode.label}</div>
                    <div className="text-[11px] font-mono text-slate-500 mt-1 truncate">{selectedNode.id}</div>
                  </div>
                  <StatusPill status={selectedNode.status.charAt(0).toUpperCase() + selectedNode.status.slice(1)} />
                </div>

                <p className="text-[13px] text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-100">
                  {selectedNode.desc}
                </p>

                <div className="grid grid-cols-2 gap-3 pt-1">
                  <Metric label="Type" value={selectedNode.type} />
                  <Metric label="Status" value={selectedNode.status} />
                </div>

                {/* Edges IO Data */}
                <div className="space-y-4 pt-2 border-t border-slate-100">
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Upstream (Inputs)</div>
                    <div className="space-y-1.5">
                      {EDGES.filter((e) => e.to === selectedNode.id).map((e, i) => (
                        <div key={i} className="flex items-center gap-2 text-[11px] bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-2">
                          <ArrowRight size={12} className="text-slate-400 shrink-0" />
                          <span className="font-mono text-slate-600 truncate">{e.from}</span>
                        </div>
                      ))}
                      {EDGES.filter((e) => e.to === selectedNode.id).length === 0 && (
                        <div className="text-[11px] text-slate-400 italic">No upstream nodes (Root Node)</div>
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2">Downstream (Outputs)</div>
                    <div className="space-y-1.5">
                      {EDGES.filter((e) => e.from === selectedNode.id).map((e, i) => (
                        <div key={i} className="flex items-center gap-2 text-[11px] bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-2">
                          <ArrowRight size={12} className="text-slate-400 shrink-0" />
                          <span className="font-mono text-slate-600 truncate">{e.to}</span>
                        </div>
                      ))}
                      {EDGES.filter((e) => e.from === selectedNode.id).length === 0 && (
                        <div className="text-[11px] text-slate-400 italic">No downstream nodes (Terminal Node)</div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center text-slate-400 py-10 text-sm">No node selected</div>
            )}
          </div>
        </div>
      </div>

      {/* Task Logs Terminal */}
      <div className="shadow-sm border border-slate-800 bg-[#0f172a] rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-slate-800 bg-[#1e293b] flex justify-between items-center">
          <div className="flex items-center gap-2 text-slate-300">
            <Terminal size={14} className="text-orange-500" /> 
            <span className="text-[13px] font-semibold">Node Logs: {selectedTaskId}</span>
          </div>
          {loadingLogs && <Loader2 size={14} className="text-orange-500 animate-spin" />}
        </div>
        
        <div 
          ref={logRef} 
          className="p-5 font-mono text-[12px] leading-relaxed overflow-y-auto text-slate-300 custom-scrollbar" 
          style={{ height: 260 }}
        >
          {loadingLogs ? (
            <div className="space-y-3">
              <div className="h-4 w-1/3 bg-slate-800 rounded animate-pulse" />
              <div className="h-4 w-1/2 bg-slate-800 rounded animate-pulse" />
              <div className="h-4 w-1/4 bg-slate-800 rounded animate-pulse" />
            </div>
          ) : (
            <pre className="whitespace-pre-wrap">{taskLogs}</pre>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- DAG Canvas Component ---------------- */
function DagCanvas({ 
  nodes, 
  edges, 
  selected, 
  onSelect 
}: {
  nodes: DagNode[];
  edges: Edge[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const NODE_W = 150;
  const NODE_H = 44;

  return (
    <div className="relative w-full h-full">
      {/* 1. Draw SVG Edges */}
      <svg className="absolute inset-0 pointer-events-none" width="100%" height="100%">
        {edges.map((e, i) => {
          const fromNode = nodes.find((n) => n.id === e.from);
          const toNode = nodes.find((n) => n.id === e.to);
          
          if (!fromNode || !toNode) return null;

          // Connect from right-center to left-center
          const startX = fromNode.x + NODE_W;
          const startY = fromNode.y + NODE_H / 2;
          const endX = toNode.x;
          const endY = toNode.y + NODE_H / 2;
          
          // Determine bezier curve controls for a smooth horizontal S-curve
          const midX = (startX + endX) / 2;

          return (
            <g key={i}>
              <path
                d={`M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`}
                fill="none"
                stroke={statusColor[fromNode.status]}
                strokeWidth="2"
                strokeOpacity="0.4"
                className="transition-all duration-500"
              />
              <circle cx={endX} cy={endY} r="3" fill={statusColor[fromNode.status]} opacity="0.6" />
            </g>
          );
        })}
      </svg>

      {/* 2. Render HTML Nodes Positioned Over SVG */}
      {nodes.map((n) => {
        const isSelected = selected === n.id;
        const Icon = nodeTypeIcon[n.type] || Database;
        const color = statusColor[n.status];

        return (
          <div
            key={n.id}
            onClick={() => onSelect(n.id)}
            className={`absolute flex items-center gap-2.5 px-3 py-2 rounded-lg cursor-pointer transition-all shadow-sm border bg-white hover:-translate-y-0.5 ${
              isSelected 
                ? "ring-2 ring-orange-500 border-orange-300 shadow-md z-20" 
                : "border-slate-200 hover:border-slate-300 hover:shadow z-10"
            }`}
            style={{ 
              left: n.x, 
              top: n.y,
              width: NODE_W,
              height: NODE_H,
              borderLeft: `4px solid ${color}`
            }}
          >
            <div className="shrink-0 flex items-center justify-center relative">
              <Icon size={14} style={{ color: color }} />
              {n.status === "running" && (
                <span className="absolute -top-1 -right-1 flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-sky-500"></span>
                </span>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] font-bold text-slate-700 truncate select-none leading-tight">
                {n.label}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}