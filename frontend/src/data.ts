export const AIRFLOW_DAG_ID = "mlops_fraud_pipeline_v1";
export const AIRFLOW_URL = "http://localhost:8080";
export const AIRFLOW_USERNAME = "admin";
export const AIRFLOW_DAG_ID_DEFAULT_RUN =
  "manual__2026-07-20T17:29:02.110651+00:00";

export const GRAFANA_URL = "http://localhost:3000";
export const GRAFANA_DASHBOARD_NAME = "GNN Dashboard";

export const TRIAGE_API_URL =
  "http://127.0.0.1:8000/api/v1/triage/evaluate";

export const PREDICTIONS_PATH = "/final_fraud_predictions.csv";

export type NotifType = "success" | "warning" | "danger" | "info";

export type Notification = {
  id: string;
  type: NotifType;
  title: string;
  msg: string;
  time: string;
};

export type PageKey = "audit" | "pipeline" | "triage" | "telemetry";

export const DEFAULT_NOTIFICATIONS: Notification[] = [
  {
    id: "n1",
    type: "success",
    title: "Pipeline Online",
    msg: `DAG ${AIRFLOW_DAG_ID} is active and healthy.`,
    time: "2m ago",
  },
  {
    id: "n2",
    type: "warning",
    title: "Drift Detected",
    msg: "Wasserstein drift score exceeded threshold on a monitored feature.",
    time: "1h ago",
  },
  {
    id: "n3",
    type: "info",
    title: "Model Retrained",
    msg: "GNN embedding model deployed to production.",
    time: "3h ago",
  },
];

export const TRIAGE_DEFAULTS = {
  file_id: 10016,
  numero_dossier: "EP052469EX01DM",
  emetteur: "OLAMCAM",
  valeur_fob: 439227.84,
  is_anomaly_ip: true,
};

export const RISK_FILTER_OPTIONS = [
  "All Records",
  "Risk: Critical (≥0.85)",
  "Risk: High (0.70–0.84)",
  "Risk: Medium / Threshold (0.60–0.69)",
  "Risk: Low (<0.60)",
];

export const NODE_STATE_CONFIG: Record<
  string,
  { label: string; color: string }
> = {
  success: { label: "OK", color: "#16a34a" },
  running: { label: "RUN", color: "#f97316" },
  failed: { label: "ERR", color: "#dc2626" },
  queued: { label: "WAIT", color: "#94a3b8" },
  skipped: { label: "SKIP", color: "#94a3b8" },
  up_for_retry: { label: "RETRY", color: "#f97316" },
};

export const NAV_ITEMS: { key: PageKey; label: string }[] = [
  { key: "audit", label: "Audit Ledger" },
  { key: "pipeline", label: "Pipeline & Execution" },
  { key: "triage", label: "Live Triage" },
  { key: "telemetry", label: "Telemetry & Drift" },
];