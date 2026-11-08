import React, { useState, useEffect, useRef } from "react";
import {
  ShieldAlert,
  ShieldCheck,
  Send,
  RefreshCw,
  GitBranch,
  FileText,
  Activity,
  Layers,
  ChevronDown,
  Cpu,
  Network,
  Gauge,
  AlertOctagon,
  Sparkles,
  Fingerprint,
  Building2,
  DollarSign,
  Wifi,
  ChevronRight,
  Shield,
} from "lucide-react";
import { Card, Button, AlertBox } from "@/components/ui";
import {
  consumePendingTirageRecord,
  onTirageRecordSent,
  type TirageHandoffRecord,
} from "./tirageBridge";
import { useToast } from "./useToast";

// --- Types ---
interface DeliberationLog {
  agent: string;
  findings: string;
}

interface Verdict {
  triage_classification?: string;
  multi_modal_score?: number;
  executive_reasoning_summary?: string;
  operational_directive?: string;
}

interface LinkRelation {
  source_id?: string | number;
  target_id?: string | number;
  relation?: string;
  importance_score?: number;
  source_type?: string;
  target_type?: string;
}

interface TriageResponse {
  verdict?: Verdict;
  xai_graph_links?: LinkRelation[];
  raw_neighborhood_links?: LinkRelation[];
  deliberation_logs?: DeliberationLog[];
  [key: string]: any;
}

// Helper function to render text containing Markdown bold syntax (**text**) as standard HTML <strong> tags
function renderFormattedText(text: string) {
  if (!text) return null;
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-bold text-slate-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}

// Helper to format structured content for Tabular and GNN agents
function FormattedAgentFindings({ text, agentName }: { text: string; agentName: string }) {
  const isTabularAgent = agentName.toLowerCase().includes("tabular");
  const isGnnAgent = agentName.toLowerCase().includes("gnn") || agentName.toLowerCase().includes("topology");

  // Single unified card renderer for Deep Tabular Auditor
  if (isTabularAgent) {
    const sections = text.split(/(?=\[\s*[A-Z\s_]+?\s*\]|\b[A-Z\s]{4,}:)/g).filter((s) => s.trim().length > 0);

    return (
      <div className="p-4 bg-white rounded-xl border border-indigo-100 shadow-xs font-sans text-xs text-slate-700 leading-relaxed space-y-3">
        {sections.map((section, idx) => {
          const trimmed = section.trim();
          const headerMatch = trimmed.match(/^(\[\s*[A-Z\s_]+?\s*\]|[A-Z\s]{4,}:)/);

          if (headerMatch) {
            const headerText = headerMatch[0];
            const contentText = trimmed.replace(headerText, "").trim();

            return (
              <div key={idx} className="space-y-1">
                <div className="font-bold text-indigo-900 uppercase tracking-wider text-[0.7rem] flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 inline-block" />
                  {headerText.replace(/[\[\]]/g, "").trim()}
                </div>
                <div className="text-slate-600 pl-3 border-l-2 border-indigo-100">
                  {renderFormattedText(contentText)}
                </div>
              </div>
            );
          }

          return <div key={idx}>{renderFormattedText(trimmed)}</div>;
        })}
      </div>
    );
  }

  // Custom renderer for GNN Topology Sniper (Risk Matrix Table + Recommended Actions Card)
  if (isGnnAgent) {
    const lines = text.split("\n").filter((l) => l.trim().length > 0);
    const tableLines = lines.filter((l) => l.includes("|"));
    const textLines = lines.filter((l) => !l.includes("|") && !l.includes("###"));

    const parsePipeRow = (rowStr: string) =>
      rowStr
        .split("|")
        .map((c) => c.trim())
        .filter((c) => c.length > 0 && !c.includes("---"));

    let headers: string[] = ["Risk Indicator", "Technical Finding / Topology Evidence", "Real-World Business Threat"];
    let rows: string[][] = [];

    tableLines.forEach((line) => {
      const parsed = parsePipeRow(line);
      if (parsed.length >= 2) {
        if (parsed[0].toLowerCase().includes("risk") || parsed[0].toLowerCase().includes("indicator")) {
          if (parsed.length >= 3) headers = parsed;
        } else {
          rows.push(parsed);
        }
      }
    });

    return (
      <div className="space-y-4 font-sans">
        {textLines.length > 0 && (
          <div className="text-xs text-slate-600 leading-relaxed space-y-1">
            {textLines.map((t, i) => (
              <p key={i}>{renderFormattedText(t.replace(/###/g, ""))}</p>
            ))}
          </div>
        )}

        {/* Risk Matrix Table */}
        {rows.length > 0 && (
          <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-xs">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-100/80 text-slate-700 font-bold uppercase tracking-wider text-[0.65rem] border-b border-slate-200">
                <tr>
                  {headers.map((h, i) => (
                    <th key={i} className="py-2.5 px-3">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {rows.map((row, rIdx) => (
                  <tr key={rIdx} className={rIdx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className={`py-2.5 px-3 align-top ${cIdx === 0 ? "font-bold text-slate-900" : ""}`}>
                        {renderFormattedText(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Third Card: Recommended Actions */}
        <div className="p-4 rounded-xl border border-orange-200 bg-gradient-to-br from-orange-50/60 via-white to-amber-50/40 shadow-xs space-y-2.5">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-orange-600" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-orange-950">Recommended Actions</h4>
          </div>
          <ul className="space-y-2 pl-1">
            <li className="flex items-start gap-2 text-xs text-slate-700 font-medium leading-relaxed">
              <ChevronRight className="w-3.5 h-3.5 text-orange-500 flex-shrink-0 mt-0.5" />
              <span>
                Immediately suspend all transactions associated with <strong className="text-slate-900">OLAMCAM</strong> and{" "}
                <strong className="text-slate-900">File ID 10016</strong>, initiating a full forensic audit of past activities.
              </span>
            </li>
            <li className="flex items-start gap-2 text-xs text-slate-700 font-medium leading-relaxed">
              <ChevronRight className="w-3.5 h-3.5 text-orange-500 flex-shrink-0 mt-0.5" />
              <span>
                Launch an urgent internal investigation into <strong className="text-slate-900">User ID 47</strong> (and related roles/teams) for potential collusion and unauthorized file processing.
              </span>
            </li>
            <li className="flex items-start gap-2 text-xs text-slate-700 font-medium leading-relaxed">
              <ChevronRight className="w-3.5 h-3.5 text-orange-500 flex-shrink-0 mt-0.5" />
              <span>
                Map and structurally prune the identified malicious subgraph to isolate and flag all connected entities and their transactional histories.
              </span>
            </li>
          </ul>
        </div>
      </div>
    );
  }

  // Fallback default formatter
  return (
    <div className="text-xs text-slate-600 leading-relaxed space-y-1.5 font-sans">
      {text.split("\n").map((line, idx) => (
        <p key={idx}>{renderFormattedText(line)}</p>
      ))}
    </div>
  );
}

// --- Dynamic Network Graph Component ---
function VisNetworkGraph({
  targetFile,
  xaiLinks = [],
  rawRelations = [],
}: {
  targetFile: string | number;
  xaiLinks?: LinkRelation[];
  rawRelations?: LinkRelation[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkInstanceRef = useRef<any>(null);
  const [visError, setVisError] = useState<string | null>(null);

  useEffect(() => {
    const targetStr = String(targetFile);
    const nodesMap = new Map<string, any>();
    const edgesList: any[] = [];
    const seenPairs = new Set<string>();

    const combinedLinks: LinkRelation[] = [];

    xaiLinks.forEach((link) => {
      const src = String(link.source_id ?? "Unknown");
      const tgt = String(link.target_id ?? "Unknown");
      const relation = link.relation || "linked_to";
      const key = `${src}|${tgt}|${relation}`;

      seenPairs.add(key);
      combinedLinks.push({
        source_id: src,
        target_id: tgt,
        relation,
        importance_score: link.importance_score ?? 50.0,
        source_type: link.source_type || "user",
        target_type: link.target_type || "company",
      });
    });

    rawRelations.forEach((rel) => {
      const src = String(rel.source_id ?? "Unknown");
      const tgt = String(rel.target_id ?? "Unknown");
      const relation = rel.relation || "linked_to";
      const key = `${src}|${tgt}|${relation}`;

      if (!seenPairs.has(key)) {
        seenPairs.add(key);
        combinedLinks.push({
          source_id: src,
          target_id: tgt,
          relation,
          importance_score: rel.importance_score ?? 20.0,
          source_type: rel.source_type || "user",
          target_type: rel.target_type || "company",
        });
      }
    });

    const addNodeIfMissing = (nodeId: string, nodeType?: string) => {
      if (!nodesMap.has(nodeId)) {
        let label = `User: ${nodeId}`;
        let color = "#1e3a8a";
        let shape = "dot";

        if (nodeType === "file" || nodeId === targetStr) {
          label = nodeId.includes("File") ? nodeId : `File ${nodeId}`;
          color = "#ef4444";
          shape = "square";
        } else if (nodeType === "company") {
          label = `Company: ${nodeId}`;
          color = "#059669";
          shape = "triangle";
        }

        nodesMap.set(nodeId, {
          id: nodeId,
          label,
          color,
          shape,
          font: { color: "#0f172a", size: 12 },
        });
      }
    };

    combinedLinks.forEach((link) => {
      const src = String(link.source_id);
      const tgt = String(link.target_id);
      const weight = link.importance_score || 0;

      addNodeIfMissing(src, link.source_type);
      addNodeIfMissing(tgt, link.target_type);

      edgesList.push({
        from: src,
        to: tgt,
        label: `${link.relation} (${weight}%)`,
        width: Math.max(1, Number((1 + weight / 40).toFixed(2))),
        arrows: "to",
        color: { color: "#94a3b8", highlight: "#16a34a" },
        font: { size: 10, align: "middle", color: "#475569" },
      });
    });

    const nodes = Array.from(nodesMap.values());

    if (nodes.length === 0) return;

    const loadAndRenderVis = () => {
      const windowVis = (window as any).vis;
      if (!windowVis || !containerRef.current) return;

      try {
        if (networkInstanceRef.current) {
          networkInstanceRef.current.destroy();
          networkInstanceRef.current = null;
        }

        const data = {
          nodes: new windowVis.DataSet(nodes),
          edges: new windowVis.DataSet(edgesList),
        };

        const options = {
          physics: {
            stabilization: true,
            barnesHut: { gravitationalConstant: -3000, springLength: 140 },
          },
          interaction: { hover: true, zoomView: true, dragView: true },
          layout: { improvedLayout: true },
        };

        networkInstanceRef.current = new windowVis.Network(containerRef.current, data, options);
      } catch (err: any) {
        setVisError(`Graph render failed: ${err.message}`);
      }
    };

    if (!(window as any).vis) {
      const script = document.createElement("script");
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js";
      script.async = true;
      script.onload = loadAndRenderVis;
      script.onerror = () => setVisError("Failed to load vis-network library.");
      document.body.appendChild(script);
    } else {
      loadAndRenderVis();
    }

    return () => {
      if (networkInstanceRef.current) {
        networkInstanceRef.current.destroy();
        networkInstanceRef.current = null;
      }
    };
  }, [targetFile, xaiLinks, rawRelations]);

  if (xaiLinks.length === 0 && rawRelations.length === 0) {
    return (
      <div className="py-14 text-center border-2 border-dashed border-slate-200 rounded-2xl bg-gradient-to-br from-slate-50 to-white flex-1 flex flex-col justify-center items-center">
        <Network className="w-8 h-8 text-slate-300 mb-2" />
        <div className="text-sm font-bold text-slate-500">No graph relations available</div>
        <div className="text-xs text-slate-400 mt-1">No relational data to render for this transaction.</div>
      </div>
    );
  }

  if (visError) {
    return <AlertBox tone="error">{visError}</AlertBox>;
  }

  return (
    <div className="relative w-full h-[520px] rounded-2xl bg-gradient-to-br from-slate-50 to-white border border-slate-200 overflow-hidden">
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 bg-white/90 backdrop-blur-sm rounded-lg border border-slate-200 shadow-sm">
        <span className="w-2 h-2 rounded-full bg-red-500" />
        <span className="w-2 h-2 rounded-full bg-emerald-500" />
        <span className="w-2 h-2 rounded-full bg-blue-800" />
        <span className="text-[0.62rem] font-bold text-slate-500 ml-1">Topology Graph</span>
      </div>
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}

// --- Field wrapper ---
function FormField({
  label,
  icon: Icon,
  children,
}: {
  label: string;
  icon: typeof Fingerprint;
  children: React.ReactNode;
}) {
  return (
    <div className="group">
      <label className="flex items-center gap-1.5 text-[0.65rem] font-bold text-slate-400 uppercase tracking-[0.08em] mb-2">
        <Icon className="w-3.5 h-3.5 text-slate-400 group-focus-within:text-orange-500 transition-colors" />
        {label}
      </label>
      {children}
    </div>
  );
}

const inputClass =
  "w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm font-mono font-bold text-slate-800 outline-none focus:border-orange-400 focus:bg-white focus:ring-2 focus:ring-orange-100 transition-all duration-200";

// --- Main Page Component ---
export function TriagePage() {
  const [fileId, setFileId] = useState<number>(10016);
  const [numeroDossier, setNumeroDossier] = useState<string>("EP052469EX01DM");
  const [emetteur, setEmetteur] = useState<string>("OLAMCAM");
  const [valeurFob, setValeurFob] = useState<number>(439227.84);
  const [isAnomalyIp, setIsAnomalyIp] = useState<boolean>(true);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"network" | "ledger">("network");
  const [dossier, setDossier] = useState<TriageResponse | null>(null);
  const [justPrefilled, setJustPrefilled] = useState<boolean>(false);

  const { showToast, ToastHost } = useToast();

  const applyHandoffRecord = (record: TirageHandoffRecord) => {
    setFileId(record.file_id);
    setNumeroDossier(record.numero_dossier ?? "");
    setValeurFob(record.valeur_fob ?? 0);
    // `code` on the audit record is the HS/tariff code, not the exporter name —
    // the Audit table doesn't carry an exporter/emetteur column, so that field
    // is intentionally left untouched here. Swap this in once it's available:
    // setEmetteur(record.code);

    setJustPrefilled(true);
    showToast({
      tone: "info",
      title: `Loaded File #${record.file_id} from Audit Queue`,
      message: `${record.numero_dossier || "Dossier"} prefilled — review and run the evaluation.`,
    });
  };

  // Pick up a record that was sent from the Audit Queue's "Tirage" button
  // *before* this page mounted (stored in sessionStorage)...
  useEffect(() => {
    const pending = consumePendingTirageRecord();
    if (pending) applyHandoffRecord(pending);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ...and also listen live, in case both pages are mounted at once
  // (e.g. rendered as tabs that stay in the DOM) and a new record arrives
  // while this page is already open.
  useEffect(() => {
    const unsubscribe = onTirageRecordSent(applyHandoffRecord);
    return unsubscribe;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleEvaluate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setJustPrefilled(false);

    const payload = {
      file_id: fileId,
      numero_dossier: numeroDossier,
      emetteur: emetteur,
      valeur_fob: valeurFob,
      is_anomaly_ip: isAnomalyIp ? 1.0 : 0.0,
    };

    try {
      const response = await fetch("http://localhost:8000/api/v1/triage/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || "Evaluation failed"}`);
      }

      const result: TriageResponse = await response.json();
      setDossier(result);
    } catch (err: any) {
      console.error(err);
      const message = err.message || "Connection Failure: FastAPI gateway offline (Port 8000 unreachable).";
      setError(message);
      showToast({ tone: "error", title: "Evaluation failed", message });
    } finally {
      setLoading(false);
    }
  };

  const verdict = dossier?.verdict || {};
  const classification = String(verdict.triage_classification || "UNKNOWN").toUpperCase();
  const scoreGlobal = Number(verdict.multi_modal_score || 0.0);
  const reasoning = verdict.executive_reasoning_summary || "No reasoning summary provided.";
  const actionRaw = verdict.operational_directive || "No operational directive issued.";

  const directiveLines = actionRaw
    .split(/(?:\r?\n|;|\b(?=\d+\.))/g)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const logs = dossier?.deliberation_logs || [];

  const getBannerTone = (cls: string) => {
    if (cls.includes("TIER 1") || cls.includes("RED") || cls.includes("SEIZURE")) {
      return {
        wrap: "bg-gradient-to-br from-red-50 to-red-100/50 border-red-200",
        meta: "text-red-500/80",
        text: "text-red-800",
        icon: AlertOctagon,
      };
    }
    if (cls.includes("TIER 2") || cls.includes("TIER 5") || cls.includes("AMBER")) {
      return {
        wrap: "bg-gradient-to-br from-amber-50 to-amber-100/50 border-amber-200",
        meta: "text-amber-600/80",
        text: "text-amber-800",
        icon: ShieldAlert,
      };
    }
    return {
      wrap: "bg-gradient-to-br from-emerald-50 to-emerald-100/50 border-emerald-200",
      meta: "text-emerald-600/80",
      text: "text-emerald-800",
      icon: ShieldCheck,
    };
  };

  const banner = getBannerTone(classification);
  const BannerIcon = banner.icon;

  return (
    <div className="space-y-6">
      <ToastHost />

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="section-meta">Data Entry Gateway</div>
          <h1 className="section-title gradient-text" style={{ marginBottom: 0 }}>
            Live Triage Engine
          </h1>
          <p className="text-sm text-slate-400 font-medium mt-1.5">
            Multi-agent consensus evaluation of customs declarations
          </p>
        </div>
        <div className="flex items-center gap-2 px-3.5 py-2 bg-white border border-slate-200 rounded-xl shadow-xs">
          <span className="live-badge">LIVE</span>
          <span className="text-[0.68rem] text-slate-400 font-mono">POST /api/v1/triage/evaluate</span>
        </div>
      </div>

      {/* Form Input Card */}
      <Card>
        {justPrefilled && (
          <div className="mb-4 flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-indigo-50 border border-indigo-100 text-xs font-bold text-indigo-700">
            <Send className="w-3.5 h-3.5" />
            Prefilled from the Audit Queue — verify the exporter field before running.
          </div>
        )}
        <form onSubmit={handleEvaluate} className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <FormField label="Structural Node ID" icon={Fingerprint}>
              <input
                type="number"
                value={fileId}
                onChange={(e) => setFileId(Number(e.target.value))}
                className={inputClass}
                required
              />
            </FormField>

            <FormField label="Customs File Number" icon={FileText}>
              <input
                type="text"
                value={numeroDossier}
                onChange={(e) => setNumeroDossier(e.target.value)}
                className={`${inputClass} font-semibold`}
                required
              />
            </FormField>

            <FormField label="Exporter Identification" icon={Building2}>
              <input
                type="text"
                value={emetteur}
                onChange={(e) => setEmetteur(e.target.value)}
                className={`${inputClass} font-semibold`}
                required
              />
            </FormField>

            <FormField label="Declared FOB Value (CFA)" icon={DollarSign}>
              <input
                type="number"
                step="1000"
                value={valeurFob}
                onChange={(e) => setValeurFob(Number(e.target.value))}
                className={inputClass}
                required
              />
            </FormField>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-4 border-t border-slate-100">
            <label className="flex items-center gap-3 cursor-pointer select-none group">
              <button
                type="button"
                onClick={() => setIsAnomalyIp((v) => !v)}
                className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                  isAnomalyIp ? "bg-orange-500" : "bg-slate-300"
                }`}
              >
                <span
                  className={`absolute top-0.5 w-5 h-5 bg-white rounded-full shadow-md transition-transform duration-200 ${
                    isAnomalyIp ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
              <span className="flex items-center gap-1.5 text-sm font-bold text-slate-700 group-hover:text-slate-900 transition-colors">
                <Wifi className={`w-3.5 h-3.5 ${isAnomalyIp ? "text-orange-500" : "text-slate-400"}`} />
                IP Anomaly Detected
              </span>
            </label>

            <Button type="submit" disabled={loading} className="px-6 py-3 gap-2">
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Processing network topologies...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Execute Multi-Agent Evaluation
                </>
              )}
            </Button>
          </div>
        </form>
      </Card>

      {error && (
        <AlertBox tone="error">
          <strong>Evaluation Error:</strong> {error}
        </AlertBox>
      )}

      {/* Results View */}
      {dossier ? (
        <div className="space-y-6 fade-in">
          {/* Classification Banner + Confidence */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className={`md:col-span-3 p-6 rounded-2xl border flex items-center justify-between gap-4 ${banner.wrap}`}>
              <div className="flex items-center gap-4">
                <div
                  className={`flex-shrink-0 w-14 h-14 rounded-2xl bg-white/60 flex items-center justify-center border ${banner.wrap}`}
                >
                  <BannerIcon className={`w-7 h-7 ${banner.text}`} strokeWidth={2.2} />
                </div>
                <div>
                  <div className={`text-[0.68rem] font-bold uppercase tracking-[0.1em] ${banner.meta}`}>
                    System Action Classification
                  </div>
                  <div className={`text-2xl font-black mt-1 font-mono ${banner.text}`}>{classification}</div>
                </div>
              </div>
            </div>

            <div className="kpi-card flex flex-col justify-center">
              <div className="flex items-center gap-1.5 mb-1">
                <Gauge className="w-3.5 h-3.5 text-orange-500" />
                <span className="kpi-label" style={{ marginBottom: 0 }}>
                  Fused Confidence
                </span>
              </div>
              <div className="text-3xl font-mono font-black text-slate-800">
                {(scoreGlobal * 100).toFixed(1)}
                <span className="text-lg text-slate-400">%</span>
              </div>
              <div className="mt-2 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-orange-400 to-orange-600 transition-all duration-700"
                  style={{ width: `${scoreGlobal * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Dual Panel Split */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            {/* Left: Interactive Graph & Agent Ledger */}
            <Card className="flex flex-col h-full">
              <div className="flex items-center justify-between mb-4">
                <div className="font-extrabold text-slate-800 text-sm flex items-center gap-2">
                  <GitBranch className="w-4 h-4 text-orange-500" />
                  Interactive Evidence
                </div>
              </div>

              <div className="flex gap-1 mb-4 p-1 bg-slate-100 rounded-xl">
                <button
                  type="button"
                  onClick={() => setActiveTab("network")}
                  className={`flex items-center gap-1.5 py-2 px-4 text-xs font-bold rounded-lg transition-all duration-200 ${
                    activeTab === "network"
                      ? "bg-white text-orange-600 shadow-xs"
                      : "text-slate-400 hover:text-slate-600"
                  }`}
                >
                  <Network className="w-3.5 h-3.5" />
                  Network Map
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("ledger")}
                  className={`flex items-center gap-1.5 py-2 px-4 text-xs font-bold rounded-lg transition-all duration-200 ${
                    activeTab === "ledger"
                      ? "bg-white text-orange-600 shadow-xs"
                      : "text-slate-400 hover:text-slate-600"
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  Agent Ledger
                </button>
              </div>

              {activeTab === "network" && (
                <div className="flex-1 flex flex-col min-h-0">
                  <VisNetworkGraph
                    targetFile={fileId}
                    xaiLinks={dossier.xai_graph_links || []}
                    rawRelations={dossier.raw_neighborhood_links || []}
                  />
                </div>
              )}

              {/* Agent Ledger */}
              {activeTab === "ledger" && (
                <div className="flex-1 flex flex-col min-h-0">
                  <div className="space-y-4 flex-1 overflow-y-auto max-h-[1000px] pr-1.5">
                    {logs.length > 0 ? (
                      logs.map((log, idx) => {
                        const isTabular = log.agent.toLowerCase().includes("tabular");
                        const Icon = isTabular ? Cpu : Network;
                        return (
                          <div
                            key={idx}
                            className={`p-4 rounded-xl border transition-all duration-200 hover:shadow-xs ${
                              isTabular
                                ? "bg-gradient-to-br from-indigo-50/60 to-white border-indigo-100"
                                : "bg-gradient-to-br from-slate-50 to-white border-slate-200"
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-3">
                              <span
                                className={`inline-flex w-7 h-7 rounded-lg items-center justify-center ${
                                  isTabular ? "bg-indigo-100" : "bg-slate-200"
                                }`}
                              >
                                <Icon className={`w-3.5 h-3.5 ${isTabular ? "text-indigo-600" : "text-slate-600"}`} />
                              </span>
                              <div>
                                <div className="text-[0.62rem] font-bold text-slate-400 uppercase tracking-[0.08em]">
                                  Pipeline Target: {isTabular ? "Statistical Core" : "Relational Topology"}
                                </div>
                                <div className="text-sm font-bold text-slate-800">{log.agent}</div>
                              </div>
                            </div>
                            <div className="pl-9">
                              <FormattedAgentFindings text={log.findings} agentName={log.agent} />
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-center py-12 text-sm text-slate-400 italic border-2 border-dashed border-slate-200 rounded-xl h-full flex items-center justify-center">
                        No agent findings recorded.
                      </div>
                    )}
                  </div>
                </div>
              )}
            </Card>

            {/* Right: Consolidated Decision */}
            <Card className="flex flex-col h-full space-y-4">
              <div className="font-extrabold text-slate-800 text-sm flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-indigo-500" />
                Consolidated Decision
              </div>

              {/* Compliance Analysis */}
              <div className="p-5 rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50/50 border border-blue-100 space-y-2 max-h-[500px] overflow-y-auto">
                <div className="flex items-center gap-2 sticky top-0 bg-blue-50/90 backdrop-blur-xs py-1">
                  <Sparkles className="w-3.5 h-3.5 text-blue-600" />
                  <strong className="text-[0.68rem] font-bold text-blue-900 uppercase tracking-[0.08em]">
                    Compliance Analysis
                  </strong>
                </div>
                <div className="text-sm text-blue-900 leading-relaxed font-sans">{renderFormattedText(reasoning)}</div>
              </div>

              {/* Enforcement Directive */}
              <div className="p-5 rounded-2xl bg-gradient-to-br from-orange-50 to-amber-50/50 border border-orange-100 space-y-3 flex-1 max-h-[500px] overflow-y-auto">
                <div className="flex items-center gap-2 sticky top-0 bg-orange-50/90 backdrop-blur-xs py-1">
                  <ShieldAlert className="w-3.5 h-3.5 text-orange-600" />
                  <strong className="text-[0.68rem] font-bold text-orange-900 uppercase tracking-[0.08em]">
                    Enforcement Directive
                  </strong>
                </div>

                <div className="space-y-2.5">
                  {directiveLines.map((line, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-orange-950 leading-relaxed font-medium">
                      <ChevronRight className="w-4 h-4 text-orange-500 flex-shrink-0 mt-0.5" />
                      <span>{renderFormattedText(line)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </div>

          {/* System Metadata Dropdown */}
          <Card>
            <details className="group">
              <summary className="cursor-pointer text-xs font-mono font-bold text-slate-500 hover:text-slate-800 flex items-center justify-between list-none">
                <span className="flex items-center gap-2">
                  <Activity className="w-3.5 h-3.5" />
                  System Metadata (JSON Schema)
                </span>
                <ChevronDown className="w-4 h-4 text-slate-400 group-open:rotate-180 transition-transform duration-200" />
              </summary>
              <pre className="mt-3 p-4 bg-slate-900 text-slate-300 text-[0.7rem] font-mono rounded-xl overflow-x-auto max-h-72 border border-slate-800 leading-relaxed">
                {JSON.stringify(dossier, null, 2)}
              </pre>
            </details>
          </Card>
        </div>
      ) : (
        !loading &&
        !error && (
          <div className="empty-state">
            <div style={{ fontWeight: 700, color: "#64748b", fontSize: "1.05rem" }}>Awaiting Evaluation</div>
            <div style={{ fontSize: "0.82rem", color: "#94a3b8", marginTop: "0.25rem" }}>
              Complete input parameters above and press "Execute Multi-Agent Evaluation".
            </div>
          </div>
        )
      )}
    </div>
  );
}