import { useState, useEffect } from "react";
import {
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Database,
  RefreshCw,
  CheckSquare,
  Square,
  Pin,
  PinOff,
  Send,
} from "lucide-react";
import { Card, Button, AlertBox } from "@/components/ui";
import { sendRecordToTirage } from "./tirageBridge";
import { useToast } from "./useToast";

interface PredictionRecord {
  file_id: number;
  numero_dossier: string;
  code: string;
  valeur_fob: number;
  final_fraud_score: number;
  final_verdict?: string;
  reviewed?: number; // 0 = unreviewed, 1 = reviewed
  pinned?: number;   // 0 = unpinned, 1 = pinned to top
}

interface ApiResponse {
  records: PredictionRecord[];
  total_records: number;
  avg_score: number;
  page: number;
  total_pages: number;
}

interface AuditPageProps {
  // Optional override. If not provided, AuditPage uses TirageContext directly
  // so it works with zero wiring from the parent component.
  onSendToTirage?: (record: PredictionRecord) => void;
  // Optional: switch the parent's active tab to "Triage" after sending.
  onNavigateToTirage?: () => void;
}

export function AuditPage({ onSendToTirage, onNavigateToTirage }: AuditPageProps) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All Records");
  const [reviewedFilter, setReviewedFilter] = useState("all"); // 'all' | 'reviewed' | 'unreviewed'
  const [page, setPage] = useState(1);
  const [limit] = useState(10);

  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { showToast, ToastHost } = useToast();

  const fetchRecords = async () => {
    setLoading(true);
    setError(null);
    try {
      const queryParams = new URLSearchParams({
        search: search.trim(),
        filter,
        reviewed_filter: reviewedFilter,
        page: page.toString(),
        limit: limit.toString(),
      });

      const response = await fetch(
        `http://localhost:8000/api/v1/audit/predictions?${queryParams.toString()}`
      );

      if (!response.ok) {
        throw new Error(`API responded with status ${response.status}`);
      }

      const result: ApiResponse = await response.json();
      setData(result);
    } catch (err: any) {
      setError(
        err.message || "Could not fetch predicted records from PostgreSQL."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, [filter, reviewedFilter, page]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchRecords();
  };

  // --- Toggle Reviewed Status (PUT Request) ---
  const toggleReviewed = async (fileId: number, currentReviewed: number = 0) => {
    const nextReviewed = currentReviewed === 1 ? 0 : 1;

    setData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        records: prev.records.map((r) =>
          r.file_id === fileId ? { ...r, reviewed: nextReviewed } : r
        ),
      };
    });

    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/audit/predictions/${fileId}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reviewed: nextReviewed }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update review status (HTTP ${response.status})`);
      }
    } catch (err: any) {
      console.error("Error updating review status:", err);
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          records: prev.records.map((r) =>
            r.file_id === fileId ? { ...r, reviewed: currentReviewed } : r
          ),
        };
      });
      showToast({
        tone: "error",
        title: "Couldn't update review status",
        message: `File #${fileId} — the change was rolled back.`,
      });
    }
  };

  // --- Toggle Pinned Status (PUT Request) ---
  const togglePinned = async (fileId: number, currentPinned: number = 0) => {
    const nextPinned = currentPinned === 1 ? 0 : 1;

    setData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        records: prev.records.map((r) =>
          r.file_id === fileId ? { ...r, pinned: nextPinned } : r
        ),
      };
    });

    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/audit/predictions/${fileId}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pinned: nextPinned }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update pin status (HTTP ${response.status})`);
      }
    } catch (err: any) {
      console.error("Error updating pin status:", err);
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          records: prev.records.map((r) =>
            r.file_id === fileId ? { ...r, pinned: currentPinned } : r
          ),
        };
      });
      showToast({
        tone: "error",
        title: "Couldn't update pin status",
        message: `File #${fileId} — the change was rolled back.`,
      });
    }
  };

  // --- Action: Send Row Data to Tirage Form ---
  const handleSendToTirage = (record: PredictionRecord) => {
    // Let a parent override the behavior if it wants to (e.g. custom routing),
    // otherwise hand the record off via TirageContext — no prop drilling needed.
    if (onSendToTirage) {
      onSendToTirage(record);
    } else {
      sendRecordToTirage({
        file_id: record.file_id,
        numero_dossier: record.numero_dossier,
        code: record.code,
        valeur_fob: record.valeur_fob,
        final_fraud_score: record.final_fraud_score,
      });
    }

    showToast({
      tone: "success",
      title: `Record #${record.file_id} sent to Triage`,
      message: `${record.numero_dossier || "Dossier"} is now prefilled in the Live Triage Engine form.`,
    });

    onNavigateToTirage?.();
  };

  const getRiskBadge = (score: number) => {
    if (score >= 0.85) {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-extrabold bg-red-100 text-red-700 border border-red-200">
          <ShieldAlert className="w-3.5 h-3.5" /> Critical (
          {(score * 100).toFixed(1)}%)
        </span>
      );
    }
    if (score >= 0.7) {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-extrabold bg-orange-100 text-orange-700 border border-orange-200">
          <AlertTriangle className="w-3.5 h-3.5" /> High (
          {(score * 100).toFixed(1)}%)
        </span>
      );
    }
    if (score >= 0.6) {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-extrabold bg-amber-100 text-amber-700 border border-amber-200">
          <AlertTriangle className="w-3.5 h-3.5" /> Medium (
          {(score * 100).toFixed(1)}%)
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-extrabold bg-emerald-100 text-emerald-700 border border-emerald-200">
        <ShieldCheck className="w-3.5 h-3.5" /> Low (
        {(score * 100).toFixed(1)}%)
      </span>
    );
  };

  const rawRecords = data?.records || [];

  const filteredRecords = rawRecords.filter((r) => {
    if (reviewedFilter === "reviewed") return r.reviewed === 1;
    if (reviewedFilter === "unreviewed") return (r.reviewed || 0) === 0;
    return true;
  });

  const sortedRecords = [...filteredRecords].sort((a, b) => {
    const pinA = a.pinned || 0;
    const pinB = b.pinned || 0;
    return pinB - pinA;
  });

  const totalPages = data?.total_pages || 1;
  const totalRecords = data?.total_records || 0;
  const avgScore = data?.avg_score || 0.0;

  return (
    <div className="space-y-6">
      <ToastHost />

      <div>
        <div className="section-meta">PostgreSQL Analytics Ledger</div>
        <h1 className="section-title gradient-text">High-Risk Auditing Queue</h1>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="flex items-center justify-between">
          <div>
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Total Recorded Inferences
            </div>
            <div className="text-2xl font-mono font-black text-slate-800 mt-1">
              {totalRecords.toLocaleString()}
            </div>
          </div>
          <Database className="w-8 h-8 text-slate-300" />
        </Card>

        <Card className="flex items-center justify-between">
          <div>
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Average Risk Score
            </div>
            <div className="text-2xl font-mono font-black text-slate-800 mt-1">
              {(avgScore * 100).toFixed(1)}%
            </div>
          </div>
          <AlertTriangle className="w-8 h-8 text-amber-400" />
        </Card>

        <Card className="flex items-center justify-between">
          <div>
            <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Active Page Filter
            </div>
            <div className="text-base font-bold text-slate-700 mt-1">
              {filter} • {reviewedFilter === "all" ? "All Review States" : reviewedFilter}
            </div>
          </div>
          <Filter className="w-8 h-8 text-indigo-400" />
        </Card>
      </div>

      {/* Filter & Search Bar */}
      <Card>
        <div className="flex flex-col md:flex-row gap-3 items-center justify-between">
          <form
            onSubmit={handleSearchSubmit}
            className="relative w-full md:w-80 flex items-center"
          >
            <Search className="absolute left-3.5 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search File ID, Dossier, or Code..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm font-medium text-slate-700 outline-none focus:border-orange-400 focus:bg-white transition-colors"
            />
          </form>

          <div className="flex flex-wrap items-center gap-2 w-full md:w-auto">
            <select
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setPage(1);
              }}
              className="px-3.5 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm font-bold text-slate-700 outline-none focus:border-orange-400 transition-colors"
            >
              <option value="All Records">All Risk Levels</option>
              <option value="Critical Risk">Critical Risk (≥85%)</option>
              <option value="High Risk">High Risk (70% - 85%)</option>
              <option value="Medium Risk">Medium Risk (60% - 70%)</option>
              <option value="Low Risk">Low Risk (&lt;60%)</option>
            </select>

            <select
              value={reviewedFilter}
              onChange={(e) => {
                setReviewedFilter(e.target.value);
                setPage(1);
              }}
              className="px-3.5 py-2 bg-slate-50 border border-slate-200 rounded-xl text-sm font-bold text-slate-700 outline-none focus:border-orange-400 transition-colors"
            >
              <option value="all">All Statuses</option>
              <option value="unreviewed">Unreviewed Only</option>
              <option value="reviewed">Reviewed Only</option>
            </select>

            <Button
              type="button"
              onClick={fetchRecords}
              disabled={loading}
              className="p-2.5"
            >
              <RefreshCw
                className={`w-4 h-4 ${loading ? "animate-spin" : ""}`}
              />
            </Button>
          </div>
        </div>
      </Card>

      {error && (
        <AlertBox tone="error">
          <strong>Database Connection Error</strong>
          <br />
          {error}
        </AlertBox>
      )}

      {/* Data Table */}
      <Card className="p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-slate-500 uppercase text-[0.7rem] tracking-wider font-extrabold border-b border-slate-200">
              <tr>
                <th className="py-3 px-3 text-center w-12">Pin</th>
                <th className="py-3 px-4">File ID</th>
                <th className="py-3 px-4">Dossier Number</th>
                <th className="py-3 px-4">Code</th>
                <th className="py-3 px-4 text-right">FOB Value</th>
                <th className="py-3 px-4 text-center">Fraud Risk Score</th>
                <th className="py-3 px-4 text-center w-24">Reviewed</th>
                <th className="py-3 px-4 text-center w-28">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-400">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-orange-500" />
                    Querying PostgreSQL database...
                  </td>
                </tr>
              ) : sortedRecords.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-400">
                    No matching records found in PostgreSQL ledger.
                  </td>
                </tr>
              ) : (
                sortedRecords.map((row) => {
                  const isPinned = row.pinned === 1;
                  const isReviewed = row.reviewed === 1;

                  return (
                    <tr
                      key={row.file_id}
                      className={`transition-colors ${
                        isPinned
                          ? "bg-amber-50/60 hover:bg-amber-100/50"
                          : "hover:bg-slate-50/80"
                      }`}
                    >
                      <td className="py-3 px-3 text-center">
                        <button
                          type="button"
                          onClick={() => togglePinned(row.file_id, row.pinned)}
                          title={isPinned ? "Unpin row" : "Pin row to top"}
                          className={`p-1.5 rounded-lg transition-colors ${
                            isPinned
                              ? "text-amber-600 bg-amber-100 hover:bg-amber-200"
                              : "text-slate-300 hover:text-slate-500 hover:bg-slate-100"
                          }`}
                        >
                          {isPinned ? (
                            <Pin className="w-4 h-4 fill-amber-500" />
                          ) : (
                            <PinOff className="w-4 h-4" />
                          )}
                        </button>
                      </td>

                      <td className="py-3 px-4 font-mono font-bold text-slate-800">
                        #{row.file_id}
                        {isPinned && (
                          <span className="ml-2 text-[0.62rem] font-sans font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-200 text-amber-800">
                            Pinned
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 font-medium text-slate-700">
                        {row.numero_dossier || "—"}
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-600">
                        {row.code || "—"}
                      </td>
                      <td className="py-3 px-4 text-right font-mono font-medium">
                        {row.valeur_fob
                          ? `$${Number(row.valeur_fob).toLocaleString()}`
                          : "—"}
                      </td>
                      <td className="py-3 px-4 text-center">
                        {getRiskBadge(row.final_fraud_score)}
                      </td>

                      <td className="py-3 px-4 text-center">
                        <button
                          type="button"
                          onClick={() =>
                            toggleReviewed(row.file_id, row.reviewed)
                          }
                          title={
                            isReviewed
                              ? "Mark as unreviewed"
                              : "Mark as reviewed"
                          }
                          className={`inline-flex items-center justify-center p-1.5 rounded-lg transition-all ${
                            isReviewed
                              ? "text-emerald-600 bg-emerald-50 hover:bg-emerald-100"
                              : "text-slate-300 hover:text-slate-500 hover:bg-slate-100"
                          }`}
                        >
                          {isReviewed ? (
                            <CheckSquare className="w-5 h-5 text-emerald-600" />
                          ) : (
                            <Square className="w-5 h-5" />
                          )}
                        </button>
                      </td>

                      <td className="py-3 px-4 text-center">
                        <button
                          type="button"
                          onClick={() => handleSendToTirage(row)}
                          title="Test this record in Tirage form"
                          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-bold text-indigo-600 bg-indigo-50 hover:bg-indigo-100 hover:text-indigo-700 transition-colors"
                        >
                          <Send className="w-3.5 h-3.5" />
                          <span>Tirage</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div className="p-4 bg-slate-50 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 font-bold">
          <div>
            Page {page} of {totalPages} ({totalRecords} items)
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              disabled={page <= 1 || loading}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1 text-xs"
            >
              <ChevronLeft className="w-3.5 h-3.5 mr-1" /> Previous
            </Button>
            <Button
              type="button"
              disabled={page >= totalPages || loading}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1 text-xs"
            >
              Next <ChevronRight className="w-3.5 h-3.5 ml-1" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}