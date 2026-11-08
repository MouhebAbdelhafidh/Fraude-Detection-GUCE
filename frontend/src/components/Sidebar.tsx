import { useState } from "react";
import {
  ScrollText,
  Workflow,
  Radar,
  Activity,
  Settings,
  Menu,
  X,
  LogOut,
  User,
} from "lucide-react";
import type { PageKey } from "@/types";
import { NAV_ITEMS } from "@/data";
import { AIRFLOW_URL, GRAFANA_URL, TRIAGE_API_URL } from "@/data";
import guceLogo from "@/assets/guce_logo.png";

const ICONS: Record<PageKey, typeof ScrollText> = {
  audit: ScrollText,
  pipeline: Workflow,
  triage: Radar,
  telemetry: Activity,
};

export function Sidebar({
  page,
  setPage,
  open,
  onClose,
}: {
  page: PageKey;
  setPage: (p: PageKey) => void;
  open: boolean;
  onClose: () => void;
}) {
  return (
    <>
      {open && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-30 lg:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-72 bg-white border-r border-slate-200 flex flex-col transition-transform duration-300 ${
          open ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        <div className="px-5 py-4 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <img
              src={guceLogo}
              alt="GUCE Logo"
              className="w-16 h-16 object-contain shrink-0"
            />
            <div>
              <div className="text-sm font-black text-slate-900 leading-tight">
                GUCE Control
              </div>
              <div className="text-[0.68rem] text-slate-500 font-medium">
                Fraud Intelligence Unit
              </div>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
          <div className="px-3 pb-2 text-[0.62rem] uppercase tracking-[0.12em] text-slate-400 font-bold">
            Navigation
          </div>
          {NAV_ITEMS.map((item) => {
            const Icon = ICONS[item.key];
            const active = page === item.key;
            return (
              <button
                key={item.key}
                onClick={() => {
                  setPage(item.key);
                  onClose();
                }}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-left text-sm font-bold transition-all duration-200 ${
                  active
                    ? "bg-gradient-to-br from-orange-500 to-orange-600 text-white shadow-lg shadow-orange-500/25"
                    : "text-slate-600 bg-slate-50 hover:bg-orange-50 hover:text-orange-700 border border-slate-200 hover:border-orange-300"
                }`}
              >
                <Icon className="w-4.5 h-4.5 flex-shrink-0" strokeWidth={2.2} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </aside>
    </>
  );
}

export function TopBar({
  onMenu,
  settingsOpen,
  onOpenSettings,
  onCloseSettings,
  onLogout,
}: {
  onMenu: () => void;
  settingsOpen: boolean;
  onOpenSettings: () => void;
  onCloseSettings: () => void;
  onLogout?: () => void;
}) {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [highContrast, setHighContrast] = useState(false);
  const [timeRange, setTimeRange] = useState("Last 6h");

  // Safely read user from localStorage or fallback
  const savedUser = localStorage.getItem("user");
  const user = savedUser ? JSON.parse(savedUser) : { username: "Admin" };

  return (
    <header className="sticky top-0 z-20 bg-white/85 backdrop-blur-xl border-b border-slate-200">
      <div className="flex items-center justify-between px-5 lg:px-8 py-3">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenu}
            className="lg:hidden p-2 -ml-2 rounded-lg hover:bg-slate-100"
          >
            <Menu className="w-5 h-5 text-slate-600" />
          </button>
          <div className="flex items-center gap-2.5">
            <img
              src={guceLogo}
              alt="GUCE Logo"
              className="w-14 h-14 object-contain lg:hidden shrink-0"
            />
            <div>
              <div className="text-[1.05rem] font-black text-slate-900 leading-tight">
                GUCE Customs Fraud Control
              </div>
              <div className="hidden sm:block text-[0.68rem] text-slate-400 font-medium">
                Control Center
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="live-badge hidden sm:inline">LIVE</span>

          <button
            onClick={settingsOpen ? onCloseSettings : onOpenSettings}
            className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 transition-colors"
          >
            <Settings className="w-5 h-5 text-slate-600" />
          </button>
        </div>
      </div>

      {settingsOpen && (
        <div className="absolute right-5 top-16 w-80 bg-white border border-slate-200 rounded-2xl shadow-2xl z-50 fade-in p-4">
          <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
            <span className="font-black text-slate-800">System Settings</span>
            <button
              onClick={onCloseSettings}
              className="p-1 rounded-lg hover:bg-slate-100"
            >
              <X className="w-4 h-4 text-slate-500" />
            </button>
          </div>

          {/* User Account Info Header */}
          <div className="flex items-center gap-3 bg-slate-50 rounded-xl p-2.5 mb-3 border border-slate-100">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-orange-100 text-orange-700 font-bold text-sm">
              <User className="w-4 h-4" />
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-[0.62rem] font-bold text-slate-400 uppercase tracking-wider">
                Signed in as
              </span>
              <span className="text-xs font-bold text-slate-800 truncate">
                {user.username || "Admin"}
              </span>
            </div>
          </div>

          <div className="space-y-3">
            <label className="flex items-center justify-between text-sm font-semibold text-slate-700">
              Auto-refresh Telemetry
              <button
                onClick={() => setAutoRefresh((v) => !v)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  autoRefresh ? "bg-orange-500" : "bg-slate-300"
                }`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                    autoRefresh ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </label>
            <label className="flex items-center justify-between text-sm font-semibold text-slate-700">
              High Contrast Mode
              <button
                onClick={() => setHighContrast((v) => !v)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  highContrast ? "bg-orange-500" : "bg-slate-300"
                }`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                    highContrast ? "translate-x-5" : "translate-x-0.5"
                  }`}
                />
              </button>
            </label>
            <div>
              <div className="text-sm font-semibold text-slate-700 mb-1.5">
                Default Time Range
              </div>
              <div className="flex gap-1.5">
                {["Last 15m", "Last 1h", "Last 6h", "Last 24h"].map((r) => (
                  <button
                    key={r}
                    onClick={() => setTimeRange(r)}
                    className={`px-2 py-1 rounded-lg text-xs font-bold transition-colors ${
                      timeRange === r
                        ? "bg-orange-500 text-white"
                        : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                    }`}
                  >
                    {r.replace("Last ", "")}
                  </button>
                ))}
              </div>
            </div>

            {/* Sign Out Button */}
            <div className="pt-3 border-t border-slate-100">
              <button
                onClick={() => {
                  onCloseSettings();
                  if (onLogout) {
                    onLogout();
                  } else {
                    localStorage.removeItem("user");
                    window.location.reload();
                  }
                }}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-red-50 text-red-600 hover:bg-red-100 text-xs font-bold transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Sign Out</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}