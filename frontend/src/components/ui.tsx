import { useState, useRef, useEffect, type ReactNode } from "react";
import { ChevronDown, X } from "lucide-react";

export function Card({
  title,
  subtitle,
  children,
  className = "",
  action,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
}) {
  return (
    <div className={`light-card fade-in ${className}`}>
      {(title || action) && (
        <div className="light-card-header">
          <div>
            {title && <div className="light-card-title">{title}</div>}
            {subtitle && <div className="light-card-sub">{subtitle}</div>}
          </div>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function Select({
  value,
  onChange,
  options,
  className = "",
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  className?: string;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = options.find((o) => o === value);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-700 hover:border-orange-300 hover:bg-orange-50/40 transition-colors"
      >
        <span className={current ? "" : "text-slate-400"}>{current ?? placeholder ?? "Select..."}</span>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="absolute z-30 mt-1.5 w-full bg-white border border-slate-200 rounded-xl shadow-xl py-1.5 max-h-72 overflow-auto">
          {options.map((o) => (
            <button
              key={o}
              type="button"
              onClick={() => {
                onChange(o);
                setOpen(false);
              }}
              className={`w-full text-left px-3.5 py-2 text-sm font-medium hover:bg-orange-50 transition-colors ${
                o === value ? "text-orange-700 bg-orange-50/60 font-bold" : "text-slate-600"
              }`}
            >
              {o}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  className = "",
  disabled = false,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger" | "subtle";
  className?: string;
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  const styles = {
    primary:
      "bg-gradient-to-br from-orange-500 to-orange-600 text-white shadow-lg shadow-orange-500/30 hover:shadow-orange-500/40 hover:-translate-y-px",
    ghost:
      "bg-white text-slate-600 border border-slate-200 hover:border-orange-300 hover:text-orange-700 hover:bg-orange-50/40",
    danger:
      "bg-red-50 text-red-600 border border-red-200 hover:bg-red-100",
    subtle:
      "bg-slate-100 text-slate-700 hover:bg-slate-200 border border-transparent",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function EmptyState({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="empty-state">
      <div style={{ fontWeight: 700, color: "#64748b", fontSize: "1.05rem" }}>{title}</div>
      <div style={{ fontSize: "0.82rem", color: "#94a3b8", marginTop: "0.25rem" }}>{subtitle}</div>
    </div>
  );
}

export function AlertBox({
  tone,
  children,
}: {
  tone: "error" | "warning";
  children: ReactNode;
}) {
  return <div className={`alert-box ${tone}`}>{children}</div>;
}

export function Modal({
  open,
  onClose,
  children,
  title,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  title: string;
}) {
  if (!open) return null;
  return (
    <>
      <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 fade-in" onClick={onClose} />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg pointer-events-auto slide-in">
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
            <h3 className="font-black text-slate-900">{title}</h3>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-100">
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>
          <div className="p-6">{children}</div>
        </div>
      </div>
    </>
  );
}
