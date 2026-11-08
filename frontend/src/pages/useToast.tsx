import { useCallback, useRef, useState } from "react";
import { CheckCircle2, XCircle, AlertTriangle, Info, X } from "lucide-react";

// ---------------------------------------------------------------------------
// Self-contained toast system.
// No <Provider> needed — just call useToast() inside any page component and
// render the returned <ToastHost /> once, anywhere in that page's JSX tree.
// ---------------------------------------------------------------------------

type ToastTone = "success" | "error" | "warning" | "info";

interface ToastInput {
  tone?: ToastTone;
  title: string;
  message?: string;
  duration?: number; // ms, default 4000
}

interface ToastItem extends Required<Omit<ToastInput, "duration">> {
  id: number;
  duration: number;
}

const TONE_STYLES: Record<
  ToastTone,
  { wrap: string; icon: typeof CheckCircle2; iconColor: string; bar: string }
> = {
  success: {
    wrap: "bg-white border-emerald-200",
    icon: CheckCircle2,
    iconColor: "text-emerald-500",
    bar: "bg-emerald-500",
  },
  error: {
    wrap: "bg-white border-red-200",
    icon: XCircle,
    iconColor: "text-red-500",
    bar: "bg-red-500",
  },
  warning: {
    wrap: "bg-white border-amber-200",
    icon: AlertTriangle,
    iconColor: "text-amber-500",
    bar: "bg-amber-500",
  },
  info: {
    wrap: "bg-white border-indigo-200",
    icon: Info,
    iconColor: "text-indigo-500",
    bar: "bg-indigo-500",
  },
};

function ToastCard({
  toast,
  onDismiss,
}: {
  toast: ToastItem;
  onDismiss: (id: number) => void;
}) {
  const [leaving, setLeaving] = useState(false);
  const styles = TONE_STYLES[toast.tone];
  const Icon = styles.icon;

  const handleClose = () => {
    setLeaving(true);
    window.setTimeout(() => onDismiss(toast.id), 180);
  };

  return (
    <div
      className={`pointer-events-auto relative w-80 max-w-[calc(100vw-2rem)] rounded-2xl border shadow-lg overflow-hidden transition-all duration-200 ease-out ${
        styles.wrap
      } ${leaving ? "opacity-0 translate-x-4" : "opacity-100 translate-x-0"}`}
      style={{ animation: leaving ? undefined : "toast-in 220ms ease-out" }}
      role="alert"
    >
      <div className="flex items-start gap-3 p-4 pr-9">
        <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${styles.iconColor}`} />
        <div className="min-w-0">
          <div className="text-sm font-bold text-slate-800 leading-snug">
            {toast.title}
          </div>
          {toast.message && (
            <div className="text-xs text-slate-500 font-medium mt-0.5 leading-relaxed">
              {toast.message}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={handleClose}
          className="absolute top-3 right-3 text-slate-300 hover:text-slate-500 transition-colors"
          aria-label="Dismiss notification"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="h-1 w-full bg-slate-100">
        <div
          className={`h-full ${styles.bar}`}
          style={{
            animation: leaving
              ? undefined
              : `toast-shrink ${toast.duration}ms linear forwards`,
          }}
        />
      </div>
    </div>
  );
}

export function useToast() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const idRef = useRef(0);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    ({ tone = "info", title, message = "", duration = 4000 }: ToastInput) => {
      const id = ++idRef.current;
      setToasts((prev) => [...prev, { id, tone, title, message, duration }]);
      window.setTimeout(() => dismissToast(id), duration + 200);
    },
    [dismissToast]
  );

  const ToastHost = useCallback(
    () => (
      <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-2.5 pointer-events-none">
        {toasts.map((t) => (
          <ToastCard key={t.id} toast={t} onDismiss={dismissToast} />
        ))}
        <style>{`
          @keyframes toast-in {
            from { opacity: 0; transform: translateX(16px); }
            to { opacity: 1; transform: translateX(0); }
          }
          @keyframes toast-shrink {
            from { width: 100%; }
            to { width: 0%; }
          }
        `}</style>
      </div>
    ),
    [toasts, dismissToast]
  );

  return { showToast, ToastHost };
}
