// ---------------------------------------------------------------------------
// Plain (non-React) bridge for handing a record from AuditPage to TriagePage.
// Works no matter how the two pages are composed in your app — no <Provider>
// wrapping, no prop drilling. Uses sessionStorage (survives tab-switches /
// remounts) + a CustomEvent (delivers instantly if TriagePage is already
// mounted, e.g. rendered in a hidden tab).
// ---------------------------------------------------------------------------

export interface TirageHandoffRecord {
  file_id: number;
  numero_dossier: string;
  code: string;
  valeur_fob: number;
  final_fraud_score: number;
}

const STORAGE_KEY = "tirage_pending_record";
const EVENT_NAME = "tirage:record-sent";

export function sendRecordToTirage(record: TirageHandoffRecord) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(record));
  } catch {
    // sessionStorage unavailable (e.g. private browsing) — the CustomEvent
    // below still delivers the record if TriagePage is already mounted.
  }
  window.dispatchEvent(new CustomEvent<TirageHandoffRecord>(EVENT_NAME, { detail: record }));
}

/** Read + clear any record that was sent before TriagePage mounted. */
export function consumePendingTirageRecord(): TirageHandoffRecord | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(STORAGE_KEY);
    return JSON.parse(raw) as TirageHandoffRecord;
  } catch {
    return null;
  }
}

/** Subscribe to records sent while TriagePage is already mounted. Returns an unsubscribe fn. */
export function onTirageRecordSent(
  callback: (record: TirageHandoffRecord) => void
): () => void {
  const handler = (e: Event) => {
    const detail = (e as CustomEvent<TirageHandoffRecord>).detail;
    if (detail) callback(detail);
  };
  window.addEventListener(EVENT_NAME, handler);
  return () => window.removeEventListener(EVENT_NAME, handler);
}
