import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import { api } from "../api";
import type { RunRecord } from "../types";

const POLL_INTERVAL_MS = 3000;

export interface RailSelection {
  ids: string[];
  onSelect: (run: RunRecord) => void;
  label?: string;
}

interface ContextValue {
  runs: RunRecord[];
  loading: boolean;
  error: string;
  refresh: () => Promise<void>;
  selection: RailSelection;
  registerSelection: (selection: RailSelection | null) => void;
}

const defaultSelection: RailSelection = {
  ids: [],
  onSelect: () => undefined,
  label: undefined,
};

const RunHistoryContext = createContext<ContextValue | null>(null);

export function RunHistoryProvider({ children }: { children: ReactNode }) {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [selection, setSelection] = useState<RailSelection>(defaultSelection);
  const mountedRef = useRef<boolean>(true);

  const refresh = useCallback(async () => {
    try {
      const next = await api.listRuns();
      if (!mountedRef.current) {
        return;
      }
      setRuns(next);
      setError("");
    } catch (reason: unknown) {
      if (!mountedRef.current) {
        return;
      }
      setError(reason instanceof Error ? reason.message : "Failed to load runs");
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();
    const interval = window.setInterval(() => {
      void refresh();
    }, POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(interval);
    };
  }, [refresh]);

  const registerSelection = useCallback((next: RailSelection | null) => {
    setSelection(next ?? defaultSelection);
  }, []);

  const value = useMemo<ContextValue>(
    () => ({ runs, loading, error, refresh, selection, registerSelection }),
    [runs, loading, error, refresh, selection, registerSelection],
  );

  return <RunHistoryContext.Provider value={value}>{children}</RunHistoryContext.Provider>;
}

export function useRunHistory(): ContextValue {
  const ctx = useContext(RunHistoryContext);
  if (!ctx) {
    throw new Error("useRunHistory must be used inside RunHistoryProvider");
  }
  return ctx;
}

export function useRailSelection(selection: RailSelection | null) {
  const { registerSelection } = useRunHistory();
  useEffect(() => {
    registerSelection(selection);
    return () => registerSelection(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.ids.join("|"), selection?.onSelect, selection?.label, registerSelection]);
}
