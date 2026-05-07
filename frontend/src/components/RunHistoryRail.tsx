import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { useRunHistory } from "./RunHistoryContext";
import TopologyBadge from "./TopologyBadge";
import type { RunRecord } from "../types";

const STATUS_DOT: Record<string, string> = {
  running: "live-dot live-dot--running",
  queued: "live-dot live-dot--idle",
  stopping: "live-dot live-dot--idle",
  failed: "live-dot live-dot--error",
  completed: "live-dot",
  stopped: "live-dot live-dot--idle",
};

const STATUS_TEXT: Record<string, string> = {
  running: "running",
  queued: "queued",
  stopping: "stopping",
  failed: "failed",
  completed: "done",
  stopped: "stopped",
};

function formatTime(value: number | null): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value * 1000);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function durationLabel(run: RunRecord): string {
  if (!run.started_at) {
    return "—";
  }
  const end = run.finished_at ?? Date.now() / 1000;
  const seconds = Math.max(0, end - run.started_at);
  if (seconds < 60) {
    return `${seconds.toFixed(0)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}m ${rest.toString().padStart(2, "0")}s`;
}

export default function RunHistoryRail() {
  const { runs, loading, error, refresh, selection } = useRunHistory();
  const [collapsed, setCollapsed] = useState<boolean>(false);

  const sorted = useMemo(() => {
    return [...runs].sort((left, right) => {
      const leftStart = left.started_at ?? 0;
      const rightStart = right.started_at ?? 0;
      return rightStart - leftStart;
    });
  }, [runs]);

  if (collapsed) {
    return (
      <aside className="flex w-9 shrink-0 flex-col items-center border-r border-white/[0.06] bg-zinc-950/40 py-3">
        <button
          type="button"
          className="btn-ghost"
          aria-label="Expand run history"
          onClick={() => setCollapsed(false)}
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
        <span
          className="mt-3 text-[10px] uppercase tracking-[0.2em] text-zinc-500"
          style={{ writingMode: "vertical-rl" }}
        >
          History · {runs.length}
        </span>
      </aside>
    );
  }

  return (
    <aside className="flex w-[260px] shrink-0 flex-col border-r border-white/[0.06] bg-zinc-950/40">
      <div className="flex items-center justify-between border-b border-white/[0.06] px-3.5 py-2.5">
        <div className="flex items-center gap-2">
          <span className="label">Runs</span>
          <span className="tabular text-[11px] text-zinc-400">{runs.length}</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            className="btn-ghost"
            onClick={() => void refresh()}
            disabled={loading}
            aria-label="Refresh runs"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "opacity-50" : ""}`} />
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setCollapsed(true)}
            aria-label="Collapse run history"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {selection.label ? (
        <div className="border-b border-white/[0.06] px-3.5 py-2 text-[11px] text-zinc-500">
          {selection.label}
        </div>
      ) : null}

      {error ? (
        <div className="border-b border-rose-500/20 bg-rose-500/5 px-3.5 py-2 text-[11px] text-rose-300">
          {error}
        </div>
      ) : null}

      <div className="flex-1 overflow-y-auto px-2 py-2">
        {sorted.length === 0 ? (
          <div className="px-2 py-6 text-[12px] text-zinc-500">
            {loading ? "Loading runs…" : "No runs yet. Launch one from Studio."}
          </div>
        ) : (
          <ul className="space-y-1">
            {sorted.map((run) => {
              const isSelected = selection.ids.includes(run.run_id);
              const dotClass = STATUS_DOT[run.status] ?? "live-dot live-dot--idle";
              const statusText = STATUS_TEXT[run.status] ?? run.status;
              const throughput = Number(run.latest_metrics?.throughput_samples_sec ?? NaN);
              return (
                <li key={run.run_id}>
                  <button
                    type="button"
                    onClick={() => selection.onSelect(run)}
                    className={`w-full rounded-lg border px-2.5 py-2 text-left transition ${
                      isSelected
                        ? "border-emerald-500/40 bg-emerald-500/[0.06]"
                        : "border-white/[0.04] bg-transparent hover:border-white/[0.10] hover:bg-white/[0.02]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-1.5">
                        <span className={dotClass} />
                        <span className="font-mono text-[10.5px] text-zinc-300">
                          {run.run_id.slice(0, 8)}
                        </span>
                      </div>
                      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                        {statusText}
                      </span>
                    </div>
                    <div className="mt-1.5 flex items-center justify-between">
                      <TopologyBadge topology={run.mode} />
                      <span className="tabular text-[10.5px] text-zinc-500">
                        {durationLabel(run)}
                      </span>
                    </div>
                    <div className="mt-1.5 flex items-center justify-between font-mono text-[10.5px]">
                      <span className="text-zinc-500">{formatTime(run.started_at)}</span>
                      <span className="tabular text-zinc-400">
                        {Number.isFinite(throughput) ? `${throughput.toFixed(0)} s/s` : "—"}
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
