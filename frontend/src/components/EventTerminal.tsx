import { useEffect, useRef } from "react";
import type { EventLogEntry } from "../hooks/useRun";

interface Props {
  events: EventLogEntry[];
  emptyHint?: string;
  maxHeight?: number;
}

const TYPE_COLORS: Record<string, string> = {
  "run.started": "#10b981",
  "run.completed": "#10b981",
  "run.failed": "#f43f5e",
  "run.stopped": "#fbbf24",
  "run.stopping": "#fbbf24",
  "controller.switch": "#c4b5fd",
  "train.epoch_summary": "#60a5fa",
  "train.iteration": "#a1a1aa",
  "telemetry.snapshot": "#a1a1aa",
  "run.epoch_started": "#71717a",
  "worker.epoch_summary": "#71717a",
};

function formatTime(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) {
    return "--:--:--";
  }
  const date = new Date(ts * 1000);
  return date.toLocaleTimeString([], { hour12: false });
}

export default function EventTerminal({ events, emptyHint, maxHeight = 240 }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) {
      return;
    }
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [events.length]);

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-center justify-between">
        <span className="label">Event Stream</span>
        <span className="terminal text-[10px] text-zinc-500">{events.length} events</span>
      </div>
      <div
        ref={ref}
        className="terminal flex-1 overflow-y-auto rounded-lg border border-white/[0.06] bg-black/80 px-3 py-2.5 text-[11px] leading-[1.55] text-zinc-300"
        style={{ maxHeight }}
      >
        {events.length === 0 ? (
          <div className="text-zinc-600">
            {emptyHint ?? "Waiting for events…"}
          </div>
        ) : (
          events.map((entry) => (
            <div key={entry.id} className="flex gap-2.5">
              <span className="shrink-0 text-zinc-600">{formatTime(entry.ts)}</span>
              <span
                className="shrink-0 font-medium"
                style={{ color: TYPE_COLORS[entry.type] ?? "#a1a1aa" }}
              >
                {entry.type}
              </span>
              {entry.detail ? <span className="text-zinc-400">{entry.detail}</span> : null}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
