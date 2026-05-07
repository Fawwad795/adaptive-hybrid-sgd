import { useMemo } from "react";
import type { EpochSummary, SwitchRecord } from "../hooks/useRun";

interface Props {
  epochs: number;
  summaries: EpochSummary[];
  switches: SwitchRecord[];
  initialTopology?: string;
}

const TOPOLOGY_FILL: Record<string, string> = {
  rar: "rgba(16, 185, 129, 0.18)",
  ps: "rgba(245, 158, 11, 0.18)",
  hybrid: "rgba(167, 139, 250, 0.18)",
  default: "rgba(255, 255, 255, 0.04)",
};

const TOPOLOGY_BORDER: Record<string, string> = {
  rar: "rgba(16, 185, 129, 0.35)",
  ps: "rgba(245, 158, 11, 0.35)",
  hybrid: "rgba(167, 139, 250, 0.35)",
  default: "rgba(255, 255, 255, 0.06)",
};

export default function SwitchTimeline({ epochs, summaries, switches, initialTopology }: Props) {
  const cells = useMemo(() => {
    const list: Array<{ epoch: number; topology: string }> = [];
    let current = String(initialTopology ?? "rar").toLowerCase();
    for (let epoch = 1; epoch <= epochs; epoch += 1) {
      const summary = summaries.find((entry) => entry.epoch === epoch);
      if (summary?.topology) {
        current = String(summary.topology).toLowerCase();
      }
      list.push({ epoch, topology: current });
    }
    return list;
  }, [epochs, summaries, initialTopology]);

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between">
        <span className="label">Switch Timeline</span>
        <span className="text-[10px] text-zinc-500">
          {switches.length === 0 ? "no switches" : `${switches.length} switch${switches.length === 1 ? "" : "es"}`}
        </span>
      </div>
      <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${epochs}, minmax(0, 1fr))` }}>
        {cells.map((cell) => {
          const fill = TOPOLOGY_FILL[cell.topology] ?? TOPOLOGY_FILL.default;
          const border = TOPOLOGY_BORDER[cell.topology] ?? TOPOLOGY_BORDER.default;
          const switchAtEpoch = switches.find((swap) => swap.epoch === cell.epoch);
          return (
            <div
              key={cell.epoch}
              className="relative flex h-12 flex-col justify-between rounded-md border px-2 py-1.5"
              style={{ backgroundColor: fill, borderColor: border }}
            >
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-zinc-300">
                {cell.topology}
              </span>
              <span className="tabular text-[11px] text-zinc-500">e{cell.epoch}</span>
              {switchAtEpoch ? (
                <span className="absolute -top-1.5 right-1.5 rounded-sm border border-violet-400/40 bg-zinc-950 px-1 py-[1px] font-mono text-[9px] text-violet-300">
                  {String(switchAtEpoch.from).toUpperCase()}→{String(switchAtEpoch.to).toUpperCase()}
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
