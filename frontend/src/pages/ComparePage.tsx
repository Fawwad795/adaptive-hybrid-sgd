import { useEffect, useMemo, useState } from "react";
import { Layers, Trash2, Users } from "lucide-react";
import { api } from "../api";
import KpiTile from "../components/KpiTile";
import { useRailSelection } from "../components/RunHistoryContext";
import StressTrioLauncher, { type OverlayRunPayload } from "../components/StressTrioLauncher";
import ThroughputChart, { type ThroughputSeries } from "../components/ThroughputChart";
import TopologyBadge from "../components/TopologyBadge";
import type { EpochSummary, SwitchRecord } from "../hooks/useRun";
import type { EventEnvelope, Preset, RunMode, RunRecord } from "../types";

const TOPOLOGY_COLORS: Record<RunMode, string> = {
  rar: "#10b981",
  ps: "#f59e0b",
  hybrid: "#a78bfa",
};

const STRESS_WINDOW = [3, 4, 5];

type Mode = "trio" | "pick";

interface OverlayEntry {
  runId: string;
  presetId?: string;
  topology: RunMode;
  summaries: EpochSummary[];
  switches: SwitchRecord[];
  source: "trio" | "pick";
}

function summariesAvgInWindow(summaries: EpochSummary[], window: number[]): number {
  const points = summaries.filter((entry) => window.includes(entry.epoch));
  if (points.length === 0) {
    return 0;
  }
  const sum = points.reduce((total, entry) => total + entry.throughput_samples_sec, 0);
  return sum / points.length;
}

export default function ComparePage() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [mode, setMode] = useState<Mode>("trio");
  const [overlay, setOverlay] = useState<Record<string, OverlayEntry>>({});
  const [pickSelection, setPickSelection] = useState<string[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    void api
      .getPresets()
      .then(setPresets)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Failed to load presets.");
      });
  }, []);

  // Replay summaries for a run via the event-stream backlog.
  function loadRunSummaries(run: RunRecord) {
    if (overlay[run.run_id]) {
      return;
    }
    const summaries: EpochSummary[] = [];
    const switches: SwitchRecord[] = [];

    function flush() {
      setOverlay((current) => ({
        ...current,
        [run.run_id]: {
          runId: run.run_id,
          topology: run.mode,
          summaries: [...summaries].sort((a, b) => a.epoch - b.epoch),
          switches: [...switches],
          source: "pick",
        },
      }));
    }

    flush();

    const source = api.connectEvents(run.run_id, (event: EventEnvelope) => {
      const data = event.data as Record<string, unknown>;
      if (event.type === "train.epoch_summary") {
        const next = summaries.filter((entry) => entry.epoch !== Number(data.epoch));
        next.push({
          epoch: Number(data.epoch ?? 0),
          topology: String(data.topology ?? ""),
          throughput_samples_sec: Number(data.throughput_samples_sec ?? 0),
          val_acc: Number(data.val_acc ?? 0),
          train_loss: Number(data.train_loss ?? 0),
          avg_compute_ms: Number(data.avg_compute_ms ?? 0),
          avg_comm_ms: Number(data.avg_comm_ms ?? 0),
          ...data,
        });
        next.sort((a, b) => a.epoch - b.epoch);
        summaries.splice(0, summaries.length, ...next);
        flush();
      } else if (event.type === "controller.switch") {
        switches.push({
          epoch: Number(data.epoch ?? 0),
          from: String(data.from ?? ""),
          to: String(data.to ?? ""),
          reason: (data.reason as Record<string, unknown>) ?? {},
          ...data,
        });
        flush();
      }
    });
    source.onerror = () => {
      source.close();
    };
    // Close after 6s — enough for the full persistent backlog (all epoch
    // summaries + switches) to replay over SSE even for a 6-epoch run.
    window.setTimeout(() => source.close(), 6000);
  }

  // Wire the rail's click handler based on mode.
  useRailSelection(
    useMemo(() => {
      if (mode === "trio") {
        return {
          ids: Object.values(overlay).filter((entry) => entry.source === "trio").map((entry) => entry.runId),
          onSelect: () => undefined,
          label: "Trio mode — runs auto-load from launcher",
        };
      }
      return {
        ids: pickSelection,
        onSelect: (run: RunRecord) => {
          setPickSelection((current) => {
            if (current.includes(run.run_id)) {
              setOverlay((entries) => {
                const next = { ...entries };
                delete next[run.run_id];
                return next;
              });
              return current.filter((id) => id !== run.run_id);
            }
            const limited = [...current, run.run_id].slice(-3);
            const dropped = current.filter((id) => !limited.includes(id));
            if (dropped.length > 0) {
              setOverlay((entries) => {
                const next = { ...entries };
                dropped.forEach((id) => delete next[id]);
                return next;
              });
            }
            loadRunSummaries(run);
            return limited;
          });
        },
        label: `Pick mode — choose up to 3 runs (${pickSelection.length}/3)`,
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [mode, pickSelection, overlay]),
  );

  function handleTrioStart() {
    setError("");
    setOverlay((current) => {
      const next: Record<string, OverlayEntry> = {};
      Object.entries(current).forEach(([key, entry]) => {
        if (entry.source !== "trio") {
          next[key] = entry;
        }
      });
      return next;
    });
  }

  function handleTrioUpdate(payload: OverlayRunPayload) {
    setOverlay((current) => ({
      ...current,
      [payload.runId]: {
        runId: payload.runId,
        topology: payload.topology,
        presetId: payload.presetId,
        summaries: payload.summaries,
        switches: payload.switches,
        source: "trio",
      },
    }));
  }

  function clearOverlay() {
    setOverlay({});
    setPickSelection([]);
  }

  const overlayList = useMemo(() => Object.values(overlay), [overlay]);

  const series: ThroughputSeries[] = useMemo(
    () =>
      overlayList.map((entry) => ({
        key: entry.runId,
        label:
          entry.source === "trio"
            ? `${entry.topology.toUpperCase()} · ${entry.presetId ?? entry.runId.slice(0, 8)}`
            : `${entry.topology.toUpperCase()} · ${entry.runId.slice(0, 8)}`,
        color: TOPOLOGY_COLORS[entry.topology] ?? "#a1a1aa",
        data: entry.summaries.map((summary) => ({
          epoch: summary.epoch,
          throughput: summary.throughput_samples_sec,
        })),
      })),
    [overlayList],
  );

  const allSwitches = useMemo(
    () =>
      overlayList.flatMap((entry) =>
        entry.switches.map((swap) => ({
          epoch: swap.epoch,
          from: swap.from,
          to: swap.to,
          label: `${entry.topology.toUpperCase()}: ${swap.from.toUpperCase()}→${swap.to.toUpperCase()}`,
        })),
      ),
    [overlayList],
  );

  const stressedAvg = useMemo(() => {
    const result: Partial<Record<RunMode, { value: number; runId: string }>> = {};
    overlayList.forEach((entry) => {
      const value = summariesAvgInWindow(entry.summaries, STRESS_WINDOW);
      if (value > 0) {
        result[entry.topology] = { value, runId: entry.runId };
      }
    });
    return result;
  }, [overlayList]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 px-5 py-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-[18px] font-semibold tracking-tight text-zinc-100">Compare</h1>
          <p className="mt-0.5 text-[12.5px] text-zinc-400">
            Overlay multiple runs on a shared throughput axis. Trio runs use the same scenario;
            pick mode lets you compare any three finished runs.
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-md border border-white/[0.06] bg-zinc-950 p-0.5">
          <ModeButton active={mode === "trio"} onClick={() => setMode("trio")} icon={<Layers className="h-3.5 w-3.5" />}>
            Stress Trio
          </ModeButton>
          <ModeButton active={mode === "pick"} onClick={() => setMode("pick")} icon={<Users className="h-3.5 w-3.5" />}>
            Pick Three
          </ModeButton>
        </div>
      </div>

      {mode === "trio" ? (
        <StressTrioLauncher
          presets={presets}
          onStart={handleTrioStart}
          onUpdate={handleTrioUpdate}
          onComplete={() => undefined}
          onError={(message) => setError(message)}
        />
      ) : (
        <div className="surface-lg flex flex-col gap-3 px-4 py-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <span className="label">Pick Three Runs</span>
              <p className="mt-1 text-[12.5px] text-zinc-400">
                Click any three runs in the left rail to overlay them. Long-completed runs may show
                only their most recent epoch summaries.
              </p>
            </div>
            <button type="button" className="btn-secondary" onClick={clearOverlay}>
              <Trash2 className="h-3.5 w-3.5" />
              Clear
            </button>
          </div>
          <div className="flex items-center gap-2">
            {pickSelection.length === 0 ? (
              <span className="text-[12px] text-zinc-500">No runs selected.</span>
            ) : (
              pickSelection.map((id) => {
                const entry = overlay[id];
                return (
                  <span
                    key={id}
                    className="flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-black/40 px-2 py-1"
                  >
                    <TopologyBadge topology={entry?.topology} />
                    <span className="font-mono text-[10.5px] text-zinc-300">{id.slice(0, 8)}</span>
                  </span>
                );
              })
            )}
          </div>
        </div>
      )}

      {error ? (
        <div className="rounded-md border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-[12px] text-rose-300">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <KpiTile
          label="Runs Overlaid"
          value={overlayList.length}
          hint={mode === "trio" ? "trio mode" : "pick mode"}
        />
        <KpiTile
          label="Stressed Avg · RAR"
          value={stressedAvg.rar ? stressedAvg.rar.value.toFixed(0) : "—"}
          hint="samples/sec, ep 3–5"
        />
        <KpiTile
          label="Stressed Avg · PS"
          value={stressedAvg.ps ? stressedAvg.ps.value.toFixed(0) : "—"}
          hint="samples/sec, ep 3–5"
        />
        <KpiTile
          label="Stressed Avg · Hybrid"
          value={stressedAvg.hybrid ? stressedAvg.hybrid.value.toFixed(0) : "—"}
          hint="samples/sec, ep 3–5"
        />
      </div>

      <div className="surface-lg flex flex-col px-4 py-3.5">
        <div className="flex items-center justify-between">
          <div>
            <span className="label">Throughput Overlay</span>
            <p className="mt-1 text-[12.5px] text-zinc-400">
              Stressed window (epochs 3–5) shaded. Switches marked per series.
            </p>
          </div>
          <span className="font-mono text-[11px] text-zinc-500">
            {overlayList.length} series
          </span>
        </div>
        <div className="mt-3">
          <ThroughputChart
            series={series}
            yKey="throughput"
            stressedEpochs={STRESS_WINDOW}
            switches={allSwitches}
            height={340}
            yLabel="samples/sec"
            showLegend
          />
        </div>
      </div>

      {overlayList.length > 0 ? (
        <div className="surface-lg px-4 py-3.5">
          <span className="label">Overlay Detail</span>
          <div className="mt-3 grid gap-2">
            {overlayList.map((entry) => (
              <div
                key={entry.runId}
                className="flex items-center justify-between gap-3 rounded-lg border border-white/[0.06] bg-black/40 px-3 py-2"
              >
                <div className="flex items-center gap-2.5">
                  <TopologyBadge topology={entry.topology} />
                  <span className="font-mono text-[11px] text-zinc-300">
                    {entry.presetId ?? entry.runId.slice(0, 8)}
                  </span>
                  <span className="text-[10.5px] text-zinc-500">
                    {entry.summaries.length} epoch{entry.summaries.length === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="flex items-center gap-3 font-mono text-[10.5px] text-zinc-400">
                  <span>switches: {entry.switches.length}</span>
                  <span className="tabular">
                    avg{" "}
                    {(
                      entry.summaries.reduce(
                        (total, summary) => total + summary.throughput_samples_sec,
                        0,
                      ) / Math.max(1, entry.summaries.length)
                    ).toFixed(0)}{" "}
                    s/s
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ModeButton({
  active,
  onClick,
  children,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
  icon: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] font-medium transition ${
        active ? "bg-white/[0.06] text-zinc-100" : "text-zinc-400 hover:text-zinc-200"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}
