import { useEffect, useMemo, useState } from "react";
import { Play, Square } from "lucide-react";
import { api } from "../api";
import EventTerminal from "../components/EventTerminal";
import KpiTile from "../components/KpiTile";
import ScenarioForm from "../components/ScenarioForm";
import SwitchTimeline from "../components/SwitchTimeline";
import ThroughputChart, { type ThroughputSeries } from "../components/ThroughputChart";
import TopologyBadge from "../components/TopologyBadge";
import { useRailSelection } from "../components/RunHistoryContext";
import { useRun } from "../hooks/useRun";
import type { Preset, RunRecord, RunRequest } from "../types";

const DEFAULT_REQUEST: RunRequest = {
  mode: "hybrid",
  model: "logreg",
  dataset: "mnist",
  epochs: 5,
  lr: 0.01,
  batch_size: 64,
  num_workers: 4,
  seed: 42,
  ps_discipline: "bsp",
  initial_topology: "rar",
  scenario: {
    straggler_epochs: [3, 4],
    straggler_rank: 3,
    straggler_factor: 3,
    base_compute_ms: 5,
    bandwidth_epochs: [],
    throttle_ms: 0,
  },
};

function cloneRequest(request: RunRequest): RunRequest {
  return JSON.parse(JSON.stringify(request)) as RunRequest;
}

export default function RunStudioPage() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [request, setRequest] = useState<RunRequest>(cloneRequest(DEFAULT_REQUEST));
  const [selectedPreset, setSelectedPreset] = useState<string>("");
  const [presetError, setPresetError] = useState<string>("");
  const {
    activeRun,
    setActiveRun,
    events,
    iterations,
    summaries,
    telemetry,
    switches,
    error,
    busy,
    runIsActive,
    pulseTick,
    startRun,
    stopRun,
  } = useRun();

  useEffect(() => {
    void api
      .getPresets()
      .then((rows) => {
        setPresets(rows);
        if (rows.length > 0) {
          setSelectedPreset(rows[0].id);
          setRequest(cloneRequest(rows[0].request));
        }
      })
      .catch((reason: unknown) => {
        setPresetError(reason instanceof Error ? reason.message : "Failed to load presets.");
      });
  }, []);

  // Wire the run-history rail: clicking any run loads its summaries here.
  useRailSelection(
    useMemo(
      () => ({
        ids: activeRun ? [activeRun.run_id] : [],
        onSelect: (run: RunRecord) => {
          setActiveRun(run);
        },
        label: "Click a run to load it",
      }),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [activeRun?.run_id, setActiveRun],
    ),
  );

  const latestSummary = summaries[summaries.length - 1] ?? activeRun?.latest_metrics ?? {};
  const throughputSeries: ThroughputSeries[] = useMemo(
    () => [
      {
        key: "throughput",
        label: "samples / sec",
        color: "#10b981",
        data: summaries.map((entry) => ({
          epoch: entry.epoch,
          throughput: entry.throughput_samples_sec,
        })),
      },
    ],
    [summaries],
  );

  const commComputeSeries: ThroughputSeries[] = useMemo(
    () => [
      {
        key: "comm",
        label: "comm ms",
        color: "#f59e0b",
        data: summaries.map((entry) => ({
          epoch: entry.epoch,
          comm: entry.avg_comm_ms,
        })),
      },
      {
        key: "compute",
        label: "compute ms",
        color: "#34d399",
        data: summaries.map((entry) => ({
          epoch: entry.epoch,
          compute: entry.avg_compute_ms,
        })),
      },
    ],
    [summaries],
  );

  const lossSeries: ThroughputSeries[] = useMemo(
    () => [
      {
        key: "loss",
        label: "loss",
        color: "#60a5fa",
        data: iterations.map((point) => ({ epoch: point.point, loss: point.loss })),
      },
    ],
    [iterations],
  );

  const switchMarkers = useMemo(
    () =>
      switches.map((swap) => ({
        epoch: swap.epoch,
        from: swap.from,
        to: swap.to,
      })),
    [switches],
  );

  const stressedEpochs = request.scenario.straggler_epochs;

  function handlePresetChange(presetId: string) {
    setSelectedPreset(presetId);
    const preset = presets.find((entry) => entry.id === presetId);
    if (preset) {
      setRequest(cloneRequest(preset.request));
    }
  }

  async function handleStart() {
    await startRun(request);
  }

  async function handleStop() {
    await stopRun();
  }

  const currentTopology = (
    activeRun?.current_topology ??
    (latestSummary as Record<string, unknown>).topology ??
    request.initial_topology
  )?.toString();

  const throughputValue = Number(
    (latestSummary as Record<string, unknown>).throughput_samples_sec ?? 0,
  );
  const valAccValue = Number((latestSummary as Record<string, unknown>).val_acc ?? 0) * 100;
  const lagValue = Number(telemetry.lag_ratio ?? 0);

  const statusLabel = activeRun?.status ?? "idle";
  const epochLabel = activeRun?.latest_epoch ?? Number((latestSummary as Record<string, unknown>).epoch ?? 0);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-[340px_minmax(0,1fr)] gap-0">
      {/* Left column: scenario form */}
      <aside className="flex min-h-0 flex-col border-r border-white/[0.06] bg-zinc-950/40 px-4 py-4">
        <div className="flex items-center justify-between">
          <span className="label">Scenario</span>
          <span className="font-mono text-[11px] text-zinc-500">
            {runIsActive ? `live · ${pulseTick}` : statusLabel}
          </span>
        </div>

        <div className="mt-4 flex-1 overflow-y-auto pr-1">
          {presetError ? (
            <div className="mb-3 rounded-md border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-[11.5px] text-rose-300">
              {presetError}
            </div>
          ) : null}
          <ScenarioForm
            presets={presets}
            selectedPresetId={selectedPreset}
            onSelectPreset={handlePresetChange}
            request={request}
            onChangeRequest={(updater) => setRequest(updater)}
            disabled={runIsActive || busy}
          />

          {error ? (
            <div className="mt-4 rounded-md border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-[11.5px] text-rose-300">
              {error}
            </div>
          ) : null}
        </div>

        <div className="mt-4 flex items-center gap-2 border-t border-white/[0.06] pt-4">
          <button
            type="button"
            onClick={() => void handleStart()}
            disabled={busy || runIsActive}
            className="btn-primary flex-1"
          >
            <Play className="h-3.5 w-3.5" />
            Start run
          </button>
          <button
            type="button"
            onClick={() => void handleStop()}
            disabled={busy || !runIsActive}
            className="btn-secondary"
          >
            <Square className="h-3.5 w-3.5" />
            Stop
          </button>
        </div>
      </aside>

      {/* Right column: KPIs + charts + timeline + terminal */}
      <section className="flex min-h-0 flex-col gap-4 px-5 py-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <KpiTile
            label="Throughput"
            value={throughputValue > 0 ? throughputValue.toFixed(1) : "—"}
            hint="samples / sec"
          />
          <KpiTile
            label="Val Accuracy"
            value={valAccValue > 0 ? `${valAccValue.toFixed(1)}%` : "—"}
            hint={`epoch ${epochLabel || 0}`}
          />
          <KpiTile
            label="Topology"
            value={
              <TopologyBadge topology={currentTopology ?? undefined} size="md" />
            }
            hint={
              activeRun
                ? `mode: ${activeRun.mode.toUpperCase()}`
                : `mode: ${request.mode.toUpperCase()}`
            }
          />
          <KpiTile
            label="Switches"
            value={switches.length}
            hint={lagValue > 0 ? `lag ratio ${lagValue.toFixed(2)}` : "no switches yet"}
          />
        </div>

        <div className="surface-lg flex flex-col px-4 py-3.5">
          <div className="flex items-center justify-between">
            <div>
              <span className="label">Throughput</span>
              <p className="mt-1 text-[12.5px] text-zinc-400">
                Epoch summaries · stressed window shaded · switches marked
              </p>
            </div>
            <span className="tabular font-mono text-[11px] text-zinc-500">
              {summaries.length} epoch{summaries.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="mt-3">
            <ThroughputChart
              series={throughputSeries}
              yKey="throughput"
              stressedEpochs={stressedEpochs}
              switches={switchMarkers}
              height={280}
              yLabel="samples/sec"
            />
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <div className="surface-lg flex flex-col px-4 py-3.5">
            <div className="flex items-center justify-between">
              <span className="label">Comm vs Compute</span>
              <span className="font-mono text-[11px] text-zinc-500">avg ms / epoch</span>
            </div>
            <div className="mt-3">
              <ThroughputChart
                series={commComputeSeries}
                yKey="value"
                stressedEpochs={stressedEpochs}
                switches={switchMarkers}
                height={220}
                showLegend
              />
            </div>
          </div>

          <div className="surface-lg flex flex-col px-4 py-3.5">
            <div className="flex items-center justify-between">
              <span className="label">Live Loss</span>
              <span className="font-mono text-[11px] text-zinc-500">
                {iterations.length} pts
              </span>
            </div>
            <div className="mt-3">
              <ThroughputChart
                series={lossSeries}
                yKey="loss"
                xKey="epoch"
                height={220}
                yLabel="loss"
              />
            </div>
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
          <div className="surface-lg flex flex-col px-4 py-3.5">
            <SwitchTimeline
              epochs={request.epochs}
              summaries={summaries}
              switches={switches}
              initialTopology={request.initial_topology}
            />
          </div>
          <div className="surface-lg flex h-[260px] flex-col px-4 py-3.5">
            <EventTerminal
              events={events}
              emptyHint="Launch a run to see events stream in."
              maxHeight={210}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
