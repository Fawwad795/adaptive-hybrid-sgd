import { useEffect, useRef, useState } from "react";
import { Play, X } from "lucide-react";
import { api } from "../api";
import type { EpochSummary, SwitchRecord } from "../hooks/useRun";
import type { EventEnvelope, Preset, RunMode, RunRecord } from "../types";

const TERMINAL_STATUSES = new Set(["completed", "failed", "stopped"]);
const POLL_INTERVAL_MS = 750;

const TRIO_DEFINITIONS = [
  {
    label: "Stress A",
    description: "Strong straggler only",
    a: "stress-a-rar",
    p: "stress-a-ps",
    h: "stress-a-hybrid",
  },
  {
    label: "Stress B",
    description: "Straggler + comm throttle",
    a: "stress-b-rar",
    p: "stress-b-ps",
    h: "stress-b-hybrid",
  },
] as const;

export type TrioPhase = "idle" | "running" | "done" | "error";

export interface OverlayRunPayload {
  topology: RunMode;
  runId: string;
  presetId: string;
  summaries: EpochSummary[];
  switches: SwitchRecord[];
}

interface StressTrioLauncherProps {
  presets: Preset[];
  onStart: () => void;
  onUpdate: (payload: OverlayRunPayload) => void;
  onComplete: (payloads: OverlayRunPayload[]) => void;
  onError: (message: string) => void;
}

interface ActiveTrio {
  trio: typeof TRIO_DEFINITIONS[number];
  presetIds: string[];
  status: Record<string, "queued" | "running" | "done" | "error">;
  currentIndex: number;
}

function findPreset(presets: Preset[], id: string): Preset | null {
  return presets.find((preset) => preset.id === id) ?? null;
}

async function pollUntilTerminal(runId: string, signal: AbortSignal): Promise<RunRecord> {
  while (!signal.aborted) {
    const run = await api.getRun(runId);
    if (TERMINAL_STATUSES.has(run.status)) {
      return run;
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  throw new Error("aborted");
}

function streamRun(
  runId: string,
  onSummary: (summary: EpochSummary) => void,
  onSwitch: (record: SwitchRecord) => void,
  signal: AbortSignal,
): EventSource {
  const source = api.connectEvents(runId, (event: EventEnvelope) => {
    if (signal.aborted) {
      return;
    }
    const data = event.data as Record<string, unknown>;
    if (event.type === "train.epoch_summary") {
      onSummary({
        epoch: Number(data.epoch ?? 0),
        topology: String(data.topology ?? ""),
        throughput_samples_sec: Number(data.throughput_samples_sec ?? 0),
        val_acc: Number(data.val_acc ?? 0),
        train_loss: Number(data.train_loss ?? 0),
        avg_compute_ms: Number(data.avg_compute_ms ?? 0),
        avg_comm_ms: Number(data.avg_comm_ms ?? 0),
        ...data,
      });
    } else if (event.type === "controller.switch") {
      onSwitch({
        epoch: Number(data.epoch ?? 0),
        from: String(data.from ?? ""),
        to: String(data.to ?? ""),
        reason: (data.reason as Record<string, unknown>) ?? {},
        ...data,
      });
    }
  });
  source.onerror = () => {
    source.close();
  };
  return source;
}

export default function StressTrioLauncher({
  presets,
  onStart,
  onUpdate,
  onComplete,
  onError,
}: StressTrioLauncherProps) {
  const [active, setActive] = useState<ActiveTrio | null>(null);
  const [phase, setPhase] = useState<TrioPhase>("idle");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  function cancel() {
    abortRef.current?.abort();
    setPhase("idle");
    setActive(null);
  }

  async function runTrio(trio: typeof TRIO_DEFINITIONS[number]) {
    if (phase === "running") {
      return;
    }
    const presetIds = [trio.a, trio.p, trio.h];
    const missing = presetIds.find((id) => !findPreset(presets, id));
    if (missing) {
      onError(`Preset "${missing}" is not registered with the backend.`);
      return;
    }
    onStart();
    const controller = new AbortController();
    abortRef.current = controller;

    const status: Record<string, "queued" | "running" | "done" | "error"> = {
      [trio.a]: "queued",
      [trio.p]: "queued",
      [trio.h]: "queued",
    };
    setActive({ trio, presetIds, status, currentIndex: 0 });
    setPhase("running");

    const collected: OverlayRunPayload[] = [];

    try {
      for (let index = 0; index < presetIds.length; index += 1) {
        if (controller.signal.aborted) {
          throw new Error("aborted");
        }
        const presetId = presetIds[index];
        const preset = findPreset(presets, presetId);
        if (!preset) {
          throw new Error(`Preset ${presetId} not found.`);
        }

        status[presetId] = "running";
        setActive({ trio, presetIds, status: { ...status }, currentIndex: index });

        const run = await api.createRun(preset.request);
        const summaries: EpochSummary[] = [];
        const switches: SwitchRecord[] = [];

        const pushUpdate = () => {
          onUpdate({
            topology: preset.request.mode,
            runId: run.run_id,
            presetId,
            summaries: [...summaries],
            switches: [...switches],
          });
        };

        const source = streamRun(
          run.run_id,
          (summary) => {
            const next = summaries.filter((entry) => entry.epoch !== summary.epoch);
            next.push(summary);
            next.sort((left, right) => left.epoch - right.epoch);
            summaries.splice(0, summaries.length, ...next);
            pushUpdate();
          },
          (record) => {
            switches.push(record);
            pushUpdate();
          },
          controller.signal,
        );

        try {
          await pollUntilTerminal(run.run_id, controller.signal);
        } finally {
          source.close();
        }

        const finalRun = await api.getRun(run.run_id);
        if (finalRun.status === "failed") {
          status[presetId] = "error";
          setActive({ trio, presetIds, status: { ...status }, currentIndex: index });
          throw new Error(finalRun.error ?? `${presetId} failed`);
        }

        status[presetId] = "done";
        setActive({ trio, presetIds, status: { ...status }, currentIndex: index });
        collected.push({
          topology: preset.request.mode,
          runId: run.run_id,
          presetId,
          summaries: [...summaries],
          switches: [...switches],
        });
      }

      setPhase("done");
      onComplete(collected);
    } catch (reason: unknown) {
      if (controller.signal.aborted) {
        setPhase("idle");
        return;
      }
      setPhase("error");
      onError(reason instanceof Error ? reason.message : "Trio run failed.");
    } finally {
      abortRef.current = null;
    }
  }

  return (
    <div className="surface-lg flex flex-col gap-4 px-4 py-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="label">Stress Trio</span>
          <p className="mt-1 text-[12.5px] text-zinc-400">
            Runs RAR, PS, and Hybrid sequentially under the same scenario, then overlays them.
          </p>
        </div>
        {phase === "running" ? (
          <button type="button" className="btn-secondary" onClick={cancel}>
            <X className="h-3.5 w-3.5" />
            Cancel
          </button>
        ) : null}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {TRIO_DEFINITIONS.map((trio) => {
          const isActive = active?.trio.label === trio.label && phase === "running";
          return (
            <div
              key={trio.label}
              className={`flex flex-col gap-3 rounded-xl border px-3.5 py-3 transition ${
                isActive
                  ? "border-emerald-500/40 bg-emerald-500/[0.06]"
                  : "border-white/[0.06] bg-black/30"
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[13px] font-semibold tracking-tight text-zinc-100">
                    {trio.label}
                  </div>
                  <div className="mt-0.5 text-[11px] text-zinc-500">{trio.description}</div>
                </div>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => void runTrio(trio)}
                  disabled={phase === "running"}
                >
                  <Play className="h-3.5 w-3.5" />
                  Run
                </button>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <TrioSlot
                  label="RAR"
                  preset={trio.a}
                  status={isActive ? active!.status[trio.a] : "queued"}
                  active={isActive}
                  color="#10b981"
                />
                <TrioSlot
                  label="PS"
                  preset={trio.p}
                  status={isActive ? active!.status[trio.p] : "queued"}
                  active={isActive}
                  color="#f59e0b"
                />
                <TrioSlot
                  label="Hybrid"
                  preset={trio.h}
                  status={isActive ? active!.status[trio.h] : "queued"}
                  active={isActive}
                  color="#a78bfa"
                />
              </div>
            </div>
          );
        })}
      </div>

      {phase === "done" ? (
        <div className="rounded-md border border-emerald-500/20 bg-emerald-500/[0.06] px-3 py-2 text-[12px] text-emerald-300">
          Trio complete. All three runs are overlaid below.
        </div>
      ) : null}
    </div>
  );
}

function TrioSlot({
  label,
  preset,
  status,
  active,
  color,
}: {
  label: string;
  preset: string;
  status: "queued" | "running" | "done" | "error";
  active: boolean;
  color: string;
}) {
  let dotClass = "live-dot live-dot--idle";
  if (active && status === "running") {
    dotClass = "live-dot live-dot--running";
  } else if (status === "done") {
    dotClass = "live-dot";
  } else if (status === "error") {
    dotClass = "live-dot live-dot--error";
  }
  return (
    <div className="rounded-md border border-white/[0.05] bg-black/40 px-2 py-1.5">
      <div className="flex items-center justify-between">
        <span
          className="font-mono text-[10.5px] uppercase tracking-[0.16em]"
          style={{ color }}
        >
          {label}
        </span>
        <span className={dotClass} />
      </div>
      <div className="mt-1 truncate font-mono text-[10px] text-zinc-500">{preset}</div>
      <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-400">
        {status}
      </div>
    </div>
  );
}
