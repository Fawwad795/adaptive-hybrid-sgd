import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { EventEnvelope, RunRecord, RunRequest } from "../types";

const TERMINAL_STATUSES = new Set(["completed", "failed", "stopped"]);
const POLL_INTERVAL_MS = 750;

export interface IterationPoint {
  point: number;
  epoch: number;
  loss: number;
  acc: number;
  topology: string;
}

export interface EpochSummary {
  epoch: number;
  topology: string;
  throughput_samples_sec: number;
  val_acc: number;
  train_loss: number;
  avg_compute_ms: number;
  avg_comm_ms: number;
  [key: string]: unknown;
}

export interface SwitchRecord {
  epoch: number;
  from: string;
  to: string;
  reason: Record<string, unknown>;
  [key: string]: unknown;
}

export interface EventLogEntry {
  id: number;
  type: string;
  detail: string;
  ts: number;
}

export interface UseRunResult {
  activeRun: RunRecord | null;
  setActiveRun: (run: RunRecord | null) => void;
  events: EventLogEntry[];
  iterations: IterationPoint[];
  summaries: EpochSummary[];
  telemetry: Record<string, unknown>;
  switches: SwitchRecord[];
  error: string;
  busy: boolean;
  runIsActive: boolean;
  pulseTick: number;
  startRun: (request: RunRequest) => Promise<RunRecord | null>;
  stopRun: () => Promise<void>;
  reset: () => void;
}

function formatEventDetail(envelope: EventEnvelope): string {
  const data = envelope.data as Record<string, unknown>;
  if (typeof data.error === "string") {
    return data.error;
  }
  if (typeof data.from === "string" && typeof data.to === "string") {
    return `${String(data.from).toUpperCase()} -> ${String(data.to).toUpperCase()} @ epoch ${String(
      data.epoch ?? "-",
    )}`;
  }
  if (typeof data.topology === "string") {
    return `${String(data.topology).toUpperCase()} epoch ${String(data.epoch ?? "-")}`;
  }
  if (typeof data.epoch === "number" || typeof data.epoch === "string") {
    return `epoch ${String(data.epoch)}`;
  }
  return "";
}

export function useRun(): UseRunResult {
  const [activeRun, setActiveRun] = useState<RunRecord | null>(null);
  const [events, setEvents] = useState<EventLogEntry[]>([]);
  const [iterations, setIterations] = useState<IterationPoint[]>([]);
  const [summaries, setSummaries] = useState<EpochSummary[]>([]);
  const [telemetry, setTelemetry] = useState<Record<string, unknown>>({});
  const [switches, setSwitches] = useState<SwitchRecord[]>([]);
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [pulseTick, setPulseTick] = useState<number>(0);
  const eventCounterRef = useRef<number>(0);

  const reset = useCallback(() => {
    setEvents([]);
    setIterations([]);
    setSummaries([]);
    setTelemetry({});
    setSwitches([]);
    setError("");
    eventCounterRef.current = 0;
  }, []);

  const handleEvent = useCallback((envelope: EventEnvelope) => {
    const data = envelope.data as Record<string, unknown>;
    eventCounterRef.current += 1;
    const id = eventCounterRef.current;
    setEvents((current) =>
      [
        ...current,
        {
          id,
          type: envelope.type,
          detail: formatEventDetail(envelope),
          ts: envelope.ts,
        },
      ].slice(-200),
    );

    if (envelope.type === "train.iteration") {
      setIterations((current) => [
        ...current,
        {
          point: current.length + 1,
          epoch: Number(data.display_epoch ?? data.epoch ?? 0),
          loss: Number(data.loss ?? 0),
          acc: Number(data.acc ?? 0) * 100,
          topology: String(data.topology ?? ""),
        },
      ]);
    }

    if (envelope.type === "train.epoch_summary") {
      setSummaries((current) => {
        const next = current.filter((entry) => entry.epoch !== Number(data.epoch));
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
        return next.sort((left, right) => left.epoch - right.epoch);
      });
      setPulseTick((tick) => tick + 1);
    }

    if (envelope.type === "telemetry.snapshot") {
      setTelemetry(data);
    }

    if (envelope.type === "controller.switch") {
      setSwitches((current) => [
        ...current,
        {
          epoch: Number(data.epoch ?? 0),
          from: String(data.from ?? ""),
          to: String(data.to ?? ""),
          reason: (data.reason as Record<string, unknown>) ?? {},
          ...data,
        },
      ]);
    }
  }, []);

  // SSE wiring — re-subscribes whenever the active run id changes.
  useEffect(() => {
    const runId = activeRun?.run_id;
    if (!runId) {
      return;
    }
    const source = api.connectEvents(runId, handleEvent);
    source.onerror = () => {
      source.close();
    };
    return () => source.close();
  }, [activeRun?.run_id, handleEvent]);

  // 750ms polling for active runs only — terminal runs are static.
  useEffect(() => {
    const runId = activeRun?.run_id;
    if (!runId) {
      return;
    }
    if (TERMINAL_STATUSES.has(activeRun!.status)) {
      return;
    }
    const interval = window.setInterval(() => {
      setPulseTick((tick) => tick + 1);
      void api
        .getRun(runId)
        .then((next) => setActiveRun(next))
        .catch(() => undefined);
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [activeRun?.run_id, activeRun?.status, activeRun]);

  const runIsActive = useMemo(
    () => activeRun !== null && !TERMINAL_STATUSES.has(activeRun.status),
    [activeRun],
  );

  const startRun = useCallback(
    async (request: RunRequest): Promise<RunRecord | null> => {
      setBusy(true);
      setError("");
      reset();
      try {
        const run = await api.createRun(request);
        setActiveRun(run);
        return run;
      } catch (reason: unknown) {
        setError(reason instanceof Error ? reason.message : "Failed to start run.");
        return null;
      } finally {
        setBusy(false);
      }
    },
    [reset],
  );

  const stopRun = useCallback(async () => {
    if (!activeRun) {
      return;
    }
    setBusy(true);
    try {
      const run = await api.stopRun(activeRun.run_id);
      setActiveRun(run);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Failed to stop run.");
    } finally {
      setBusy(false);
    }
  }, [activeRun]);

  const setActiveRunSafe = useCallback((run: RunRecord | null) => {
    reset();
    setActiveRun(run);
  }, [reset]);

  return {
    activeRun,
    setActiveRun: setActiveRunSafe,
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
    reset,
  };
}

export const POLL_INTERVAL = POLL_INTERVAL_MS;
