import type { ComparisonResponse, EventEnvelope, Preset, RunRecord, RunRequest } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE,
  getPresets: () => request<Preset[]>("/presets"),
  getComparisons: () => request<ComparisonResponse>("/results/comparisons"),
  listRuns: () => request<RunRecord[]>("/runs"),
  createRun: (payload: RunRequest) =>
    request<RunRecord>("/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getRun: (runId: string) => request<RunRecord>(`/runs/${runId}`),
  stopRun: (runId: string) =>
    request<RunRecord>(`/runs/${runId}/stop`, {
      method: "POST",
    }),
  connectEvents(runId: string, onEvent: (event: EventEnvelope) => void) {
    const source = new EventSource(`${API_BASE}/runs/${runId}/events`);
    const eventTypes = [
      "run.started",
      "run.epoch_started",
      "train.iteration",
      "worker.epoch_summary",
      "train.epoch_summary",
      "telemetry.snapshot",
      "controller.switch",
      "run.completed",
      "run.failed",
      "run.stopping",
      "run.stopped",
    ];
    eventTypes.forEach((type) => {
      source.addEventListener(type, (message) => {
        onEvent(JSON.parse((message as MessageEvent<string>).data));
      });
    });
    return source;
  },
};
