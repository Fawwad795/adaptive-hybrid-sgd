export type RunMode = "ps" | "rar" | "hybrid";
export type RunStatus = "queued" | "running" | "stopping" | "completed" | "failed" | "stopped";
export type Topology = "ps" | "rar";

export interface ScenarioConfig {
  straggler_epochs: number[];
  straggler_rank: number;
  straggler_factor: number;
  base_compute_ms: number;
  bandwidth_epochs: number[];
  throttle_ms: number;
}

export interface RunRequest {
  mode: RunMode;
  model: "logreg" | "cnn";
  dataset: "mnist" | "cifar10";
  epochs: number;
  lr: number;
  batch_size: number;
  num_workers: number;
  seed: number;
  ps_discipline: "bsp" | "ssp" | "async";
  initial_topology: Topology;
  scenario: ScenarioConfig;
}

export interface Preset {
  id: string;
  title: string;
  description: string;
  accent: string;
  request: RunRequest;
}

export interface ArtifactSummary {
  logs: string[];
  checkpoints: string[];
  plots: string[];
  tables: string[];
}

export interface RunRecord {
  run_id: string;
  status: RunStatus;
  mode: RunMode;
  config: RunRequest;
  started_at: number | null;
  finished_at: number | null;
  current_topology: string | null;
  latest_epoch: number;
  latest_metrics: Record<string, unknown>;
  switches: Record<string, unknown>[];
  artifacts: ArtifactSummary;
  error: string | null;
}

export interface EventEnvelope<T = Record<string, unknown>> {
  type: string;
  run_id: string;
  ts: number;
  data: T;
}

export interface ComparisonSeries {
  key: string;
  label: string;
  points: Array<Record<string, number | string | boolean>>;
}

export interface ComparisonDataset {
  id: string;
  title: string;
  description: string;
  source: string;
  x_key: string;
  y_key: string;
  series: ComparisonSeries[];
}

export interface ComparisonResponse {
  datasets: ComparisonDataset[];
  available_files: string[];
}
