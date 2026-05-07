from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RunMode = Literal["ps", "rar", "hybrid"]
RunStatus = Literal["queued", "running", "stopping", "completed", "failed", "stopped"]


class ScenarioConfig(BaseModel):
    straggler_epochs: list[int] = Field(default_factory=list)
    straggler_rank: int = 3
    straggler_factor: float = 3.0
    base_compute_ms: float = 5.0
    bandwidth_epochs: list[int] = Field(default_factory=list)
    throttle_ms: float = 0.0


class RunRequest(BaseModel):
    mode: RunMode
    model: Literal["logreg", "cnn"] = "logreg"
    dataset: Literal["mnist", "cifar10"] = "mnist"
    epochs: int = Field(default=5, ge=1, le=20)
    lr: float = Field(default=0.01, gt=0)
    batch_size: int = Field(default=64, ge=1)
    num_workers: int = Field(default=4, ge=1, le=16)
    seed: int = 42
    ps_discipline: Literal["bsp", "ssp", "async"] = "bsp"
    initial_topology: Literal["ps", "rar"] = "rar"
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)


class EventEnvelope(BaseModel):
    type: str
    run_id: str
    ts: float
    data: dict[str, Any]


class ArtifactSummary(BaseModel):
    logs: list[str] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    plots: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)


class RunRecordResponse(BaseModel):
    run_id: str
    status: RunStatus
    mode: RunMode
    config: dict[str, Any]
    started_at: float | None = None
    finished_at: float | None = None
    current_topology: str | None = None
    latest_epoch: int = 0
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    switches: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: ArtifactSummary = Field(default_factory=ArtifactSummary)
    error: str | None = None


class Preset(BaseModel):
    id: str
    title: str
    description: str
    accent: str
    request: RunRequest


class ComparisonSeries(BaseModel):
    key: str
    label: str
    points: list[dict[str, Any]] = Field(default_factory=list)


class ComparisonDataset(BaseModel):
    id: str
    title: str
    description: str
    source: str
    x_key: str
    y_key: str
    series: list[ComparisonSeries] = Field(default_factory=list)


class ComparisonResponse(BaseModel):
    datasets: list[ComparisonDataset] = Field(default_factory=list)
    available_files: list[str] = Field(default_factory=list)
