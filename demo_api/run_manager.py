from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from demo_api.event_stream import EVENT_STREAM
from demo_api.runtime import execute_run
from demo_api.schemas import ArtifactSummary, RunRecordResponse


@dataclass
class RunRecord:
    run_id: str
    mode: str
    config: dict[str, Any]
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    current_topology: str | None = None
    latest_epoch: int = 0
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    switches: list[dict[str, Any]] = field(default_factory=list)
    artifacts: ArtifactSummary = field(default_factory=ArtifactSummary)
    error: str | None = None
    thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    def to_response(self) -> RunRecordResponse:
        return RunRecordResponse(
            run_id=self.run_id,
            status=self.status,
            mode=self.mode,
            config=self.config,
            started_at=self.started_at,
            finished_at=self.finished_at,
            current_topology=self.current_topology,
            latest_epoch=self.latest_epoch,
            latest_metrics=self.latest_metrics,
            switches=self.switches,
            artifacts=self.artifacts,
            error=self.error,
        )


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}

    def start_run(self, request) -> RunRecordResponse:
        run_id = f"demo-{uuid.uuid4().hex[:8]}"
        record = RunRecord(
            run_id=run_id,
            mode=request.mode,
            config=request.model_dump(),
            current_topology=request.mode if request.mode != "hybrid" else request.initial_topology,
        )
        thread = threading.Thread(target=self._run_job, args=(record, request), daemon=True)
        record.thread = thread
        with self._lock:
            self._runs[run_id] = record
        thread.start()
        return record.to_response()

    def list_runs(self) -> list[RunRecordResponse]:
        with self._lock:
            return [record.to_response() for record in self._runs.values()]

    def get_run(self, run_id: str) -> RunRecordResponse | None:
        with self._lock:
            record = self._runs.get(run_id)
            return record.to_response() if record else None

    def stop_run(self, run_id: str) -> RunRecordResponse | None:
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return None
            if record.status in {"completed", "failed", "stopped"}:
                return record.to_response()
            record.status = "stopping"
            record.stop_event.set()
        self._publish(record, "run.stopping", {"message": "Stop requested."})
        return record.to_response()

    def _run_job(self, record: RunRecord, request) -> None:
        with self._lock:
            record.status = "running"
            record.started_at = time.time()
        try:
            artifacts = execute_run(record.run_id, request, lambda event_type, data: self._publish(record, event_type, data), record.stop_event)
            with self._lock:
                record.finished_at = time.time()
                if record.status == "stopping":
                    record.status = "stopped"
                elif record.status != "failed":
                    record.status = "completed"
                record.artifacts = ArtifactSummary(**artifacts)
        except Exception as exc:
            with self._lock:
                record.status = "failed"
                record.finished_at = time.time()
                record.error = str(exc)

    def _publish(self, record: RunRecord, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            if event_type == "train.epoch_summary":
                record.latest_epoch = int(data.get("epoch", record.latest_epoch))
                record.latest_metrics = data
                record.current_topology = data.get("topology", record.current_topology)
            elif event_type == "controller.switch":
                record.switches.append(data)
                record.current_topology = data.get("to", record.current_topology)
            elif event_type == "run.started":
                record.current_topology = data.get("initial_topology", record.current_topology)
            elif event_type == "run.completed":
                record.finished_at = time.time()
            elif event_type == "run.failed":
                record.error = data.get("error")
            elif event_type == "run.stopped":
                record.finished_at = time.time()
        EVENT_STREAM.publish(record.run_id, event_type, data)


RUN_MANAGER = RunManager()
