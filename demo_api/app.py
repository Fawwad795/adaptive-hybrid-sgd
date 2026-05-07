from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from demo_api.comparisons import load_comparisons
from demo_api.event_stream import EVENT_STREAM
from demo_api.presets import get_demo_presets
from demo_api.run_manager import RUN_MANAGER
from demo_api.schemas import ComparisonResponse, Preset, RunRecordResponse, RunRequest


app = FastAPI(
    title="Adaptive Hybrid SGD Demo API",
    version="1.0.0",
    description="Judge-facing live demo backend for PS, RAR, and adaptive hybrid runs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/presets", response_model=list[Preset])
def get_presets() -> list[Preset]:
    return get_demo_presets()


@app.get("/runs", response_model=list[RunRecordResponse])
def list_runs() -> list[RunRecordResponse]:
    return RUN_MANAGER.list_runs()


@app.post("/runs", response_model=RunRecordResponse)
def create_run(request: RunRequest) -> RunRecordResponse:
    return RUN_MANAGER.start_run(request)


@app.get("/runs/{run_id}", response_model=RunRecordResponse)
def get_run(run_id: str) -> RunRecordResponse:
    record = RUN_MANAGER.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@app.post("/runs/{run_id}/stop", response_model=RunRecordResponse)
def stop_run(run_id: str) -> RunRecordResponse:
    record = RUN_MANAGER.stop_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


@app.get("/runs/{run_id}/events")
def stream_events(run_id: str) -> StreamingResponse:
    if not RUN_MANAGER.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(
        EVENT_STREAM.iter_sse(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/results/comparisons", response_model=ComparisonResponse)
def get_comparisons() -> ComparisonResponse:
    return load_comparisons()
