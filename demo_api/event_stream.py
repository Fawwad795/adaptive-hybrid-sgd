from __future__ import annotations

import json
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Iterator


# Event types that must survive indefinitely for history replay.
# Iteration events are high-frequency and can be evicted once the buffer fills.
_PERSISTENT_TYPES = frozenset({
    "run.started",
    "train.epoch_summary",
    "worker.epoch_summary",
    "telemetry.snapshot",
    "controller.switch",
    "run.completed",
    "run.failed",
    "run.stopped",
    "run.stopping",
})


class RunEventStream:
    def __init__(self, buffer_size: int = 512) -> None:
        self._lock = threading.Lock()
        # Rolling buffer for high-frequency events (train.iteration etc.)
        self._events: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=buffer_size))
        # Never-evicted list for epoch summaries, switches, and lifecycle events.
        self._persistent: dict[str, list[dict]] = defaultdict(list)
        self._subscribers: dict[str, list[queue.Queue]] = defaultdict(list)

    def publish(self, run_id: str, event_type: str, data: dict) -> dict:
        event = {
            "type": event_type,
            "run_id": run_id,
            "ts": time.time(),
            "data": data,
        }
        with self._lock:
            self._events[run_id].append(event)
            if event_type in _PERSISTENT_TYPES:
                self._persistent[run_id].append(event)
            subscribers = list(self._subscribers[run_id])
        for subscriber in subscribers:
            subscriber.put(event)
        return event

    def subscribe(self, run_id: str) -> tuple[queue.Queue, list[dict]]:
        subscriber: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers[run_id].append(subscriber)
            # Merge persistent (always complete) + rolling (recent iterations),
            # deduplicate by object identity, then sort by timestamp.
            persistent = self._persistent.get(run_id, [])
            rolling = list(self._events.get(run_id, ()))
            seen: set[int] = set()
            merged: list[dict] = []
            for event in persistent:
                seen.add(id(event))
                merged.append(event)
            for event in rolling:
                if id(event) not in seen:
                    merged.append(event)
            backlog = sorted(merged, key=lambda e: e["ts"])
        return subscriber, backlog

    def unsubscribe(self, run_id: str, subscriber: queue.Queue) -> None:
        with self._lock:
            listeners = self._subscribers.get(run_id, [])
            if subscriber in listeners:
                listeners.remove(subscriber)

    def iter_sse(self, run_id: str) -> Iterator[str]:
        subscriber, backlog = self.subscribe(run_id)
        try:
            for event in backlog:
                yield self._format_sse(event)
            while True:
                try:
                    event = subscriber.get(timeout=15)
                    yield self._format_sse(event)
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            self.unsubscribe(run_id, subscriber)

    @staticmethod
    def _format_sse(event: dict) -> str:
        return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


EVENT_STREAM = RunEventStream()
