from __future__ import annotations

from demo_api.schemas import Preset, RunRequest, ScenarioConfig


def get_demo_presets() -> list[Preset]:
    return [
        Preset(
            id="ps-quick",
            title="Static PS Baseline",
            description="A short, stable MNIST run that shows centralized synchronization.",
            accent="cyan",
            request=RunRequest(
                mode="ps",
                epochs=4,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
            ),
        ),
        Preset(
            id="rar-quick",
            title="Static RAR Baseline",
            description="The same short run using Ring AllReduce for comparison.",
            accent="emerald",
            request=RunRequest(
                mode="rar",
                epochs=4,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
            ),
        ),
        Preset(
            id="hybrid-straggler",
            title="Hybrid Straggler Demo",
            description="Starts in RAR, injects a mid-run straggler, and switches at epoch boundaries.",
            accent="violet",
            request=RunRequest(
                mode="hybrid",
                epochs=5,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                initial_topology="rar",
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4],
                    straggler_rank=3,
                    straggler_factor=3.0,
                ),
            ),
        ),
        Preset(
            id="ps-bandwidth",
            title="Bandwidth Pressure",
            description="Applies communication delay in the middle epochs to surface comm-sensitive behavior.",
            accent="amber",
            request=RunRequest(
                mode="ps",
                epochs=4,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                scenario=ScenarioConfig(
                    bandwidth_epochs=[2, 3],
                    throttle_ms=10.0,
                ),
            ),
        ),

        # ── Stress Level A: strong straggler only ─────────────────────────────
        Preset(
            id="stress-a-rar",
            title="Stress A: Static RAR",
            description="Static RAR under a strong 3-epoch straggler. Expect ring barriers to amplify the slowdown.",
            accent="emerald",
            request=RunRequest(
                mode="rar",
                epochs=6,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4, 5],
                    straggler_rank=3,
                    straggler_factor=30.0,
                    base_compute_ms=5.0,
                ),
            ),
        ),
        Preset(
            id="stress-a-ps",
            title="Stress A: Static PS",
            description="Static PS under the same 3-epoch straggler. PS absorbs straggler cost at one sync point per batch.",
            accent="cyan",
            request=RunRequest(
                mode="ps",
                epochs=6,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4, 5],
                    straggler_rank=3,
                    straggler_factor=30.0,
                    base_compute_ms=5.0,
                ),
            ),
        ),
        Preset(
            id="stress-a-hybrid",
            title="Stress A: Hybrid",
            description="Hybrid starts in RAR, switches to PS when the straggler lag is detected, then tracks PS robustness.",
            accent="violet",
            request=RunRequest(
                mode="hybrid",
                epochs=6,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                initial_topology="rar",
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4, 5],
                    straggler_rank=3,
                    straggler_factor=30.0,
                    base_compute_ms=5.0,
                ),
            ),
        ),

        # ── Stress Level B: straggler + communication throttle ────────────────
        # throttle_ms applies inside every ring-barrier step in RAR (6x per batch
        # with 4 workers), but only once per batch in PS — so the gap is largest here.
        Preset(
            id="stress-b-rar",
            title="Stress B: Static RAR",
            description="Static RAR under straggler + communication delay. Ring barriers compound both costs per batch.",
            accent="emerald",
            request=RunRequest(
                mode="rar",
                epochs=6,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4, 5],
                    straggler_rank=3,
                    straggler_factor=30.0,
                    base_compute_ms=5.0,
                    bandwidth_epochs=[3, 4, 5],
                    throttle_ms=8.0,
                ),
            ),
        ),
        Preset(
            id="stress-b-ps",
            title="Stress B: Static PS",
            description="Static PS under the same straggler + throttle. PS sees throttle once per batch, not per ring step.",
            accent="cyan",
            request=RunRequest(
                mode="ps",
                epochs=6,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4, 5],
                    straggler_rank=3,
                    straggler_factor=30.0,
                    base_compute_ms=5.0,
                    bandwidth_epochs=[3, 4, 5],
                    throttle_ms=8.0,
                ),
            ),
        ),
        Preset(
            id="stress-b-hybrid",
            title="Stress B: Hybrid",
            description="Hybrid under straggler + throttle. Switches from RAR to PS early and tracks PS throughput during stress.",
            accent="violet",
            request=RunRequest(
                mode="hybrid",
                epochs=6,
                model="logreg",
                dataset="mnist",
                num_workers=4,
                seed=42,
                initial_topology="rar",
                scenario=ScenarioConfig(
                    straggler_epochs=[3, 4, 5],
                    straggler_rank=3,
                    straggler_factor=30.0,
                    base_compute_ms=5.0,
                    bandwidth_epochs=[3, 4, 5],
                    throttle_ms=8.0,
                ),
            ),
        ),
    ]
