import { ChevronDown, ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";
import type { Preset, RunRequest, ScenarioConfig } from "../types";

interface Props {
  presets: Preset[];
  selectedPresetId: string;
  onSelectPreset: (presetId: string) => void;
  request: RunRequest;
  onChangeRequest: (updater: (current: RunRequest) => RunRequest) => void;
  disabled?: boolean;
}

function makeWindow(epochs: number): number[] {
  if (epochs <= 2) {
    return [epochs];
  }
  const start = Math.max(2, Math.floor(epochs / 2));
  return Array.from(new Set([start, Math.min(epochs, start + 1)]));
}

export default function ScenarioForm({
  presets,
  selectedPresetId,
  onSelectPreset,
  request,
  onChangeRequest,
  disabled = false,
}: Props) {
  const [scenarioOpen, setScenarioOpen] = useState<boolean>(true);

  const scenarioActive = useMemo(
    () =>
      request.scenario.straggler_epochs.length > 0 ||
      request.scenario.bandwidth_epochs.length > 0,
    [request.scenario.straggler_epochs.length, request.scenario.bandwidth_epochs.length],
  );

  function updateScenario(update: (scenario: ScenarioConfig) => ScenarioConfig) {
    onChangeRequest((current) => ({ ...current, scenario: update(current.scenario) }));
  }

  return (
    <div className="space-y-4">
      <Field label="Preset">
        <select
          value={selectedPresetId}
          onChange={(event) => onSelectPreset(event.target.value)}
          className="ui-input"
          disabled={disabled}
        >
          {presets.length === 0 ? (
            <option value="">Loading presets…</option>
          ) : null}
          {presets.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.title}
            </option>
          ))}
        </select>
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field label="Mode">
          <select
            value={request.mode}
            onChange={(event) =>
              onChangeRequest((current) => ({
                ...current,
                mode: event.target.value as RunRequest["mode"],
              }))
            }
            className="ui-input"
            disabled={disabled}
          >
            <option value="ps">PS</option>
            <option value="rar">RAR</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </Field>
        <Field label="Topology">
          <select
            value={request.initial_topology}
            onChange={(event) =>
              onChangeRequest((current) => ({
                ...current,
                initial_topology: event.target.value as RunRequest["initial_topology"],
              }))
            }
            className="ui-input"
            disabled={disabled}
          >
            <option value="rar">RAR</option>
            <option value="ps">PS</option>
          </select>
        </Field>
        <Field label="Epochs">
          <input
            type="number"
            min={1}
            max={20}
            value={request.epochs}
            onChange={(event) =>
              onChangeRequest((current) => ({ ...current, epochs: Number(event.target.value) }))
            }
            className="ui-input"
            disabled={disabled}
          />
        </Field>
        <Field label="Workers">
          <input
            type="number"
            min={1}
            max={16}
            value={request.num_workers}
            onChange={(event) =>
              onChangeRequest((current) => ({
                ...current,
                num_workers: Number(event.target.value),
              }))
            }
            className="ui-input"
            disabled={disabled}
          />
        </Field>
        <Field label="Seed">
          <input
            type="number"
            value={request.seed}
            onChange={(event) =>
              onChangeRequest((current) => ({ ...current, seed: Number(event.target.value) }))
            }
            className="ui-input"
            disabled={disabled}
          />
        </Field>
        <Field label="Batch">
          <input
            type="number"
            min={1}
            value={request.batch_size}
            onChange={(event) =>
              onChangeRequest((current) => ({ ...current, batch_size: Number(event.target.value) }))
            }
            className="ui-input"
            disabled={disabled}
          />
        </Field>
      </div>

      <div className="rounded-xl border border-white/[0.06] bg-black/40">
        <button
          type="button"
          onClick={() => setScenarioOpen((current) => !current)}
          className="flex w-full items-center justify-between px-3.5 py-2.5 text-left"
        >
          <div className="flex items-center gap-2">
            {scenarioOpen ? (
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-zinc-500" />
            )}
            <span className="label">Scenario</span>
          </div>
          {scenarioActive ? (
            <span className="rounded-md border border-white/[0.08] bg-white/[0.03] px-1.5 py-0.5 text-[10px] uppercase tracking-[0.18em] text-zinc-400">
              active
            </span>
          ) : null}
        </button>

        {scenarioOpen ? (
          <div className="space-y-3 border-t border-white/[0.06] px-3.5 py-3.5">
            <Toggle
              checked={request.scenario.straggler_epochs.length > 0}
              label="Straggler in middle epochs"
              disabled={disabled}
              onChange={(checked) =>
                updateScenario((scenario) => ({
                  ...scenario,
                  straggler_epochs: checked ? makeWindow(request.epochs) : [],
                }))
              }
            />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Factor">
                <input
                  type="number"
                  step={0.5}
                  min={1}
                  value={request.scenario.straggler_factor}
                  onChange={(event) =>
                    updateScenario((scenario) => ({
                      ...scenario,
                      straggler_factor: Number(event.target.value),
                    }))
                  }
                  className="ui-input"
                  disabled={disabled}
                />
              </Field>
              <Field label="Rank">
                <input
                  type="number"
                  min={0}
                  value={request.scenario.straggler_rank}
                  onChange={(event) =>
                    updateScenario((scenario) => ({
                      ...scenario,
                      straggler_rank: Number(event.target.value),
                    }))
                  }
                  className="ui-input"
                  disabled={disabled}
                />
              </Field>
            </div>
            <Toggle
              checked={request.scenario.bandwidth_epochs.length > 0}
              label="Comm throttle in middle epochs"
              disabled={disabled}
              onChange={(checked) =>
                updateScenario((scenario) => ({
                  ...scenario,
                  bandwidth_epochs: checked ? makeWindow(request.epochs) : [],
                }))
              }
            />
            <Field label="Throttle (ms)">
              <input
                type="number"
                min={0}
                step={0.5}
                value={request.scenario.throttle_ms}
                onChange={(event) =>
                  updateScenario((scenario) => ({
                    ...scenario,
                    throttle_ms: Number(event.target.value),
                  }))
                }
                className="ui-input"
                disabled={disabled}
              />
            </Field>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <div className="mt-1.5">{children}</div>
    </label>
  );
}

function Toggle({
  checked,
  label,
  onChange,
  disabled,
}: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label
      className={`flex items-center justify-between text-sm text-zinc-300 ${
        disabled ? "opacity-50" : ""
      }`}
    >
      <span>{label}</span>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 shrink-0 rounded-full border transition ${
          checked
            ? "border-emerald-500/40 bg-emerald-500"
            : "border-white/10 bg-zinc-800"
        }`}
        aria-pressed={checked}
      >
        <span
          className={`absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white transition ${
            checked ? "left-[18px]" : "left-0.5"
          }`}
        />
      </button>
    </label>
  );
}
