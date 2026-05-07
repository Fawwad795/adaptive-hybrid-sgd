import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import type { ComparisonDataset } from "../types";

const SERIES_COLORS = ["#10b981", "#f59e0b", "#a78bfa", "#60a5fa", "#f472b6"];

function mergeDataset(dataset: ComparisonDataset): Array<Record<string, string | number>> {
  const bucket = new Map<string, Record<string, string | number>>();
  dataset.series.forEach((series) => {
    series.points.forEach((point) => {
      const xValue = String(point[dataset.x_key]);
      const row = bucket.get(xValue) ?? { [dataset.x_key]: point[dataset.x_key] as string | number };
      row[series.key] = point[dataset.y_key] as number;
      bucket.set(xValue, row);
    });
  });
  return Array.from(bucket.values());
}

export default function BenchmarksPage() {
  const [comparisons, setComparisons] = useState<ComparisonDataset[]>([]);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    void api
      .getComparisons()
      .then((response) => {
        setComparisons(response.datasets);
        setAvailableFiles(response.available_files);
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Failed to load benchmark data.");
      });
  }, []);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 px-5 py-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-[18px] font-semibold tracking-tight text-zinc-100">Benchmarks</h1>
          <p className="mt-0.5 text-[12.5px] text-zinc-400">
            Precomputed comparison tables loaded from{" "}
            <span className="font-mono text-zinc-300">results/tables</span>.
          </p>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] text-zinc-400">
          <span className="tabular text-zinc-100">{comparisons.length}</span>
          <span className="text-zinc-500">datasets</span>
          <span className="text-zinc-700">·</span>
          <span className="tabular text-zinc-100">{availableFiles.length}</span>
          <span className="text-zinc-500">files</span>
        </div>
      </div>

      {error ? (
        <div className="rounded-md border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-[12px] text-rose-300">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="space-y-4">
          {comparisons.length === 0 ? (
            <div className="surface-lg flex flex-col gap-2 px-4 py-8 text-center">
              <span className="label mx-auto">no datasets</span>
              <p className="mx-auto max-w-md text-[12.5px] text-zinc-400">
                Generate CSV tables in <span className="font-mono text-zinc-300">results/tables</span>{" "}
                and they will render here.
              </p>
            </div>
          ) : (
            comparisons.map((dataset) => {
              const chartData = mergeDataset(dataset);
              return (
                <div key={dataset.id} className="surface-lg px-4 py-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="text-[14px] font-semibold tracking-tight text-zinc-100">
                        {dataset.title}
                      </h3>
                      <p className="mt-1 text-[12px] text-zinc-400">{dataset.description}</p>
                    </div>
                    <span className="font-mono text-[10.5px] uppercase tracking-[0.18em] text-zinc-500">
                      {dataset.source}
                    </span>
                  </div>
                  <div className="mt-3">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                        <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey={dataset.x_key}
                          stroke="#52525b"
                          tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "JetBrains Mono" }}
                          tickLine={false}
                          axisLine={{ stroke: "#27272a" }}
                        />
                        <YAxis
                          stroke="#52525b"
                          tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "JetBrains Mono" }}
                          tickLine={false}
                          axisLine={false}
                          width={50}
                        />
                        <Tooltip
                          cursor={{ stroke: "#3f3f46", strokeDasharray: "3 3" }}
                          contentStyle={{
                            background: "#09090b",
                            border: "1px solid rgba(255,255,255,0.08)",
                            borderRadius: 8,
                            fontFamily: "JetBrains Mono",
                            fontSize: 11,
                          }}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: 11, color: "#a1a1aa", paddingTop: 6 }}
                          iconType="plainline"
                        />
                        {dataset.series.map((series, index) => (
                          <Line
                            key={series.key}
                            type="monotone"
                            dataKey={series.key}
                            name={series.label}
                            stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                            strokeWidth={2}
                            dot={{ r: 2.5, strokeWidth: 0 }}
                            activeDot={{ r: 4, strokeWidth: 0 }}
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })
          )}
        </div>

        <div className="surface-lg flex h-fit flex-col px-4 py-4">
          <span className="label">CSV Inventory</span>
          <div className="mt-3 space-y-1.5">
            {availableFiles.length === 0 ? (
              <p className="text-[12px] text-zinc-500">No CSV benchmark tables detected yet.</p>
            ) : (
              availableFiles.map((file) => (
                <div
                  key={file}
                  className="rounded-md border border-white/[0.05] bg-black/40 px-2.5 py-1.5 font-mono text-[11px] text-zinc-300"
                >
                  {file}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
