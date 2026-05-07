import { useMemo } from "react";
import {
  CartesianGrid,
  Label,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ThroughputSeries {
  key: string;
  label: string;
  color: string;
  data: Array<Record<string, number | string>>;
}

interface SwitchMarker {
  epoch: number;
  from?: string;
  to?: string;
  label?: string;
}

interface Props {
  series: ThroughputSeries[];
  yKey: string;
  xKey?: string;
  height?: number;
  stressedEpochs?: number[];
  switches?: SwitchMarker[];
  yLabel?: string;
  showLegend?: boolean;
}

const AXIS_COLOR = "#52525b";
const GRID_COLOR = "#27272a";

function buildStressedRanges(epochs: number[] | undefined): Array<[number, number]> {
  if (!epochs || epochs.length === 0) {
    return [];
  }
  const sorted = Array.from(new Set(epochs)).sort((a, b) => a - b);
  const ranges: Array<[number, number]> = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (let i = 1; i < sorted.length; i += 1) {
    const value = sorted[i];
    if (value === prev + 1) {
      prev = value;
      continue;
    }
    ranges.push([start - 0.5, prev + 0.5]);
    start = value;
    prev = value;
  }
  ranges.push([start - 0.5, prev + 0.5]);
  return ranges;
}

export default function ThroughputChart({
  series,
  yKey,
  xKey = "epoch",
  height = 280,
  stressedEpochs,
  switches,
  yLabel,
  showLegend = false,
}: Props) {
  const stressedRanges = useMemo(() => buildStressedRanges(stressedEpochs), [stressedEpochs]);

  const merged = useMemo(() => {
    const bucket = new Map<number | string, Record<string, number | string>>();
    series.forEach((line) => {
      line.data.forEach((point) => {
        const x = point[xKey] as number | string;
        const row = bucket.get(x) ?? { [xKey]: x };
        row[line.key] = point[yKey] as number;
        bucket.set(x, row);
      });
    });
    return Array.from(bucket.values()).sort((left, right) => {
      return Number(left[xKey]) - Number(right[xKey]);
    });
  }, [series, yKey, xKey]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
        <CartesianGrid stroke={GRID_COLOR} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey={xKey}
          stroke={AXIS_COLOR}
          tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={{ stroke: GRID_COLOR }}
          allowDecimals={false}
          type="number"
          domain={["dataMin", "dataMax"]}
        />
        <YAxis
          stroke={AXIS_COLOR}
          tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={false}
          width={50}
          label={
            yLabel
              ? {
                  value: yLabel,
                  angle: -90,
                  position: "insideLeft",
                  fill: "#52525b",
                  fontSize: 10,
                  fontFamily: "Inter",
                  style: { textTransform: "uppercase", letterSpacing: "0.18em" },
                }
              : undefined
          }
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
        {showLegend ? (
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#a1a1aa", paddingTop: 6 }}
            iconType="plainline"
          />
        ) : null}
        {stressedRanges.map(([from, to], index) => (
          <ReferenceArea
            key={`stress-${index}`}
            x1={from}
            x2={to}
            fill="#f43f5e"
            fillOpacity={0.06}
            stroke="#f43f5e"
            strokeOpacity={0.18}
            strokeDasharray="3 3"
            ifOverflow="hidden"
          />
        ))}
        {(switches ?? []).map((swap, index) => (
          <ReferenceLine
            key={`switch-${index}-${swap.epoch}`}
            x={swap.epoch}
            stroke="#a78bfa"
            strokeDasharray="2 4"
            strokeOpacity={0.65}
          >
            <Label
              value={
                swap.label ??
                (swap.from && swap.to
                  ? `${swap.from.toUpperCase()}→${swap.to.toUpperCase()}`
                  : "switch")
              }
              position="top"
              fill="#c4b5fd"
              fontSize={10}
              fontFamily="JetBrains Mono"
            />
          </ReferenceLine>
        ))}
        {series.map((line) => (
          <Line
            key={line.key}
            type="monotone"
            dataKey={line.key}
            name={line.label}
            stroke={line.color}
            strokeWidth={2}
            dot={{ r: 2.5, fill: line.color, strokeWidth: 0 }}
            activeDot={{ r: 4, strokeWidth: 0 }}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
