import type { CSSProperties } from "react";

type TopologyKey = "rar" | "ps" | "hybrid";

const COLOR_MAP: Record<TopologyKey, { bg: string; fg: string; border: string }> = {
  rar: { bg: "rgba(16, 185, 129, 0.10)", fg: "#34d399", border: "rgba(16, 185, 129, 0.30)" },
  ps: { bg: "rgba(245, 158, 11, 0.10)", fg: "#fbbf24", border: "rgba(245, 158, 11, 0.30)" },
  hybrid: { bg: "rgba(167, 139, 250, 0.10)", fg: "#c4b5fd", border: "rgba(167, 139, 250, 0.30)" },
};

interface Props {
  topology: string | null | undefined;
  size?: "sm" | "md";
  className?: string;
}

export default function TopologyBadge({ topology, size = "sm", className = "" }: Props) {
  const key = String(topology ?? "").toLowerCase();
  const palette =
    key === "rar" || key === "ps" || key === "hybrid"
      ? COLOR_MAP[key as TopologyKey]
      : { bg: "rgba(255, 255, 255, 0.04)", fg: "#a1a1aa", border: "rgba(255, 255, 255, 0.12)" };

  const style: CSSProperties = {
    backgroundColor: palette.bg,
    color: palette.fg,
    borderColor: palette.border,
  };

  const padding = size === "md" ? "px-2.5 py-1 text-[11px]" : "px-2 py-0.5 text-[10px]";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border font-mono uppercase tracking-[0.14em] ${padding} ${className}`}
      style={style}
    >
      <span className="h-1 w-1 rounded-full" style={{ backgroundColor: palette.fg }} />
      {topology ? String(topology).toUpperCase() : "—"}
    </span>
  );
}
