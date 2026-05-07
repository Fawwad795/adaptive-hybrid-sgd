import { NavLink } from "react-router-dom";

interface Props {
  apiBase: string;
  apiOnline: boolean | null;
  runCount: number;
  activeRunCount: number;
}

const NAV = [
  { to: "/", label: "Studio", end: true },
  { to: "/compare", label: "Compare", end: false },
  { to: "/benchmarks", label: "Benchmarks", end: false },
];

export default function Header({ apiBase, apiOnline, runCount, activeRunCount }: Props) {
  let dotClass = "live-dot live-dot--idle";
  let dotLabel = "checking";
  if (apiOnline === true) {
    dotClass = activeRunCount > 0 ? "live-dot live-dot--running" : "live-dot";
    dotLabel = activeRunCount > 0 ? `${activeRunCount} running` : "online";
  } else if (apiOnline === false) {
    dotClass = "live-dot live-dot--error";
    dotLabel = "offline";
  }

  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-black/95 backdrop-blur-[2px]">
      <div className="mx-auto flex h-12 w-full max-w-[1600px] items-center justify-between px-5">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2.5">
            <div className="h-5 w-5 rounded-md border border-white/[0.08] bg-gradient-to-br from-emerald-500/40 to-emerald-700/30" />
            <span className="text-[13px] font-semibold tracking-tight text-zinc-100">
              Adaptive Hybrid SGD
            </span>
          </div>
          <nav className="flex items-center gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-md px-2.5 py-1 text-[12.5px] font-medium transition ${
                    isActive
                      ? "bg-white/[0.06] text-zinc-100"
                      : "text-zinc-400 hover:text-zinc-200"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-2 sm:flex">
            <span className="font-mono text-[11px] text-zinc-500">{apiBase.replace(/^https?:\/\//, "")}</span>
            <span className={dotClass} />
            <span className="text-[11px] text-zinc-400">{dotLabel}</span>
          </div>
          <div className="flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-zinc-950 px-2 py-0.5 font-mono text-[11px] text-zinc-300">
            <span className="text-zinc-500">runs</span>
            <span className="tabular text-zinc-100">{runCount}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
