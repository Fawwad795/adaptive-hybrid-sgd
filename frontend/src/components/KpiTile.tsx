import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  trailing?: ReactNode;
  className?: string;
}

export default function KpiTile({ label, value, hint, trailing, className = "" }: Props) {
  return (
    <div
      className={`flex flex-col justify-between rounded-xl border border-white/[0.06] bg-zinc-950 px-4 py-3.5 ${className}`}
    >
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        {trailing ? <div className="text-zinc-500">{trailing}</div> : null}
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="tabular text-2xl font-semibold tracking-tight text-zinc-100 lg:text-[28px]">
          {value}
        </span>
      </div>
      {hint ? <div className="mt-1.5 text-[11px] text-zinc-500">{hint}</div> : null}
    </div>
  );
}
