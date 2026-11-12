"use client";

import { Hint } from "@/components/ui/controls";
import { cn } from "@/lib/utils";

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint: string;
  tone?: "default" | "alert";
}) {
  return (
    <Hint label={hint}>
      <button
        type="button"
        className={cn(
          "glass w-full rounded-2xl p-3 sm:p-3.5 lg:p-4 text-left transition-all hover:border-slate-300",
          tone === "alert" && "border-rose-400/30"
        )}
      >
        <span className="block text-[11px] sm:text-xs uppercase tracking-[0.12em] text-slate-500 truncate">
          {label}
        </span>
        <span className="mt-1 sm:mt-1.5 block font-mono text-xl sm:text-2xl text-slate-900 font-semibold tracking-tight">
          {value}
        </span>
      </button>
    </Hint>
  );
}
