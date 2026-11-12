import { cn } from "@/lib/utils";

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "cyan" | "violet" | "amber" | "rose" | "green";
  className?: string;
}) {
  const tones = {
    neutral: "bg-slate-100 text-slate-700",
    cyan: "bg-cyan-50 text-cyan-800",
    violet: "bg-violet-50 text-violet-800",
    amber: "bg-amber-50 text-amber-800",
    rose: "bg-rose-50 text-rose-800",
    green: "bg-emerald-50 text-emerald-800",
  };
  return <span className={cn("inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium", tones[tone], className)}>{children}</span>;
}
