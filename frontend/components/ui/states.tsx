import { cn } from "@/lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-2xl bg-slate-200", className)} />;
}

export function Progress({ value }: { value: number }) {
  const width = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="h-2 overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuenow={width} aria-valuemin={0} aria-valuemax={100}>
      <div className="h-full rounded-full bg-gradient-to-r from-violet-500 via-blue-500 to-cyan-400" style={{ width: `${width}%` }} />
    </div>
  );
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: React.ReactNode }) {
  return (
    <div className="glass rounded-2xl px-6 py-12 text-center">
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted">{body}</p>
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-5 py-4">
      <h3 className="font-semibold text-rose-800">{title}</h3>
      <p className="mt-1 text-sm leading-6 text-rose-700">{body}</p>
    </div>
  );
}

export function SampleBanner({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900">
      <strong className="font-semibold">Sample data. </strong>
      {text}
    </div>
  );
}
