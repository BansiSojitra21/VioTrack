"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Activity,
  Clapperboard,
  LayoutDashboard,
  Menu,
  Radio,
  Scan,
  Settings,
  ShieldAlert,
  X,
} from "lucide-react";
import { liveStreamUrl } from "@/lib/api";
import { loadPreferences } from "@/lib/settings";
import { useTrafficData } from "@/components/providers/data-provider";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/videos", label: "Video Management", icon: Clapperboard },
  { href: "/detection", label: "Vehicle Detection", icon: Scan },
  { href: "/violations", label: "Traffic Violations", icon: ShieldAlert },
  { href: "/analytics", label: "Analytics", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const { data } = useTrafficData();
  const stream = liveStreamUrl();
  const items = stream ? [...links.slice(0, 2), { href: "/live", label: "Live Analysis", icon: Radio }, ...links.slice(2)] : links;

  useEffect(() => {
    document.documentElement.dataset.accent = loadPreferences().accent;
    const timer = window.setTimeout(() => setMounted(true), 0);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen">
      <aside className={cn("no-print fixed inset-y-0 left-0 z-40 w-60 border-r border-slate-200 bg-white/95 p-3.5 shadow-sm backdrop-blur-xl transition md:translate-x-0", open ? "translate-x-0" : "-translate-x-full")}>
        <div className="mb-6 flex items-center justify-between px-2">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-tr from-violet-600 to-cyan-500 text-white shadow-sm ring-1 ring-black/5">
              <ShieldAlert size={16} strokeWidth={2.4} />
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-700">Smart city</p>
              <p className="text-lg font-bold tracking-tight text-slate-900">VioTrack</p>
            </div>
          </Link>
          <button className="rounded-lg p-2 text-slate-500 md:hidden" onClick={() => setOpen(false)} aria-label="Close navigation">
            <X size={16} />
          </button>
        </div>
        <nav className="space-y-1">
          {items.map((item) => {
            const active = mounted && (pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href)));
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={cn(
                  "flex items-center gap-3 rounded-xl px-3 py-2 text-sm text-slate-600 hover:bg-slate-100",
                  active && "bg-violet-50 text-slate-900",
                )}
              >
                <Icon size={16} className={active ? "text-violet-700" : "text-slate-400"} />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="absolute bottom-4 left-3.5 right-3.5 rounded-2xl border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-500">
          {data?.source === "live" ? "Inference API connected. Records come from processed video." : "Demo mode. Connect the Python API to run YOLO."}
        </div>
      </aside>
      <div className="md:pl-60">
        <header className="no-print sticky top-0 z-30 flex items-center justify-between border-b border-slate-200 bg-white/85 px-4 py-3 backdrop-blur-xl md:px-8">
          <button className="rounded-xl border border-slate-200 p-2 md:hidden" onClick={() => setOpen(true)} aria-label="Open navigation">
            <Menu size={18} />
          </button>
          <p className="text-sm text-slate-500">Potential violations are for human review. VioTrack does not issue fines.</p>
          <span className={cn("rounded-full px-3 py-1 text-xs", data?.source === "live" ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800")}>
            {data?.source === "live" ? "Live API" : data ? "Sample data" : "Connecting"}
          </span>
        </header>
        <main className="px-4 py-6 md:px-8 md:py-8">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({
  kicker,
  title,
  description,
  actions,
}: {
  kicker: string;
  title: string;
  description: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-end">
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-cyan-700">{kicker}</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight">{title}</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">{description}</p>
      </div>
      {actions}
    </div>
  );
}
