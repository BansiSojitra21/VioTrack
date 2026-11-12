"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Summary, TrafficSeries } from "@/lib/types";
import { classColor, titleCase } from "@/lib/utils";

const tooltipStyle = {
  background: "#ffffff",
  border: "1px solid rgba(15,23,42,0.1)",
  borderRadius: 12,
  color: "#0f172a",
};

export function AnalyticsCharts({ summary, traffic }: { summary: Summary; traffic: TrafficSeries }) {
  const categories = Object.entries(summary.vehicle_counts).map(([name, value]) => ({ name: titleCase(name), value, fill: classColor(name) }));
  const violations = Object.entries(summary.violation_counts).map(([name, value]) => ({ name: titleCase(name), value }));
  const peak = traffic.hourly.reduce((best, row) => (row.violations > best.violations ? row : best), traffic.hourly[0] ?? { hour: 0, violations: 0 });

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <ChartCard title="Violations over time" subtitle="Daily counts from stored records">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={traffic.daily}>
            <CartesianGrid stroke="rgba(15,23,42,0.08)" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={(value) => String(value).slice(5)} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey="violations" stroke="#22d3ee" strokeWidth={2.4} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Vehicle categories" subtitle="Unique tracks by class when ids exist">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={categories} dataKey="value" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={3} tabIndex={-1}>
              {categories.map((entry) => (
                <Cell key={entry.name} fill={entry.fill} tabIndex={-1} className="outline-none focus:outline-none" />
              ))}
            </Pie>
            <Legend />
            <Tooltip contentStyle={tooltipStyle} />
          </PieChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Violation categories" subtitle={summary.most_common_violation ? `Most common: ${titleCase(summary.most_common_violation)}` : "No violations stored"}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={violations}>
            <CartesianGrid stroke="rgba(15,23,42,0.08)" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} interval={0} angle={-18} height={54} textAnchor="end" />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Hourly activity" subtitle={`Peak hour ${String(peak.hour).padStart(2, "0")}:00 with ${peak.violations} potential violations`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={traffic.hourly}>
            <CartesianGrid stroke="rgba(15,23,42,0.08)" vertical={false} />
            <XAxis dataKey="hour" tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="violations" fill="#22d3ee" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Traffic density" subtitle="Detections in 10-second buckets">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={traffic.density}>
            <CartesianGrid stroke="rgba(15,23,42,0.08)" vertical={false} />
            <XAxis dataKey="t" tick={{ fill: "#94a3b8", fontSize: 11 }} unit="s" />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="detections" fill="#38bdf8" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
      <ChartCard title="Vehicle flow timeline" subtitle="Vehicle detections per minute across monitored duration">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={traffic.vehicles_by_minute}>
            <CartesianGrid stroke="rgba(15,23,42,0.08)" vertical={false} />
            <XAxis dataKey="minute" tick={{ fill: "#94a3b8", fontSize: 11 }} unit="m" />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey="vehicles" stroke="#10b981" strokeWidth={2.4} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="glass rounded-2xl p-3 sm:p-4 min-w-0 overflow-hidden">
      <h3 className="font-medium truncate">{title}</h3>
      <p className="mb-3 text-xs text-muted truncate">{subtitle}</p>
      <div className="h-56 sm:h-64 w-full">{children}</div>
    </section>
  );
}
