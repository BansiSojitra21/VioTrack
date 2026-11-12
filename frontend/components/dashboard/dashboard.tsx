"use client";

import Link from "next/link";
import { AnalyticsCharts } from "@/components/charts/analytics-charts";
import { StatCard } from "@/components/dashboard/stat-card";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { Badge } from "@/components/ui/badge";
import { SampleBanner, Skeleton, ErrorState } from "@/components/ui/states";
import { formatClock, titleCase } from "@/lib/utils";

export function Dashboard() {
  const { status, data, error } = useTrafficData();
  if (status === "loading") {
    return (
      <div className="grid gap-4 md:grid-cols-4">
        {Array.from({ length: 11 }).map((_, index) => (
          <Skeleton key={index} className="h-28" />
        ))}
      </div>
    );
  }
  if (status === "error" || !data) return <ErrorState title="Dashboard unavailable" body={error || "The inference API did not return data."} />;
  const summary = data.summary;
  const accuracy = summary.detection_accuracy == null ? "—" : `${Math.round(summary.detection_accuracy * 1000) / 10}%`;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Command center"
        title="Traffic operations"
        description="Vehicle counts, potential violations, and the status of every module. Speed and accuracy appear only when calibration or human review exists."
      />
      {data.source === "sample" && data.notice ? <SampleBanner text={data.notice} /> : null}

      {/* Top: Arranged & Responsive KPIs (1 on mobile, 2 on tablet, strictly 4 in a row on desktop) */}
      <section className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Videos analyzed" value={String(summary.videos_analyzed)} hint="Completed analysis jobs. Uploads that have not finished are excluded." />
        <StatCard label="Unique vehicles tracked" value={String(summary.vehicles_detected)} hint="Unique vehicle IDs scoped to their analysis job and video." />
        <StatCard label="Visible in latest frame" value={String(summary.vehicles_visible_now ?? 0)} hint="Tracked vehicles in the latest processed frame of the newest job." />
        <StatCard label="Vehicle detections" value={String(summary.vehicle_detections_total ?? 0)} hint="Valid vehicle detections across processed frames; one vehicle can appear in many frames." />
        <StatCard label="Potential violations" value={String(summary.violations_total)} hint="Flags waiting for a person. These are not penalties." tone="alert" />
        <StatCard label="Confirmed violations" value={String(summary.violations_confirmed ?? 0)} hint="Potential violations reviewed and confirmed by a person." tone="alert" />
        <StatCard label="Red light" value={String(summary.red_light)} hint="Crossings of a configured stop line during a verified red interval." />
        <StatCard label="Wrong lane" value={String(summary.wrong_lane)} hint="Tracks whose heading opposes a configured lane direction." />
        <StatCard label="Helmet" value={String(summary.helmet)} hint="Requires a custom helmet model. COCO weights cannot produce this class." />
        <StatCard label="Overspeeding" value={String(summary.overspeed)} hint="Counted only when camera calibration is marked validated." />
        <StatCard label="Review accuracy" value={accuracy} hint={summary.detection_accuracy_note} />
      </section>

      {/* Bottom: 2 Columns of Charts */}
      <section>
        <AnalyticsCharts summary={summary} traffic={data.traffic} />
      </section>

      {/* Bottom: Table */}
      <section className="glass rounded-2xl p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-medium">Recent potential violations</h2>
          <Link href="/violations" className="text-sm text-cyan-700">
            Open table
          </Link>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Type</th>
                <th>Vehicle</th>
                <th>Track</th>
                <th>Time</th>
                <th>Confidence</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {data.violations.slice(0, 5).map((row) => (
                <tr key={row.id}>
                  <td>{titleCase(row.violation_type)}</td>
                  <td>{titleCase(row.vehicle_category)}</td>
                  <td className="font-mono">{row.tracking_id ?? "—"}</td>
                  <td>{formatClock(row.timestamp_sec)}</td>
                  <td>{row.confidence.toFixed(2)}</td>
                  <td>
                    {row.sample ? <Badge tone="amber">Sample</Badge> : row.video_name}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
