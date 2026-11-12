"use client";

import { AnalyticsCharts } from "@/components/charts/analytics-charts";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { Button } from "@/components/ui/button";
import { ErrorState, SampleBanner, Skeleton } from "@/components/ui/states";
import { classColor, titleCase } from "@/lib/utils";

export function AnalyticsView() {
  const { status, data, error } = useTrafficData();
  if (status === "loading") return <Skeleton className="h-96" />;
  if (status === "error" || !data) return <ErrorState title="Analytics unavailable" body={error || "Could not load analytics."} />;

  function exportCsv() {
    const lines = ["section,key,value,sample"];
    Object.entries(data!.summary.vehicle_counts).forEach(([key, value]) => lines.push(`vehicle,${key},${value},${data!.source === "sample"}`));
    Object.entries(data!.summary.violation_counts).forEach(([key, value]) => lines.push(`violation,${key},${value},${data!.source === "sample"}`));
    data!.traffic.hourly.forEach((row) => lines.push(`hour,${row.hour},${row.violations},${data!.source === "sample"}`));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "traffic-analytics.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const maxHour = data.traffic.hourly.reduce((best, row) => (row.violations > best.violations ? row : best), { hour: 0, violations: 0 });

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Data science"
        title="Traffic analytics"
        description="Counts are aggregated with Pandas on the API when it is connected. The timing heat strip uses violation timestamps. It is not a map of the intersection."
        actions={<Button onClick={exportCsv}>Export dataset</Button>}
      />
      {data.source === "sample" && data.notice ? <SampleBanner text={data.notice} /> : null}
      <AnalyticsCharts summary={data.summary} traffic={data.traffic} />
      <section className="grid gap-4 lg:grid-cols-2">
        <div className="glass rounded-2xl p-4">
          <h2 className="font-medium">Violation timing heat strip</h2>
          <p className="mb-4 text-sm text-muted">A spatial heatmap would need a site map and a homography. Those are not invented here. Darker cells are busier hours.</p>
          <div className="grid grid-cols-12 gap-2">
            {data.traffic.hourly.map((row) => {
              const max = Math.max(...data.traffic.hourly.map((item) => item.violations), 1);
              return (
                <div key={row.hour} className="rounded-lg px-1 py-3 text-center text-[10px]" style={{ background: `rgba(34,211,238,${0.08 + (row.violations / max) * 0.85})` }} title={`${row.hour}:00`}>
                  {row.hour}
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-sm text-slate-600">Peak hour in this dataset: {String(maxHour.hour).padStart(2, "0")}:00.</p>
        </div>
        <div className="glass rounded-2xl p-4">
          <h2 className="font-medium">Movement tracks</h2>
          <p className="mb-3 text-sm text-muted">
            {data.source === "sample" ? "Sample paths drawn for the layout. They are not trajectories from a tracker." : "Paths are centroids from the latest stored detections, normalized per frame size when a single video is available."}
          </p>
          <svg viewBox="0 0 100 100" className="h-64 w-full rounded-xl bg-slate-100">
            {(data.source === "sample" ? data.tracks : tracksFrom(data.detections)).map((track) => (
              <polyline
                key={track.id}
                fill="none"
                stroke={classColor(track.class_name)}
                strokeWidth="1.4"
                points={track.points.map((point) => `${point[0] * 100},${point[1] * 100}`).join(" ")}
              />
            ))}
          </svg>
        </div>
      </section>
      <section className="glass rounded-2xl p-5">
        <h2 className="text-lg font-medium">Why a row was flagged</h2>
        <div className="mt-4 grid gap-3">
          {data.violations.slice(0, 4).map((row) => (
            <article key={row.id} className="rounded-xl border border-slate-200 p-4 text-sm leading-6">
              <p className="font-medium">{titleCase(row.violation_type)} · {titleCase(row.vehicle_category)} · confidence {row.confidence.toFixed(2)}</p>
              <p>Object: track {row.tracking_id ?? "unassigned"}.</p>
              <p>Rule: {row.evidence.rule}</p>
              <p>Evidence: {row.evidence.explanation}</p>
              <p className="text-muted">Manual review is required because the system stores potential violations, and low-confidence helmet predictions stay marked for a person.</p>
            </article>
          ))}
          {data.violations.length === 0 ? <p className="text-sm text-muted">Run an analysis with rules configured to populate this section.</p> : null}
        </div>
      </section>
    </div>
  );
}

function tracksFrom(detections: { tracking_id: number | null; class_name: string; bbox: number[] }[]) {
  const groups = new Map<number, { id: number; class_name: string; points: [number, number][] }>();
  for (const detection of detections) {
    if (detection.tracking_id == null) continue;
    const current = groups.get(detection.tracking_id) ?? { id: detection.tracking_id, class_name: detection.class_name, points: [] };
    const [x1, y1, x2, y2] = detection.bbox;
    const maxX = Math.max(...detections.map((item) => item.bbox[2]), 1);
    const maxY = Math.max(...detections.map((item) => item.bbox[3]), 1);
    current.points.push([(x1 + x2) / 2 / maxX, (y1 + y2) / 2 / maxY]);
    groups.set(detection.tracking_id, current);
  }
  return [...groups.values()];
}
