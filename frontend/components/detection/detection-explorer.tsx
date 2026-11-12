"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { toast } from "sonner";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { Button } from "@/components/ui/button";
import { ErrorState, SampleBanner, Skeleton } from "@/components/ui/states";
import { apiBase, downloadApi } from "@/lib/api";
import { classColor, formatVideoTimestamp, titleCase } from "@/lib/utils";

export function DetectionExplorer() {
  const { status, data, error } = useTrafficData();
  const [klass, setKlass] = useState("all");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    if (!data) return [];
    const normalizedQuery = query.trim().toLowerCase();
    return data.detections.filter((row) => {
      const matchesClass = klass === "all" || row.class_name === klass;
      const matchesQuery = !normalizedQuery || [row.class_name, row.tracking_id, row.plate_number]
        .some((value) => String(value ?? "").toLowerCase().includes(normalizedQuery));
      return matchesClass && matchesQuery;
    });
  }, [data, klass, query]);

  if (status === "loading") return <Skeleton className="h-96" />;
  if (status === "error" || !data) return <ErrorState title="Detections unavailable" body={error || "Could not load detections."} />;
  const liveData = data;

  async function exportCsv() {
    if (liveData.source !== "sample") {
      if (!apiBase()) {
        toast.error("CSV export requires a connection to the inference API.");
        return;
      }
      try {
        await downloadApi("/api/reports/export/csv?kind=detections", "detections.csv");
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "CSV export failed.");
      }
      return;
    }

    const header = ["detection_id", "job_id", "video_id", "frame_number", "timestamp_sec", "class_name", "confidence", "tracking_id", "plate_number"];
    const body = rows.map((row) =>
      [row.id, row.job_id, row.video_id, row.frame_number, row.timestamp_sec, row.class_name, row.confidence, row.tracking_id ?? "", row.plate_number ?? ""]
        .map((value) => `"${String(value).replaceAll('"', '""')}"`)
        .join(","),
    );
    const blob = new Blob([[header.join(","), ...body].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "sample-detections.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const classes = ["all", ...new Set(liveData.detections.map((row) => row.class_name))];

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Perception"
        title="Vehicle detection"
        description="Review stored vehicle detections, tracks, and recognized plate numbers."
        actions={
          <div className="flex w-full flex-col gap-2 sm:flex-row md:w-auto md:items-center">
            <label className="relative block w-full sm:w-80">
              <Search size={16} aria-hidden="true" className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search track or plate…"
                aria-label="Search detections by class, track, or plate" className="w-full rounded-xl border border-slate-200 bg-white py-2.5 pl-9 pr-3 text-sm outline-none transition placeholder:text-slate-400 focus:border-violet-400 focus:ring-4 focus:ring-violet-100" />
            </label>
            <Button className="shrink-0" onClick={() => void exportCsv()}>Export CSV</Button>
          </div>
        }
      />
      {data.source === "sample" && data.notice ? <SampleBanner text={data.notice} /> : null}
      <div className="glass table-wrap rounded-2xl">
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 text-sm">
          <p className="font-medium text-slate-700">Detection records</p>
          <p className="text-muted" aria-live="polite">{rows.length} {rows.length === 1 ? "result" : "results"}</p>
        </div>
        <table className="data">
          <thead>
            <tr>
              <th>
                <label className="inline-flex items-center gap-2">
                  <span>Class</span>
                  <select
                    className="h-8 rounded-lg border border-slate-200 bg-white px-2 text-xs font-normal normal-case tracking-normal text-slate-700"
                    value={klass}
                    onChange={(event) => setKlass(event.target.value)}
                    aria-label="Filter detections by class"
                  >
                    <option value="all">All classes</option>
                    {classes.filter((item) => item !== "all").map((item) => (
                      <option key={item} value={item}>{titleCase(item)}</option>
                    ))}
                  </select>
                </label>
              </th>
              <th>Track</th>
              <th>Plate</th>
              <th>Confidence</th>
              <th>Frame</th>
              <th>Time (video)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td style={{ color: classColor(row.class_name) }}>{titleCase(row.class_name)}</td>
                <td>{row.tracking_id ?? "—"}</td>
                <td className="font-mono">{row.plate_number ?? "—"}</td>
                <td>{row.confidence.toFixed(2)}</td>
                <td>{row.frame_number}</td>
                <td className="font-mono">{formatVideoTimestamp(row.timestamp_sec)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <p className="font-medium text-slate-700">{data.detections.length === 0 ? "No detections yet" : "No matching detections"}</p>
            <p className="mt-1 text-sm text-muted">{data.detections.length === 0 ? "Run a video analysis to see detected vehicles here." : "Try another search or class filter."}</p>
            {query || klass !== "all" ? <button type="button" onClick={() => { setQuery(""); setKlass("all"); }} className="mt-3 text-sm font-medium text-violet-700 hover:text-violet-900">Clear filters</button> : null}
          </div>
        ) : null}
      </div>
      <p className="text-sm text-muted">
        Active model: {data.model.name}. {data.model.loaded ? "Weights are loaded." : data.model.error}
      </p>
    </div>
  );
}
