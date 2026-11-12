"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { Button } from "@/components/ui/button";
import { Modal } from "@/components/ui/modal";
import { Input } from "@/components/ui/input";
import { ErrorState, SampleBanner, Skeleton } from "@/components/ui/states";
import { apiBase, downloadApi, mediaUrl } from "@/lib/api";
import type { ViolationRecord } from "@/lib/types";
import { formatClock, formatDate, formatWhen, titleCase } from "@/lib/utils";

const types = ["all", "red_light", "wrong_lane", "helmet", "overspeed"];

export function ViolationBoard() {
  const { status, data, error } = useTrafficData();
  const [query, setQuery] = useState("");
  const [type, setType] = useState("all");
  const [day, setDay] = useState("");
  const [minConfidence, setMinConfidence] = useState(0);
  const [sort, setSort] = useState<{ key: keyof ViolationRecord; dir: "asc" | "desc" }>({ key: "created_at", dir: "desc" });
  const [open, setOpen] = useState<ViolationRecord | null>(null);

  const rows = useMemo(() => {
    if (!data) return [];
    const filtered = data.violations.filter((row) => {
      const haystack = `${row.id} ${row.video_name} ${row.vehicle_category} ${row.tracking_id ?? ""} ${row.evidence.plate_number ?? ""}`.toLowerCase();
      if (query && !haystack.includes(query.toLowerCase())) return false;
      if (type !== "all" && row.violation_type !== type) return false;
      if (day && !row.created_at.startsWith(day)) return false;
      if (row.confidence < minConfidence) return false;
      return true;
    });
    return filtered.sort((a, b) => {
      const left = a[sort.key];
      const right = b[sort.key];
      const order = left === right ? 0 : left != null && right != null && left > right ? 1 : -1;
      return sort.dir === "asc" ? order : -order;
    });
  }, [data, day, minConfidence, query, sort, type]);

  if (status === "loading") return <Skeleton className="h-96" />;
  if (status === "error" || !data) return <ErrorState title="Violations unavailable" body={error || "Could not load violations."} />;
  const isSample = data.source === "sample";

  function sortBy(key: keyof ViolationRecord) {
    setSort((current) => ({ key, dir: current.key === key && current.dir === "asc" ? "desc" : "asc" }));
  }

  function exportCsv() {
    const header = ["violation_id", "tracking_id", "plate_number", "vehicle_category", "violation_type", "timestamp_sec", "confidence", "video", "review_status", "sample", "explanation"];
    const body = rows.map((row) =>
      [row.id, row.tracking_id ?? "", row.evidence.plate_number ?? "", row.vehicle_category, row.violation_type, row.timestamp_sec, row.confidence, row.video_name, row.review_status, row.sample, row.evidence.explanation]
        .map((value) => `"${String(value).replaceAll('"', '""')}"`)
        .join(","),
    );
    const blob = new Blob([[header.join(","), ...body].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = data?.source === "sample" ? "sample-violations.csv" : "violations.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function exportPdf() {
    if (isSample) {
      window.print();
      return;
    }
    if (!apiBase()) {
      toast.error("PDF export requires a connection to the inference API.");
      return;
    }
    try {
      await downloadApi("/api/reports/export/pdf", "violations-report.pdf");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "PDF export failed.");
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Review queue"
        title="Traffic violations"
        description="Every row is a potential violation. Review keeps it, and false positive removes it from the confirmed set. Nothing here is a legal penalty."
        actions={
          <div className="no-print flex gap-2">
            <Button variant="secondary" onClick={exportCsv}>Export CSV</Button>
            <Button onClick={() => void exportPdf()}>{data.source === "sample" ? "Print / Save PDF" : "Export PDF"}</Button>
          </div>
        }
      />
      {data.source === "sample" && data.notice ? <SampleBanner text={data.notice} /> : null}
      <div className="no-print grid gap-3 md:grid-cols-4">
        <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search id, track, or video" aria-label="Search violations" />
        <select className="h-10 rounded-xl border border-slate-200 bg-white px-3 text-sm" value={type} onChange={(event) => setType(event.target.value)} aria-label="Violation type">
          {types.map((item) => (
            <option key={item} value={item}>
              {item === "all" ? "All types" : titleCase(item)}
            </option>
          ))}
        </select>
        <Input type="date" value={day} onChange={(event) => setDay(event.target.value)} aria-label="Filter by date" />
        <label className="text-xs text-muted">
          Minimum confidence {minConfidence.toFixed(2)}
          <input className="mt-2 w-full" type="range" min={0} max={1} step={0.01} value={minConfidence} onChange={(event) => setMinConfidence(Number(event.target.value))} />
        </label>
      </div>
      <div className="glass table-wrap rounded-2xl">
        <table className="data">
          <thead>
            <tr>
              {[
                ["id", "Violation ID"],
                ["tracking_id", "Track"],
                ["vehicle_category", "Vehicle"],
                ["violation_type", "Type"],
                ["created_at", "Date"],
                ["timestamp_sec", "Timestamp"],
                ["confidence", "Confidence"],
                ["video_name", "Video"],
              ].map(([key, label]) => (
                <th key={key}>
                  <button onClick={() => sortBy(key as keyof ViolationRecord)}>{label}</button>
                </th>
              ))}
              <th>Plate</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id}>
                <td className="font-mono text-xs">{row.id.slice(0, 10)}</td>
                <td>{row.tracking_id ?? "—"}</td>
                <td>{titleCase(row.vehicle_category)}</td>
                <td>{titleCase(row.violation_type)}</td>
                <td>{formatDate(row.created_at)}</td>
                <td>{formatClock(row.timestamp_sec)}</td>
                <td>{row.confidence.toFixed(2)}</td>
                <td>{row.video_name}</td>
                <td className="font-mono">{row.evidence.plate_number || "Not read"}</td>
                <td>
                  <Button size="sm" variant="secondary" onClick={() => setOpen(row)}>
                    View
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 ? <p className="p-6 text-sm text-muted">No records match these filters.</p> : null}
      </div>
      <Modal open={Boolean(open)} onOpenChange={(next) => { if (!next) setOpen(null); }} title="Violation evidence" description="The explanation states which rule fired and why a person should look at it.">
        {open ? (
          <div className="space-y-3 text-sm leading-6">
            {open.evidence.image && mediaUrl(open.evidence.image) ? (
              // Evidence frames are served by the inference API.
              // eslint-disable-next-line @next/next/no-img-element
              <img src={mediaUrl(open.evidence.image) ?? ""} alt="Annotated evidence frame" className="max-h-72 w-full rounded-xl object-contain bg-black" />
            ) : (
              <p className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-muted">No evidence image was stored for this record.</p>
            )}
            <p><strong>Plate number.</strong> {open.evidence.plate_number || "Not read"}</p>
            <p><strong>Rule.</strong> {open.evidence.rule}</p>
            <p>{open.evidence.explanation}</p>
            <p className="text-muted">Recorded {formatWhen(open.created_at)} · frame {open.frame_number} · model {open.evidence.model || "n/a"}</p>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
