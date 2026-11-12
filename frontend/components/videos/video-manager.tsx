"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/input";
import { Modal } from "@/components/ui/modal";
import { Toggle } from "@/components/ui/controls";
import { EmptyState, ErrorState, SampleBanner, Skeleton } from "@/components/ui/states";
import { apiBase, apiSend, downloadApi, mediaUrl, startAnalysis } from "@/lib/api";
import { loadPreferences } from "@/lib/settings";
import type { AnalysisConfig, VideoRecord } from "@/lib/types";
import { formatBytes, formatClock, formatWhen } from "@/lib/utils";

const tones = {
  pending: "neutral",
  processing: "cyan",
  cancelling: "cyan",
  cancelled: "neutral",
  completed: "green",
  failed: "rose",
} as const;

interface StopLineSuggestion {
  found: boolean;
  line: number[][] | null;
  score: number;
  preview_width: number;
  preview_height: number;
  preview_jpeg: string;
}

function parsePairs(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const points = trimmed.split(/\s+/).map((part) => part.split(",").map(Number));
  if (points.some((pair) => pair.length !== 2 || pair.some((value) => !Number.isFinite(value)))) {
    throw new Error("Coordinates must look like 0.08,0.62 0.92,0.62");
  }
  return points;
}

function createSignalTimeline(durationSec: number): NonNullable<AnalysisConfig["signal_timeline"]> {
  const timeline: NonNullable<AnalysisConfig["signal_timeline"]> = [];
  const duration = Math.max(0, durationSec);
  const initialGreenEnd = Math.min(2, duration);
  if (initialGreenEnd > 0) {
    timeline.push({ start: 0, end: initialGreenEnd, state: "green" });
  }

  for (let cycleStart = 2; cycleStart < duration; cycleStart += 15) {
    const redEnd = Math.min(cycleStart + 5, duration);
    timeline.push({ start: cycleStart, end: redEnd, state: "red" });
    const greenEnd = Math.min(cycleStart + 15, duration);
    if (redEnd < greenEnd) {
      timeline.push({ start: redEnd, end: greenEnd, state: "green" });
    }
  }
  return timeline;
}

function formatStopLine(line: number[][]) {
  return line
    .map((point) => point.map((coordinate) => coordinate.toFixed(4)).join(","))
    .join(" ");
}

function parseStopLineEndpoints(text: string): [[number, number], [number, number]] | null {
  const parts = text.trim().split(/\s+/);
  if (parts.length !== 2 || !text.trim()) return null;
  const points = parts.map((part) => part.split(",").map(Number));
  if (points.some((point) => point.length !== 2 || point.some((coordinate) => !Number.isFinite(coordinate)))) {
    return null;
  }
  return [
    [points[0][0], points[0][1]],
    [points[1][0], points[1][1]],
  ];
}

export function VideoManager() {
  const router = useRouter();
  const { status, data, error, refresh } = useTrafficData();
  const [analyzingVideo, setAnalyzingVideo] = useState<string | null>(null);
  const [cancellingJob, setCancellingJob] = useState<string | null>(null);
  const [advancedVideo, setAdvancedVideo] = useState<VideoRecord | null>(null);
  const [progress, setProgress] = useState<string | null>(null);
  const [stopLine, setStopLine] = useState("");
  const [stopLineDrawn, setStopLineDrawn] = useState(false);
  const [stopLineSuggestion, setStopLineSuggestion] = useState<StopLineSuggestion | null>(null);
  const [suggestingStopLine, setSuggestingStopLine] = useState(false);
  const [stopLineRetry, setStopLineRetry] = useState(0);
  const [drawingStopLine, setDrawingStopLine] = useState(false);
  const [lineDrawingStart, setLineDrawingStart] = useState<[number, number] | null>(null);
  const [lineDrawingEnd, setLineDrawingEnd] = useState<[number, number] | null>(null);
  const [modules, setModules] = useState({
    red: true,
    lane: true,
    speed: true,
    helmet: true,
    annotated: true,
  });

  useEffect(() => {
    if (!advancedVideo || !modules.red) return;

    const videoId = advancedVideo.id;
    let cancelled = false;

    async function loadStopLine() {
      try {
        const suggestion = await apiSend<StopLineSuggestion>(
          `/api/videos/${videoId}/stop-line/suggest`,
          { method: "POST" },
        );
        if (cancelled) return;
        setStopLineSuggestion(suggestion);
        if (suggestion.found && suggestion.line) {
          setStopLine(formatStopLine(suggestion.line));
          setStopLineDrawn(true);
        }
      } catch (err) {
        if (!cancelled) {
          toast.error(err instanceof Error ? err.message : "Could not detect a stop line automatically.");
        }
      } finally {
        if (!cancelled) setSuggestingStopLine(false);
      }
    }

    void loadStopLine();
    return () => {
      cancelled = true;
    };
  }, [advancedVideo, modules.red, stopLineRetry]);

  if (status === "loading") return <Skeleton className="h-80" />;
  if (status === "error" || !data) return <ErrorState title="Videos unavailable" body={error || "Could not load the video library."} />;

  const maxMb = data.health?.max_upload_mb ?? 200;
  const maxDuration = data.health?.max_duration_sec ?? 180;

  async function upload(files: File[]) {
    if (!apiBase() || data?.source !== "live") {
      toast.error("The inference API is not connected. The file was not uploaded and no detection was run.");
      return;
    }
    for (const file of files) {
      if (file.size > maxMb * 1024 * 1024) {
        toast.error(`${file.name} is over the ${maxMb} MB limit.`);
        continue;
      }
      const body = new FormData();
      body.append("file", file);
      setProgress(`Uploading ${file.name}`);
      try {
        await apiSend("/api/videos/upload", { method: "POST", body });
        toast.success(`${file.name} stored.`);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Upload failed.");
      }
    }
    setProgress(null);
    await refresh();
  }

  async function remove(video: VideoRecord) {
    if (video.sample || !apiBase()) {
      toast.error("Sample clips cannot be deleted. They exist only in the browser demo.");
      return;
    }
    if (video.latest_job && ["pending", "processing", "cancelling"].includes(video.latest_job.status)) {
      toast.error("Cancel the active analysis before deleting this video.");
      return;
    }
    if (!window.confirm(`Delete "${video.original_name}" and its analysis results?`)) return;
    try {
      await apiSend(`/api/videos/${video.id}`, { method: "DELETE" });
      toast.success("Video deleted.");
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not delete this video.");
      await refresh();
    }
  }

  async function cancelAnalysis(video: VideoRecord) {
    const activeJob = video.latest_job;
    if (!activeJob || !apiBase()) {
      toast.error("The active analysis job could not be found.");
      return;
    }
    setCancellingJob(activeJob.id);
    try {
      await apiSend(`/api/analysis/${activeJob.id}/cancel`, { method: "POST" });
      toast.success("Analysis cancellation requested. Its partial detections will be cleared.");
      await refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not cancel this analysis.");
    } finally {
      setCancellingJob(null);
    }
  }

  async function analyze(video: VideoRecord, customConfig?: AnalysisConfig) {
    if (video.sample || !apiBase()) {
      toast.error("Analysis needs a video stored on the inference API. Sample rows are not sent to YOLO.");
      return;
    }
    setAnalyzingVideo(video.id);
    const prefs = loadPreferences();
    try {
      const defaults: AnalysisConfig = {
        conf: prefs.conf,
        iou: prefs.iou,
        imgsz: prefs.imgsz,
        track_buffer: prefs.trackBuffer,
        debug_mode: prefs.debugMode,
        frame_stride: prefs.frameStride,
        target_fps: prefs.targetFps,
        max_frames: Math.min(
          prefs.maxFrames,
          Math.max(1, Math.ceil((video.duration_sec ?? maxDuration) * (video.fps || 25))),
        ),
        stop_line: null,
        signal_timeline: null,
        lanes: null,
        calibration: null,
        enable_red_light: true,
        enable_wrong_lane: true,
        enable_overspeed: true,
        enable_helmet: true,
        save_annotated: prefs.saveAnnotated,
        model_name: null,
      };
      await startAnalysis(video.id, customConfig ?? defaults);
      await refresh();
      toast.success("Analysis started. You can view progress or cancel it here.");
      setAdvancedVideo(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not start analysis.");
    } finally {
      setAnalyzingVideo(null);
    }
  }

  function configureAnalysis(video: VideoRecord) {
    const prefs = loadPreferences();
    setStopLine("");
    setStopLineDrawn(false);
    setStopLineSuggestion(null);
    setDrawingStopLine(false);
    setSuggestingStopLine(true);
    setLineDrawingStart(null);
    setLineDrawingEnd(null);
    setModules({
      red: true,
      lane: true,
      speed: true,
      helmet: true,
      annotated: prefs.saveAnnotated,
    });
    setAdvancedVideo(video);
  }

  function analyzeWithAdvancedSettings() {
    if (!advancedVideo) return;
    try {
      const stopLinePoints = parsePairs(stopLine);
      if (stopLinePoints && stopLinePoints.length !== 2) {
        throw new Error("A stop line must have exactly two points.");
      }
      if (modules.red && (!stopLineDrawn || !stopLinePoints)) {
        throw new Error("A stop line must be detected automatically or drawn on the video frame before running the red-light check.");
      }
      const prefs = loadPreferences();
      const config: AnalysisConfig = {
        conf: prefs.conf,
        iou: prefs.iou,
        imgsz: prefs.imgsz,
        track_buffer: prefs.trackBuffer,
        debug_mode: prefs.debugMode,
        frame_stride: prefs.frameStride,
        target_fps: prefs.targetFps,
        max_frames: Math.min(
          prefs.maxFrames,
          Math.max(1, Math.ceil((advancedVideo.duration_sec ?? maxDuration) * (advancedVideo.fps || 25))),
        ),
        stop_line: modules.red ? stopLinePoints : null,
        signal_timeline: modules.red
          ? createSignalTimeline(
              advancedVideo.duration_sec && advancedVideo.duration_sec > 0
                ? advancedVideo.duration_sec
                : maxDuration,
            )
          : null,
        lanes: null,
        calibration: null,
        enable_red_light: modules.red,
        enable_wrong_lane: modules.lane,
        enable_overspeed: modules.speed,
        enable_helmet: modules.helmet,
        save_annotated: modules.annotated,
        model_name: null,
      };
      void analyze(advancedVideo, config);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Analysis settings are invalid.");
    }
  }

  function setRedLightEnabled(checked: boolean) {
    if (checked) {
      setStopLine("");
      setStopLineDrawn(false);
      setStopLineSuggestion(null);
      setDrawingStopLine(false);
      setSuggestingStopLine(true);
      setLineDrawingStart(null);
      setLineDrawingEnd(null);
    } else {
      setSuggestingStopLine(false);
    }
    setModules((current) => ({ ...current, red: checked }));
  }

  function pointOnPreview(event: React.PointerEvent<HTMLDivElement>): [number, number] {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [
      Number(Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)).toFixed(4)),
      Number(Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)).toFixed(4)),
    ];
  }

  async function downloadJobReport(jobId: string, format: "csv" | "pdf") {
    if (!apiBase()) {
      toast.error("Analysis report downloads require a connection to the inference API.");
      return;
    }
    const path = format === "csv"
      ? `/api/reports/export/csv?kind=violations&job_id=${jobId}`
      : `/api/reports/export/pdf?job_id=${jobId}`;
    try {
      await downloadApi(path, `${jobId}-report.${format}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not download the analysis report.");
    }
  }

  const manualLine = parseStopLineEndpoints(stopLine);
  const previewLine = lineDrawingStart && lineDrawingEnd
    ? [lineDrawingStart, lineDrawingEnd]
    : manualLine ?? (stopLineSuggestion?.found ? stopLineSuggestion.line : null);
  const previewLineNormalized = previewLine?.map(([x, y]) => [
    x <= 1 ? x : x / (stopLineSuggestion?.preview_width ?? 1),
    y <= 1 ? y : y / (stopLineSuggestion?.preview_height ?? 1),
  ]);

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Library"
        title="Video management"
        description={`Upload MP4, AVI, MOV, or WebM up to ${maxMb} MB and ${maxDuration} seconds. All violation checks are enabled by default; configure camera-specific evidence for red-light, wrong-lane, and speed checks.`}
      />
      {data.source === "sample" && data.notice ? <SampleBanner text={data.notice} /> : null}
      <label
        className="glass block cursor-pointer rounded-3xl border-dashed px-6 py-10 text-center"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          void upload(Array.from(event.dataTransfer.files));
        }}
      >
        <input
          className="sr-only"
          type="file"
          accept="video/mp4,video/avi,video/quicktime,video/webm,.mp4,.avi,.mov,.webm"
          multiple
          onChange={(event) => {
            void upload(Array.from(event.target.files ?? []));
            event.target.value = "";
          }}
        />
        <span className="block text-lg font-medium">Drop surveillance clips here</span>
        <span className="mt-2 block text-sm text-muted">Multiple files are uploaded one after another. Progress appears below the drop zone.</span>
        {progress ? <span className="mt-3 block text-sm text-cyan-700">{progress}</span> : null}
      </label>
      {data.videos.length === 0 ? (
        <EmptyState title="No videos yet" body="Upload a traffic clip, then start analysis with one click." />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {data.videos.map((video) => {
            const latestJob = video.latest_job;
            const annotatedJob =
              latestJob?.status === "completed" && latestJob.annotated_ready ? latestJob : null;
            const previewSrc = mediaUrl(
              annotatedJob ? `/api/analysis/${annotatedJob.id}/annotated` : video.media_path,
            );
            const analysisActive = Boolean(
              latestJob && ["pending", "processing", "cancelling"].includes(latestJob.status),
            );
            return (
              <article key={video.id} className="glass rounded-2xl p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="font-medium">{video.original_name}</h2>
                    <p className="mt-1 text-xs text-muted">
                      {formatBytes(video.size_bytes)} · {formatClock(video.duration_sec)} · {video.width ?? "?"}×{video.height ?? "?"} · {video.fps ?? "?"} fps
                    </p>
                  </div>
                  <Badge tone={tones[video.status]}>{video.sample ? `Sample ${video.status}` : video.status}</Badge>
                </div>
                <div className="mt-4 overflow-hidden rounded-xl bg-slate-100">
                  {video.sample || !apiBase() ? (
                    <div className="flex h-40 items-center justify-center px-6 text-center text-sm text-slate-500">
                      Preview is available after a real file is stored on the inference API.
                    </div>
                  ) : (
                    <video className="h-40 w-full bg-black object-contain" src={previewSrc ?? undefined} controls preload="metadata" />
                  )}
                </div>
                {!video.sample && apiBase() ? (
                  <p className="mt-2 text-xs text-muted">
                    {annotatedJob
                      ? "YOLO boxes are included in this playback."
                      : latestJob?.status === "completed"
                        ? "Annotated playback was not saved. Open analysis to review detections."
                        : "Original video preview. Run analysis to generate YOLO boxes."}
                  </p>
                ) : null}
                <p className="mt-3 text-xs text-slate-500">Added {formatWhen(video.created_at)} · storage {video.storage}</p>
                {latestJob?.error ? <p className="mt-2 text-sm text-rose-700">{latestJob.error}</p> : null}
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button
                    disabled={analyzingVideo !== null || analysisActive}
                    onClick={() => void analyze(video)}
                  >
                    {analyzingVideo === video.id ? "Starting analysis" : "Analyze"}
                  </Button>
                  <Button variant="secondary" disabled={analyzingVideo !== null || analysisActive} onClick={() => configureAnalysis(video)}>
                    Advanced options
                  </Button>
                  {latestJob ? (
                    <Button variant="secondary" onClick={() => router.push(`/analysis/${latestJob.id}`)}>
                      {latestJob.status === "processing"
                        ? `View progress · ${Math.round(latestJob.progress * 100)}%`
                        : latestJob.status === "pending"
                          ? "View progress"
                          : latestJob.status === "cancelling"
                            ? "View cancellation progress"
                            : latestJob.status === "cancelled"
                              ? "Open cancelled analysis"
                              : "Open analysis"}
                    </Button>
                  ) : null}
                  {latestJob && ["pending", "processing", "cancelling"].includes(latestJob.status) ? (
                    <Button
                      variant="danger"
                      disabled={cancellingJob !== null || latestJob.status === "cancelling"}
                      onClick={() => void cancelAnalysis(video)}
                    >
                      {cancellingJob === latestJob.id || latestJob.status === "cancelling" ? "Cancelling…" : "Cancel analysis"}
                    </Button>
                  ) : null}
                  <Button
                    variant="danger"
                    disabled={analysisActive}
                    onClick={() => void remove(video)}
                  >
                    Delete
                  </Button>
                </div>
              </article>
            );
          })}
        </div>
      )}
      <section className="glass rounded-2xl p-5">
        <h2 className="text-xl font-semibold">Analysis history</h2>
        <p className="mt-1 text-sm text-muted">Open previous analyses or download their violation CSV and PDF reports.</p>
        <div className="mt-4 space-y-3">
          {data.history.length === 0 ? <p className="text-sm text-muted">No completed analyses yet.</p> : null}
          {data.history.map((item) => (
            <article key={item.job_id} className="rounded-xl border border-slate-200 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="font-medium">{item.video_name}</h3>
                  <p className="text-xs text-muted">Finished {formatWhen(item.finished_at)} · job {item.job_id}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant="ghost" onClick={() => router.push(`/analysis/${item.job_id}`)}>Open analysis</Button>
                  <Button size="sm" variant="secondary" disabled={data.source !== "live"} onClick={() => void downloadJobReport(item.job_id, "csv")}>CSV</Button>
                  <Button size="sm" disabled={data.source !== "live"} onClick={() => void downloadJobReport(item.job_id, "pdf")}>PDF</Button>
                </div>
              </div>
              <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-3">
                <div>Vehicles: {item.summary?.vehicles_detected ?? "—"}</div>
                <div>Potential violations: {item.summary?.violations_total ?? "—"}</div>
                <div>Model: {item.summary?.model?.name ?? "—"}</div>
              </dl>
            </article>
          ))}
        </div>
      </section>
      <Modal
        open={Boolean(advancedVideo)}
        onOpenChange={(open) => {
          if (!open) setAdvancedVideo(null);
        }}
        title="Advanced analysis options"
        description="The app automatically loads a frame and detects the stop line for red-light analysis."
      >
        <div className="grid items-stretch gap-4 md:grid-cols-2">
          <section className="glass space-y-4 rounded-2xl p-4">
            <div>
              <h3 className="text-sm font-semibold">Road &amp; signal setup</h3>
              <p className="mt-1 text-xs text-muted">The stop line is detected automatically when this dialog opens.</p>
            </div>
            <div className="space-y-2">
              <Field label="Stop line coordinates" hint="Detected coordinates are filled automatically.">
                <Input value={stopLine} disabled placeholder={suggestingStopLine ? "Detecting stop line…" : "No stop line detected"} />
              </Field>
              {stopLineSuggestion ? (
                <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-2">
                  <div
                    className={`relative touch-none ${drawingStopLine ? "cursor-crosshair" : ""}`}
                    onPointerDown={(event) => {
                      if (!drawingStopLine) return;
                      const point = pointOnPreview(event);
                      setLineDrawingStart(point);
                      setLineDrawingEnd(point);
                      event.currentTarget.setPointerCapture(event.pointerId);
                    }}
                    onPointerMove={(event) => {
                      if (drawingStopLine && lineDrawingStart) {
                        setLineDrawingEnd(pointOnPreview(event));
                      }
                    }}
                    onPointerUp={(event) => {
                      if (!drawingStopLine || !lineDrawingStart) return;
                      const end = pointOnPreview(event);
                      const distance = Math.hypot(
                        end[0] - lineDrawingStart[0],
                        end[1] - lineDrawingStart[1],
                      );
                      if (distance > 0.02) {
                        setLineDrawingEnd(end);
                        setStopLine(formatStopLine([lineDrawingStart, end]));
                        setStopLineDrawn(true);
                        setDrawingStopLine(false);
                        setLineDrawingStart(null);
                      } else {
                        setLineDrawingStart(null);
                        setLineDrawingEnd(null);
                      }
                    }}
                    onPointerCancel={() => {
                      setLineDrawingStart(null);
                      setLineDrawingEnd(null);
                    }}
                  >
                    <Image
                      src={`data:image/jpeg;base64,${stopLineSuggestion.preview_jpeg}`}
                      alt="Video frame with the automatically detected stop line"
                      width={stopLineSuggestion.preview_width}
                      height={stopLineSuggestion.preview_height}
                      unoptimized
                      className="block h-auto w-full rounded-lg"
                    />
                    {previewLineNormalized ? (
                      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
                        <line
                          x1={previewLineNormalized[0][0]}
                          y1={previewLineNormalized[0][1]}
                          x2={previewLineNormalized[1][0]}
                          y2={previewLineNormalized[1][1]}
                          stroke="#ef4444"
                          strokeWidth="0.005"
                          strokeDasharray="0.015 0.008"
                        />
                      </svg>
                    ) : null}
                  </div>
                  {suggestingStopLine ? (
                    <p className="text-xs text-muted">Detecting stop line…</p>
                  ) : stopLineDrawn ? (
                    <p className="text-xs text-emerald-700">Stop line detected automatically and applied.</p>
                  ) : stopLineSuggestion.found && stopLineSuggestion.line ? (
                    <p className="text-xs text-muted">A line was detected. Drag to replace it if needed.</p>
                  ) : (
                    <p className="text-xs text-amber-700">No reliable line was detected automatically. Draw one on the frame to continue.</p>
                  )}
                  {!suggestingStopLine ? (
                    <Button
                      variant={drawingStopLine ? "danger" : "secondary"}
                      onClick={() => {
                        setDrawingStopLine(!drawingStopLine);
                        setLineDrawingStart(null);
                        setLineDrawingEnd(null);
                      }}
                    >
                      {drawingStopLine ? "Cancel adjustment" : stopLineDrawn ? "Adjust stop line" : "Draw stop line"}
                    </Button>
                  ) : null}
                </div>
              ) : null}
              {stopLineDrawn && !drawingStopLine ? (
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="secondary"
                    onClick={() => {
                      setStopLine("");
                      setStopLineDrawn(false);
                      setLineDrawingStart(null);
                      setLineDrawingEnd(null);
                    }}
                  >
                    Clear line
                  </Button>
                </div>
              ) : null}
              {suggestingStopLine && !stopLineSuggestion ? (
                <p className="text-xs text-muted">Loading video frame and detecting stop line…</p>
              ) : null}
              {!suggestingStopLine && !stopLineSuggestion ? (
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-xs text-amber-700">The video frame could not be loaded automatically.</p>
                  <Button variant="secondary" onClick={() => setStopLineRetry((current) => current + 1)}>
                    Retry detection
                  </Button>
                </div>
              ) : null}
            </div>
          </section>
          <section className="glass space-y-4 rounded-2xl p-4">
            <div>
              <h3 className="text-sm font-semibold">Violation checks</h3>
              <p className="mt-1 text-xs text-muted">Choose which checks to include in this analysis.</p>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              <Toggle checked={modules.red} onCheckedChange={setRedLightEnabled} label="Red-light" />
              <Toggle checked={modules.lane} onCheckedChange={(checked) => setModules((current) => ({ ...current, lane: checked }))} label="Wrong-lane" />
              <Toggle checked={modules.speed} onCheckedChange={(checked) => setModules((current) => ({ ...current, speed: checked }))} label="Overspeed" />
              <Toggle checked={modules.helmet} onCheckedChange={(checked) => setModules((current) => ({ ...current, helmet: checked }))} label="Helmet" />
            </div>
            <p className="text-xs leading-5 text-muted">
              Wrong-lane checks need lane geometry configured by the system. Overspeed checks need validated camera calibration. Limits: cars &gt; 70 km/h; buses, trucks, and two-wheelers &gt; 60 km/h.
            </p>
          </section>
        </div>
        <section className="glass mt-4 flex items-center justify-between gap-4 rounded-2xl p-4">
          <div>
            <h3 className="text-sm font-semibold">Output</h3>
            <p className="mt-1 text-xs text-muted">Save a video with detections drawn on it for playback.</p>
          </div>
          <Toggle checked={modules.annotated} onCheckedChange={(checked) => setModules((current) => ({ ...current, annotated: checked }))} label="Save video" />
        </section>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="secondary" onClick={() => setAdvancedVideo(null)}>
            Cancel
          </Button>
          <Button disabled={analyzingVideo !== null} onClick={analyzeWithAdvancedSettings}>
            {analyzingVideo ? "Starting" : "Run with these options"}
          </Button>
        </div>
      </Modal>
    </div>
  );
}
