"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorState, SampleBanner, Skeleton } from "@/components/ui/states";
import { apiBase, apiSend, downloadApi, mediaUrl } from "@/lib/api";
import {
  DISPLAY_VEHICLE_CLASSES,
  boxesAtPlayhead,
  displayPlate,
  groupDetections,
  isValidBBox,
  medianGap,
  overlayLabel,
  sanitizeConfidence,
  stabilizeOverlay,
  type StableTrackState,
} from "@/lib/playback-boxes";
import type { DetectionRecord, JobRecord, VideoRecord, ViolationRecord } from "@/lib/types";
import { formatClock, titleCase } from "@/lib/utils";

interface Results {
  job: JobRecord;
  detections: DetectionRecord[];
  violations: ViolationRecord[];
}

function ccw(p1: [number, number], p2: [number, number], p3: [number, number]): boolean {
  return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0]);
}

function segmentsIntersect(
  a1: [number, number],
  a2: [number, number],
  b1: [number, number],
  b2: [number, number],
): boolean {
  return ccw(a1, b1, b2) !== ccw(a2, b1, b2) && ccw(a1, a2, b1) !== ccw(a1, a2, b2);
}

interface PlaybackDetection {
  class_name: string;
  class_id?: number;
  confidence: number;
  bbox: number[];
  tracking_id?: number | null;
  track_id?: number | null;
}

export function AnalysisWorkspace({ jobId }: { jobId: string }) {
  const { status, data, error } = useTrafficData();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionsRef = useRef<DetectionRecord[]>([]);
  const previewedDuringAnalysisRef = useRef(false);
  const replayWithDetectionsStartedRef = useRef(false);
  const liveTracksRef = useRef<Map<number, DetectionRecord[]>>(new Map());
  const stableTracksRef = useRef<Map<number, StableTrackState>>(new Map());
  const savedOverlayLatchRef = useRef(false);
  const captureBusyRef = useRef(false);
  const lastCaptureAtRef = useRef(-1);
  const sessionGenRef = useRef(0);
  const jobStatusRef = useRef<JobRecord["status"] | undefined>(undefined);
  const captureNoticeRef = useRef(false);
  const [liveDetections, setLiveDetections] = useState<DetectionRecord[]>([]);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [video, setVideo] = useState<VideoRecord | null>(null);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [detections, setDetections] = useState<DetectionRecord[]>([]);
  const [playbackMode, setPlaybackMode] = useState<"overlay" | "annotated">("overlay");
  const [violations, setViolations] = useState<ViolationRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [now, setNow] = useState(0);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const replayWithDetections = useCallback(() => {
    const media = videoRef.current;
    if (!media || replayWithDetectionsStartedRef.current) return;
    replayWithDetectionsStartedRef.current = true;
    media.currentTime = 0;
    setNow(0);
    void media.play().catch((playError: unknown) => {
      if (playError instanceof DOMException && playError.name === "AbortError") return;
      replayWithDetectionsStartedRef.current = false;
      toast.error("Analysis is complete. Press Play to review the video with detections.");
    });
  }, []);
  useEffect(() => {
    previewedDuringAnalysisRef.current = false;
    replayWithDetectionsStartedRef.current = false;
    liveTracksRef.current = new Map();
    stableTracksRef.current = new Map();
    savedOverlayLatchRef.current = false;
    captureBusyRef.current = false;
    lastCaptureAtRef.current = -1;
    sessionGenRef.current += 1;
    captureNoticeRef.current = false;
    setLiveDetections([]);
  }, [jobId]);
  const sampleView = useMemo(() => {
    if (!data || data.source !== "sample") return null;
    const match = data.videos.map((item) => item.latest_job).find((item) => item?.id === jobId) ?? null;
    const source = data.videos.find((item) => item.latest_job?.id === jobId) ?? null;
    return {
      job: match,
      video: source,
      detections: data.detections.filter((item) => item.job_id === jobId),
      violations: data.violations.filter((item) => item.job_id === jobId),
    };
  }, [data, jobId]);

  useEffect(() => {
    if (status !== "ready" || !data || data.source === "sample") return;
    let stop = false;
    const tick = async () => {
      try {
        const next = await apiSend<JobRecord>(`/api/analysis/${jobId}/status`);
        if (stop) return;
        setJob(next);

        if (next.video && !video) {
          setVideo(next.video);
        } else if (!video && next.video_id) {
          const matched = data?.videos.find((v) => v.id === next.video_id);
          if (matched) {
            setVideo(matched);
          } else {
            try {
              const fetched = await apiSend<VideoRecord>(`/api/videos/${next.video_id}`);
              if (!stop && fetched) setVideo(fetched);
            } catch {
              // Non-critical if fetch fails
            }
          }
        }

        // Poll live detections and violations while processing so boxes render in real time.
        // Re-read a short overlap because OFFSET pagination is unstable when new rows
        // arrive at timestamps that already contain detections.
        if (next.status === "processing" || next.status === "pending") {
          try {
            const latestTimestamp = detectionsRef.current.reduce(
              (latest, item) => Math.max(latest, item.timestamp_sec),
              0,
            );
            const fromSec = Math.max(0, latestTimestamp - 1);
            const [detRes, vioRes] = await Promise.all([
              apiSend<{ items: DetectionRecord[] }>(
                `/api/detections?job_id=${jobId}&from_sec=${fromSec}&limit=20000&offset=0`,
              ),
              apiSend<{ items: ViolationRecord[] }>(`/api/violations?video_id=${next.video_id}`),
            ]);
            if (!stop) {
              if (detRes?.items && detRes.items.length > 0) {
                const merged = new Map(detectionsRef.current.map((item) => [item.id, item]));
                for (const item of detRes.items) merged.set(item.id, item);
                const updated = [...merged.values()].sort(
                  (a, b) => a.timestamp_sec - b.timestamp_sec || a.id.localeCompare(b.id),
                );
                detectionsRef.current = updated;
                setDetections(updated);
              }
              if (vioRes?.items) {
                setViolations(vioRes.items.filter((v) => v.job_id === jobId));
              }
            }
          } catch {
            // Ignore during processing
          }
        }

        if (next.status === "completed") {
          const results = await apiSend<Results>(`/api/analysis/${jobId}/results`);
          if (stop) return;
          const allDetections: DetectionRecord[] = [];
          for (let offset = 0; ; offset += 20000) {
            if (stop) return;
            const page = await apiSend<{ items: DetectionRecord[] }>(
              `/api/detections?job_id=${jobId}&limit=20000&offset=${offset}`,
            );
            if (stop) return;
            allDetections.push(...page.items);
            if (page.items.length < 20000) break;
          }
          detectionsRef.current = allDetections;
          setDetections(allDetections);
          setViolations(results.violations);
          return;
        }

        if (next.status === "failed") {
          try {
            const allDetections: DetectionRecord[] = [];
            for (let offset = 0; ; offset += 20000) {
              if (stop) return;
              const page = await apiSend<{ items: DetectionRecord[] }>(
                `/api/detections?job_id=${jobId}&limit=20000&offset=${offset}`,
              );
              if (stop) return;
              allDetections.push(...page.items);
              if (page.items.length < 20000) break;
            }
            detectionsRef.current = allDetections;
            setDetections(allDetections);
            const vioRes = await apiSend<{ items: ViolationRecord[] }>(
              `/api/violations?video_id=${next.video_id}`,
            );
            if (!stop) setViolations(vioRes.items.filter((item) => item.job_id === jobId));
          } catch (err) {
            if (!stop) {
              toast.error(
                err instanceof Error
                  ? `Could not load partial detections: ${err.message}`
                  : "Could not load partial detections.",
              );
            }
          }
          return;
        }
        window.setTimeout(() => void tick(), 1200);
      } catch (err) {
        if (!stop) setLoadError(err instanceof Error ? err.message : "Could not read the analysis job.");
      }
    };
    void tick();
    return () => {
      stop = true;
    };
  }, [data, jobId, status, video]);

  const viewJob = data?.source === "sample" ? sampleView?.job ?? null : job;
  jobStatusRef.current = viewJob?.status;
  const viewVideo =
    data?.source === "sample"
      ? sampleView?.video ?? null
      : video ??
        job?.video ??
        data?.videos.find((item) => item.id === viewJob?.video_id || item.latest_job?.id === jobId) ??
        null;

  const viewDetections = useMemo(
    () => (data?.source === "sample" ? sampleView?.detections ?? [] : detections),
    [data?.source, detections, sampleView],
  );
  const overlayDetections = viewDetections;
  const viewViolations = useMemo(
    () => (data?.source === "sample" ? sampleView?.violations ?? [] : violations),
    [data?.source, sampleView, violations],
  );
  const detectionsByTrack = useMemo(() => groupDetections(overlayDetections), [overlayDetections]);
  const detectionGap = useMemo(() => medianGap(detectionsByTrack.byTrack), [detectionsByTrack]);
  const liveGrouped = useMemo(() => groupDetections(liveDetections), [liveDetections]);
  const liveGap = useMemo(() => medianGap(liveGrouped.byTrack), [liveGrouped]);
  const storedUntil = useMemo(
    () => viewDetections.reduce((max, item) => Math.max(max, item.timestamp_sec), -1),
    [viewDetections],
  );

  const fps = viewVideo?.fps || 25;
  const stride = viewJob?.config?.frame_stride ?? 1;
  const sampleFps = Math.min(viewJob?.config?.target_fps ?? fps / stride, fps);
  const duration =
    viewVideo?.duration_sec ||
    videoDuration ||
    Math.max(0, ...viewDetections.map((item) => item.timestamp_sec));

  // Analysis reads frames sequentially, including frames where YOLO found nothing.
  // Use its processed-frame watermark so playback cannot outrun the detector.
  const processingThrough =
    viewJob?.status === "processing" || viewJob?.status === "pending" || viewJob?.status === "failed"
      ? Math.max(0, viewJob.frames_processed - 1) / sampleFps
      : duration;

  // Retain the violation start time so a vehicle changes to red at the
  // crossing, rather than being red throughout the earlier part of its track.
  const redLightViolationTimes = useMemo(() => {
    const violationTimes = new Map<number, number>();
    const markViolation = (trackId: number, time: number) => {
      const previous = violationTimes.get(trackId);
      if (previous == null || time < previous) violationTimes.set(trackId, time);
    };
    for (const v of viewViolations) {
      if (v.tracking_id != null) markViolation(v.tracking_id, v.timestamp_sec);
    }
    const stopLine = viewJob?.config?.stop_line;
    const timeline = viewJob?.config?.signal_timeline;
    if (stopLine && stopLine.length === 2 && timeline && overlayDetections.length > 0) {
      const vw = viewVideo?.width || 1;
      const vh = viewVideo?.height || 1;
      const sl1: [number, number] = [
        stopLine[0][0] <= 1 ? stopLine[0][0] * vw : stopLine[0][0],
        stopLine[0][1] <= 1 ? stopLine[0][1] * vh : stopLine[0][1],
      ];
      const sl2: [number, number] = [
        stopLine[1][0] <= 1 ? stopLine[1][0] * vw : stopLine[1][0],
        stopLine[1][1] <= 1 ? stopLine[1][1] * vh : stopLine[1][1],
      ];

      const tracksMap = new Map<number, { centroid: [number, number]; time: number }[]>();
      for (const d of overlayDetections) {
        if (d.tracking_id == null) continue;
        const c: [number, number] = [(d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2];
        if (!tracksMap.has(d.tracking_id)) tracksMap.set(d.tracking_id, []);
        tracksMap.get(d.tracking_id)!.push({ centroid: c, time: d.timestamp_sec });
      }

      for (const [tid, pts] of tracksMap.entries()) {
        pts.sort((a, b) => a.time - b.time);
        for (let i = 0; i < pts.length - 1; i++) {
          const p1 = pts[i];
          const p2 = pts[i + 1];
          let isRed = false;
          for (const [index, seg] of timeline.entries()) {
            const isLast = index === timeline.length - 1;
            if (
              seg.start <= p2.time &&
              (p2.time < seg.end || (isLast && p2.time === seg.end)) &&
              seg.state === "red"
            ) {
              isRed = true;
              break;
            }
          }
          if (isRed && segmentsIntersect(p1.centroid, p2.centroid, sl1, sl2)) {
            markViolation(tid, p2.time);
            break;
          }
        }
      }
    }
    return violationTimes;
  }, [overlayDetections, viewViolations, viewJob?.config, viewVideo]);

  const effectiveNow = now;
  const analysisActive =
    viewJob?.status === "pending" || viewJob?.status === "processing";
  // While analysis is running, always draw from live YOLO so boxes appear
  // immediately — do not wait for the worker to pass each playhead frame.
  // After the job finishes, use saved tracks (with hysteresis near the watermark).
  if (analysisActive) {
    savedOverlayLatchRef.current = false;
  } else if (
    data?.source === "sample" ||
    viewJob?.status === "completed" ||
    viewJob?.status === "cancelled" ||
    viewJob?.status === "failed"
  ) {
    savedOverlayLatchRef.current = true;
  } else if (storedUntil >= 0 && effectiveNow <= storedUntil + 0.25) {
    savedOverlayLatchRef.current = true;
  } else if (storedUntil < 0 || effectiveNow > storedUntil + 0.55) {
    savedOverlayLatchRef.current = false;
  }
  const useSavedTracks = savedOverlayLatchRef.current;
  const overlayModeRef = useRef(useSavedTracks);
  if (overlayModeRef.current !== useSavedTracks) {
    // Keep existing visible tracks during handoff so boxes do not blink off.
    overlayModeRef.current = useSavedTracks;
  }

  const visible = useMemo(() => {
    // Keep detection recall reasonably high; the stabilizer rejects one-frame FPs.
    const displayThreshold = useSavedTracks
      ? (viewJob?.config?.conf ?? 0.25)
      : Math.max(viewJob?.config?.conf ?? 0.25, 0.28);
    const liveRaw = boxesAtPlayhead(
      liveGrouped.byTrack,
      liveGrouped.untracked,
      effectiveNow,
      Math.max(liveGap, 0.12),
      displayThreshold,
    );
    const savedRaw = boxesAtPlayhead(
      detectionsByTrack.byTrack,
      detectionsByTrack.untracked,
      effectiveNow,
      detectionGap,
      displayThreshold,
    );
    // During analysis prefer live; if a live round is still in flight, fall back to
    // any saved rows that already arrived so the overlay is never blank.
    const raw = useSavedTracks ? savedRaw : liveRaw.length > 0 ? liveRaw : savedRaw;
    const stabilized = stabilizeOverlay(stableTracksRef.current, raw, effectiveNow, {
      persistSec: useSavedTracks ? 0.55 : 0.7,
      alpha: 0.8,
      // Show the first live hit immediately (was 2 → boxes lagged until a 2nd frame).
      confirmHits: 1,
      minDisplayConf: 0.35,
      frameW: viewVideo?.width ?? 0,
      frameH: viewVideo?.height ?? 0,
    });
    stableTracksRef.current = stabilized.state;
    return stabilized.boxes;
  }, [
    detectionGap,
    detectionsByTrack,
    effectiveNow,
    liveGap,
    liveGrouped,
    useSavedTracks,
    viewJob?.config?.conf,
    viewVideo?.height,
    viewVideo?.width,
  ]);

  const tracks = useSavedTracks
    ? detectionsByTrack.byTrack.size
    : Math.max(liveGrouped.byTrack.size, detectionsByTrack.byTrack.size);

  useEffect(() => {
    const active = viewJob?.status === "pending" || viewJob?.status === "processing";
    if (!active || data?.source === "sample" || !apiBase()) return;

    let stopped = false;
    let timer = 0;
    const schedule = (delay: number) => {
      timer = window.setTimeout(() => {
        void capture();
      }, delay);
    };

    const capture = async () => {
      if (stopped) return;
      const videoEl = videoRef.current;
      const status = jobStatusRef.current;
      if (!videoEl || videoEl.ended) {
        if (!stopped) schedule(250);
        return;
      }
      if (status !== "pending" && status !== "processing") return;
      const time = videoEl.currentTime;
      // Always score live frames while the job is running so the overlay does not
      // wait for the background worker to process the current playhead.
      const coveredUntil = detectionsRef.current.reduce((max, item) => Math.max(max, item.timestamp_sec), -1);
      const needsLive =
        status === "pending" ||
        status === "processing" ||
        coveredUntil < 0 ||
        time > coveredUntil + 0.2;
      if (!needsLive) {
        schedule(isVideoPlaying ? 200 : 400);
        return;
      }
      const minInterval = videoEl.paused ? 0.12 : 0.04;
      if (
        captureBusyRef.current ||
        (lastCaptureAtRef.current >= 0 && Math.abs(time - lastCaptureAtRef.current) < minInterval)
      ) {
        schedule(videoEl.paused ? 180 : 40);
        return;
      }
      if (!videoEl.videoWidth || !videoEl.videoHeight) {
        schedule(200);
        return;
      }
      if (lastCaptureAtRef.current >= 0 && Math.abs(time - lastCaptureAtRef.current) > 1.8) {
        sessionGenRef.current += 1;
        liveTracksRef.current = new Map();
        // Keep stable overlay state so boxes do not vanish on seek recovery.
      }

      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      try {
        context.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      } catch {
        if (!captureNoticeRef.current) {
          captureNoticeRef.current = true;
          toast.error("The browser blocked frame capture, so boxes ahead of the saved analysis cannot be drawn.");
        }
        return;
      }
      captureBusyRef.current = true;
      lastCaptureAtRef.current = time;
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.72));
      if (!blob || stopped) {
        captureBusyRef.current = false;
        if (!stopped) schedule(200);
        return;
      }
      const body = new FormData();
      body.append("file", blob, "frame.jpg");
      // Lower model conf for recall; frontend confirmation + minDisplayConf remove ghosts.
      body.append("conf", String(Math.max(viewJob?.config?.conf ?? 0.25, 0.25)));
      body.append("iou", String(viewJob?.config?.iou ?? 0.5));
      body.append("session_id", `playback-${jobId}-${sessionGenRef.current}`);
      const mediaFps = viewVideo?.fps || 25;
      body.append("frame_number", String(Math.max(0, Math.round(time * mediaFps))));
      body.append("timestamp_sec", String(time));
      try {
        const result = await apiSend<{ items: PlaybackDetection[] }>("/api/detect/frame", { method: "POST", body });
        if (stopped) return;
        const playhead = videoRef.current?.currentTime ?? time;
        if (Math.abs(playhead - time) > 0.8) return;
        if (jobStatusRef.current !== "pending" && jobStatusRef.current !== "processing") return;
        const previous = liveTracksRef.current;
        const nextMap = new Map<number, DetectionRecord[]>();
        for (const item of result.items) {
          const trackId = item.tracking_id ?? item.track_id;
          if (trackId == null || !DISPLAY_VEHICLE_CLASSES.has(item.class_name)) continue;
          const confidence = sanitizeConfidence(item.confidence);
          if (confidence == null || confidence < 0.25) continue;
          if (!isValidBBox(item.bbox)) continue;
          const record: DetectionRecord = {
            id: `live-${sessionGenRef.current}-${trackId}-${time.toFixed(3)}`,
            job_id: jobId,
            video_id: viewJob?.video_id ?? "",
            frame_number: Math.max(0, Math.round(time * mediaFps)),
            timestamp_sec: time,
            class_name: item.class_name,
            class_id: item.class_id ?? 0,
            confidence,
            bbox: item.bbox,
            tracking_id: -Math.abs(trackId),
            plate_number: null,
            sample: false,
          };
          const history = previous.get(trackId) ?? [];
          const last = history[history.length - 1];
          const points =
            last && Math.abs(last.timestamp_sec - time) < 0.05
              ? [...history.slice(0, -1), record]
              : [...history, record];
          nextMap.set(trackId, points.slice(-8));
        }
        // Keep missing tracks for ~0.7s so boxes stay continuous between live API rounds.
        for (const [trackId, points] of previous.entries()) {
          if (nextMap.has(trackId)) continue;
          const last = points[points.length - 1];
          if (last && time - last.timestamp_sec <= 0.7) nextMap.set(trackId, points.slice(-8));
        }
        liveTracksRef.current = nextMap;
        setLiveDetections([...nextMap.values()].flat());
      } catch {
        if (!captureNoticeRef.current) {
          captureNoticeRef.current = true;
          toast.error("Live boxes could not be scored. Saved boxes still appear as the background job reaches each moment.");
        }
      } finally {
        captureBusyRef.current = false;
        if (!stopped) schedule(videoEl.paused ? 120 : 30);
      }
    };

    // Fire immediately so the first paint after opening the workspace gets boxes.
    schedule(0);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [
    data?.source,
    isVideoPlaying,
    jobId,
    viewJob?.config?.conf,
    viewJob?.config?.iou,
    viewJob?.status,
    viewJob?.video_id,
    viewVideo?.fps,
  ]);

  const draw = useCallback(function draw() {
    const canvas = canvasRef.current;
    const media = videoRef.current;
    if (!canvas || !media) return;

    const width = media.clientWidth || media.offsetWidth;
    const height = media.clientHeight || media.offsetHeight;
    if (!width || !height) {
      requestAnimationFrame(() => draw());
      return;
    }
    // Only resize when needed — resetting width/height every frame causes flicker.
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);

    if (playbackMode === "annotated") {
      return;
    }

    const vw = media.videoWidth || viewVideo?.width;
    const vh = media.videoHeight || viewVideo?.height;
    if (!vw || !vh || vw <= 0 || vh <= 0) return;
    const objectFit = getComputedStyle(media).objectFit;
    const containScale = Math.min(width / vw, height / vh);
    const scale =
      objectFit === "cover"
        ? Math.max(width / vw, height / vh)
        : objectFit === "none"
          ? 1
          : objectFit === "scale-down"
            ? Math.min(1, containScale)
            : containScale;
    const sx = objectFit === "fill" ? width / vw : scale;
    const sy = objectFit === "fill" ? height / vh : scale;
    const contentWidth = vw * sx;
    const contentHeight = vh * sy;
    const offsetX = (width - contentWidth) / 2;
    const offsetY = (height - contentHeight) / 2;
    const contentRight = offsetX + contentWidth;
    const contentBottom = offsetY + contentHeight;

    const currentNow = effectiveNow;
    const job = viewJob;

    // ── Stop line ──────────────────────────────────────────────────────────────
    if (
      job?.config?.enable_red_light !== false &&
      job?.config?.stop_line &&
      job.config.stop_line.length === 2
    ) {
      const [[p1x, p1y], [p2x, p2y]] = job.config.stop_line;
      const rawX1 = p1x <= 1 ? p1x * vw : p1x;
      const rawY1 = p1y <= 1 ? p1y * vh : p1y;
      const rawX2 = p2x <= 1 ? p2x * vw : p2x;
      const rawY2 = p2y <= 1 ? p2y * vh : p2y;
      const lx1 = offsetX + rawX1 * sx;
      const ly1 = offsetY + rawY1 * sy;
      const lx2 = offsetX + rawX2 * sx;
      const ly2 = offsetY + rawY2 * sy;

      let signalState: "red" | "green" | "yellow" | null = null;
      if (job.config.signal_timeline) {
        for (const [index, seg] of job.config.signal_timeline.entries()) {
          const isLast = index === job.config.signal_timeline.length - 1;
          if (
            seg.start <= currentNow &&
            (currentNow < seg.end || (isLast && currentNow === seg.end))
          ) {
            signalState = seg.state;
            break;
          }
        }
      }
      const signalColor =
        signalState === "red"
          ? "#ef4444"
          : signalState === "green"
            ? "#10b981"
            : signalState === "yellow"
              ? "#f59e0b"
              : "#94a3b8";
      const signalLabel =
        signalState === "red"
          ? "STOP LINE (RED LIGHT)"
          : signalState === "green"
            ? "STOP LINE (GREEN)"
            : signalState === "yellow"
              ? "STOP LINE (YELLOW)"
              : "STOP LINE (SIGNAL NOT CONFIGURED)";

      context.save();
      context.beginPath();
      context.moveTo(lx1, ly1);
      context.lineTo(lx2, ly2);
      context.strokeStyle = signalColor;
      context.lineWidth = 4;
      context.setLineDash([8, 6]);
      context.stroke();
      context.fillStyle = signalColor;
      context.font = "bold 12px sans-serif";
      const labelX = Math.min(Math.max(lx1 + 10, offsetX), contentRight - 190);
      const labelY = ly1 - 8 >= offsetY + 12 ? ly1 - 8 : ly1 + 18;
      context.fillText(signalLabel, labelX, Math.min(labelY, contentBottom - 4));
      context.restore();
    }

    // ── Lane corridors (wrong-lane geometry) ───────────────────────────────────
    if (job?.config?.enable_wrong_lane !== false && job?.config?.lanes?.length) {
      context.save();
      context.setLineDash([6, 5]);
      context.lineWidth = 2;
      context.strokeStyle = "rgba(14, 165, 233, 0.75)";
      context.fillStyle = "rgba(14, 165, 233, 0.08)";
      for (const lane of job.config.lanes) {
        if (!lane.polygon || lane.polygon.length < 3) continue;
        context.beginPath();
        lane.polygon.forEach((point, index) => {
          const px = point[0] <= 1 ? point[0] * vw : point[0];
          const py = point[1] <= 1 ? point[1] * vh : point[1];
          const x = offsetX + px * sx;
          const y = offsetY + py * sy;
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.closePath();
        context.fill();
        context.stroke();
      }
      context.setLineDash([]);
      context.fillStyle = "rgba(14, 165, 233, 0.95)";
      context.font = "bold 11px sans-serif";
      context.fillText("LANES (wrong-lane check)", offsetX + 10, Math.min(offsetY + 16, contentBottom - 4));
      context.restore();
    }

    // ── Vehicle bounding boxes ─────────────────────────────────────────────────
    context.save();
    context.beginPath();
    context.rect(offsetX, offsetY, contentWidth, contentHeight);
    context.clip();
    const placedLabels: { x: number; y: number; width: number; height: number }[] = [];
    for (const detection of visible) {
      if (!isValidBBox(detection.bbox, vw, vh)) continue;
      if (sanitizeConfidence(detection.confidence) == null) continue;
      const [x1, y1, x2, y2] = detection.bbox;
      const boxX = offsetX + x1 * sx;
      const boxY = offsetY + y1 * sy;
      const boxWidth = (x2 - x1) * sx;
      const boxHeight = (y2 - y1) * sy;
      if (![boxX, boxY, boxWidth, boxHeight].every((value) => Number.isFinite(value))) continue;
      if (boxWidth < 4 || boxHeight < 4) continue;
      const violationAt = detection.tracking_id == null
        ? undefined
        : redLightViolationTimes.get(detection.tracking_id);
      const isViolator = violationAt != null && effectiveNow >= violationAt;

      // 🔴 Red = rule violator   🟢 Green = compliant
      const boxColor = isViolator ? "#ef4444" : "#22c55e";
      context.strokeStyle = boxColor;
      context.lineWidth = isViolator ? 4 : 3;
      context.shadowColor = "#0f172a";
      context.shadowBlur = 3;
      context.strokeRect(boxX, boxY, boxWidth, boxHeight);
      context.shadowBlur = 0;

      const { primary: label, plate } = overlayLabel(detection);

      const labelFont = "bold 12px sans-serif";
      context.font = labelFont;
      const textWidth = context.measureText(label).width;
      const plateFont = "bold 12px monospace";
      context.font = plateFont;
      const plateWidth = plate ? context.measureText(plate).width : 0;
      const labelWidth = Math.max(textWidth, plateWidth) + 10;
      const labelHeight = plate ? 38 : 18;
      const clampX = (x: number) => Math.min(Math.max(offsetX, x), Math.max(offsetX, contentRight - labelWidth));
      const clampY = (y: number) => Math.min(Math.max(offsetY, y), Math.max(offsetY, contentBottom - labelHeight));
      const candidates = [
        [clampX(boxX), clampY(boxY - labelHeight - 6)],
        [clampX(boxX), clampY(boxY + boxHeight + 6)],
        [clampX(boxX - labelWidth - 6), clampY(boxY)],
        [clampX(boxX + boxWidth + 6), clampY(boxY)],
      ];
      const labelPosition = candidates.find(([x, y]) =>
        placedLabels.every((placed) =>
          x + labelWidth + 3 <= placed.x ||
          placed.x + placed.width + 3 <= x ||
          y + labelHeight + 3 <= placed.y ||
          placed.y + placed.height + 3 <= y,
        ),
      ) ?? candidates[0];
      const [labelX, labelTop] = labelPosition;
      placedLabels.push({ x: labelX, y: labelTop, width: labelWidth, height: labelHeight });

      context.fillStyle = isViolator ? "#ef4444" : "#15803d";
      context.fillRect(labelX, labelTop, labelWidth, 18);
      context.fillStyle = "#ffffff";
      context.font = labelFont;
      context.fillText(label, labelX + 5, labelTop + 14);

      if (plate) {
        const plateTop = labelTop + 20;
        context.fillStyle = isViolator ? "#fee2e2" : "#dcfce7";
        context.fillRect(labelX, plateTop, labelWidth, 18);
        context.fillStyle = isViolator ? "#991b1b" : "#166534";
        context.font = plateFont;
        context.fillText(plate, labelX + 5, plateTop + 14);
      }
    }
    context.restore();
  }, [
    effectiveNow,
    redLightViolationTimes,
    viewJob,
    viewVideo,
    visible,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const media = videoRef.current;
    if (!media || !isVideoPlaying) return;

    // Prefer decoded-frame timing where available. Keep the overlay moving in
    // browsers without it by sampling the playhead on animation frames.
    if (!media.requestVideoFrameCallback) {
      let animationFrame = 0;
      const updatePlayhead = () => {
        setNow(media.currentTime);
        animationFrame = requestAnimationFrame(updatePlayhead);
      };
      animationFrame = requestAnimationFrame(updatePlayhead);
      return () => cancelAnimationFrame(animationFrame);
    }

    let callbackId = 0;
    const updatePlayhead = () => {
      setNow(media.currentTime);
      callbackId = media.requestVideoFrameCallback(updatePlayhead);
    };
    callbackId = media.requestVideoFrameCallback(updatePlayhead);
    return () => media.cancelVideoFrameCallback(callbackId);
  }, [isVideoPlaying]);

  // The first pass is a responsive preview. After it ends, replay once with the
  // complete synchronized detection timeline if background analysis has finished.
  useEffect(() => {
    const media = videoRef.current;
    if (
      media?.ended &&
      viewJob?.status === "completed" &&
      detections.length > 0 &&
      previewedDuringAnalysisRef.current
    ) {
      replayWithDetections();
    }
  }, [detections.length, replayWithDetections, viewJob?.status]);

  // Redraw whenever the video element is resized (initial layout, responsive breakpoints)
  useEffect(() => {
    const media = videoRef.current;
    if (!media) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(media);
    return () => ro.disconnect();
  }, [draw]);

  if (status === "loading") return <Skeleton className="h-96" />;
  if (status === "error" || !data) return <ErrorState title="Analysis unavailable" body={error || "Could not load analysis."} />;
  if (data.source === "sample" && sampleView && !sampleView.job) {
    return <ErrorState title="Job not found" body="That sample job is not in the demo library." />;
  }
  if (loadError) return <ErrorState title="Job not found" body={loadError} />;
  if (!viewJob) return <Skeleton className="h-96" />;

  const sample = data.source === "sample" || viewJob.sample;
  const mediaPath =
    viewVideo?.media_path ??
    (viewJob?.video_id ? `/api/videos/${viewJob.video_id}/media` : null);
  const src = sample ? null : mediaUrl(mediaPath);

  return (
    <div className="space-y-5">
      <PageHeader
        kicker="Workspace"
        title={viewVideo?.original_name ?? (viewJob?.video_id ? `Video ${viewJob.video_id.slice(0, 8)}` : "Analysis")}
        description="Review video playback with synchronized vehicle detections, tracked trajectories, and flagged violations."
        actions={
          <div className="flex gap-2">
            <Button variant="secondary" asChild>
              <Link href="/videos">Back to videos</Link>
            </Button>
            {viewJob.config.debug_mode && apiBase() ? (
              <Button variant="secondary" asChild>
                <Link
                  href={mediaUrl(`/api/analysis/${viewJob.id}/debug`) ?? "#"}
                  target="_blank"
                  rel="noreferrer"
                >
                  View debug frames
                </Link>
              </Button>
            ) : null}
            {viewJob.annotated_ready && apiBase() ? (
              <Button
                onClick={() =>
                  void downloadApi(`/api/analysis/${viewJob.id}/annotated`, `${viewJob.id}-annotated.mp4`).catch((err) =>
                    toast.error(err instanceof Error ? err.message : "Annotated video is unavailable."),
                  )
                }
              >
                Download annotated video
              </Button>
            ) : null}
          </div>
        }
      />
      {sample ? <SampleBanner text="This workspace is showing sample records. There is no source video and the boxes below were not predicted by YOLO." /> : null}
      {viewJob.error ? <ErrorState title="Analysis failed" body={viewJob.error} /> : null}
      <div className="grid gap-3 md:grid-cols-4">
        <Metric label="Tracked vehicles" value={String(tracks || viewJob.summary?.vehicles_detected || 0)} />
        <Metric label="Visible now" value={String(visible.length)} />
        <Metric label="Potential violations" value={String(viewViolations.length)} />
        <Metric label="Analyzed" value={`${viewJob.frames_processed}/${viewJob.frames_total || "—"} frames`} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="space-y-3">
          <div className="relative overflow-hidden rounded-2xl bg-black">
            {src ? (
              <video
                ref={videoRef}
                src={src}
                crossOrigin="anonymous"
                controls
                className="block max-h-[500px] w-full"
                preload="auto"
                onLoadedMetadata={(event) => {
                  const d = event.currentTarget.duration;
                  if (Number.isFinite(d) && d > 0) {
                    setVideoDuration(d);
                  }
                  draw();
                }}
                onTimeUpdate={(event) => {
                  const media = event.currentTarget;
                  setNow(media.currentTime);
                  draw();
                }}
                onSeeked={(event) => {
                  const time = event.currentTarget.currentTime;
                  if (Math.abs(time - now) > 0.45) {
                    stableTracksRef.current = new Map();
                  }
                  if (lastCaptureAtRef.current < 0 || Math.abs(time - lastCaptureAtRef.current) > 0.8) {
                    sessionGenRef.current += 1;
                    liveTracksRef.current = new Map();
                    setLiveDetections([]);
                    lastCaptureAtRef.current = -1;
                  }
                  setNow(time);
                }}
                onPlay={() => {
                  if (viewJob.status === "processing" || viewJob.status === "pending") {
                    previewedDuringAnalysisRef.current = true;
                  }
                  setIsVideoPlaying(true);
                }}
                onPause={() => setIsVideoPlaying(false)}
                onEnded={() => {
                  setIsVideoPlaying(false);
                  if (viewJob.status === "completed" && detections.length > 0) {
                    replayWithDetections();
                  }
                }}
              />
            ) : (
              <div className="flex h-80 items-center justify-center bg-slate-100 px-8 text-center text-sm text-slate-500">
                {sample ? "Demo mode has no video file to play." : "The stored video could not be previewed."}
              </div>
            )}
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
          </div>
          {viewJob.status === "processing" || viewJob.status === "pending" ? (
            <p className="px-1 text-xs text-muted">
              Boxes follow each vehicle continuously (track ID, confidence, plate). While analysis is still catching up, a live YOLO pass fills ahead of the saved timeline; afterward the overlay uses the stored tracks.
            </p>
          ) : viewJob.status === "failed" && viewJob.frames_processed < viewJob.frames_total ? (
            <p className="px-1 text-xs text-amber-700">
              Analysis stopped at {formatClock(processingThrough)} ({viewJob.frames_processed} of {viewJob.frames_total} frames). Boxes are unavailable after that point; restart the analysis to process the rest.
            </p>
          ) : null}
        </section>
        <aside className="glass max-h-[760px] overflow-auto rounded-2xl p-3">
          <h2 className="font-medium">
            Detections at {formatClock(visible.length > 0 ? visible[0].timestamp_sec : now)}
            {visible.length === 0 ? " — none visible" : ""}
          </h2>
          <ul className="mt-3 space-y-2">
            {visible.length === 0 ? (
              <li className="text-sm text-muted">
                {viewJob.status === "processing" || viewJob.status === "pending"
                  ? "Waiting for live boxes for this playhead…"
                  : "No tracked vehicles near this time."}
              </li>
            ) : null}
            {visible.map((item) => {
              const violationAt = item.tracking_id == null ? undefined : redLightViolationTimes.get(item.tracking_id);
              const isViolator = violationAt != null && now >= violationAt;
              const trackId =
                item.tracking_id == null
                  ? "—"
                  : item.tracking_id < 0
                    ? Math.abs(item.tracking_id)
                    : item.tracking_id;
              return (
                <li
                  key={item.id}
                  className={`rounded-xl border p-2.5 text-sm transition ${
                    isViolator ? "border-rose-400 bg-rose-50/70" : "border-slate-200 bg-white"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-semibold ${isViolator ? "text-rose-700" : "text-emerald-700"}`}>
                      {titleCase(item.class_name === "motorcycle" ? "bike" : item.class_name)} #{trackId} ·{" "}
                      {Math.round(item.confidence * 100)}%
                    </span>
                    {isViolator ? (
                      <Badge tone="rose">RED LIGHT</Badge>
                    ) : (
                      <Badge tone="green">OK</Badge>
                    )}
                  </div>
                  <p className={`mt-1 font-mono text-xs font-semibold ${isViolator ? "text-rose-900" : "text-slate-800"}`}>
                    Plate: {displayPlate(item.plate_number) || "—"}
                  </p>
                </li>
              );
            })}
          </ul>
          <h2 className="mt-6 font-medium">Potential violations ({viewViolations.length})</h2>
          <ul className="mt-3 space-y-2">
            {viewViolations.length === 0 ? (
              <li className="text-sm text-muted">
                {viewJob.status === "processing" || viewJob.status === "pending"
                  ? "Evaluating violations as frames process…"
                  : "No violations flagged for this job."}
              </li>
            ) : null}
            {viewViolations.map((item) => (
              <li key={item.id} className="rounded-xl border border-rose-200 bg-rose-50/40 p-2.5 text-sm">
                <p className="font-medium text-rose-900">{titleCase(item.violation_type)} · track {item.tracking_id ?? "—"}</p>
                <p className="mt-1 text-xs leading-5 text-slate-600">{item.evidence.explanation}</p>
                <p className="mt-1 text-xs text-muted">At {formatClock(item.timestamp_sec)} (frame {item.frame_number})</p>
              </li>
            ))}
          </ul>
          {viewJob.summary?.modules ? (
            <div className="mt-6 space-y-2">
              <h2 className="font-medium">Module status</h2>
              {viewJob.summary.modules.map((module) => (
                <p key={module.name} className="text-xs leading-5 text-slate-600">
                  <span className="text-cyan-700 font-medium">{titleCase(module.name)}</span> · {module.status}. {module.message}
                </p>
              ))}
            </div>
          ) : null}
          {viewJob.summary?.processing_seconds ? <p className="mt-4 text-xs text-muted">Processed in {viewJob.summary.processing_seconds}s.</p> : null}
        </aside>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-2xl p-4">
      <p className="text-xs uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-xl">{value}</p>
    </div>
  );
}
