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
import type { DetectionRecord, JobRecord, VideoRecord, ViolationRecord } from "@/lib/types";
import { formatClock, titleCase } from "@/lib/utils";

interface Results {
  job: JobRecord;
  detections: DetectionRecord[];
  violations: ViolationRecord[];
}

const DISPLAY_VEHICLE_CLASSES = new Set(["car", "motorcycle", "bus", "truck", "bicycle"]);

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

function detectionIoU(a: number[], b: number[]): number {
  const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const intersection = width * height;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(areaA + areaB - intersection, 1);
}

export function AnalysisWorkspace({ jobId }: { jobId: string }) {
  const { status, data, error } = useTrafficData();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const detectionsRef = useRef<DetectionRecord[]>([]);
  const previewedDuringAnalysisRef = useRef(false);
  const replayWithDetectionsStartedRef = useRef(false);
  const [job, setJob] = useState<JobRecord | null>(null);
  const [video, setVideo] = useState<VideoRecord | null>(null);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [detections, setDetections] = useState<DetectionRecord[]>([]);
  const [liveDetections, setLiveDetections] = useState<DetectionRecord[]>([]);
  const liveHistoryRef = useRef<DetectionRecord[]>([]);
  const nextLiveTrackIdRef = useRef(-1);
  const liveSessionIdRef = useRef<string | null>(null);
  const liveFrameNumberRef = useRef(0);
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
  const viewVideo =
    data?.source === "sample"
      ? sampleView?.video ?? null
      : video ??
        job?.video ??
        data?.videos.find((item) => item.id === viewJob?.video_id || item.latest_job?.id === jobId) ??
        null;

  useEffect(() => {
    setLiveDetections([]);
    liveHistoryRef.current = [];
    nextLiveTrackIdRef.current = -1;
    liveSessionIdRef.current = null;
    liveFrameNumberRef.current = 0;
    if (!isVideoPlaying || data?.source === "sample" || !apiBase()) return;
    let stopped = false;
    let inFlight = false;
    const media = videoRef.current;
    if (!media) return;
    const liveSessionId = liveSessionIdRef.current ?? globalThis.crypto?.randomUUID?.() ?? `playback-${jobId}-${Date.now()}`;
    liveSessionIdRef.current = liveSessionId;

    const inferPlaybackFrame = async () => {
      if (stopped || inFlight || media.paused || !media.videoWidth || !media.videoHeight) return;
      inFlight = true;
      const capturedAt = media.currentTime;
      const frameNumber = liveFrameNumberRef.current++;
      const sourceWidth = media.videoWidth;
      const sourceHeight = media.videoHeight;
      const scale = Math.min(1, 960 / sourceWidth);
      const frameCanvas = document.createElement("canvas");
      frameCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
      frameCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
      const context = frameCanvas.getContext("2d");
      if (!context) {
        inFlight = false;
        return;
      }
      try {
        context.drawImage(media, 0, 0, frameCanvas.width, frameCanvas.height);
        const blob = await new Promise<Blob | null>((resolve) =>
          frameCanvas.toBlob(resolve, "image/jpeg", 0.78),
        );
        if (!blob || stopped) return;
        const body = new FormData();
        body.append("file", blob, "playback-frame.jpg");
        body.append("conf", String(viewJob?.config.conf ?? 0.25));
        body.append("iou", String(viewJob?.config.iou ?? 0.5));
        body.append("session_id", liveSessionId);
        body.append("frame_number", String(frameNumber));
        body.append("timestamp_sec", String(capturedAt));
        const result = await apiSend<{
          items: {
            class_name: string;
            class_id: number;
            confidence: number;
            bbox: number[];
            tracking_id?: number | null;
            track_id?: number | null;
          }[];
        }>("/api/detect/frame", { method: "POST", body });
        if (stopped || media.paused) return;

        const previousByTrack = new Map<number, DetectionRecord>();
        for (const item of liveHistoryRef.current) {
          if (item.timestamp_sec > capturedAt || item.tracking_id == null) continue;
          const current = previousByTrack.get(item.tracking_id);
          if (!current || item.timestamp_sec > current.timestamp_sec) {
            previousByTrack.set(item.tracking_id, item);
          }
        }
        const previous = [...previousByTrack.values()];
        const matched = new Set<number>();
        const frameDetections: DetectionRecord[] = result.items.map((item, index) => {
          const bbox = item.bbox.map((coordinate) => coordinate / scale);
          let bestIndex = -1;
          let bestIou = 0.15;
          previous.forEach((candidate, candidateIndex) => {
            if (matched.has(candidateIndex) || candidate.class_name !== item.class_name) return;
            const overlap = detectionIoU(candidate.bbox, bbox);
            if (overlap > bestIou) {
              bestIou = overlap;
              bestIndex = candidateIndex;
            }
          });
          const trackingId = item.tracking_id ?? item.track_id ?? (bestIndex >= 0
            ? previous[bestIndex].tracking_id!
            : nextLiveTrackIdRef.current--);
          if (bestIndex >= 0) matched.add(bestIndex);
          return {
            id: `live-${jobId}-${capturedAt.toFixed(3)}-${index}`,
            job_id: jobId,
            video_id: viewJob?.video_id ?? "",
            frame_number: frameNumber,
            frame_index: frameNumber,
            timestamp_sec: capturedAt,
            timestamp: capturedAt,
            class_name: item.class_name,
            class_id: item.class_id,
            confidence: item.confidence,
            bbox,
            tracking_id: trackingId,
            track_id: trackingId,
            sample: false,
          };
        });
        const historyByTrack = new Map<number, DetectionRecord[]>();
        for (const item of [...liveHistoryRef.current, ...frameDetections]) {
          if (item.tracking_id == null) continue;
          const history = historyByTrack.get(item.tracking_id) ?? [];
          history.push(item);
          historyByTrack.set(item.tracking_id, history);
        }
        liveHistoryRef.current = [...historyByTrack.values()].flatMap((history) =>
          history
            .filter((item) => capturedAt - item.timestamp_sec <= 6)
            .sort((a, b) => a.timestamp_sec - b.timestamp_sec)
            .slice(-2),
        );
        setLiveDetections(liveHistoryRef.current);
      } catch {
        // Playback inference is best effort; background analysis keeps running.
      } finally {
        inFlight = false;
      }
    };

    void inferPlaybackFrame();
    const interval = window.setInterval(() => void inferPlaybackFrame(), 400);
    return () => {
      stopped = true;
      window.clearInterval(interval);
    };
  }, [data?.source, isVideoPlaying, jobId, viewJob?.config.conf, viewJob?.config.iou, viewJob?.video_id]);
  const viewDetections = useMemo(
    () => (data?.source === "sample" ? sampleView?.detections ?? [] : detections),
    [data?.source, detections, sampleView],
  );
  const overlayDetections = viewDetections;
  const viewViolations = useMemo(
    () => (data?.source === "sample" ? sampleView?.violations ?? [] : violations),
    [data?.source, sampleView, violations],
  );
  const detectionsByTrack = useMemo(() => {
    const ordered = [...overlayDetections].sort((a, b) => a.timestamp_sec - b.timestamp_sec);
    const byTrack = new Map<number, DetectionRecord[]>();
    const untracked: DetectionRecord[] = [];
    for (const detection of ordered) {
      if (!DISPLAY_VEHICLE_CLASSES.has(detection.class_name)) continue;
      if (detection.tracking_id == null) {
        untracked.push(detection);
        continue;
      }
      const track = byTrack.get(detection.tracking_id);
      if (track) track.push(detection);
      else byTrack.set(detection.tracking_id, [detection]);
    }
    return { byTrack, untracked };
  }, [overlayDetections]);

  // Compute actual median inter-detection gap from the data, not from configured FPS.
  // This makes box visibility work regardless of frame stride setting.
  const detectionGap = useMemo(() => {
    const gaps: number[] = [];
    for (const trackDets of detectionsByTrack.byTrack.values()) {
      for (let i = 1; i < trackDets.length; i++) {
        const g = trackDets[i].timestamp_sec - trackDets[i - 1].timestamp_sec;
        if (g > 0 && g < 10) gaps.push(g);
      }
    }
    if (gaps.length === 0) return 0.5; // safe default: 500ms
    gaps.sort((a, b) => a - b);
    return gaps[Math.floor(gaps.length / 2)]; // median gap
  }, [detectionsByTrack]);

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

  const visible = useMemo(() => {
    const liveByTrack = new Map<number, DetectionRecord[]>();
    for (const item of liveDetections) {
      if (item.tracking_id == null || item.timestamp_sec > effectiveNow) continue;
      const history = liveByTrack.get(item.tracking_id) ?? [];
      history.push(item);
      liveByTrack.set(item.tracking_id, history);
    }
    const liveNow = [...liveByTrack.values()]
      .map((history) => history.sort((a, b) => a.timestamp_sec - b.timestamp_sec))
      .filter((history) => effectiveNow - history[history.length - 1].timestamp_sec <= 3.5)
      .map((history) => {
          const item = history[history.length - 1];
          const previous = history.length > 1 ? history[history.length - 2] : undefined;
          if (!previous || item.timestamp_sec <= previous.timestamp_sec) return item;
          const seconds = item.timestamp_sec - previous.timestamp_sec;
          return {
            ...item,
            timestamp_sec: effectiveNow,
            bbox: item.bbox.map((coordinate, index) =>
              coordinate + ((coordinate - previous.bbox[index]) / seconds) * Math.min(2.5, Math.max(0, effectiveNow - item.timestamp_sec)),
            ),
          };
      })
      .filter((item) => item.confidence >= (viewJob?.config?.conf ?? 0.25));
    if (liveNow.length > 0) {
      return liveNow;
    }
    const { byTrack, untracked } = detectionsByTrack;
    const displayThreshold = viewJob?.config?.conf ?? 0.25;
    // Keep boxes tied to nearby analyzed timestamps, interpolating only across
    // short gaps in the same stored track.
    const snapWindow = Math.max(0.25, detectionGap * 0.5);
    const interpolationGap = Math.max(2.0, detectionGap * 3);
    const active: DetectionRecord[] = [];

    for (const trackDetections of byTrack.values()) {
      const trackSnapWindow = snapWindow;
      const trackInterpolationGap = interpolationGap;
      // Binary search: find insertion point for effectiveNow
      let lo = 0;
      let hi = trackDetections.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (trackDetections[mid].timestamp_sec < effectiveNow) lo = mid + 1;
        else hi = mid;
      }

      const before = lo > 0 ? trackDetections[lo - 1] : null;
      const after = lo < trackDetections.length ? trackDetections[lo] : null;

      // Case 1: interpolate between two nearby detections
      if (
        before &&
        after &&
        after.timestamp_sec - before.timestamp_sec <= trackInterpolationGap
      ) {
        const span = after.timestamp_sec - before.timestamp_sec;
        if (span === 0) {
          active.push(after);
          continue;
        }
        const t = Math.max(0, Math.min(1, (effectiveNow - before.timestamp_sec) / span));
        const interpolated = {
          ...before,
          timestamp_sec: effectiveNow,
          bbox: before.bbox.map(
            (coord, i) => coord + (after.bbox[i] - coord) * t,
          ),
        };
        if (interpolated.confidence >= displayThreshold) active.push(interpolated);
        continue;
      }

      // Snap to a nearby observation; never draw a live box beyond its last detected frame.
      const nearest =
        before && after
          ? effectiveNow - before.timestamp_sec < after.timestamp_sec - effectiveNow
            ? before
            : after
          : before ?? after;
      if (
        nearest &&
        nearest.confidence >= displayThreshold &&
        Math.abs(nearest.timestamp_sec - effectiveNow) <= trackSnapWindow
      ) {
        active.push(nearest);
      }
    }

    // Untracked: snap within window
    for (const detection of untracked) {
      if (
        detection.confidence >= displayThreshold &&
        Math.abs(detection.timestamp_sec - effectiveNow) <= snapWindow
      )
        active.push(detection);
    }

    return active;
  }, [detectionsByTrack, detectionGap, effectiveNow, liveDetections, viewJob?.config?.conf]);

  const tracks = detectionsByTrack.byTrack.size;

  const draw = useCallback(function draw() {
    const canvas = canvasRef.current;
    const media = videoRef.current;
    if (!canvas || !media) return;

    // Size the canvas to the video's display rect.
    // Fall back to offsetWidth if clientWidth isn't ready yet.
    const width = media.clientWidth || media.offsetWidth;
    const height = media.clientHeight || media.offsetHeight;
    if (!width || !height) {
      requestAnimationFrame(() => draw());
      return;
    }
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);

    // Scale factors: from raw video-pixel bbox coordinates → canvas display pixels.
    // Priority: live video intrinsic size > DB video metadata > canvas size (1:1 fallback).
    const vw = media.videoWidth || viewVideo?.width || width;
    const vh = media.videoHeight || viewVideo?.height || height;
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

    // ── Vehicle bounding boxes ─────────────────────────────────────────────────
    context.save();
    context.beginPath();
    context.rect(offsetX, offsetY, contentWidth, contentHeight);
    context.clip();
    const placedLabels: { x: number; y: number; width: number; height: number }[] = [];
    for (const detection of visible) {
      const [x1, y1, x2, y2] = detection.bbox;
      const boxX = offsetX + x1 * sx;
      const boxY = offsetY + y1 * sy;
      const boxWidth = (x2 - x1) * sx;
      const boxHeight = (y2 - y1) * sy;
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

      const label = [
        detection.class_name.toUpperCase(),
        detection.tracking_id != null
          ? detection.tracking_id < 0
            ? `LIVE #${Math.abs(detection.tracking_id)}`
            : `#${detection.tracking_id}`
          : "",
        isViolator ? "⚠ VIOLATION" : "",
        `CONF ${Math.round(detection.confidence * 100)}%`,
      ]
        .filter(Boolean)
        .join(" ");

      const labelFont = "bold 12px sans-serif";
      context.font = labelFont;
      const plate = detection.plate_number ? `PLATE: ${detection.plate_number}` : null;
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
              Playback runs independently. YOLO checks sampled playback frames with stable live track IDs; boxes follow vehicles and coast briefly through missed frames while background analysis continues. When analysis finishes, the video replays with the saved detection timeline.
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
                  ? "YOLO has not saved a vehicle detection near this timestamp yet. Boxes will fill in as background analysis reaches this point."
                  : "No vehicle detections are stored near this timestamp."}
              </li>
            ) : null}
            {visible.map((item) => {
              const violationAt = item.tracking_id == null ? undefined : redLightViolationTimes.get(item.tracking_id);
              const isViolator = violationAt != null && now >= violationAt;
              return (
                <li
                  key={item.id}
                  className={`rounded-xl border p-2.5 text-sm transition ${
                    isViolator ? "border-rose-400 bg-rose-50/70" : "border-slate-200 bg-white"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-semibold ${isViolator ? "text-rose-700" : "text-emerald-700"}`}>
                      {titleCase(item.class_name)}
                    </span>
                    {isViolator ? (
                      <Badge tone="rose">RED LIGHT VIOLATION</Badge>
                    ) : (
                      <Badge tone="green">COMPLIANT</Badge>
                    )}
                  </div>
                  <p className="mt-1 text-xs text-muted">
                    Track {item.tracking_id == null ? "—" : item.tracking_id < 0 ? "Live" : item.tracking_id} ·{" "}
                    {Math.round(item.confidence * 100)}% confidence · frame {item.frame_number}
                  </p>
                  <p className={`mt-1 font-mono text-xs font-semibold ${isViolator ? "text-rose-900" : "text-slate-800"}`}>
                    Plate: {item.plate_number || "Not recognized"}
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
