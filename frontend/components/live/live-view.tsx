"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { PageHeader } from "@/components/shell/app-shell";
import { Button } from "@/components/ui/button";
import { EmptyState, ErrorState } from "@/components/ui/states";
import { apiBase, apiSend, liveStreamUrl } from "@/lib/api";
import { classColor } from "@/lib/utils";

interface FrameDetection {
  class_name: string;
  confidence: number;
  bbox: number[];
}

export function LiveView() {
  const stream = liveStreamUrl();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rows, setRows] = useState<FrameDetection[] | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const media = videoRef.current;
    if (!media || !stream) return;
    media.src = stream;
  }, [stream]);

  if (!stream) {
    return (
      <EmptyState
        title="Live analysis is not configured"
        body="Set NEXT_PUBLIC_LIVE_STREAM_URL to a stream the browser can play. The navigation item stays hidden until that variable exists. This page does not invent live detections."
      />
    );
  }

  async function snapshot() {
    const media = videoRef.current;
    if (!media) return;
    if (!apiBase()) {
      toast.error("The inference API is not connected. The snapshot was not scored.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = media.videoWidth;
    canvas.height = media.videoHeight;
    const context = canvas.getContext("2d");
    if (!context || !canvas.width) {
      toast.error("The stream has no frame yet.");
      return;
    }
    const capturedAt = media.currentTime;
    setRows(null);
    const overlay = canvasRef.current;
    overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    try {
      context.drawImage(media, 0, 0);
    } catch {
      toast.error("The browser blocked frame capture. Use a CORS-enabled stream or upload a recording. No detections were created.");
      return;
    }
    setBusy(true);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) {
      setBusy(false);
      toast.error("Could not encode the frame.");
      return;
    }
    const body = new FormData();
    body.append("file", blob, "frame.jpg");
    try {
      const result = await apiSend<{ items: FrameDetection[] }>("/api/detect/frame", { method: "POST", body });
      if (media.currentTime - capturedAt > 0.5) {
        setRows(null);
        toast.message("The stream moved while YOLO was processing. Capture the current frame again.");
        return;
      }
      setRows(result.items);
      paint(result.items, media);
      if (result.items.length === 0) toast.message("YOLO returned no traffic objects on this frame.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Detection failed.");
    } finally {
      setBusy(false);
    }
  }

  function paint(items: FrameDetection[], media: HTMLVideoElement) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = media.clientWidth;
    canvas.height = media.clientHeight;
    const context = canvas.getContext("2d");
    if (!context || !media.videoWidth) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const sx = canvas.width / media.videoWidth;
    const sy = canvas.height / media.videoHeight;
    for (const item of items) {
      const [x1, y1, x2, y2] = item.bbox;
      context.strokeStyle = classColor(item.class_name);
      context.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      context.fillStyle = classColor(item.class_name);
      context.fillText(`${item.class_name} ${item.confidence.toFixed(2)}`, x1 * sx, Math.max(12, y1 * sy - 4));
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Stream"
        title="Live analysis"
        description="The player only previews the configured stream. Boxes appear after you capture a frame and the YOLO service answers."
        actions={
          <Button disabled={busy} onClick={() => void snapshot()}>
            {busy ? "Scoring frame" : "Detect current frame"}
          </Button>
        }
      />
      <div className="relative overflow-hidden rounded-2xl bg-black">
        <video ref={videoRef} className="max-h-[560px] w-full" controls muted playsInline />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      </div>
      {rows ? (
        <ul className="glass space-y-2 rounded-2xl p-4 text-sm">
          {rows.length === 0 ? <li>The model returned an empty list.</li> : null}
          {rows.map((row, index) => (
            <li key={`${row.class_name}-${index}`}>
              {row.class_name} · {row.confidence.toFixed(2)}
            </li>
          ))}
        </ul>
      ) : (
        <ErrorState title="No live scores yet" body="Press Detect current frame after the stream starts. If capture is blocked, the page stops instead of guessing." />
      )}
    </div>
  );
}
