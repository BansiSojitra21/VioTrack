"use client";

import { useEffect, useState } from "react";
import { useTrafficData } from "@/components/providers/data-provider";
import { PageHeader } from "@/components/shell/app-shell";
import { RangeField, Toggle } from "@/components/ui/controls";
import { ErrorState, Skeleton } from "@/components/ui/states";
import { apiBase } from "@/lib/api";
import { defaultPreferences, loadPreferences, savePreferences, type Preferences } from "@/lib/settings";

export function SettingsView() {
  const { status, data, error } = useTrafficData();
  const [prefs, setPrefs] = useState<Preferences>(defaultPreferences);

  useEffect(() => {
    const timer = window.setTimeout(() => setPrefs(loadPreferences()), 0);
    return () => window.clearTimeout(timer);
  }, []);

  if (status === "loading") return <Skeleton className="h-80" />;
  if (status === "error" || !data) return <ErrorState title="Settings unavailable" body={error || "Could not load settings."} />;

  function update(next: Preferences) {
    setPrefs(next);
    savePreferences(next);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Preferences"
        title="Settings"
        description="Configure your analysis thresholds and visual preferences. Changes are saved automatically."
      />
      <section className="grid gap-4 lg:grid-cols-2">
        <article className="glass space-y-5 rounded-2xl p-5">
          <h2 className="font-medium">Detection sensitivity</h2>
          <RangeField label="Confidence threshold" min={0.05} max={0.9} step={0.01} value={prefs.conf} onChange={(conf) => update({ ...prefs, conf })} />
          <p className="text-xs text-muted">Lower thresholds can detect more small or distant vehicles, but may add false positives.</p>
          <RangeField label="IoU threshold" min={0.1} max={0.9} step={0.01} value={prefs.iou} onChange={(iou) => update({ ...prefs, iou })} />
          <label className="block text-sm">
            YOLO image size
            <select
              className="mt-2 h-10 w-full rounded-xl border border-slate-200 bg-white px-3"
              value={prefs.imgsz}
              onChange={(event) => update({ ...prefs, imgsz: Number(event.target.value) })}
            >
              {[640, 960, 1280].map((size) => <option key={size} value={size}>{size} px</option>)}
            </select>
            <span className="text-xs text-muted">Larger inputs can help small vehicles, with higher memory use and slower inference.</span>
          </label>
          <label className="block text-sm">
            Analysis FPS
            <select
              className="mt-2 h-10 w-full rounded-xl border border-slate-200 bg-white px-3"
              value={prefs.targetFps ?? ""}
              onChange={(event) =>
                update({ ...prefs, targetFps: event.target.value ? Number(event.target.value) : null })
              }
            >
              <option value="">Use source FPS</option>
              {[10, 15, 20, 25, 30].map((fps) => (
                <option key={fps} value={fps}>{fps} FPS</option>
              ))}
            </select>
            <span className="text-xs text-muted">Samples the source evenly at the selected rate; this overrides frame stride.</span>
          </label>
          <label className="block text-sm">
            Frame stride
            <input
              className="mt-2 w-full"
              type="range"
              min={1}
              max={5}
              step={1}
              value={prefs.frameStride}
              onChange={(event) => update({ ...prefs, frameStride: Number(event.target.value) })}
            />
            <span className="text-xs text-muted">Process every {prefs.frameStride} frame(s). Tracking is smoothest at 1.</span>
          </label>
          <label className="block text-sm">
            Lost track buffer
            <input
              className="mt-2 w-full"
              type="range"
              min={10}
              max={120}
              step={5}
              value={prefs.trackBuffer}
              onChange={(event) => update({ ...prefs, trackBuffer: Number(event.target.value) })}
            />
            <span className="text-xs text-muted">Keep an ID available for {prefs.trackBuffer} analyzed frames during brief occlusions.</span>
          </label>
          <label className="block text-sm">
            Tracking algorithm
            <select
              className="mt-2 h-10 w-full rounded-xl border border-slate-200 bg-white px-3"
              value={prefs.trackerType ?? "bytetrack"}
              onChange={(event) =>
                update({ ...prefs, trackerType: event.target.value as "bytetrack" | "botsort" })
              }
            >
              <option value="bytetrack">ByteTrack (Fast, high-association multi-stage matching)</option>
              <option value="botsort">BoT-SORT (Motion-compensated tracking, robust for occlusions)</option>
            </select>
            <span className="text-xs text-muted">Ultralytics multi-object tracking backend for vehicle trajectory persistence.</span>
          </label>
          <label className="block text-sm">
            Max frames to analyze
            <input
              className="mt-2 h-10 w-full rounded-xl border border-slate-200 bg-white px-3"
              type="number"
              min={1}
              max={20000}
              value={prefs.maxFrames}
              onChange={(event) => update({ ...prefs, maxFrames: Number(event.target.value) })}
            />
          </label>
          <Toggle checked={prefs.saveAnnotated} onCheckedChange={(saveAnnotated) => update({ ...prefs, saveAnnotated })} label="Save annotated video for playback" />
          <Toggle checked={prefs.debugMode} onCheckedChange={(debugMode) => update({ ...prefs, debugMode })} label="Save debug frames for new analyses" />
          {prefs.debugMode ? <p className="text-xs text-muted">Stores the first 10 analyzed frames, their tracked output, and detection metadata under that job.</p> : null}
        </article>

        <article className="glass space-y-5 rounded-2xl p-5">
          <h2 className="font-medium">System & appearance</h2>
          <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-4 text-sm space-y-2">
            <div className="flex justify-between">
              <span className="text-muted">Connection:</span>
              <span className="font-medium text-emerald-700">{data.source === "live" ? "Live service connected" : "Demo preview mode"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted">API Endpoint:</span>
              <span className="font-mono text-xs">{apiBase() || "Local preview"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted">Violation Engine:</span>
              <span className="text-xs font-medium">Red-light, Wrong-lane, Speed, Helmet</span>
            </div>
          </div>

          <div>
            <p className="mb-2 text-sm font-medium">Theme Accent</p>
            <div className="flex gap-2">
              {(["violet", "cyan", "blue"] as const).map((accent) => (
                <button
                  key={accent}
                  onClick={() => update({ ...prefs, accent })}
                  className={`rounded-full px-4 py-1.5 text-sm font-medium capitalize transition ${
                    prefs.accent === accent
                      ? "bg-slate-900 text-white shadow-sm"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {accent}
                </button>
              ))}
            </div>
          </div>
        </article>
      </section>
    </div>
  );
}
