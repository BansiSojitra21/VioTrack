/**
 * Continuity check against a completed job from the local API.
 */
import assert from "node:assert/strict";
import { boxesAtPlayhead, groupDetections, medianGap } from "../lib/playback-boxes.ts";
import type { DetectionRecord } from "../lib/types.ts";

const jobId = process.argv[2] ?? "0bef8a9e5f094ca9af65e4395598b0e2";
const base = process.env.API_URL ?? "http://localhost:8000";

const response = await fetch(`${base}/api/detections?job_id=${jobId}&limit=20000`);
assert.equal(response.ok, true, `API ${response.status}`);
const payload = (await response.json()) as { items: DetectionRecord[] };
assert.ok(payload.items.length > 0, "job has detections");

const grouped = groupDetections(payload.items);
const gap = medianGap(grouped.byTrack);
const maxTime = payload.items.reduce((max, item) => Math.max(max, item.timestamp_sec), 0);

let emptyFrames = 0;
let multiSameTrack = 0;
const samples = 240;
for (let i = 0; i <= samples; i += 1) {
  const time = (maxTime * i) / samples;
  const boxes = boxesAtPlayhead(grouped.byTrack, grouped.untracked, time, gap, 0.25);
  if (boxes.length === 0) emptyFrames += 1;
  const ids = boxes.map((box) => box.tracking_id).filter((id): id is number => id != null);
  if (new Set(ids).size !== ids.length) multiSameTrack += 1;
}

assert.equal(multiSameTrack, 0, "never more than one box per track id");
// A busy clip should almost never be empty once analysis has detections.
assert.ok(emptyFrames < samples * 0.05, `too many empty playheads: ${emptyFrames}/${samples + 1}`);

console.log(
  JSON.stringify(
    {
      detections: payload.items.length,
      tracks: grouped.byTrack.size,
      gap: Number(gap.toFixed(3)),
      emptyFrames,
      samples: samples + 1,
      ok: true,
    },
    null,
    2,
  ),
);
