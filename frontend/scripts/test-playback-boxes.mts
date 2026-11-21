import assert from "node:assert/strict";
import {
  boxesAtPlayhead,
  boxesOverlapTooMuch,
  displayClassLabel,
  easeBBox,
  expandTwoWheelerBBox,
  groupDetections,
  isValidBBox,
  lerpBBox,
  medianGap,
  overlayLabel,
  preferVehicleBox,
  sanitizeConfidence,
  sanitizeDetection,
  sizesCompatible,
  stabilizeOverlay,
  translateBBox,
} from "../lib/playback-boxes.ts";
import type { DetectionRecord } from "../lib/types.ts";

function det(
  partial: Partial<DetectionRecord> &
    Pick<DetectionRecord, "id" | "timestamp_sec" | "bbox" | "tracking_id" | "confidence">,
): DetectionRecord {
  return {
    job_id: "job",
    video_id: "video",
    frame_number: Math.round(partial.timestamp_sec * 25),
    class_name: "car",
    class_id: 2,
    sample: false,
    plate_number: null,
    ...partial,
  };
}

assert.equal(sanitizeConfidence(Infinity), null);
assert.equal(sanitizeConfidence(NaN), null);
assert.equal(sanitizeConfidence(1.2), null);
assert.equal(sanitizeConfidence(0.87), 0.87);
assert.equal(isValidBBox([0, 0, 5000, 5000], 1920, 1080), false);
assert.equal(
  overlayLabel({ tracking_id: 2707, confidence: Infinity, plate_number: null, class_name: "car" }).primary,
  "Car #2707",
);
assert.ok(
  !overlayLabel({ tracking_id: 1, confidence: Infinity, plate_number: null, class_name: "car" }).primary.includes(
    "Infinity",
  ),
);

const sparse = [
  det({ id: "a", tracking_id: 7, timestamp_sec: 1.0, confidence: 0.9, bbox: [100, 100, 200, 200], plate_number: null }),
  det({ id: "b", tracking_id: 7, timestamp_sec: 3.0, confidence: 0.85, bbox: [300, 120, 400, 220], plate_number: "ABC123" }),
  det({ id: "c", tracking_id: 7, timestamp_sec: 5.0, confidence: 0.8, bbox: [500, 140, 600, 240], plate_number: null }),
];

const { byTrack, untracked } = groupDetections(sparse);
const gap = medianGap(byTrack);

const mid = boxesAtPlayhead(byTrack, untracked, 2.0, gap, 0.25);
assert.equal(mid.length, 1, "expected one box between sparse frames");
assert.equal(mid[0].tracking_id, 7);
assert.equal(mid[0].plate_number, "ABC123");

let missing = 0;
let lastCx = -1;
for (let t = 1.0; t <= 5.0; t += 0.05) {
  const boxes = boxesAtPlayhead(byTrack, untracked, t, gap, 0.25);
  if (boxes.length !== 1) missing += 1;
  else {
    const cx = (boxes[0].bbox[0] + boxes[0].bbox[2]) * 0.5;
    if (lastCx >= 0) {
      assert.ok(cx >= lastCx - 1e-3, `box must move forward with the vehicle at t=${t}`);
    }
    lastCx = cx;
  }
}
assert.equal(missing, 0, `boxes should stay continuous; missing at ${missing} timestamps`);
assert.ok(lastCx > 500, "box should travel with the track across the clip");

const clutter = groupDetections([
  det({ id: "t1", tracking_id: 3, timestamp_sec: 1.0, confidence: 0.95, bbox: [10, 10, 110, 110] }),
  det({ id: "u1", tracking_id: null, timestamp_sec: 1.0, confidence: 0.7, bbox: [20, 20, 100, 100] }),
  det({ id: "t1b", tracking_id: 3, timestamp_sec: 1.0, confidence: 0.6, bbox: [12, 12, 112, 112] }),
]);
const one = boxesAtPlayhead(clutter.byTrack, clutter.untracked, 1.0, 0.5, 0.25);
assert.equal(one.length, 1);

const stacked = groupDetections([
  det({ id: "a7", tracking_id: 7, timestamp_sec: 1.0, confidence: 0.63, bbox: [400, 300, 520, 420] }),
  det({ id: "a9", tracking_id: 9, timestamp_sec: 1.0, confidence: 0.57, bbox: [410, 310, 515, 415] }),
  det({ id: "ghost", tracking_id: 10, timestamp_sec: 1.0, confidence: 0.89, bbox: [50, 700, 400, 1000] }),
  det({ id: "ghost2", tracking_id: 11, timestamp_sec: 1.0, confidence: 0.62, bbox: [80, 720, 380, 980] }),
]);
const cleaned = boxesAtPlayhead(stacked.byTrack, stacked.untracked, 1.0, 0.5, 0.25);
assert.equal(cleaned.length, 2);

assert.equal(overlayLabel({ tracking_id: 12, confidence: 0.876, plate_number: "MH12AB1234", class_name: "car" }).plate, "MH12AB1234");
assert.equal(overlayLabel({ tracking_id: 1, confidence: 0.9, plate_number: "JM", class_name: "car" }).plate, null);
assert.equal(
  overlayLabel({ tracking_id: 1139, confidence: 0.72, plate_number: null, class_name: "motorcycle" }).primary,
  "Bike #1139  72%",
);
assert.equal(displayClassLabel("truck"), "Truck");

const shortBike = [100, 200, 160, 260];
const expandedBike = expandTwoWheelerBBox(shortBike, "motorcycle", 1920, 1080);
assert.ok(expandedBike[3] - expandedBike[1] > shortBike[3] - shortBike[1], "bike box must grow taller");
assert.ok(expandedBike[1] < shortBike[1], "bike box must grow upward for the rider");
assert.ok(expandedBike[3] - expandedBike[1] < 100, "bike pad must stay mild, not full-frame");
const carBox = [100, 200, 260, 320];
assert.deepEqual(expandTwoWheelerBBox(carBox, "car", 1920, 1080), carBox, "cars must stay unchanged");
const alreadyTall = expandTwoWheelerBBox([100, 100, 160, 220], "motorcycle", 1920, 1080);
assert.deepEqual(alreadyTall, [100, 100, 160, 220], "already-tall bike boxes must not keep growing");
// Giant bike boxes (previous bug) must be rejected so cars are not NMS-suppressed.
const giantBike = sanitizeDetection(
  det({
    id: "giant",
    tracking_id: 123,
    timestamp_sec: 12,
    confidence: 0.87,
    bbox: [50, 40, 700, 900],
    class_name: "motorcycle",
  }),
  1920,
  1080,
);
assert.equal(giantBike, null, "full-frame bike boxes must be dropped");
const carOk = sanitizeDetection(
  det({
    id: "car1",
    tracking_id: 9,
    timestamp_sec: 12,
    confidence: 0.9,
    bbox: [800, 500, 980, 680],
    class_name: "car",
  }),
  1920,
  1080,
);
assert.ok(carOk, "normal cars must still display");

// Hatchback double-fired as car + truck must keep the car label.
const carDet = det({
  id: "c537",
  tracking_id: 537,
  timestamp_sec: 33,
  confidence: 0.86,
  bbox: [1400, 420, 1700, 700],
  class_name: "car",
});
const truckDet = det({
  id: "t549",
  tracking_id: 549,
  timestamp_sec: 33,
  confidence: 0.65,
  bbox: [1380, 400, 1720, 720],
  class_name: "truck",
});
assert.equal(preferVehicleBox(truckDet, carDet).class_name, "car", "overlapping car+truck must prefer car");

const early = boxesAtPlayhead(byTrack, untracked, 0.0, gap, 0.25);
assert.equal(early.length, 0);

const afterLast = boxesAtPlayhead(byTrack, untracked, 5.3, gap, 0.25);
assert.equal(afterLast.length, 1);

const translated = translateBBox([500, 140, 600, 240], [500, 140, 600, 240], 4.8, 5.0, 5.3);
assert.ok(Math.abs(translated[2] - translated[0] - 100) < 1e-6);

// Truck approach: tiny far box must not instantly become a huge near box.
const far = [900, 500, 930, 530];
const near = [700, 400, 950, 620];
assert.equal(sizesCompatible(far, near), false);
const eased = easeBBox(far, near, 0.1, 0.5);
const [fw, fh] = [eased[2] - eased[0], eased[3] - eased[1]];
assert.ok(fw < 80, `size should grow gradually, got width ${fw}`);
assert.ok(fh < 80, `size should grow gradually, got height ${fh}`);

const midLerp = lerpBBox([100, 100, 200, 200], [300, 100, 400, 200], 0.5);
assert.ok(Math.abs((midLerp[0] + midLerp[2]) * 0.5 - 250) < 1e-6, "lerp center must track halfway");
const endLerp = lerpBBox([100, 100, 200, 200], [300, 140, 400, 240], 1);
assert.deepEqual(endLerp, [300, 140, 400, 240], "lerp at t=1 must reach the next detection");

assert.equal(boxesOverlapTooMuch([10, 10, 100, 100], [20, 20, 90, 90]), true);

// Nested window crop inside full car (#2562 inside #2648) must count as the same vehicle.
const carBody = [900, 420, 1180, 700];
const carWindow = [940, 440, 1040, 520];
assert.equal(boxesOverlapTooMuch(carBody, carWindow), true);

const nestedPair = groupDetections([
  det({
    id: "2648",
    tracking_id: 2648,
    timestamp_sec: 59.0,
    confidence: 0.91,
    bbox: carBody,
  }),
  det({
    id: "2562",
    tracking_id: 2562,
    timestamp_sec: 59.0,
    confidence: 0.66,
    bbox: carWindow,
    plate_number: "ACR597",
  }),
]);
const nestedAtPlayhead = boxesAtPlayhead(nestedPair.byTrack, nestedPair.untracked, 59.0, 0.5, 0.25);
assert.equal(nestedAtPlayhead.length, 1, "nested window+body must collapse to one box");
assert.equal(nestedAtPlayhead[0].plate_number, "ACR597");
assert.equal(nestedAtPlayhead[0].tracking_id, 2648, "keep the full-body track id");
assert.deepEqual(nestedAtPlayhead[0].bbox, carBody);

const nestedStable = stabilizeOverlay(
  new Map(),
  nestedAtPlayhead,
  59.0,
  { confirmHits: 1, minDisplayConf: 0.38, frameW: 1920, frameH: 1080 },
);
assert.equal(nestedStable.boxes.length, 1, "stabilizer must show one box for nested IDs");
assert.equal(nestedStable.boxes[0].plate_number, "ACR597");

assert.equal(sanitizeDetection(det({ id: "bad", tracking_id: 1, timestamp_sec: 1, confidence: Infinity, bbox: [1, 1, 50, 50] })), null);

let stable = new Map();
const ghost = stabilizeOverlay(
  stable,
  [det({ id: "g", tracking_id: 3, timestamp_sec: 1.0, confidence: 0.5, bbox: [10, 10, 80, 120] })],
  1.0,
  { confirmHits: 2, minDisplayConf: 0.38, persistSec: 0.4, frameW: 1920, frameH: 1080 },
);
assert.equal(ghost.boxes.length, 0);
stable = ghost.state;

const hit1 = stabilizeOverlay(
  stable,
  [det({ id: "s1", tracking_id: 4, timestamp_sec: 1.0, confidence: 0.9, bbox: [100, 100, 180, 160] })],
  1.0,
  { confirmHits: 2, minDisplayConf: 0.38, persistSec: 0.4, frameW: 1920, frameH: 1080 },
);
assert.equal(hit1.boxes.length, 0);
stable = hit1.state;

const hit2 = stabilizeOverlay(
  stable,
  [det({ id: "s2", tracking_id: 4, timestamp_sec: 1.1, confidence: 0.88, bbox: [120, 105, 200, 165] })],
  1.1,
  { confirmHits: 2, minDisplayConf: 0.38, persistSec: 0.4, frameW: 1920, frameH: 1080 },
);
assert.equal(hit2.boxes.length, 1);
stable = hit2.state;

const occluded = stabilizeOverlay(stable, [], 1.3, {
  confirmHits: 2,
  minDisplayConf: 0.38,
  persistSec: 0.55,
  frameW: 1920,
  frameH: 1080,
});
assert.equal(occluded.boxes.length, 1);
assert.ok(occluded.boxes[0].confidence >= 0.38);

// Gaps between detections must not blink the box off while the vehicle is still on screen.
let steady = new Map();
const gapTrack = [
  det({ id: "m1", tracking_id: 20, timestamp_sec: 2.0, confidence: 0.9, bbox: [200, 200, 300, 300] }),
  det({ id: "m2", tracking_id: 20, timestamp_sec: 2.5, confidence: 0.88, bbox: [240, 205, 340, 305] }),
];
for (const sample of gapTrack) {
  const step = stabilizeOverlay(steady, [sample], sample.timestamp_sec, {
    confirmHits: 1,
    minDisplayConf: 0.35,
    persistSec: 0.55,
    alpha: 0.8,
    frameW: 1920,
    frameH: 1080,
  });
  steady = step.state;
  assert.equal(step.boxes.length, 1, `expected steady box at t=${sample.timestamp_sec}`);
}
let blinks = 0;
for (let t = 2.0; t <= 3.0; t += 0.05) {
  const atGap = t > 2.5 && t < 2.5 + 0.5;
  const raw =
    t <= 2.5
      ? [gapTrack[t < 2.25 ? 0 : 1]]
      : [];
  const step = stabilizeOverlay(steady, raw, t, {
    confirmHits: 1,
    minDisplayConf: 0.35,
    persistSec: 0.55,
    alpha: 0.8,
    frameW: 1920,
    frameH: 1080,
  });
  steady = step.state;
  if (atGap && step.boxes.length !== 1) blinks += 1;
  if (!atGap && t <= 2.5 && step.boxes.length !== 1) blinks += 1;
}
assert.equal(blinks, 0, "box must stay visible across detector gaps (no blink)");

// Broken full-frame / Infinity detections must never become visible boxes.
const broken = stabilizeOverlay(
  new Map(),
  [
    det({ id: "inf", tracking_id: 2707, timestamp_sec: 2, confidence: Infinity, bbox: [0, 0, 1920, 1080] }),
    det({ id: "huge", tracking_id: 2654, timestamp_sec: 2, confidence: 0.9, bbox: [-100, -100, 3000, 3000] }),
  ],
  2,
  { confirmHits: 1, minDisplayConf: 0.38, frameW: 1920, frameH: 1080 },
);
assert.equal(broken.boxes.length, 0, "Infinity/full-frame boxes must be rejected");

console.log("playback-boxes tests passed");
