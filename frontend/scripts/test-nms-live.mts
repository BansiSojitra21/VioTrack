import { boxesAtPlayhead, groupDetections, medianGap } from '../lib/playback-boxes.ts';

const jobId = '15e6dc84a23e4c33a516af893b314cd8';
const res = await fetch('http://localhost:8000/api/detections?job_id=' + jobId + '&from_sec=30.5&limit=5000');
const payload = await res.json();
const near = payload.items.filter((d) => d.timestamp_sec >= 30.8 && d.timestamp_sec <= 31.2);
console.log('raw near 31s', near.length);
const grouped = groupDetections(near);
const gap = medianGap(grouped.byTrack);
const boxes = boxesAtPlayhead(grouped.byTrack, grouped.untracked, 31.0, gap, 0.25);
console.log('visible', boxes.length, 'tracks', grouped.byTrack.size);
function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const aa = Math.max(0, a[2]-a[0]) * Math.max(0, a[3]-a[1]);
  const bb = Math.max(0, b[2]-b[0]) * Math.max(0, b[3]-b[1]);
  return inter / Math.max(aa + bb - inter, 1);
}
let overlaps = 0;
for (let i = 0; i < boxes.length; i++) {
  for (let j = i + 1; j < boxes.length; j++) {
    if (iou(boxes[i].bbox, boxes[j].bbox) >= 0.3) overlaps++;
  }
}
console.log('overlap_pairs', overlaps);
if (overlaps > 0) throw new Error('overlaps remain');
console.log('saved overlay clean at 31s');
