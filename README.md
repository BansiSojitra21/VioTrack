# TrafficAI

TrafficAI is a traffic-monitoring workspace. The Next.js app is the dashboard. A separate FastAPI service runs YOLO, tracks vehicles, and flags **potential** violations for a person to review. It does not issue fines, and it does not invent detections when the model or a required rule input is missing.

The browser app can deploy on Vercel. Video files and model inference stay on the Python service (Render, Railway, or any host that can run Docker or a long-lived Python process).

## What you can do

- Review an analytics dashboard, violation review table, traffic charts, and reports.
- Upload MP4, AVI, MOV, or WebM (default limits: 200 MB and 180 seconds).
- Run YOLO11 (or another Ultralytics `.pt`) on a stored video.
- Start analysis with violation checks enabled by default. Red-light checks need a confirmed stop line and actual signal timing, wrong-lane checks need a lane polygon and travel direction, and overspeed checks need validated camera calibration.
- Track cars, motorcycles, buses, trucks, bicycles, and pedestrians when the weights include those COCO classes.
- Attempt plate recognition for every tracked vehicle on each processed frame. The highest-confidence readable plate is stored on that vehicle's detections and any associated violation. Recognition runs locally; ALPR models download on first use, unreadable plates are left blank, and frame stride or a selected target FPS samples frames for detection and plate recognition.
- Recognized plate numbers are stored on vehicle detections and on matching violation evidence when the associated vehicle is identified.
- Flag red-light crossings, wrong-lane travel, and overspeeding only when the matching evidence is configured.
- Attach custom helmet weights. Official COCO weights do not contain helmet classes.
- Export CSV and PDF reports from real jobs.
- Open `/docs` on the API for the interactive reference.

Demo mode is the default when `NEXT_PUBLIC_API_URL` is empty. Every sample figure is labeled. Uploading or starting analysis in that mode stops with an error instead of pretending a model ran.

## Repository layout

```text
frontend/     Next.js App Router, TypeScript, Tailwind CSS, shadcn-style UI
backend/      FastAPI, Ultralytics YOLO + persistent ByteTrack, OpenCV
backend/app/weights/     put .pt files and optional evaluation.json here
```

UI primitives follow the shadcn/ui pattern: Radix, `class-variance-authority`, and Tailwind. `frontend/components.json` records that setup.

## Rules the detector will and will not apply

| Module | Runs when | Otherwise |
| --- | --- | --- |
| Vehicle detection | `YOLO_WEIGHTS` points at a `.pt` file on disk, or `ALLOW_WEIGHT_DOWNLOAD=true` for an official Ultralytics name | Job status `failed` with the load error. No boxes are stored. |
| Red light | Enabled by default; confirm a suggested stop line (or provide one) and supply the actual signal timeline, including `red` intervals | Status `inactive` until configured. Motion alone is not a red-light violation. |
| Wrong lane | Enabled by default; a lane polygon and a travel direction in degrees are also required | Status `inactive` until configured. |
| Overspeed | Enabled by default; `calibration.validated` must be true and `meters_per_pixel` set. Cars are limited to 70 km/h; buses, trucks, and motorcycles to 60 km/h. | Speed is `unavailable` until calibrated. Pixel motion is not reported as km/h. |
| Helmet | `HELMET_WEIGHTS` is a model with a `no_helmet` class | Status `not_configured`. |

Stop lines and lane polygons may be pixels, or values from 0 to 1 which are scaled to the frame.
Stop-line auto-suggestions look for consistent wide road markings across sampled frames; review and confirm the preview. The suggestion does not identify traffic signals or infer when they are red, so red-light checks also need the actual signal timeline.

## Run it locally

You need Node.js 20+ and Python 3.11, 3.12, or 3.13. Commands below use `python` on Windows and `python3` on macOS and Linux. Use the interpreter you actually have.

### Windows

For the quickest start, double-click `start.bat` in the project folder (or run `.\start.bat` from PowerShell). It sets up missing dependencies on the first run, starts both services, and opens the dashboard at [http://localhost:3000](http://localhost:3000). Keep the two service windows open; close them to stop the app.

To start the services manually instead:

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

In a second terminal:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

Set `frontend/.env.local` to:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DEMO_MODE=false
```

Restart `npm run dev` after changing `NEXT_PUBLIC_*` values. Next.js reads them at build and dev-server start.

### macOS and Linux

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Use the same `frontend/.env.local` values as on Windows.

The API document is at [http://localhost:8000/docs](http://localhost:8000/docs). Health is `GET /api/health`.

## Model weights

Official COCO weights such as `yolo11n.pt` are not in git.

```bash
cd backend
# after the virtualenv and pip install -r requirements.txt
python -m app.scripts.download_weights yolo11n.pt
```

Or copy your own file to `backend/app/weights/your-model.pt` and set:

```text
YOLO_WEIGHTS=your-model.pt
HELMET_WEIGHTS=helmet.pt
DEVICE=auto
```

`DEVICE=auto` selects CUDA when available and otherwise uses CPU. `DEVICE=0` explicitly selects the first CUDA GPU. Inference defaults are confidence `0.25`, NMS IoU `0.50`, and image size `640`; Analysis Settings let you adjust confidence, IoU, image size, analysis FPS, frame stride, and the ByteTrack lost-track buffer. On the included overhead traffic footage, 640-pixel inference detected vehicles in the opening frame while 960-pixel inference returned none. Higher image sizes may help other views, but should be checked against the target camera because they can also reduce detections.

Each offline analysis owns one persistent Ultralytics ByteTrack session. The tracker is reset for every job, and reported IDs are scoped to that video/job. Playback overlays use stored detections with original frame indexes and source timestamps; the analysis page does not run a second, unsynchronized detector over the playing video.

Enable **Save debug frames for new analyses** in Settings to keep input and tracked-output images plus JSON metadata for the first 10 processed frames of each job. Open **View debug frames** from that analysis to inspect them. Debug artifacts are stored under the configured data directory and cleared if a job is cancelled.

`ALLOW_WEIGHT_DOWNLOAD=true` lets Ultralytics fetch a known official checkpoint the first time a job runs. Leave it false on a locked-down server and copy the file yourself.

To show real precision, recall, F1, mAP@50, and mAP@50-95, copy `backend/app/weights/evaluation.example.json` to `evaluation.json`, remove `example_only`, and replace the numbers with your validation output. Files marked `example_only` are ignored. If `evaluation.json` is absent, the model page says the scores are not reported and shows sample curves with an explicit label.

Helmet training is a separate dataset whose classes include `helmet` and `no_helmet`. Point `HELMET_WEIGHTS` at those weights. There is no fallback that guesses a helmet from a motorcycle box.

## Useful API routes

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/videos/upload` | Store a clip |
| GET | `/api/videos` | List clips and the latest job |
| GET | `/api/videos/{video_id}` | One clip |
| GET | `/api/videos/{video_id}/media` | File bytes for the player |
| DELETE | `/api/videos/{video_id}` | Delete the clip and its jobs |
| POST | `/api/analysis/start` | Queue a job |
| GET | `/api/analysis/{job_id}/status` | Progress |
| GET | `/api/analysis/{job_id}/results` | Detections and violations |
| GET | `/api/analysis/{job_id}/annotated` | Annotated MP4 when it was written |
| WS | `/api/analysis/{job_id}/ws` | Progress push; the UI also polls |
| POST | `/api/detect/frame` | One JPEG or PNG |
| GET | `/api/detections` | Paginated boxes |
| POST | `/api/detections/plates/reanalyze` | Re-read plates on stored completed videos |
| GET | `/api/detections/plates/reanalyze/{task_id}` | Plate reanalysis progress |
| GET | `/api/violations` | Review queue |
| POST | `/api/violations/{violation_id}/review` | `reviewed` or `false_positive` |
| GET | `/api/analytics/summary` | Dashboard counts |
| GET | `/api/analytics/traffic` | Chart series |
| GET | `/api/model/info` | Model card, no fake metrics |
| POST | `/api/model/probe` | Try to load weights |
| GET | `/api/reports/export/csv?kind=violations` | CSV |
| GET | `/api/reports/export/pdf` | PDF |

Analysis accepts JSON:

```json
{
  "video_id": "…",
  "config": {
    "conf": 0.25,
    "iou": 0.5,
    "imgsz": 960,
    "track_buffer": 30,
    "frame_stride": 1,
    "stop_line": [[0.08, 0.62], [0.92, 0.62]],
    "signal_timeline": [{ "start": 0, "end": 12, "state": "red" }],
    "lanes": [{ "id": "northbound", "polygon": [[0.5, 0.1], [0.8, 0.1], [0.8, 0.9], [0.5, 0.9]], "direction_deg": 90 }],
    "calibration": { "meters_per_pixel": 0.04, "validated": true }
  }
}
```

Jobs run on a background thread and report `pending`, `processing`, `completed`, or `failed`.

## Tests

From `backend` with the virtualenv active:

```bash
python -m pytest -q --basetemp .pytest-basetemp
```

The tests cover tracking association, red-light gating, wrong-lane heading, calibrated speed, helmet class checks, and a video job that must fail when weights are missing. A second job uses a stub detector to prove the red-light rule stores a real record and still refuses uncalibrated speed.

## Deploy the frontend on Vercel

1. Import the repository in Vercel.
2. Set the root directory to `frontend`.
3. Framework preset: Next.js. Build command `npm run build`. Output is the default Next.js output.
4. Environment variables:
   - `NEXT_PUBLIC_API_URL` = the public API origin, for example `https://trafficai-api.onrender.com`. Do not leave `localhost` in this value.
   - `NEXT_PUBLIC_DEMO_MODE` = `true` if the site should fall back to labeled sample data when the API is down. Set `false` to show an error instead.
   - `NEXT_PUBLIC_LIVE_STREAM_URL` = optional. The Live Analysis item appears only when this is set.
5. Redeploy after changing any `NEXT_PUBLIC_` variable.

Vercel does not store the uploaded videos and does not run PyTorch. Do not move inference into a Vercel serverless function.

There is no login, signup, or session.

## Deploy the API

### Render or Railway with Docker

`backend/Dockerfile` is the image. Set the Docker context to `backend`.

Environment:

```text
CORS_ORIGINS=https://your-app.vercel.app
YOLO_WEIGHTS=yolo11n.pt
ALLOW_WEIGHT_DOWNLOAD=true
DATA_DIR=/data
MAX_UPLOAD_MB=200
MAX_DURATION_SEC=180
```

Attach a persistent disk at `/data` if the host offers one. SQLite at `DATA_DIR/trafficai.sqlite` is the metadata store. Without a disk, uploads and jobs disappear on restart. That is the deployment-compatible alternative to a laptop folder: a mounted volume on the same API service. The app does not need a separate account database.

GPU hosts can set `DEVICE=0`; `DEVICE=auto` falls back to CPU. A small CPU instance can run `yolo11n.pt` on short clips; 960-pixel inference can be slow. Raise `FRAME_STRIDE`, lower `MAX_FRAMES`, or choose a smaller image size to cap work. Ultralytics ByteTrack requires the `lap` package, which is installed from `backend/requirements.txt`.

`backend/render.yaml` is a starting Render service definition if you keep the Docker context at `backend`.

### CORS

`CORS_ORIGINS` is a comma-separated list. Include the Vercel origin and `http://localhost:3000` while you develop. The browser calls the API directly, so the API must allow that origin.

### Cloudinary

Set `CLOUDINARY_URL` to a `cloudinary://key:secret@cloud_name` URL if you want a remote copy of an upload. The API still keeps a local file because OpenCV reads a filesystem path. If Cloudinary is unset, storage is local only.

## Live stream

Live Analysis is hidden until `NEXT_PUBLIC_LIVE_STREAM_URL` is set. The page plays that URL and can POST one captured frame to `/api/detect/frame`. If the stream does not allow canvas capture, the page reports that and does not draw boxes.

## Limits and operations

- Upload size: `MAX_UPLOAD_MB` (default 200).
- Duration: `MAX_DURATION_SEC` (default 180).
- Frames per job: `MAX_FRAMES` (default 2400) and `frame_stride`.
- Annotated videos are encoded as H.264 MP4 so they play in current browsers.
- Structured JSON logs go to stdout.

## Frontend scripts

```bash
cd frontend
npm run dev
npm run build
npm run lint
```

## Backend scripts

```bash
cd backend
uvicorn main:app --reload --port 8000
pytest
python -m app.scripts.download_weights yolo11n.pt
```
