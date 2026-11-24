Place dataset notes, class lists, and split descriptions in this folder.

## Helmet detector

Train a YOLO helmet model and point `HELMET_WEIGHTS` at the exported weights
(for example `helmet.pt` under `app/weights/`).

Public dataset used by `scripts/train_helmet.py`:

- Kaggle: [mobeenfatimah/helmet-detection-dataset-yolo-format](https://www.kaggle.com/datasets/mobeenfatimah/helmet-detection-dataset-yolo-format)
- Classes: `without_helmet` (0), `with_helmet` (1)
- Config: `app/datasets/helmet/data.yaml`

```bash
# from backend/
.venv\Scripts\python.exe scripts\train_helmet.py
```

That copies `best.pt` to `app/weights/helmet.pt`. Set `HELMET_WEIGHTS=helmet.pt`
in `.env` (already the recommended default in `.env.example`). Other violation
modules are unchanged.
