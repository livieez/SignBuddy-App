# Sign Buddy

Live ASL fingerspelling recognition — all 26 letters, combining a static
model (A-Y, minus J/Z) with a motion model (J/Z) via three independent
services.

## Quick start

```bash
docker compose up --build
```

| Service      | URL                          |
|--------------|-------------------------------|
| Frontend     | http://localhost:5173         |
| Backend      | http://localhost:8000/health  |
| ML server    | http://localhost:9000/health  |

Stop with `Ctrl+C`, or `docker compose down` from another terminal.

## Architecture — how data flows

```
┌──────────┐   WebSocket    ┌────────────┐    HTTP POST    ┌────────────┐
│ frontend │ ─────────────► │ backend-ml │ ──────────────► │  ml-server │
│ (browser)│ ◄───────────── │ (FastAPI)  │ ◄────────────── │ (FastAPI)  │
└──────────┘  letter, conf  └────────────┘  letter, conf   └────────────┘
   camera,       no ML deps,                     holds the trained
  MediaPipe,     no model in                    models in memory
  smoothing        memory                    (static v2 + motion seq)
```

**`frontend`** — captures the webcam, runs MediaPipe hand-tracking *in the
browser*, and decides per-frame whether to send:
- `{"landmarks": [...]}` — one normalized frame, for static letters, or
- `{"sequence": [[...], ...]}` — a rolling window of raw frames, when its own
  motion gate (wrist speed) detects movement — for J/Z

Smoothing and sentence-building also happen here — the backend never
remembers anything between requests.

**`backend-ml`** — a thin, stateless relay. Receives whatever payload the
frontend sent over WebSocket, forwards it as-is to `ml-server` via HTTP,
and relays the response back. Has zero ML dependencies — it never inspects
*which* payload shape it's carrying.

**`ml-server`** — the only place real inference happens. Loads a static
model (v2: raw + geometric features) and a motion model (sequence CNN over
raw landmark windows) into memory at startup. Routes each `/predict` request
to whichever model matches the payload it received.

Full request/response contract and step-by-step trace:
[backend-ml-data-flow.md](backend-ml-data-flow.md).

## Project structure

```
ASL-app/
├── docker-compose.yml
├── frontend/       React + Vite. Camera, MediaPipe, WebSocket client.
├── backend-ml/      Stateless FastAPI relay. No ML dependencies.
└── ml-server/        Loads and runs the actual trained models.
    └── models/
        ├── v2_model/    static letters
        └── seq_model/   motion letters (J/Z)
```

## Configuration

| Variable | Set on | Values | Meaning |
|---|---|---|---|
| `MODEL_MODE` | `backend-ml` | `stub` \| `remote` | `stub` = fake predictions, no `ml-server` needed. `remote` = real predictions via `ml-server`. |
| `ML_SERVER_URL` | `backend-ml` | e.g. `http://ml-server:9000` | Where to reach `ml-server`. Only used when `MODEL_MODE=remote`. |
| `STATIC_MODEL_DIR` / `STATIC_TYPE` | `ml-server` | e.g. `models/v2_model` / `v2` | Which static model to load. |
| `SEQ_MODEL_DIR` | `ml-server` | e.g. `models/seq_model` | Which motion model to load. Omit to run static-only. |

Set in `docker-compose.yml`; change and `docker compose up --build <service>`
to apply.

## Running without Docker

<details>
<summary>ml-server</summary>

```bash
cd ml-server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
STATIC_MODEL_DIR=models/v2_model STATIC_TYPE=v2 SEQ_MODEL_DIR=models/seq_model \
  uvicorn app:app --port 9000
```
</details>

<details>
<summary>backend-ml</summary>

```bash
cd backend-ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
MODEL_MODE=remote ML_SERVER_URL=http://localhost:9000 \
  uvicorn app:app --port 8000
```
</details>

<details>
<summary>frontend</summary>

```bash
cd frontend
npm install
npm run dev
```
</details>
