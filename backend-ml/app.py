"""
app.py
-------
FastAPI backend for Sign Buddy. Stateless — no per-connection memory.

    MODEL_MODE=stub                                  (default — no ML server needed)
    MODEL_MODE=remote ML_SERVER_URL=http://ml-server:9000

The WebSocket handler now forwards whatever payload dict the frontend sends
— {"landmarks": [...]} for static letters, {"sequence": [[...], ...]} for
motion letters — unchanged, straight through to the model. This backend
never inspects or cares which shape it's carrying.
"""

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from model_interface import ASLModel
from stub_model import StubASLModel

app = FastAPI(title="Sign Buddy Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_model() -> ASLModel:
    mode = os.environ.get("MODEL_MODE", "stub")
    if mode == "stub":
        return StubASLModel()

    if mode == "remote":
        from remote_model import RemoteASLModel
        ml_server_url = os.environ.get("ML_SERVER_URL")
        if not ml_server_url:
            raise ValueError("MODEL_MODE=remote requires ML_SERVER_URL to be set.")
        return RemoteASLModel(ml_server_url)

    raise ValueError(f"Unknown MODEL_MODE: {mode!r} — expected 'stub' or 'remote'.")


model: ASLModel = _load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model_info")
async def model_info():
    """
    Lets the frontend discover capability at startup — e.g. whether motion
    prediction is available, and what window size it expects — instead of
    hardcoding a guess that could silently drift out of sync with the
    actual trained sequence model.
    """
    return await model.get_info()


@app.websocket("/predict")
async def predict_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if "landmarks" not in data and "sequence" not in data:
                await websocket.send_json({"error": "expected 'landmarks' or 'sequence'"})
                continue

            letter, confidence = await model.predict(data)
            await websocket.send_json({"letter": letter, "confidence": confidence})
    except WebSocketDisconnect:
        pass
