"""
app.py
-------
The ML repo's server. Exposes:

    POST /predict
        {"landmarks": [63 floats]}          -> static letter prediction
        {"sequence": [[...], [...], ...]}    -> motion letter prediction (J/Z)
    GET  /model_info
        {"motion_available": bool, "window": int}
    GET  /health

Which one a /predict request triggers depends entirely on which key is
present in the body — the frontend decides which to send based on its own
motion gate (see WebCameraView.jsx), not this server.

Run locally
-----------
    pip install -r requirements.txt
    STATIC_MODEL_DIR=models/v2_model STATIC_TYPE=v2 \
    SEQ_MODEL_DIR=models/seq_model \
    uvicorn app:app --port 9000
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_loader import HybridModel

app = FastAPI(title="Sign Buddy ML Server (Hybrid)")


class PredictRequest(BaseModel):
    landmarks: list[float] | None = None
    sequence: list[list[float]] | None = None


class PredictResponse(BaseModel):
    letter: str
    confidence: float


def _load_model() -> HybridModel:
    static_dir = os.environ.get("STATIC_MODEL_DIR", "models/v2_model")
    static_type = os.environ.get("STATIC_TYPE", "v2")
    seq_dir_raw = os.environ.get("SEQ_MODEL_DIR")
    seq_dir = Path(seq_dir_raw) if seq_dir_raw else None

    return HybridModel(
        static_model_dir=Path(static_dir),
        static_type=static_type,
        seq_model_dir=seq_dir,
    )


model = _load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model_info")
def model_info():
    return {"motion_available": model.motion_available, "window": model.window}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.sequence is not None:
        letter, confidence = model.predict_motion(request.sequence)
    elif request.landmarks is not None:
        letter, confidence = model.predict_static(request.landmarks)
    else:
        raise HTTPException(status_code=400, detail="Provide either 'landmarks' or 'sequence'.")

    return PredictResponse(letter=letter, confidence=confidence)