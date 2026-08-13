"""
remote_model.py
-----------------
Calls the ML server for predictions. Updated to be a pure, generic
pass-through — predict() forwards whatever payload dict it's given
({"landmarks": [...]} or {"sequence": [[...], ...]}) directly to the ML
server's /predict, and relays back its response. This backend never needs
to know which shape it's carrying.
"""

import httpx

from model_interface import ASLModel


class RemoteASLModel(ASLModel):
    def __init__(self, ml_server_url: str, timeout: float = 5.0):
        self.ml_server_url = ml_server_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        print(f"[MODEL]  Using RemoteASLModel — calling {self.ml_server_url}/predict")

    async def predict(self, payload: dict) -> tuple[str, float]:
        response = await self._client.post(f"{self.ml_server_url}/predict", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["letter"], data["confidence"]

    async def get_info(self) -> dict:
        response = await self._client.get(f"{self.ml_server_url}/model_info")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()
