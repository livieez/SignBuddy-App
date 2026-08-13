"""
model_interface.py
-------------------
predict() now takes a PAYLOAD dict, not a fixed landmarks list — this lets
the same contract carry either a static single-frame request
({"landmarks": [...]}) or a motion sequence request
({"sequence": [[...], [...], ...]}) without this backend needing to know or
care which one it's forwarding. It just relays whatever the frontend sent to
the ML server and relays back whatever comes back.
"""

from abc import ABC, abstractmethod


class ASLModel(ABC):
    @abstractmethod
    async def predict(self, payload: dict) -> tuple[str, float]:
        """
        payload   : whatever dict the frontend sent — either
                    {"landmarks": [...]} (static) or
                    {"sequence": [[...], ...]} (motion).
        Returns   : (letter, confidence).
        """
        raise NotImplementedError

    async def get_info(self) -> dict:
        """Optional: model/server capability info (e.g. motion window size)."""
        return {"motion_available": False}
