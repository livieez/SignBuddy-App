"""
stub_model.py
--------------
Fake ASLModel — updated to match the generalized payload-dict contract.
Ignores whatever payload it's given (static or motion), always returns fake
data. get_info() reports motion as unavailable so the frontend knows not to
attempt sending sequences against the stub.
"""

import random
import time

from model_interface import ASLModel

_FAKE_LETTERS = ["A", "B", "C", "HELLO?", "D"]


class StubASLModel(ASLModel):
    def __init__(self):
        self._start = time.time()
        print("[MODEL]  Using StubASLModel — fake predictions, no ML server involved.")

    async def predict(self, payload: dict) -> tuple[str, float]:
        letter = _FAKE_LETTERS[int(time.time() - self._start) % len(_FAKE_LETTERS)]
        confidence = round(random.uniform(0.75, 0.99), 2)
        return letter, confidence

    async def get_info(self) -> dict:
        return {"motion_available": False}
