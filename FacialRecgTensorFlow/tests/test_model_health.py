"""Model-health checks on the SHIPPED weights (real LFW data).

Status (measured 2026-09-17, CPU):
  - my_model.keras: self-match ~= 0.5, cross-identity ~= 0.502 -> constant
    classifier. Explains README's precision 0.4011 / recall 0.9868.
  - siamesemodelv2.h5: self ~= 0.50, same-person ~= 0.86, impostor ~= 0.83 ->
    no genuine/impostor separation (likely collapsed training).

The xfail tests below pin the HEALTHY behaviour. They will start passing
after a successful retraining run (needs anchor/positive captures + GPU).
The passing test documents current degenerate output so regressions in
loading/inference still get caught.
"""
import os

import numpy as np
import pytest

from src import config
from src.preprocess import preprocess
from src.siamese import load_model

pytestmark = pytest.mark.slow

NEG = os.path.join(config.BASE_DIR, "data", "negative")
P1 = os.path.join(NEG, "Abdullah_Gul", "Abdullah_Gul_0001.jpg")
P1B = os.path.join(NEG, "Abdullah_Gul", "Abdullah_Gul_0002.jpg")
P2 = os.path.join(NEG, "Adrien_Brody", "Adrien_Brody_0001.jpg")

requires_lfw = pytest.mark.skipif(
    not (os.path.isfile(P1) and os.path.isfile(P1B) and os.path.isfile(P2)),
    reason="LFW sample images not present",
)


def _score(model, pa, pb):
    a = np.expand_dims(preprocess(pa), 0)
    b = np.expand_dims(preprocess(pb), 0)
    return float(model.predict([a, b], verbose=0)[0][0])


@requires_lfw
def test_shipped_model_loads_and_outputs_probability():
    model = load_model(config.MODEL_KERAS_PATH)
    s = _score(model, P1, P2)
    assert 0.0 < s < 1.0


@requires_lfw
@pytest.mark.xfail(
    reason="shipped my_model.keras is degenerate (constant ~0.5); "
    "remove xfail after retraining with real anchor/positive data",
    strict=False,
)
def test_healthy_model_separates_genuine_from_impostor():
    model = load_model(config.MODEL_KERAS_PATH)
    genuine = _score(model, P1, P1B)
    impostor = _score(model, P1, P2)
    assert genuine > 0.7, f"genuine too low: {genuine}"
    assert impostor < 0.4, f"impostor too high: {impostor}"
    assert genuine - impostor > 0.3
