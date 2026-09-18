"""T6: persistence — load shipped .keras, predict, save/reload roundtrip."""
import numpy as np
import pytest

from src import config
from src.siamese import load_model

pytestmark = pytest.mark.slow


def test_load_shipped_keras_and_predict():
    model = load_model(config.MODEL_KERAS_PATH)
    assert model.output_shape == (None, 1)
    rng = np.random.RandomState(0)
    x = rng.rand(2, 100, 100, 3).astype("float32")
    y = rng.rand(2, 100, 100, 3).astype("float32")
    p = model.predict([x, y], verbose=0)
    assert p.shape == (2, 1)
    assert np.all(p > 0) and np.all(p < 1)


def test_save_reload_roundtrip(tmp_path):
    from src.siamese import make_siamese_model, save_model

    model = make_siamese_model()
    rng = np.random.RandomState(2)
    x = rng.rand(1, 100, 100, 3).astype("float32")
    y = rng.rand(1, 100, 100, 3).astype("float32")
    before = model.predict([x, y], verbose=0)
    path = str(tmp_path / "roundtrip.keras")
    save_model(model, path)
    reloaded = load_model(path)
    np.testing.assert_allclose(reloaded.predict([x, y], verbose=0), before, rtol=1e-5, atol=1e-5)
