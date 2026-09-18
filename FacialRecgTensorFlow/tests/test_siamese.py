"""T3: siamese assembly — output range, symmetry, weight tying."""
import numpy as np

from src.siamese import make_siamese_model


def test_siamese_output_probability():
    model = make_siamese_model()
    rng = np.random.RandomState(0)
    x = rng.rand(2, 100, 100, 3).astype("float32")
    y = rng.rand(2, 100, 100, 3).astype("float32")
    p = model.predict([x, y], verbose=0)
    assert p.shape == (2, 1)
    assert np.all(p > 0) and np.all(p < 1)


def test_siamese_symmetric_by_construction():
    model = make_siamese_model()
    rng = np.random.RandomState(1)
    x = rng.rand(1, 100, 100, 3).astype("float32")
    y = rng.rand(1, 100, 100, 3).astype("float32")
    p_ab = model.predict([x, y], verbose=0)
    p_ba = model.predict([y, x], verbose=0)
    np.testing.assert_allclose(p_ab, p_ba, rtol=1e-5, atol=1e-5)


def test_siamese_shares_embedding_weights():
    model = make_siamese_model()
    names_in = [t.name for t in model.inputs]
    assert any("input_img" in n for n in names_in)
    assert any("validation_img" in n for n in names_in)
    assert model.output_shape == (None, 1)
    # One embedding submodel used twice => layer name appears once as submodel.
    names = [layer.name for layer in model.layers]
    assert "embedding" in names
    assert any(n == "distance" or n.startswith("l1_dist") for n in names), names
