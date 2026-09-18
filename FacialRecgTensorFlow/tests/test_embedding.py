"""T3: embedding network — shape, sigmoid bound, determinism."""
import numpy as np

from src.config import EMBEDDING_DIM
from src.embedding import make_embedding


def test_embedding_output_shape_and_bound():
    emb = make_embedding()
    assert emb.output_shape == (None, EMBEDDING_DIM)
    out = emb(np.zeros((1, 100, 100, 3), dtype="float32")).numpy()
    assert out.shape == (1, EMBEDDING_DIM)
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid head


def test_embedding_deterministic_for_same_input():
    emb = make_embedding()
    x = np.random.RandomState(0).rand(1, 100, 100, 3).astype("float32")
    a = emb(x).numpy()
    b = emb(x).numpy()
    np.testing.assert_allclose(a, b)


def test_embedding_separates_different_inputs():
    emb = make_embedding()
    rng = np.random.RandomState(3)
    x = rng.rand(1, 100, 100, 3).astype("float32")
    y = rng.rand(1, 100, 100, 3).astype("float32")
    assert not np.allclose(emb(x).numpy(), emb(y).numpy())
