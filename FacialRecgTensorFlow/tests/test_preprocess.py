"""T2: preprocessing contract — decode, resize 100x100, normalize [0,1]."""
import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.preprocess import preprocess, preprocess_strict, preprocess_twin


def _write_jpg(path, h=64, w=80, seed=0):
    rng = np.random.RandomState(seed)
    cv2.imwrite(path, rng.randint(0, 255, (h, w, 3), dtype=np.uint8))


def test_preprocess_shape_range(tmp_path):
    p = str(tmp_path / "img.jpg")
    _write_jpg(p)
    out = preprocess(p)
    assert tuple(out.shape) == (100, 100, 3)
    arr = out.numpy()
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_preprocess_accepts_tf_string_tensor(tmp_path):
    p = str(tmp_path / "img.jpg")
    _write_jpg(p)
    out = preprocess(tf.convert_to_tensor(p))
    assert tuple(out.shape) == (100, 100, 3)


def test_preprocess_strict_rejects_missing_and_empty(tmp_path):
    with pytest.raises(ValueError):
        preprocess_strict(str(tmp_path / "nope.jpg"))
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        preprocess_strict(str(empty))


def test_preprocess_twin_applies_to_both(tmp_path):
    a = str(tmp_path / "a.jpg")
    b = str(tmp_path / "b.jpg")
    _write_jpg(a, seed=1)
    _write_jpg(b, seed=2)
    ia, ib, label = preprocess_twin(a, b, 1.0)
    assert tuple(ia.shape) == (100, 100, 3)
    assert tuple(ib.shape) == (100, 100, 3)
    assert float(label) == 1.0
