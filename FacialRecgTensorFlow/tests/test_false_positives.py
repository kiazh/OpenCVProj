"""T5: false-positive suite — the core ask.

Strategy: a controllable stub model returns fixed scores so threshold,
decision-boundary, and garbage-input behaviour are tested deterministically
without waiting for real training. A separate determinism test pins the
real-model property: same pair -> same score across calls.
"""
import cv2
import numpy as np
import pytest

from src import detect as det_mod
from src.detect import Detection, NoopDetector, crop_and_preprocess, detect_and_verify
from src.verify import list_verification_images, verify


class StubModel:
    """Returns pre-set scores in gallery order, one predict call per image."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def predict(self, inputs, verbose=0):
        s = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return np.array([[s]], dtype="float32")


def _gallery_with_input(tmp_gallery, tmp_input_image):
    return tmp_gallery, tmp_input_image


def test_gitkeep_and_txt_ignored_in_gallery(tmp_gallery):
    assert ".gitkeep" not in list_verification_images(tmp_gallery)
    assert "notes.txt" not in list_verification_images(tmp_gallery)
    assert len(list_verification_images(tmp_gallery)) == 4


def test_empty_gallery_never_verifies(tmp_input_image, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    scores, verified = verify(StubModel([0.99]), tmp_input_image, str(empty))
    assert scores.size == 0
    assert verified is False


def test_threshold_boundary_accept_vs_reject(tmp_gallery, tmp_input_image):
    # 3/4 above 0.5 -> verified at (0.5, 0.5).
    scores, verified = verify(
        StubModel([0.9, 0.8, 0.7, 0.1]),
        tmp_input_image, tmp_gallery, 0.5, 0.5,
    )
    assert verified is True
    # Same scores, strict detection threshold 0.95 -> 0/4 -> rejected.
    scores2, verified2 = verify(
        StubModel([0.9, 0.8, 0.7, 0.1]),
        tmp_input_image, tmp_gallery, 0.95, 0.5,
    )
    assert verified2 is False


def test_high_precision_operating_point_rejects_borderline(tmp_gallery, tmp_input_image):
    # All scores borderline 0.55: default (0.5,0.5) accepts (FP risk),
    # raised detection threshold 0.9 rejects -> documents precision/recall knob.
    _, accept = verify(StubModel([0.55] * 4), tmp_input_image, tmp_gallery, 0.5, 0.5)
    _, reject = verify(StubModel([0.55] * 4), tmp_input_image, tmp_gallery, 0.9, 0.5)
    assert accept is True
    assert reject is False


def test_verification_threshold_requires_quorum(tmp_gallery, tmp_input_image):
    # Only 1/4 above threshold: passes detection but fails 0.5 quorum.
    _, verified = verify(
        StubModel([0.99, 0.1, 0.1, 0.1]), tmp_input_image, tmp_gallery, 0.5, 0.5
    )
    assert verified is False


def test_garbage_inputs_rejected():
    """Blank / black / white / noise crops must not verify against gallery."""
    rng = np.random.RandomState(7)
    gallery = [rng.rand(100, 100, 3).astype("float32") for _ in range(3)]
    blanks = {
        "black": np.zeros((200, 200, 3), dtype=np.uint8),
        "white": np.full((200, 200, 3), 255, dtype=np.uint8),
        "noise": rng.randint(0, 255, (200, 200, 3), dtype=np.uint8),
    }
    box = [Detection(10, 10, 110, 110, score=1.0)]
    for name, img in blanks.items():
        res = detect_and_verify(img, gallery, StubModel([0.05, 0.1, 0.02]),
                                NoopDetector(box), 0.5, 0.5)
        assert res[0][2] is False, f"garbage '{name}' falsely verified"


def test_degenerate_box_skipped_not_crash():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    gallery = [rng.rand(100, 100, 3).astype("float32")]
    bad = [Detection(40, 40, 10, 10)]  # x2<x1
    assert detect_and_verify(img, gallery, StubModel([0.99]),
                             NoopDetector(bad)) == []


def test_real_model_deterministic():
    from src.siamese import make_siamese_model

    model = make_siamese_model()
    rng = np.random.RandomState(11)
    x = rng.rand(1, 100, 100, 3).astype("float32")
    y = rng.rand(1, 100, 100, 3).astype("float32")
    np.testing.assert_allclose(
        model.predict([x, y], verbose=0), model.predict([x, y], verbose=0)
    )
