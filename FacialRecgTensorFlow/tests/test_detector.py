"""T7: object-detection front-end — IoU, NMS, Haar smoke, crop, pipeline."""
import numpy as np
import pytest

from src.detect import (
    Detection,
    HaarFaceDetector,
    NoopDetector,
    crop_and_preprocess,
    detect_and_verify,
    iou,
    non_max_suppression,
)


def test_iou_identical_and_disjoint():
    a = Detection(0, 0, 10, 10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, Detection(20, 20, 30, 30)) == pytest.approx(0.0)


def test_iou_partial():
    assert iou(Detection(0, 0, 10, 10), Detection(5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_nms_suppresses_overlap_keeps_best():
    boxes = [Detection(0, 0, 10, 10, score=0.9), Detection(1, 1, 11, 11, score=0.5),
             Detection(50, 50, 60, 60, score=0.8)]
    kept = non_max_suppression(boxes, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].score == pytest.approx(0.9)


def test_haar_detector_loads_and_handles_blank():
    det = HaarFaceDetector()
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    out = det.detect(blank)
    assert isinstance(out, list)  # blank -> almost surely [] but must not crash


def test_haar_detector_rejects_bad_cascade():
    with pytest.raises(ValueError):
        HaarFaceDetector(cascade_path="nonexistent.xml")


def test_crop_and_preprocess_shape_range():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    crop = crop_and_preprocess(img, Detection(10, 20, 110, 120))
    assert crop.shape == (100, 100, 3)
    assert crop.min() >= 0.0 and crop.max() <= 1.0


def test_crop_clamps_out_of_bounds_box():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    crop = crop_and_preprocess(img, Detection(-10, -10, 1000, 1000))
    assert crop.shape == (100, 100, 3)


class StubSiamese:
    def __init__(self, score):
        self.score = score

    def predict(self, inputs, verbose=0):
        return np.array([[self.score]], dtype="float32")


def test_detect_and_verify_accept_and_reject():
    rng = np.random.RandomState(4)
    img = rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    gallery = [rng.rand(100, 100, 3).astype("float32") for _ in range(2)]
    box = [Detection(10, 10, 110, 110)]
    ok = detect_and_verify(img, gallery, StubSiamese(0.95), NoopDetector(box), 0.5, 0.5)
    assert len(ok) == 1 and ok[0][2] is True
    bad = detect_and_verify(img, gallery, StubSiamese(0.05), NoopDetector(box), 0.5, 0.5)
    assert bad[0][2] is False


def test_detect_and_verify_no_faces_empty():
    rng = np.random.RandomState(5)
    img = rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    gallery = [rng.rand(100, 100, 3).astype("float32")]
    assert detect_and_verify(img, gallery, StubSiamese(0.99), NoopDetector([])) == []
