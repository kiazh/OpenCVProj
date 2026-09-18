"""T6: verify() + enroll() with stub model (no webcam, no TF training)."""
import os

import numpy as np

from src.verify import enroll, list_verification_images, verify


class StubModel:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def predict(self, inputs, verbose=0):
        s = self.scores[self.calls % len(self.scores)]
        self.calls += 1
        return np.array([[s]], dtype="float32")


def test_verify_accepts_on_quorum(tmp_gallery, tmp_input_image):
    scores, verified = verify(
        StubModel([0.9, 0.9, 0.9, 0.9]), tmp_input_image, tmp_gallery, 0.5, 0.5
    )
    assert len(scores) == 4
    assert verified is True


def test_verify_rejects_impostor(tmp_gallery, tmp_input_image):
    scores, verified = verify(
        StubModel([0.05, 0.1, 0.02, 0.2]), tmp_input_image, tmp_gallery, 0.5, 0.5
    )
    assert verified is False
    assert np.all(scores < 0.5)


def test_enroll_copies_only_images(tmp_path):
    src = tmp_path / "pos"
    src.mkdir()
    import cv2

    rng = np.random.RandomState(0)
    cv2.imwrite(str(src / "a.jpg"), rng.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    (src / ".gitkeep").write_text("")
    dst = tmp_path / "gallery"
    out = enroll(str(src), str(dst), max_images=200)
    assert out == ["a.jpg"]
    assert os.path.isfile(os.path.join(str(dst), "a.jpg"))
