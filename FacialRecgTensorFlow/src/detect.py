"""Object-detection front-end (Phase T7: face-detect + Siamese verify).

Design: the Siamese network verifies *cropped* faces, so a detector stage
turns "object detection" into: detect boxes -> crop -> resize 100x100 ->
Siamese verify. Any detector implementing `Detector.detect()` slots in,
from classical Haar cascades (no extra deps) to YOLO (optional).
"""
import abc

import cv2
import numpy as np


class Detection:
    """Single bounding box in xyxy pixel coordinates + confidence."""

    def __init__(self, x1, y1, x2, y2, score=1.0, label="face"):
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
        self.score = float(score)
        self.label = label

    def xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)

    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.5):
    """Greedy NMS, highest score first."""
    kept = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class Detector(abc.ABC):
    @abc.abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list:
        """Return list[Detection] for a BGR uint8 image."""


class HaarFaceDetector(Detector):
    """Classical OpenCV Haar cascade — zero extra dependencies."""

    def __init__(self, cascade_path=None, scale_factor=1.1, min_neighbors=5,
                 min_size=(30, 30)):
        cascade_path = cascade_path or (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise ValueError(f"Could not load Haar cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        dets = [
            Detection(x, y, x + w, y + h, score=1.0)
            for (x, y, w, h) in faces
        ]
        return non_max_suppression(dets)


class NoopDetector(Detector):
    """Test double returning a fixed box list."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image_bgr):
        return list(self.boxes)


def crop_and_preprocess(image_bgr, detection, size=(100, 100)):
    """Crop box -> BGR->RGB -> resize -> normalize [0,1] float32."""
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, detection.x1), max(0, detection.y1)
    x2, y2 = min(w, detection.x2), min(h, detection.y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate box: {detection.xyxy()} for image {w}x{h}")
    crop = image_bgr[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, size)
    return (crop.astype("float32") / 255.0)


def detect_and_verify(image_bgr, gallery_crops, siamese_model, detector,
                      detection_threshold=0.5, verification_threshold=0.5):
    """Full two-stage pipeline: detect faces, verify each against gallery.

    `gallery_crops` are pre-processed (100,100,3) float crops of enrolled faces.
    Returns list of (Detection, scores, verified).
    """
    detections = detector.detect(image_bgr)
    out = []
    for det in detections:
        try:
            crop = crop_and_preprocess(image_bgr, det)
        except ValueError:
            continue
        scores = []
        for ref in gallery_crops:
            s = siamese_model.predict(
                [np.expand_dims(crop, 0), np.expand_dims(ref, 0)], verbose=0
            )
            scores.append(float(s[0][0]))
        scores = np.array(scores)
        frac = float(np.mean(scores > detection_threshold)) if len(scores) else 0.0
        out.append((det, scores, bool(frac > verification_threshold)))
    return out
