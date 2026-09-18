"""Verification application — notebook cells 48-52, refactored for testability.

The notebook version hardcodes global paths and a live webcam loop, which
cannot run under pytest. This module keeps the identical decision logic but:
  - takes explicit paths / thresholds as arguments,
  - filters non-image files (excludes .gitkeep),
  - exposes `enroll()` and `verify()` as pure functions,
  - leaves the cv2 capture loop to a thin `run_live_verification()` wrapper.
"""
import os
import shutil

import numpy as np

from .config import (
    DETECTION_THRESHOLD,
    INPUT_IMAGE_PATH,
    POS_PATH,
    VERIFICATION_GALLERY_SIZE,
    VERIFICATION_IMAGES_DIR,
    VERIFICATION_THRESHOLD,
)
from .preprocess import preprocess_strict

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_verification_images(verification_dir=VERIFICATION_IMAGES_DIR):
    if not os.path.isdir(verification_dir):
        return []
    return sorted(
        f
        for f in os.listdir(verification_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def enroll(
    pos_path=POS_PATH,
    verification_dir=VERIFICATION_IMAGES_DIR,
    max_images=VERIFICATION_GALLERY_SIZE,
):
    """Copy up to `max_images` positives into the verification gallery."""
    os.makedirs(verification_dir, exist_ok=True)
    pos_images = sorted(os.listdir(pos_path))[:max_images]
    for img in pos_images:
        src = os.path.join(pos_path, img)
        if os.path.isfile(src) and img.lower().endswith(IMAGE_EXTENSIONS):
            shutil.copy(src, os.path.join(verification_dir, img))
    return list_verification_images(verification_dir)


def verify(
    model,
    input_image_path=INPUT_IMAGE_PATH,
    verification_dir=VERIFICATION_IMAGES_DIR,
    detection_threshold=DETECTION_THRESHOLD,
    verification_threshold=VERIFICATION_THRESHOLD,
):
    """Compare input image against every gallery image.

    Returns (scores array, verified bool). Two-stage decision boundary:
      detection = #{score > detection_threshold}
      verified  = (detection / N) > verification_threshold
    """
    gallery = list_verification_images(verification_dir)
    if not gallery:
        return np.array([]), False

    input_img = preprocess_strict(input_image_path)

    results = []
    for image in gallery:
        validation_img = preprocess_strict(os.path.join(verification_dir, image))
        result = model.predict(
            [np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)],
            verbose=0,
        )
        results.append(result[0][0])

    results = np.array(results)
    detection = np.sum(results > detection_threshold)
    verification = detection / len(gallery)
    verified = bool(verification > verification_threshold)
    return results, verified


def run_live_verification(model, detection_threshold=0.5, verification_threshold=0.5):
    """Thin webcam wrapper (not used in tests — requires a display + camera)."""
    import cv2

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.resize(frame, (250, 250))
        cv2.imshow("Verification", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("v"):
            os.makedirs(os.path.dirname(INPUT_IMAGE_PATH), exist_ok=True)
            if cv2.imwrite(INPUT_IMAGE_PATH, frame):
                results, verified = verify(
                    model, INPUT_IMAGE_PATH, VERIFICATION_IMAGES_DIR,
                    detection_threshold, verification_threshold,
                )
                print(f"Verified: {verified}")
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
