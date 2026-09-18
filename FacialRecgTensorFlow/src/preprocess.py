"""Image preprocessing — extracted from notebook cells 11 and 51.

Two variants exist in the notebook: a tf.data file-path version and a
stricter verification-time version with an existence guard. Both are kept
here so behaviour is identical and testable without a webcam.
"""
import os

import tensorflow as tf

from .config import IMAGE_SIZE


def preprocess(file_path):
    """Decode JPEG -> resize to 100x100 -> normalize to [0, 1].

    Accepts a string path or a tf.string tensor (for tf.data pipelines).
    Mirrors notebook cell 11 exactly.
    """
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img


def preprocess_strict(file_path):
    """Verification-time variant with missing/empty file guard (cell 51)."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise ValueError(f"Image file is missing or empty: {file_path}")
    return preprocess(file_path)


def preprocess_twin(input_img, validation_img, label):
    """Apply `preprocess` element-wise to both images of a pair (cell 13)."""
    return (preprocess(input_img), preprocess(validation_img), label)
