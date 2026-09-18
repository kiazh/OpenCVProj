"""Shared fixtures: temp image dirs, synthetic jpgs, tiny datasets."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


@pytest.fixture()
def tmp_gallery(tmp_path):
    """Directory with 4 synthetic gallery jpgs + a .gitkeep (must be ignored)."""
    d = tmp_path / "gallery"
    d.mkdir()
    rng = np.random.RandomState(0)
    for i in range(4):
        img = rng.randint(0, 255, (120, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"face_{i}.jpg"), img)
    (d / ".gitkeep").write_text("")
    (d / "notes.txt").write_text("not an image")
    return str(d)


@pytest.fixture()
def tmp_input_image(tmp_path):
    rng = np.random.RandomState(1)
    img = rng.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    p = tmp_path / "input.jpg"
    cv2.imwrite(str(p), img)
    return str(p)


@pytest.fixture()
def tiny_pair_dirs(tmp_path):
    """2 anchor + 2 positive + 2 LFW-style negative images for pipeline tests."""
    anc = tmp_path / "anchor"
    pos = tmp_path / "positive"
    neg_person = tmp_path / "negative" / "Some_Person"
    anc.mkdir()
    pos.mkdir(parents=True)
    neg_person.mkdir(parents=True)
    rng = np.random.RandomState(42)
    for i in range(2):
        cv2.imwrite(str(anc / f"a{i}.jpg"), rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(pos / f"p{i}.jpg"), rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        cv2.imwrite(
            str(neg_person / f"Some_Person_000{i}.jpg"),
            rng.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        )
    return str(anc), str(pos), str(tmp_path / "negative")
