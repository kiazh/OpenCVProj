"""T0: repo integrity — every action in later phases assumes these exist."""
import os

from src import config


def test_required_code_files_exist():
    for p in [
        os.path.join(config.BASE_DIR, "main.ipynb"),
        os.path.join(config.BASE_DIR, "README.md"),
        os.path.join(config.BASE_DIR, "..", "README.md"),
        os.path.join(config.BASE_DIR, "..", "TextScanner", "fileScanner.py"),
        os.path.join(config.BASE_DIR, "..", "LICENSE"),
    ]:
        assert os.path.isfile(p), f"missing required file: {p}"


def test_models_present():
    assert os.path.isfile(config.MODEL_KERAS_PATH), "my_model.keras missing (LFS?)"
    assert os.path.isfile(config.MODEL_H5_PATH), "siamesemodelv2.h5 missing (LFS?)"
    assert os.path.getsize(config.MODEL_KERAS_PATH) > 10_000_000, "keras model suspiciously small"


def test_data_layout():
    assert os.path.isdir(config.ANC_PATH)
    assert os.path.isdir(config.POS_PATH)
    assert os.path.isdir(config.NEG_PATH)
    # LFW negative corpus must be substantial.
    n = sum(len(files) for _, _, files in os.walk(config.NEG_PATH))
    assert n > 5000, f"negative corpus too small: {n}"
    # Anchor/positive keep-place markers (images themselves are gitignored).
    assert os.path.isfile(os.path.join(config.ANC_PATH, ".gitkeep"))
    assert os.path.isfile(os.path.join(config.POS_PATH, ".gitkeep"))


def test_hyperparameter_contract():
    assert config.IMAGE_SIZE == (100, 100)
    assert config.EMBEDDING_DIM == 4096
    assert config.LEARNING_RATE == 1e-4
    assert config.BATCH_SIZE == 16
    assert config.EPOCHS == 50
    assert config.TRAIN_SPLIT == 0.7
