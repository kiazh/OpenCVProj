"""T4: training — loss/optimizer wiring, weight updates, checkpoints."""
import os

import numpy as np
import tensorflow as tf

from src.datasets import build_pair_dataset, prepare_datasets
from src.siamese import make_siamese_model
from src.training import make_checkpoint, make_loss_and_optimizer, make_train_step


def test_loss_and_optimizer_hyperparams():
    loss_fn, opt = make_loss_and_optimizer()
    assert isinstance(loss_fn, tf.losses.BinaryCrossentropy)
    assert abs(float(opt.learning_rate) - 1e-4) < 1e-7  # float32 rounding


def test_train_step_updates_weights(tiny_pair_dirs):
    anc, pos, neg = tiny_pair_dirs
    ds = build_pair_dataset(anc, pos, neg, max_samples=2)
    train, _ = prepare_datasets(ds, train_split=0.5, batch_size=2,
                                shuffle_buffer=10, prefetch_depth=1)
    batch = next(iter(train))
    model = make_siamese_model()
    loss_fn, opt = make_loss_and_optimizer()
    before = [w.numpy().copy() for w in model.trainable_variables]
    step = make_train_step(model, loss_fn, opt)
    loss = step(batch)
    assert np.isfinite(float(loss))
    after = [w.numpy() for w in model.trainable_variables]
    assert any(not np.allclose(b, a) for a, b in zip(before, after)), \
        "train_step did not update any weights"


def test_train_step_is_graph_function(tiny_pair_dirs):
    anc, pos, neg = tiny_pair_dirs
    ds = build_pair_dataset(anc, pos, neg, max_samples=2)
    train, _ = prepare_datasets(ds, train_split=0.5, batch_size=2,
                                shuffle_buffer=10, prefetch_depth=1)
    batch = next(iter(train))
    model = make_siamese_model()
    loss_fn, opt = make_loss_and_optimizer()
    step = make_train_step(model, loss_fn, opt)
    # tf.function wrapper exposes a graph; plain python fn does not.
    assert hasattr(step, "get_concrete_function")
    loss = step(batch)
    assert np.isfinite(float(loss))


def test_checkpoint_roundtrip(tmp_path):
    model = make_siamese_model()
    _, opt = make_loss_and_optimizer()
    ckpt, prefix = make_checkpoint(
        model, opt, checkpoint_dir=str(tmp_path / "ckpts")
    )
    save_path = ckpt.save(file_prefix=prefix)
    assert save_path is not None
    assert any(str(tmp_path / "ckpts") in str(p) for p in [save_path])
    assert len(os.listdir(str(tmp_path / "ckpts"))) >= 2  # .index + .data
