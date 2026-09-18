"""T3: L1Dist layer — correctness, symmetry, gradient flow."""
import numpy as np
import tensorflow as tf

from src.distance import L1Dist


def test_l1dist_value():
    layer = L1Dist()
    a = tf.constant([[1.0, 2.0, 3.0]])
    b = tf.constant([[1.0, 0.0, 3.0]])
    np.testing.assert_allclose(layer([a, b]).numpy(), [[0.0, 2.0, 0.0]])


def test_l1dist_symmetry():
    layer = L1Dist()
    rng = np.random.RandomState(0)
    a = tf.constant(rng.rand(2, 16).astype("float32"))
    b = tf.constant(rng.rand(2, 16).astype("float32"))
    np.testing.assert_allclose(layer([a, b]).numpy(), layer([b, a]).numpy())


def test_l1dist_zero_for_identical():
    layer = L1Dist()
    a = tf.ones((1, 8))
    np.testing.assert_allclose(layer([a, a]).numpy(), np.zeros((1, 8)))


def test_l1dist_gradient_flows():
    layer = L1Dist()
    a = tf.Variable([[1.0, 2.0]])
    b = tf.constant([[0.0, 0.0]])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer([a, b]))
    grad = tape.gradient(loss, a)
    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))
