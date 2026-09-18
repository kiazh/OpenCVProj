"""Training — notebook cells 26, 28, 30, 32."""
import os

import tensorflow as tf

from .config import CHECKPOINT_DIR, LEARNING_RATE


def make_loss_and_optimizer(learning_rate=LEARNING_RATE):
    loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(learning_rate)
    return loss, opt


def make_checkpoint(model, optimizer, checkpoint_dir=CHECKPOINT_DIR):
    os.makedirs(checkpoint_dir, exist_ok=True)
    prefix = os.path.join(checkpoint_dir, "ckpt")
    return tf.train.Checkpoint(opt=optimizer, siamese_model=model), prefix


def make_train_step(model, loss_fn, optimizer):
    """Graph-compiled single gradient step (notebook cell 30)."""

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            X = batch[:2]
            y = batch[2]
            yhat = model(X, training=True)
            loss = loss_fn(y, yhat)
        tf.print(loss)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss

    return train_step


def train(data, model, epochs, checkpoint=None, checkpoint_prefix=None):
    """Epoch loop with per-epoch Recall/Precision (notebook cell 32)."""
    from keras.metrics import Precision, Recall

    loss_fn, _ = make_loss_and_optimizer()
    # Reuse the passed optimizer via checkpoint if available.
    optimizer = checkpoint.opt if checkpoint is not None else tf.keras.optimizers.Adam(LEARNING_RATE)
    train_step = make_train_step(model, loss_fn, optimizer)

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n Epoch {epoch}/{epochs}")
        progbar = tf.keras.utils.Progbar(len(data))

        r = Recall()
        p = Precision()

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = model(batch[:2], training=False)
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        history.append((float(loss.numpy()), float(r.result().numpy()), float(p.result().numpy())))

        if checkpoint is not None and epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    return history
