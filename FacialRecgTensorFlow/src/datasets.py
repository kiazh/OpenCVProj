"""Dataset construction — notebook cells 9, 13, 15."""
import tensorflow as tf

from .config import (
    ANC_PATH,
    BATCH_SIZE,
    MAX_SAMPLES_PER_SPLIT,
    NEG_PATH,
    POS_PATH,
    PREFETCH_DEPTH,
    SHUFFLE_BUFFER,
    TRAIN_SPLIT,
)
from .preprocess import preprocess_twin


def build_pair_dataset(
    anc_path=ANC_PATH,
    pos_path=POS_PATH,
    neg_path=NEG_PATH,
    max_samples=MAX_SAMPLES_PER_SPLIT,
):
    """Anchor-positive (label 1) + anchor-negative (label 0), concatenated."""
    anchor = tf.data.Dataset.list_files(anc_path + "/*.jpg").take(max_samples)
    positive = tf.data.Dataset.list_files(pos_path + "/*.jpg").take(max_samples)
    # Negatives live in per-identity subfolders (LFW layout).
    negative = tf.data.Dataset.list_files(neg_path + "/*/*.jpg").take(max_samples)

    positives = tf.data.Dataset.zip(
        (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(max_samples)))
    )
    negatives = tf.data.Dataset.zip(
        (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(max_samples)))
    )
    return positives.concatenate(negatives)


def prepare_datasets(
    data,
    train_split=TRAIN_SPLIT,
    batch_size=BATCH_SIZE,
    shuffle_buffer=SHUFFLE_BUFFER,
    prefetch_depth=PREFETCH_DEPTH,
):
    """Map -> cache -> shuffle -> 70/30 split -> batch -> prefetch (cell 15)."""
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=shuffle_buffer)

    n = len(data)
    n_train = round(n * train_split)
    n_test = round(n * (1.0 - train_split))

    train_data = data.take(n_train).batch(batch_size).prefetch(prefetch_depth)
    test_data = data.skip(n_train).take(n_test).batch(batch_size).prefetch(prefetch_depth)
    return train_data, test_data
