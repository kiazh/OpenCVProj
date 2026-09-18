"""T2: dataset assembly — labels, concatenation, 70/30 split, batching."""
import tensorflow as tf

from src.datasets import build_pair_dataset, prepare_datasets


def test_build_pair_dataset_labels(tiny_pair_dirs):
    anc, pos, neg = tiny_pair_dirs
    ds = build_pair_dataset(anc, pos, neg, max_samples=2)
    rows = list(ds.as_numpy_iterator())
    assert len(rows) == 4  # 2 positive + 2 negative
    labels = sorted(float(r[2]) for r in rows)
    assert labels == [0.0, 0.0, 1.0, 1.0]


def test_prepare_datasets_split_and_batch(tiny_pair_dirs):
    anc, pos, neg = tiny_pair_dirs
    ds = build_pair_dataset(anc, pos, neg, max_samples=2)
    train, test = prepare_datasets(
        ds, train_split=0.5, batch_size=2, shuffle_buffer=10, prefetch_depth=1
    )
    n_train = sum(1 for _ in train)
    n_test = sum(1 for _ in test)
    assert n_train + n_test >= 1
    for batch in train.take(1):
        assert len(batch) == 3
        assert batch[0].shape[1:] == (100, 100, 3)
        assert batch[0].shape[0] <= 2
