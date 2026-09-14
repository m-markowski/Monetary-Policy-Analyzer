import pandas as pd
import pytest

from src.models.pipeline import auto_k_features, chronological_split


def make_frame(n_rows: int):
    index = pd.date_range("2010-01-31", periods=n_rows, freq="ME")
    X = pd.DataFrame({"feature": range(n_rows)}, index=index)
    y = pd.Series(range(n_rows), index=index, dtype=float)
    return X, y


def test_chronological_split_keeps_order_and_purges_horizon_gap():
    X, y = make_frame(100)

    splits = chronological_split(X, y, preset="70/15/15", gap=3)
    X_train, _ = splits["Train"]
    X_valid, _ = splits["Valid"]
    X_test, _ = splits["Test"]

    assert (len(X_train), len(X_valid), len(X_test)) == (67, 12, 15)
    assert X_train.index.max() < X_valid.index.min()
    assert X_valid.index.max() < X_test.index.min()

    # Rows whose labels are realised inside the next split are dropped, not moved.
    used = X_train.index.union(X_valid.index).union(X_test.index)
    assert not X.index[67:70].isin(used).any()
    assert not X.index[82:85].isin(used).any()


def test_chronological_split_rejects_unsorted_dates():
    X, y = make_frame(50)

    with pytest.raises(ValueError):
        chronological_split(X.iloc[::-1], y.iloc[::-1])


def test_chronological_split_rejects_gap_that_empties_a_split():
    X, y = make_frame(20)  # the validation block holds 3 rows before purging

    with pytest.raises(ValueError):
        chronological_split(X, y, preset="70/15/15", gap=3)


def test_auto_k_features_is_bounded_by_smallest_fold_and_floor():
    assert auto_k_features(n_rows=240, n_features=50, n_splits=5, gap=0) == 10
    assert auto_k_features(n_rows=120, n_features=50, n_splits=5, gap=3) == 5
    assert auto_k_features(n_rows=240, n_features=3, n_splits=5, gap=0) == 3