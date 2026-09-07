import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config.settings import SEED

# Quiet TF logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def make_sequences(values: np.ndarray, lookback: int) -> np.ndarray:
    """
    Turn a 2D feature array into overlapping look-back windows for sequence models.

    Row i maps to the window ending at i; the first rows are left-padded by
    repeating the first observation so the output has one window per input row.
    Callers pass the full history so padding only affects the very first months
    of the sample.

    Args:
        values (np.ndarray): Scaled feature matrix (n_samples x n_features).
        lookback (int): Window length in rows (months on the monthly frame).

    Returns:
        np.ndarray: Windows of shape (n_samples, lookback, n_features).
    """
    n = values.shape[0]
    padded = np.vstack([np.repeat(values[:1], lookback - 1, axis=0), values])
    return np.stack([padded[i : i + lookback] for i in range(n)], axis=0)


def build_network(kind, input_shape, n_outputs, task, units, dropout, l2, learning_rate):
    """
    Compile a small regularized Keras network for one task.

    Args:
        kind (str): 'mlp' or 'lstm'.
        input_shape (tuple): (n_features,) for MLP, (lookback, n_features) otherwise.
        n_outputs (int): Number of classes (classification) or 1 (regression).
        task (str): 'regression' or 'classification'.
        units (int): Width of the main layer.
        dropout (float): Dropout rate after each block.
        l2 (float): L2 kernel penalty.
        learning_rate (float): Adam learning rate.

    Returns:
        keras.Model: The compiled model.
    """
    import keras

    reg = keras.regularizers.l2(l2)
    model = keras.Sequential([keras.layers.Input(shape=input_shape)])
    if kind == "mlp":
        model.add(keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(units // 2, activation="relu", kernel_regularizer=reg))
        model.add(keras.layers.Dropout(dropout))
    elif kind == "lstm":
        model.add(keras.layers.LSTM(units, kernel_regularizer=reg))
        model.add(keras.layers.Dropout(dropout))

    if task == "classification":
        model.add(keras.layers.Dense(n_outputs, activation="softmax"))
        loss, metrics = "sparse_categorical_crossentropy", ["accuracy"]
    else:
        model.add(keras.layers.Dense(1))
        loss, metrics = "mse", ["mae"]
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss, metrics=metrics)
    return model


class KerasEstimator:
    """
    sklearn-style wrapper around a small Keras network (MLP / LSTM).

    Standardisation and (for the sequence kinds) windowing are handled internally,
    so the estimator consumes the same 2D feature frame as the sklearn roster and
    exposes `predict` / `predict_proba` for the shared evaluation path.

    Pickling is supported: the Keras network cannot survive pickle on its own, so
    `__getstate__` serialises it with Keras' native `.keras` format and
    `__setstate__` rebuilds it, letting the registry persist the neural roster
    alongside the sklearn models in one artifact.
    """

    def __init__(
        self,
        task,
        kind="mlp",
        lookback=6,
        units=32,
        dropout=0.2,
        l2=1e-3,
        learning_rate=1e-3,
        epochs=300,
        batch_size=16,
        class_weight=True,
        random_state=SEED,
    ):
        self.task = task
        self.kind = kind
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.random_state = random_state

    def transform_inputs(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind != "lstm":
            return self.scaler_.transform(np.asarray(X, dtype=float))

        # Each window ends at a row of X and reaches back into the stored feature
        # history, so dev/test/scenario rows are preceded by their real past months
        # rather than copies of themselves. Rows of X replace history rows at the
        # same dates, which is what lets an edited scenario row take effect.
        context = pd.concat([self.history_.drop(index=X.index, errors="ignore"), X]).sort_index()
        windows = make_sequences(
            self.scaler_.transform(context.to_numpy(dtype=float)),
            self.lookback,
        )
        return windows[context.index.get_indexer(X.index)]

    def fit(self, X: pd.DataFrame, y, history: pd.DataFrame | None = None):
        import keras

        keras.utils.set_random_seed(self.random_state)
        # Seeds the Python/NumPy/TF RNGs. GPU kernels and some parallel ops stay
        # nondeterministic, so neural runs are close but not bit-identical.
        # The scaler sees training rows only; history supplies earlier feature rows
        # for the LSTM windows and carries no labels.
        self.history_ = X if history is None else history
        self.scaler_ = StandardScaler().fit(np.asarray(X, dtype=float))
        prepared = self.transform_inputs(X)
        input_shape = prepared.shape[1:]

        weights = None
        if self.task == "classification":
            self.classes_ = np.unique(y)
            n_outputs = len(self.classes_)
            target = np.searchsorted(self.classes_, y)
            if self.class_weight:
                balanced = compute_class_weight("balanced", classes=self.classes_, y=np.asarray(y))
                weights = dict(enumerate(balanced))
        else:
            n_outputs = 1
            target = np.asarray(y, dtype=float)

        self.model_ = build_network(
            self.kind, input_shape, n_outputs, self.task, self.units, self.dropout, self.l2, self.learning_rate
        )
        callbacks = [
            keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-5),
        ]
        self.model_.fit(
            prepared,
            target,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=weights,
            shuffle=False,
            verbose=0,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict(self.transform_inputs(X), verbose=0)

    def predict(self, X) -> np.ndarray:
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return self.model_.predict(self.transform_inputs(X), verbose=0).ravel()

    def __getstate__(self):
        state = self.__dict__.copy()
        network = state.pop("model_", None)
        if network is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "network.keras"
                network.save(path)
                state["model_bytes_"] = path.read_bytes()
        return state

    def __setstate__(self, state):
        import keras

        blob = state.pop("model_bytes_", None)
        self.__dict__.update(state)
        if blob is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "network.keras"
                path.write_bytes(blob)
                self.model_ = keras.models.load_model(path)


def neural_models(task, lookback=6, class_weight=True, random_state=SEED) -> dict:
    """
    Build the (unfitted) neural roster for one task.

    Args:
        task (str): 'regression' or 'classification'.
        lookback (int): Window length for the sequence models.
        class_weight (bool): Balance classes during training (classification).
        random_state (int): Seed.

    Returns:
        dict: Model name -> unfitted KerasEstimator.
    """
    common = {"class_weight": class_weight, "random_state": random_state}
    return {
        "MLP": KerasEstimator(task, kind="mlp", **common),
        "LSTM": KerasEstimator(task, kind="lstm", lookback=lookback, **common),
    }


def fit_neural_models(
    X: pd.DataFrame,
    y,
    task: str,
    *,
    history: pd.DataFrame,
    lookback: int = 6,
    class_weight: bool = True,
    random_state: int = SEED,
    progress=None,
) -> dict:
    """
    Fit the neural roster on the training split.

    Neural nets use fixed, heavily-regularized configurations with early stopping
    rather than a hyperparameter search, so this mirrors `train_roster` but skips
    the CV search.

    Args:
        X, y: Training split (y label-encoded for classification, matching the
            sklearn roster so predictions align in the shared leaderboard).
        task (str): 'regression' or 'classification'.
        history (pd.DataFrame): Every feature-complete month (labelled or not),
            date-indexed; the LSTM builds each row's window from the months preceding it.
        lookback (int): Window length for the sequence models.
        class_weight (bool): Balance classes during training (classification).
        random_state (int): Seed.
        progress (callable | None): Called as progress(done, total, name, None).

    Returns:
        dict: Model name -> fitted KerasEstimator.
    """
    roster = neural_models(task, lookback=lookback, class_weight=class_weight, random_state=random_state)

    fitted = {}
    total = len(roster)
    for done, (name, estimator) in enumerate(roster.items(), start=1):
        estimator.fit(X, y, history=history)
        fitted[name] = estimator
        if progress:
            progress(done, total, name, None)
    return fitted
