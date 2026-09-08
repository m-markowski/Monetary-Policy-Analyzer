import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, root_mean_squared_error

from config.settings import SEED
from src.models.evaluate import normalize_probabilities


def stack_meta_features(models: dict, X, task: str) -> np.ndarray:
    """
    Build the meta-feature matrix from the base models' outputs.

    Classification stacks each model's class probabilities; regression stacks each
    model's point prediction as one column.

    Args:
        models (dict): Base model name -> fitted estimator.
        X: Feature matrix to run the base models on.
        task (str): 'regression' or 'classification'.

    Returns:
        np.ndarray: Horizontally concatenated base-model outputs.
    """
    parts = []
    for model in models.values():
        if task == "classification":
            parts.append(model.predict_proba(X))
        else:
            parts.append(np.asarray(model.predict(X)).reshape(-1, 1))
    return np.hstack(parts)


class BlendEnsemble:
    """
    Weighted average of base models, weights proportional to inverse valid error.

    Classification averages probabilities; regression averages point predictions.
    """

    def __init__(self, models: dict, weights: dict, task: str):
        self.models = models
        self.weights = weights
        self.task = task

    def predict_proba(self, X) -> np.ndarray:
        acc = None
        for name, model in self.models.items():
            weighted = model.predict_proba(X) * self.weights[name]
            acc = weighted if acc is None else acc + weighted
        return acc

    def predict(self, X) -> np.ndarray:
        if self.task == "classification":
            return np.argmax(self.predict_proba(X), axis=1)
        acc = None
        for name, model in self.models.items():
            weighted = np.asarray(model.predict(X)) * self.weights[name]
            acc = weighted if acc is None else acc + weighted
        return acc


class StackEnsemble:
    """
    Meta-model trained on the base models' validation-set predictions.

    The base models were fit on train; the meta-model is fit on their valid-set
    outputs, so the test set stays untouched until final evaluation.
    """

    def __init__(self, models: dict, meta_model, task: str, n_classes: int | None = None):
        self.models = models
        self.meta_model = meta_model
        self.task = task
        self.n_classes = n_classes

    def predict_proba(self, X) -> np.ndarray:
        proba = self.meta_model.predict_proba(stack_meta_features(self.models, X, self.task))
        if self.n_classes and proba.shape[1] != self.n_classes:
            full = np.zeros((proba.shape[0], self.n_classes))
            full[:, self.meta_model.classes_.astype(int)] = proba
            return full
        return proba

    def predict(self, X) -> np.ndarray:
        return self.meta_model.predict(stack_meta_features(self.models, X, self.task))


def blend_weights(models: dict, X_valid, y_valid, task: str, labels=None) -> dict:
    """
    Compute inverse-error blending weights on the validation set.

    Weight is proportional to 1 / error (RMSE for regression, log loss for
    classification), normalised to sum to one.

    Args:
        models (dict): Base model name -> fitted estimator.
        X_valid, y_valid: Validation split.
        task (str): 'regression' or 'classification'.
        labels (list | None): Ordered label set (classification only).

    Returns:
        dict: Model name -> weight.
    """
    inv = {}
    for name, model in models.items():
        if task == "regression":
            error = root_mean_squared_error(y_valid, model.predict(X_valid))
        else:
            error = log_loss(y_valid, normalize_probabilities(model.predict_proba(X_valid)), labels=labels)
        inv[name] = 1.0 / max(error, 1e-9)
    total = sum(inv.values())
    return {name: weight / total for name, weight in inv.items()}


def build_stack(models: dict, X_valid, y_valid, task: str, n_classes=None, random_state: int = SEED):
    """
    Fit the stacking meta-model on validation-set base predictions.

    Args:
        models (dict): Base model name -> fitted estimator.
        X_valid, y_valid: Validation split.
        task (str): 'regression' or 'classification'.
        n_classes (int | None): Full class count, for probability alignment.
        random_state (int): Seed for the meta-model.

    Returns:
        StackEnsemble: The fitted stacking ensemble.
    """
    meta_X = stack_meta_features(models, X_valid, task)
    if task == "classification":
        meta = LogisticRegression(max_iter=5000, random_state=random_state)
    else:
        meta = Ridge(random_state=random_state)
    meta.fit(meta_X, y_valid)
    return StackEnsemble(models, meta, task, n_classes=n_classes)


def build_ensembles(
    models: dict,
    splits: dict,
    task: str,
    labels=None,
    random_state: int = SEED,
) -> dict:
    """
    Assemble the blend and stack ensembles from the fitted base roster.

    Args:
        models (dict): Base model name -> fitted estimator.
        splits (dict): 'Train'/'Valid'/'Test' -> (X, y); only 'Valid' is used here.
        task (str): 'regression' or 'classification'.
        labels (list | None): Ordered label set (classification only).
        random_state (int): Seed for the stacking meta-model.

    Returns:
        dict: 'ensembles' ('Blend'/'Stack' -> estimator), 'weights' (blend weights)
        and 'members' (base models used).
    """
    X_valid, y_valid = splits["Valid"]

    weights = blend_weights(models, X_valid, y_valid, task, labels=labels)
    n_classes = len(labels) if labels is not None else None
    ensembles = {"Blend": BlendEnsemble(models, weights, task)}
    skip_reason = None
    if task == "classification" and len(np.unique(y_valid)) < 2:
        skip_reason = "Stack omitted: the Dev split contains only one class."
    else:
        ensembles["Stack"] = build_stack(models, X_valid, y_valid, task, n_classes=n_classes, random_state=random_state)
    return {"ensembles": ensembles, "weights": weights, "members": list(models), "skip_reason": skip_reason}
