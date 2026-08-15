import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

def roc_auc_ovr_scorer(estimator, X, y) -> float:
    """
    Fold-safe macro one-vs-rest ROC-AUC scorer for TimeSeriesSplit CV.

    Early expanding folds can train on a subset of classes (so `predict_proba`
    returns fewer columns than the full label set) and a validation fold can miss a
    class entirely. Probabilities are aligned to the estimator's fitted classes and
    the AUC is averaged only over classes present in `y`, so a degenerate fold no
    longer collapses the whole CV score to nan. A single-class fold, where OvR AUC
    is undefined, contributes the chance value 0.5; every model sees the same folds
    so this constant does not bias selection.

    Args:
        estimator: Fitted classifier exposing `predict_proba` and `classes_`.
        X: Validation-fold features.
        y: Validation-fold labels (encoded integers, matching training).

    Returns:
        float: Macro OvR ROC-AUC over the present classes, or 0.5 if undefined.
    """
    y = np.asarray(y)
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    aucs = []
    for col, cls in enumerate(estimator.classes_):
        binary = (y == cls).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        aucs.append(roc_auc_score(binary, proba[:, col]))
    return float(np.mean(aucs)) if aucs else 0.5

REGRESSION_METRICS = ("RMSE", "MAE", "R2")
CLASSIFICATION_METRICS = ("ROC-AUC (macro/OvR)", "F1-macro", "Balanced accuracy")

# Maps a display metric to the sklearn scoring string used for CV / model selection.
# The neg_ scorers are already sign-aligned so a higher score is always better.
CV_SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
    "ROC-AUC (macro/OvR)": roc_auc_ovr_scorer,
    "F1-macro": "f1_macro",
    "Balanced accuracy": "balanced_accuracy",
}


def regression_metrics(y_true, y_pred) -> dict:
    """
    Compute the curated regression metrics for one set of predictions.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.

    Returns:
        dict: RMSE, MAE and R2 keyed by their display names.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, labels) -> dict:
    """
    Compute imbalance-aware classification metrics and per-class breakdowns.

    Args:
        y_true: Observed class labels.
        y_pred: Predicted class labels.
        labels (list): Full ordered label set (so absent classes still appear).

    Returns:
        dict: Accuracy, balanced accuracy, F1-macro, a per-class DataFrame
        (precision/recall/F1/support) and the confusion matrix as a DataFrame.
    """
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = pd.DataFrame(
        {"Precision": prec, "Recall": rec, "F1": f1, "Support": support}, index=labels
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1-macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion": pd.DataFrame(cm, index=labels, columns=labels),
    }


def roc_auc_macro_ovr(y_true, y_proba, labels) -> float | None:
    """
    Macro one-vs-rest ROC-AUC, handling the binary and multiclass cases.

    Args:
        y_true: Observed class labels.
        y_proba: Predicted class probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.

    Returns:
        float | None: Macro OvR ROC-AUC, or None if it is undefined (e.g. a class
        is absent from y_true).
    """
    y_proba = np.asarray(y_proba, dtype=float)
    try:
        if len(labels) == 2:
            binary = (np.asarray(y_true) == labels[1]).astype(int)
            return float(roc_auc_score(binary, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def probability_metrics(y_true, y_proba, labels) -> dict:
    """
    Compute calibration-style scores (log loss and multiclass Brier).

    The multiclass Brier score is the mean squared error between the one-hot
    encoded truth and the predicted probabilities, summed over classes.

    Args:
        y_true: Observed class labels.
        y_proba: Predicted class probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.

    Returns:
        dict: 'Log loss' (None if undefined) and 'Brier'.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    try:
        ll = float(log_loss(y_true, y_proba, labels=labels))
    except ValueError:
        ll = None
    onehot = np.column_stack([(y_true == lab).astype(float) for lab in labels])
    brier = float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))
    return {"Log loss": ll, "Brier": brier}


def youden_thresholds(y_true, y_proba, labels) -> dict:
    """
    Per-class one-vs-rest Youden's J thresholds, computed on the validation set.

    Kept separate from test so the operating point is chosen leakage-free.

    Args:
        y_true: Validation-set class labels.
        y_proba: Validation-set class probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.

    Returns:
        dict: Label -> probability threshold maximising TPR - FPR (0.5 when a
        class is degenerate in the validation fold).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    thresholds = {}
    for j, lab in enumerate(labels):
        binary = (y_true == lab).astype(int)
        if binary.sum() in (0, len(binary)):
            thresholds[lab] = 0.5
            continue
        fpr, tpr, thr = roc_curve(binary, y_proba[:, j])
        thresholds[lab] = float(thr[int(np.argmax(tpr - fpr))])
    return thresholds


def predict_with_thresholds(y_proba, thresholds, labels) -> np.ndarray:
    """
    Turn probabilities into class predictions using per-class OvR thresholds.

    Each column is scaled by its threshold and the argmax is taken, which
    reconciles independent OvR cut-offs into a single multiclass decision.

    Args:
        y_proba: Predicted class probabilities (n_samples x n_classes).
        thresholds (dict): Label -> threshold from `youden_thresholds`.
        labels (list): Ordered label set matching the probability columns.

    Returns:
        np.ndarray: Predicted labels.
    """
    y_proba = np.asarray(y_proba, dtype=float)
    thr = np.array([max(thresholds.get(lab, 0.5), 1e-6) for lab in labels], dtype=float)
    idx = np.argmax(y_proba / thr, axis=1) # scale to return one prediction for one sample only
    labels = np.asarray(labels)
    return labels[idx]


def roc_curve_data(y_true, y_proba, labels) -> dict:
    """
    Per-class OvR ROC curve coordinates for plotting.

    Args:
        y_true: Observed class labels.
        y_proba: Predicted class probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.

    Returns:
        dict: Label -> {'fpr', 'tpr', 'thresholds', 'auc'}.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    out = {}
    for j, lab in enumerate(labels):
        binary = (y_true == lab).astype(int)
        if binary.sum() in (0, len(binary)): # cannot compute for either lab count = 0 or lab count = len(binary)
            continue
        fpr, tpr, thr = roc_curve(binary, y_proba[:, j])
        out[lab] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thr,
            "auc": float(roc_auc_score(binary, y_proba[:, j])),
        }
    return out


def pr_curve_data(y_true, y_proba, labels) -> dict:
    """
    Per-class OvR precision-recall coordinates and average precision.

    Args:
        y_true: Observed class labels.
        y_proba: Predicted class probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.

    Returns:
        dict: Label -> Label -> {'recall', 'precision', 'ap', 'baseline'}.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    out = {}
    for j, lab in enumerate(labels):
        binary = (y_true == lab).astype(int)
        if binary.sum() in (0, len(binary)):
            continue
        precision, recall, _ = precision_recall_curve(binary, y_proba[:, j])
        out[lab] = {
            "recall": recall,
            "precision": precision,
            "ap": float(average_precision_score(binary, y_proba[:, j])),
            "baseline": float(binary.mean()),
        }
    return out


def lift_curve_data(y_true, y_proba, labels) -> dict:
    """
    Per-class OvR cumulative lift coordinates for plotting.

    Lift at a population fraction is the positive rate among the top-scored
    samples divided by the overall positive rate.

    Args:
        y_true: Observed class labels.
        y_proba: Predicted class probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.

    Returns:
        dict: Label -> {'fraction', 'lift'}.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    ranks = np.arange(1, n + 1)
    out = {}
    for j, lab in enumerate(labels):
        binary = (y_true == lab).astype(int)
        base = binary.mean()
        if base == 0:
            continue
        order = np.argsort(y_proba[:, j])[::-1]
        cum_pos = np.cumsum(binary[order])
        out[lab] = {"fraction": ranks / n, "lift": (cum_pos / ranks) / base}
    return out


def build_leaderboard(models, splits, task, labels=None, thresholds=None) -> pd.DataFrame:
    """
    Score every fitted model across the train/valid/test splits.

    Args:
        models (dict): Model name -> fitted estimator (or Pipeline).
        splits (dict): Split name ('Train'/'Valid'/'Test') -> (X, y).
        task (str): 'regression' or 'classification'.
        labels (list | None): Ordered label set (classification only).
        thresholds (dict | None): Per-class thresholds from `youden_thresholds`;
            when given, classification predictions use them instead of argmax.

    Returns:
        pd.DataFrame: One row per model, columns '<Split> <Metric>', indexed by
        model name.
    """
    records = []
    for name, model in models.items():
        row = {"Model": name}
        for split_name, (X, y) in splits.items():
            if task == "regression":
                for metric, value in regression_metrics(y, model.predict(X)).items():
                    row[f"{split_name} {metric}"] = value
            else:
                proba = model.predict_proba(X)
                if thresholds is not None:
                    pred = predict_with_thresholds(proba, thresholds, labels)
                else:
                    pred = model.predict(X)
                metrics = classification_metrics(y, pred, labels)
                row[f"{split_name} ROC-AUC (macro/OvR)"] = roc_auc_macro_ovr(y, proba, labels)
                row[f"{split_name} F1-macro"] = metrics["F1-macro"]
                row[f"{split_name} Balanced accuracy"] = metrics["Balanced accuracy"]
        records.append(row)
    return pd.DataFrame(records).set_index("Model")
