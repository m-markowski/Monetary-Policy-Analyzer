import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance
from config.settings import SEED


def final_estimator(model):
    """
    Return the last step of a Pipeline, or the estimator itself if it is bare.

    Args:
        model: A fitted sklearn estimator, Pipeline (final step named 'model'), or
            KerasEstimator.

    Returns:
        The underlying fitted estimator (the Pipeline's 'model' step, or `model` itself).
    """
    return model.named_steps["model"] if hasattr(model, "named_steps") else model

def is_tree_model(model) -> bool:
    """
    Whether the final estimator exposes tree-style feature importances.

    Args:
        model: A fitted estimator or Pipeline.

    Returns:
        bool: True for trees / boosting (usable by `shap.TreeExplainer`).
    """
    return hasattr(final_estimator(model), "feature_importances_")


def normalize_importance(values: np.ndarray, feature_names) -> pd.Series:
    """
    Scale raw importances to 0-100 and sort descending.

    Args:
        values (np.ndarray): Non-negative importance scores.
        feature_names: Index for the resulting Series.

    Returns:
        pd.Series: Importances on a 0-100 scale, largest first.
    """
    values = np.abs(np.asarray(values, dtype=float))
    top = values.max()
    scaled = 100.0 * values / top if top > 0 else values
    return pd.Series(scaled, index=feature_names).sort_values(ascending=False)


def native_importance(model, feature_names) -> pd.Series | None:
    """
    Model-native feature importance (tree importances or |coefficients|).

    Coefficients are averaged over classes for multiclass linear models.

    Args:
        model: A fitted estimator or Pipeline.
        feature_names: Feature column names in matrix order.

    Returns:
        pd.Series | None: 0-100 importances, or None if the model exposes neither
        feature_importances_ nor coef_.
    """
    est = final_estimator(model)
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    return normalize_importance(values, feature_names)


def permutation_importance_scores(
    model, X, y, scoring=None, n_repeats: int = 10, random_state: int = SEED
) -> pd.Series:
    """
    Permutation importance on a held-out split, scaled to 0-100.

    Model-agnostic, so it covers the estimators without a native importance
    (KNN, SVM, neural). Negative drops are clipped to zero.

    Args:
        model: A fitted estimator or Pipeline.
        X, y: Split to measure importance on (use valid or test, never train).
        scoring: sklearn scoring string; the estimator's default if None.
        n_repeats (int): Number of shuffles per feature.
        random_state (int): Seed.

    Returns:
        pd.Series: 0-100 importances, largest first.
    """
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    drops = np.clip(result.importances_mean, 0, None)
    return normalize_importance(drops, X.columns)


def feature_group_importance(importance: pd.Series, group_of) -> pd.Series:
    """
    Aggregate feature importance into business groups (rates/macro/market/...).

    Args:
        importance (pd.Series): Per-feature importance (e.g. from `native_importance`).
        group_of (callable): Maps a feature name to its group label.

    Returns:
        pd.Series: Per-group importance, re-scaled to 0-100, largest first.
    """
    grouped = importance.groupby(importance.index.map(group_of)).sum()
    top = grouped.max()
    if top > 0:
        grouped = 100.0 * grouped / top
    return grouped.sort_values(ascending=False)


def tree_shap(model, X, max_samples: int = 200, random_state: int = SEED) -> dict | None:
    """
    SHAP values for a tree/boosting model via `TreeExplainer`.

    X is sub-sampled for speed; the returned frame is the exact sample the SHAP
    values correspond to (so summary/dependence plots stay aligned).

    Args:
        model: A fitted estimator or Pipeline with a tree final step.
        X (pd.DataFrame): Feature matrix to explain.
        max_samples (int): Cap on rows explained.
        random_state (int): Seed for sub-sampling.

    Returns:
        dict | None: 'shap_values' (array, or list per class for multiclass),
        'features' (the sampled frame) and 'expected_value', or None if the model is
        not tree-based or the explainer fails.
    """
    if not is_tree_model(model):
        return None
    sample = X.sample(max_samples, random_state=random_state) if len(X) > max_samples else X
    try:
        explainer = shap.TreeExplainer(final_estimator(model))
        values = explainer.shap_values(sample)
    except Exception:
        return None
    return {"shap_values": values, "features": sample, "expected_value": explainer.expected_value}


def partial_dependence_data(model, X, feature: str, grid_resolution: int = 40) -> dict:
    """
    Partial dependence of the model on one feature, averaged over the sample.

    Args:
        model: A fitted estimator or Pipeline.
        X (pd.DataFrame): Feature matrix.
        feature (str): Feature to vary.
        grid_resolution (int): Number of grid points across the feature range.

    Returns:
        dict: 'grid' (feature values) and 'average' with shape (n_outputs, n_grid) —
        one row per class for classification, one row for regression.
    """
    result = partial_dependence(model, X, [feature], grid_resolution=grid_resolution, kind="average")
    grid = result.get("grid_values", result.get("values"))[0]
    return {"grid": np.asarray(grid), "average": np.asarray(result["average"])}


def misclassified_index(y_true, y_pred) -> np.ndarray:
    """
    Positional indices of misclassified rows, for local-explanation drill-down.

    Args:
        y_true: Observed labels.
        y_pred: Predicted labels.

    Returns:
        np.ndarray: Positions where prediction != truth.
    """
    return np.where(np.asarray(y_true) != np.asarray(y_pred))[0]


def local_shap(model, X_row: pd.DataFrame) -> dict | None:
    """
    Per-feature SHAP contributions for a single observation (tree models).

    Args:
        model: A fitted estimator or Pipeline with a tree final step.
        X_row (pd.DataFrame): A one-row feature frame.

    Returns:
        dict | None: 'contributions' (array, or list per class) and 'expected_value',
        or None if the model is not tree-based or the explainer fails.
    """
    if not is_tree_model(model):
        return None
    try:
        explainer = shap.TreeExplainer(final_estimator(model))
        values = explainer.shap_values(X_row)
    except Exception:
        return None
    return {"contributions": values, "expected_value": explainer.expected_value}

def shap_summary(shap_data: dict | None, class_index: int | None = None) -> pd.Series | None:
    """
    Global SHAP importance: mean absolute SHAP value per feature, scaled to 0-100.

    Handles both the list-per-class and the stacked-array multiclass outputs of
    `TreeExplainer.shap_values`.

    Args:
        shap_data (dict | None): Output of `tree_shap`.
        class_index (int | None): For multiclass, which class's values to summarise;
            ignored for a single-output (regression / binary) result.

    Returns:
        pd.Series | None: 0-100 mean-|SHAP| importance, largest first, or None.
    """
    if shap_data is None:
        return None
    values = shap_data["shap_values"]
    if isinstance(values, list):
        values = values[class_index or 0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, class_index or 0]
    mean_abs = np.abs(values).mean(axis=0)
    return normalize_importance(mean_abs, shap_data["features"].columns)


def local_shap_series(model, X_row: pd.DataFrame, class_index: int | None = None) -> pd.Series | None:
    """
    Signed per-feature SHAP contributions for one observation, as a Series.

    Args:
        model: A fitted estimator or Pipeline with a tree final step.
        X_row (pd.DataFrame): A one-row feature frame.
        class_index (int | None): For multiclass, which class to explain.

    Returns:
        pd.Series | None: Signed contributions indexed by feature, or None if the
        model is not tree-based.
    """
    data = local_shap(model, X_row)
    if data is None:
        return None
    values = data["contributions"]
    if isinstance(values, list):
        values = values[class_index or 0]
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, class_index or 0]
    return pd.Series(np.asarray(values).reshape(-1), index=X_row.columns)
