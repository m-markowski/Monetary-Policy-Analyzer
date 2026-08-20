import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import get_scorer
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

def selected_features(model, feature_names):
    """
    Feature names surviving the pipeline's in-CV selector, else the originals.

    Args:
        model: A fitted estimator or Pipeline.
        feature_names: Full feature names in matrix order.

    Returns:
        list: The kept feature names (all of them if there is no selector).
    """
    if hasattr(model, "named_steps") and "select" in model.named_steps:
        mask = model.named_steps["select"].get_support()
        return list(np.asarray(feature_names)[mask])
    return list(feature_names)


def model_matrix(model, X):
    """
    X exactly as the final estimator was fitted to see it: the pipeline's
    preprocessing (selection, scaling) applied, in the estimator's native input
    type. Feeding this straight to the estimator keeps DataFrame-vs-NumPy feature
    names consistent with training (no spurious LightGBM/sklearn name warnings).
    Returns X unchanged for a bare estimator.

    Args:
        model: A fitted estimator or Pipeline.
        X (pd.DataFrame): Feature matrix in the original column order.

    Returns:
        The transformed matrix as produced by the pipeline (or X if bare).
    """
    if not hasattr(model, "named_steps"):
        return X
    return model[:-1].transform(X)

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

    Coefficients are averaged over classes for multiclass linear models. When the
    pipeline selected a subset of features, the scores align to the survivors.

    Args:
        model: A fitted estimator or Pipeline.
        feature_names: Full feature column names in matrix order.

    Returns:
        pd.Series | None: 0-100 importances, or None if the model exposes neither
        feature_importances_ nor coef_.
    """
    est = final_estimator(model)
    names = selected_features(model, feature_names)
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    return normalize_importance(values, names)


def permutation_importance_scores(
    model, X, y, scoring=None, n_repeats: int = 10, random_state: int = SEED
) -> pd.Series:
    """
    Permutation importance on a held-out split, scaled to 0-100.

    Model-agnostic, so it covers the estimators without a native importance
    (KNN, SVM, neural). Negative drops are clipped to zero.
    Blend/Stack fall back to a plain shuffle loop because sklearn's implementation requires
    a fit method they do not have.

    Args:
        model: A fitted estimator or Pipeline.
        X, y: Split to measure importance on (use valid or test, never train).
        scoring: sklearn scoring string; the estimator's default if None.
        n_repeats (int): Number of shuffles per feature.
        random_state (int): Seed.

    Returns:
        pd.Series: 0-100 importances, largest first.
    """
    try:
        result = permutation_importance(
            model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        drops = result.importances_mean
    except TypeError:  # sklearn's InvalidParameterError subclasses TypeError
        # Blend/Stack are assembled already fitted and expose no sklearn fit, which
        # permutation_importance insists on; score the same shuffle drops directly.
        scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
        base = scorer(model, X, y)
        rng = np.random.default_rng(random_state)
        X_perm = X.copy()
        drops = np.empty(X.shape[1])
        for j, col in enumerate(X.columns):
            scores = []
            for _ in range(n_repeats):
                X_perm[col] = rng.permutation(X[col].to_numpy())
                scores.append(scorer(model, X_perm, y))
            X_perm[col] = X[col]
            drops[j] = base - np.mean(scores)
    return normalize_importance(np.clip(drops, 0, None), X.columns)


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

    X is sub-sampled for speed and pushed through the pipeline's preprocessing so
    the matrix matches the trained tree exactly (same input type as fit, so no
    feature-name warnings). The returned frame carries the surviving feature names
    for the summary/dependence plots but is never fed back to the estimator.

    Args:
        model: A fitted estimator or Pipeline with a tree final step.
        X (pd.DataFrame): Feature matrix to explain.
        max_samples (int): Cap on rows explained.
        random_state (int): Seed for sub-sampling.

    Returns:
        dict | None: 'shap_values' (array, or list per class for multiclass),
        'features' (the sampled frame, selected columns) and 'expected_value', or
        None if the model is not tree-based or the explainer fails.
    """
    if not is_tree_model(model):
        return None
    sample = X.sample(max_samples, random_state=random_state) if len(X) > max_samples else X
    matrix = model_matrix(model, sample)
    names = selected_features(model, sample.columns)
    try:
        explainer = shap.TreeExplainer(final_estimator(model))
        values = explainer.shap_values(matrix)
    except Exception:
        return None
    features = pd.DataFrame(np.asarray(matrix), index=sample.index, columns=names)
    return {"shap_values": values, "features": features, "expected_value": explainer.expected_value}


def partial_dependence_data(
    model, X, feature: str, grid_resolution: int = 40, task: str | None = None
) -> dict:
    """
    Partial dependence of the model on one feature, averaged over the sample.

    Blend/Stack expose no sklearn `fit`, which `sklearn.inspection.partial_dependence`
    insists on, so they fall back to a manual sweep: the feature is set to each grid
    value across the whole sample and the predictions averaged - the definition of
    partial dependence. The grid spans the 5th-95th percentile, matching sklearn.

    Args:
        model: A fitted estimator or Pipeline.
        X (pd.DataFrame): Feature matrix.
        feature (str): Feature to vary.
        grid_resolution (int): Number of grid points across the feature range.
        task (str | None): 'regression' or 'classification'; tells the manual
            fallback whether to average predict_proba (per class) or predict.

    Returns:
        dict: 'grid' (feature values) and 'average' with shape (n_outputs, n_grid) —
        one row per class for classification, one row for regression.
    """
    try:
        result = partial_dependence(model, X, [feature], grid_resolution=grid_resolution, kind="average")
        grid = result.get("grid_values", result.get("values"))[0]
        return {"grid": np.asarray(grid), "average": np.asarray(result["average"])}
    except TypeError:  # sklearn's InvalidParameterError subclasses TypeError
        lo, hi = np.nanpercentile(X[feature].to_numpy(dtype=float), [5.0, 95.0])
        grid = np.linspace(lo, hi, grid_resolution)
        X_mod = X.copy()
        rows = []
        for value in grid:
            X_mod[feature] = value
            if task == "classification":
                rows.append(np.asarray(model.predict_proba(X_mod), dtype=float).mean(axis=0))
            else:
                rows.append([float(np.mean(model.predict(X_mod)))])
        return {"grid": grid, "average": np.asarray(rows).T}


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
        X_row (pd.DataFrame): A one-row feature frame in the original column order.

    Returns:
        dict | None: 'contributions' (array, or list per class), 'expected_value' and
        'features' (the transformed one-row frame, selected columns), or None if the
        model is not tree-based or the explainer fails.
    """
    if not is_tree_model(model):
        return None
    matrix = model_matrix(model, X_row)
    names = selected_features(model, X_row.columns)
    try:
        explainer = shap.TreeExplainer(final_estimator(model))
        values = explainer.shap_values(matrix)
    except Exception:
        return None
    features = pd.DataFrame(np.asarray(matrix), index=X_row.index, columns=names)
    return {"contributions": values, "expected_value": explainer.expected_value, "features": features}

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
    return pd.Series(np.asarray(values).reshape(-1), index=data["features"].columns)
