import numpy as np
import pandas as pd


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame | None:
    """
    Collapse the daily master to a month-end snapshot for modelling.

    The daily master is the single source of truth and is never modified; modelling
    runs on this derived view. Levels and rates take their month-end value; daily
    engineered features (returns, moving averages, realized vols) are sampled at
    month-end rather than recomputed.

    Args:
        daily (pd.DataFrame): Date-indexed daily master dataset.

    Returns:
        pd.DataFrame | None: Month-end-indexed frame, or None if the index is not a
        DatetimeIndex.
    """
    if not isinstance(daily.index, pd.DatetimeIndex):
        return None
    return daily.resample("ME").last()


def leakage_columns(columns: list[str], target_column: str) -> list[str]:
    """
    Find columns that trivially reveal the target's own series.

    Returns the target's base column and any engineered child sharing its stem
    (moving averages, returns, vols, changes), which are near-deterministic
    functions of the target and would leak its contemporaneous value into X.

    Args:
        columns (list[str]): Candidate feature column names.
        target_column (str): The target's source column.

    Returns:
        list[str]: Columns to exclude from the feature matrix.
    """
    return [c for c in columns if c == target_column or c.startswith(f"{target_column}_")]


def build_matrix(
    monthly: pd.DataFrame,
    y: pd.Series,
    target_column: str,
    extra_exclude: list[str] | None = None,
    target_lags: tuple[int, ...] = (),
    min_rows: int = 24,
) -> dict | None:
    """
    Assemble the aligned feature matrix and target for one modelling run.

    Keeps every numeric feature except the target's own family (see
    `leakage_columns`) and any caller-supplied extras (e.g. spreads that embed the
    target as a component), aligns X to y on the shared month-end index, and drops
    rows with any missing value so training sees complete cases only.

    Args:
        monthly (pd.DataFrame): Month-end modelling frame from `to_monthly`.
        y (pd.Series): Target series (from `targets.py`), month-end-indexed.
        target_column (str): The target's source column, used for leakage exclusion.
        extra_exclude (list[str] | None): Further columns to drop from X.
        target_lags (tuple[int, ...]): Lags of the target's own series (in months) to
            add back as autoregressive features.
        min_rows (int): Minimum complete rows required to return a usable matrix.

    Returns:
        dict | None: 'X' (dates x features), 'y' (aligned target) and 'X_latest'
        (the most recent complete feature row, possibly unlabelled), or None if
        fewer than `min_rows` complete rows remain.
    """
    exclude = set(leakage_columns(list(monthly.columns), target_column))
    if extra_exclude:
        exclude.update(extra_exclude)
    features = monthly.drop(columns=[c for c in exclude if c in monthly.columns]).select_dtypes("number")
    # pct_change can produce ±inf (e.g. divide-by-zero); dropna alone keeps them and
    # StandardScaler / PowerTransformer then fail
    features = features.replace([np.inf, -np.inf], np.nan)

    if target_lags and target_column in monthly.columns:
        base = monthly[target_column]
        for lag in target_lags:
            features[f"{target_column}_lag{lag}"] = base.shift(lag)

    complete = features.dropna()
    combined = complete.join(y.rename("__target__"), how="inner").dropna()
    if combined.shape[0] < min_rows:
        return None
    target = combined.pop("__target__")
    # The last complete feature row may be newer than the last labelled row (the
    # forward target is unknown for the final `horizon` months); it is the natural
    # anchor for a live scenario prediction.
    return {"X": combined, "y": target, "X_latest": complete.iloc[[-1]]}
