import pandas as pd

# Per-economy columns used to derive categorical regimes. Each role maps to the
# friendly column name; None means the economy has no such series
# (e.g. the Sahm recession indicator is USA-only).
REGIME_COLUMNS = {
    "usa": {
        "policy_rate": "rate_ff_eff",
        "curve_spread": "sprd_10y_2y",
        "recession": "ind_sahm_realtime",
    },
    "eurozone": {
        "policy_rate": "rate_ecb_dep",
        "curve_spread": "sprd_10y_ecb",
        "recession": None,
    },
}


def policy_regime(df: pd.DataFrame, economy: str, window: int = 63, deadband: float = 0.125) -> pd.Series | None:
    """
    Label each row by monetary-policy stance over a trailing window.

    The policy rate is compared with its value `window` rows earlier. A deadband
    avoids labelling sub-step noise as a move (0.125 = half a 25bps step, so a
    single hike/cut over the window still registers).

    The default window spans ~63 trading days (~3 months, roughly two FOMC/ECB
    meetings). Spanning more than one meeting keeps a tightening/easing cycle
    coherent when the central bank skips a meeting, at the cost of some lag in
    labelling turning points.

    Args:
        df (pd.DataFrame): Master dataset containing the policy-rate column.
        economy (str): Economy identifier ('usa' or 'eurozone').
        window (int): Number of trailing rows (trading days) used to measure the rate
        change. Defaults to ~2 policy meetings to smooth single-meeting holds into
        one regime; shorten it for tighter tracking at the expense of choppier labels.
        deadband (float): Minimum absolute change (in rate points) to count as a move.

    Returns:
        pd.Series | None: Categorical labels ('Hiking', 'Holding', 'Easing'),
        or None if the policy-rate column is unavailable.
    """
    col = REGIME_COLUMNS.get(economy, {}).get("policy_rate")
    if col is None or col not in df.columns:
        return None
    delta = df[col] - df[col].shift(window)
    labels = pd.Series("Holding", index=df.index, dtype="object")
    labels[delta > deadband] = "Hiking"
    labels[delta < -deadband] = "Easing"
    labels[delta.isna()] = pd.NA
    return labels.astype("category")


def recession_regime(df: pd.DataFrame, economy: str, threshold: float = 0.5) -> pd.Series | None:
    """
    Label each row as recession or expansion from the Sahm real-time indicator.

    Args:
        df (pd.DataFrame): Master dataset containing the recession indicator.
        economy (str): Economy identifier ('usa' only as 'eurozone' has no recession indicator).
        threshold (float): Sahm value at or above which a recession is flagged.

    Returns:
        pd.Series | None: Categorical labels ('Recession', 'Expansion'),
        or None if no recession indicator exists for the economy.
    """
    col = REGIME_COLUMNS.get(economy, {}).get("recession")
    if col is None or col not in df.columns:
        return None
    labels = (df[col] >= threshold).map({True: "Recession", False: "Expansion"})
    labels[df[col].isna()] = pd.NA
    return labels.astype("category")


def curve_state(df: pd.DataFrame, economy: str) -> pd.Series | None:
    """
    Label each row by yield-curve state from a term spread.

    Args:
        df (pd.DataFrame): Master dataset containing the curve-spread column.
        economy (str): Economy identifier ('usa' or 'eurozone').

    Returns:
        pd.Series | None: Categorical labels ('Inverted', 'Normal'),
        or None if the curve-spread column is unavailable.
    """
    col = REGIME_COLUMNS.get(economy, {}).get("curve_spread")
    if col is None or col not in df.columns:
        return None
    labels = pd.Series(pd.NA, index=df.index, dtype="object")
    labels[df[col] < 0] = "Inverted"
    labels[df[col] >= 0] = "Normal"
    return labels.astype("category")


def available_regimes(df: pd.DataFrame, economy: str) -> dict[str, pd.Series]:
    """
    Build every regime label available for the given economy and dataset.

    Args:
        df (pd.DataFrame): Master dataset to label.
        economy (str): Economy identifier ('usa' or 'eurozone').

    Returns:
        dict[str, pd.Series]: Display name -> categorical label series, including
        only regimes whose source columns are present.
    """
    candidates = {
        "Policy regime": policy_regime(df, economy),
        "Recession": recession_regime(df, economy),
        "Curve state": curve_state(df, economy),
    }
    return {name: series for name, series in candidates.items() if series is not None}


def regime_source_columns(df: pd.DataFrame, economy: str) -> list[str]:
    """
    Return the base columns the available regimes are derived from.

    Useful as a default feature set for unsupervised structure analysis, so the
    data-driven clusters can be compared against the rule-based regimes on the same
    inputs.

    Args:
        df (pd.DataFrame): Master dataset to check column availability against.
        economy (str): Economy identifier ('usa' or 'eurozone').

    Returns:
        list[str]: The regime-defining columns present in the dataset.
    """
    roles = REGIME_COLUMNS.get(economy, {})
    return [col for col in roles.values() if col and col in df.columns]
