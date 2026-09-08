import pandas as pd

# Curated modelling targets per economy. Each maps a display name to its source
# column, `extra_exclude` (features that reconstruct the target from other columns,
# e.g. curve spreads that embed the policy rate, which `leakage_columns`
# stem-matching cannot catch) and `purpose` (the one-line business question the
# target answers, shown in Setup).

CURATED_TARGETS = {
    "usa": {
        "Policy rate": {
            "column": "rate_ff_eff",
            "extra_exclude": ["sprd_5y_ff", "sprd_2y_ff"],
            "purpose": "Where is the Fed heading - the forward rate call as a number",
        },
        "Sticky core inflation": {
            "column": "cpi_sticky_core",
            "extra_exclude": [],
            "purpose": "Will underlying price pressure build or fade - the slow-moving inflation the Fed reacts to.",
        },
        "Unemployment rate": {
            "column": "rate_unemployment",
            "extra_exclude": [],
            "purpose": "Is the labour market loosening - the other half of the Fed's dual mandate.",
        },
        "Nonfarm payrolls": {
            "column": "emp_nonfarm",
            "extra_exclude": [],
            "purpose": "Is job creation accelerating or stalling - the monthly employment print markets trade on.",
        },
        "Real GDP": {
            "column": "gdp_real",
            "extra_exclude": [],
            "purpose": "Is growth holding up - the broadest, quarterly activity gauge.",
        },
        "Yield curve (10y-2y)": {
            "column": "sprd_10y_2y",
            "extra_exclude": ["yld_ust_2y", "yld_ust_10y", "sprd_yld_5y2y", "sprd_yld_10y5y"],
            "purpose": "Will the curve steepen or invert - a market-priced growth/recession "
            "signal; this asks about long-end expectations, not the policy rate itself.",
        },
    },
    "eurozone": {
        "Policy rate": {
            "column": "rate_ecb_dep",
            "extra_exclude": ["sprd_10y_ecb", "psprd_ib_3m_ecb"],
            "purpose": "Where is the ECB heading - the forward rate call as a number",
        },
        "HICP price index": {
            "column": "hicp_all",
            "extra_exclude": [],
            "purpose": "How will the HICP price index change? The forecast is in index points, not an inflation rate.",
        },
        "Real GDP": {
            "column": "gdp_real",
            "extra_exclude": [],
            "purpose": "Is the euro-area economy expanding or contracting - the broadest, quarterly activity gauge.",
        },
        "Yield curve (10y-ECB)": {
            "column": "sprd_10y_ecb",
            "extra_exclude": ["yld_10y_gov", "rate_ecb_dep", "sprd_10y_ib3m", "psprd_ib_3m_ecb"],
            "purpose": "Will the yield curve steepen or flatten relative to the ECB policy rate - a market "
            "signal of changing growth, inflation, and term-premium expectations.",
        },
    },
}


def available_value_targets(df: pd.DataFrame, economy: str) -> dict[str, dict]:
    """
    List the curated regression targets present in the dataset.

    Args:
        df (pd.DataFrame): Monthly modelling frame to check columns against.
        economy (str): Economy identifier ('usa' or 'eurozone').

    Returns:
        dict[str, dict]: Display name -> target spec, keeping only targets whose
        source column exists.
    """
    specs = CURATED_TARGETS.get(economy, {})
    return {name: spec for name, spec in specs.items() if spec["column"] in df.columns}


def policy_rate_column(economy: str) -> str | None:
    """Source column of the economy's policy rate, or None if no curated policy target exists."""
    return CURATED_TARGETS.get(economy, {}).get("Policy rate", {}).get("column")


def deadband_labels(change: pd.Series, deadband: float) -> pd.Series:
    """
    Map a rate change to Hike / Hold / Cut, treating |change| <= deadband as Hold.

    Args:
        change (pd.Series): Rate change per row, NaN where it cannot be computed.
        deadband (float): Minimum absolute change (in points) to count as a move.

    Returns:
        pd.Series: Categorical labels, pd.NA where the change is missing.
    """
    labels = pd.Series("Hold", index=change.index, dtype="object")
    labels[change > deadband] = "Hike"
    labels[change < -deadband] = "Cut"
    labels[change.isna()] = pd.NA
    return labels.astype("category")


def direction_target(
    df: pd.DataFrame,
    economy: str,
    horizon: int,
    deadband: float = 0.125,
) -> pd.Series | None:
    """
    Build the forward policy-rate decision label (Hike / Hold / Cut).

    Compares the policy rate `horizon` rows ahead with today's value; a deadband of
    half a 25bps step keeps sub-step noise as a Hold.

    Args:
        df (pd.DataFrame): Monthly modelling frame containing the policy-rate column.
        economy (str): Economy identifier ('usa' or 'eurozone').
        horizon (int): Number of rows (months on the monthly frame) to look ahead.
        deadband (float): Minimum absolute rate change (in points) to count as a move.

    Returns:
        pd.Series | None: Categorical labels ('Hike', 'Hold', 'Cut'), or None if the
        policy-rate column is unavailable.
    """
    col = policy_rate_column(economy)
    if col is None or col not in df.columns:
        return None
    return deadband_labels(df[col].shift(-horizon) - df[col], deadband).rename("policy_direction")


def momentum_baseline(
    df: pd.DataFrame,
    economy: str,
    horizon: int,
    deadband: float = 0.125,
) -> pd.Series | None:
    """
    Naive trailing-momentum prediction for the direction target.

    Extrapolates the most recent move: the label at month t is the deadband sign of
    the change over the *previous* `horizon` months (rate(t) - rate(t-horizon)), so
    it uses only information available at prediction time - the feasible naive
    counterpart of `direction_target`, which looks forward.

    Args:
        df (pd.DataFrame): Monthly modelling frame containing the policy-rate column.
        economy (str): Economy identifier ('usa' or 'eurozone').
        horizon (int): Number of rows (months on the monthly frame) to look back.
        deadband (float): Minimum absolute rate change (in points) to count as a move.

    Returns:
        pd.Series | None: Categorical labels ('Hike', 'Hold', 'Cut'), or None if the
        policy-rate column is unavailable.
    """
    col = policy_rate_column(economy)
    if col is None or col not in df.columns:
        return None
    return deadband_labels(df[col] - df[col].shift(horizon), deadband).rename("momentum_baseline")


def value_target(df: pd.DataFrame, column: str, horizon: int) -> pd.Series | None:
    """
    Build the forward-change regression target from a curated series.

    The change (value `horizon` rows ahead minus today's) is modelled rather than
    the level, so a trending series does not fake a near-perfect fit.

    Args:
        df (pd.DataFrame): Monthly modelling frame containing the source column.
        column (str): Source column name (from `CURATED_TARGETS`).
        horizon (int): Number of rows (months on the monthly frame) to look ahead.

    Returns:
        pd.Series | None: The forward change named '<column>_fwd_change', or None if
        the column is missing.
    """
    if column not in df.columns:
        return None
    return (df[column].shift(-horizon) - df[column]).rename(f"{column}_fwd_change")
