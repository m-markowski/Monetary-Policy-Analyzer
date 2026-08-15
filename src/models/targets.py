import pandas as pd

# Curated modelling targets per economy. Each maps a display name to its source
# column, a `positive` flag (strictly-positive series may take a log/Yeo-Johnson
# target transform later; the policy rate and spreads never do) and `extra_exclude`
# (features that reconstruct the target from other columns, e.g. curve spreads that
# embed the policy rate, which `leakage_columns` stem-matching cannot catch).
CURATED_TARGETS = {
    "usa": {
        "Policy rate": {
            "column": "rate_ff_eff",
            "positive": False,
            "extra_exclude": ["sprd_5y_ff", "sprd_2y_ff"],
        },
        "Sticky core inflation": {"column": "cpi_sticky_core", "positive": True, "extra_exclude": []},
        "Unemployment rate": {"column": "rate_unemployment", "positive": True, "extra_exclude": []},
        "Real GDP": {"column": "gdp_real", "positive": True, "extra_exclude": []},
        "Yield curve (10y-2y)": {
            "column": "sprd_10y_2y",
            "positive": False,
            "extra_exclude": ["yld_ust_2y", "yld_ust_10y", "sprd_yld_5y2y", "sprd_yld_10y5y"],
        },
    },
    "eurozone": {
        "Policy rate": {
            "column": "rate_ecb_dep",
            "positive": False,
            "extra_exclude": ["sprd_10y_ecb", "psprd_ib_3m_ecb"],
        },
        "HICP inflation": {"column": "hicp_all", "positive": True, "extra_exclude": []},
        "Real GDP": {"column": "gdp_real", "positive": True, "extra_exclude": []},
        "Yield curve (10y-ECB)": {
            "column": "sprd_10y_ecb",
            "positive": False,
            "extra_exclude": ["yld_10y_gov", "rate_ecb_dep", "sprd_10y_ib3m", "psprd_ib_3m_ecb"],
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


def direction_target(df: pd.DataFrame, economy: str, horizon: int, deadband: float = 0.125) -> pd.Series | None:
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
    col = CURATED_TARGETS.get(economy, {}).get("Policy rate", {}).get("column")
    if col is None or col not in df.columns:
        return None
    forward_change = df[col].shift(-horizon) - df[col]
    labels = pd.Series("Hold", index=df.index, dtype="object")
    labels[forward_change > deadband] = "Hike"
    labels[forward_change < -deadband] = "Cut"
    labels[forward_change.isna()] = pd.NA
    return labels.astype("category").rename("policy_direction")


def value_target(df: pd.DataFrame, column: str, horizon: int, kind: str = "change") -> pd.Series | None:
    """
    Build a forward-looking regression target from a curated series.

    'level' returns the value `horizon` rows ahead; 'change' returns the ahead-minus-
    now difference.

    Args:
        df (pd.DataFrame): Monthly modelling frame containing the source column.
        column (str): Source column name (from `CURATED_TARGETS`).
        horizon (int): Number of rows (months on the monthly frame) to look ahead.
        kind (str): 'level' or 'change'.

    Returns:
        pd.Series | None: The forward target named '<column>_fwd_<kind>', or None if
        the column is missing or `kind` is unrecognised.
    """
    if column not in df.columns:
        return None
    forward = df[column].shift(-horizon)
    if kind == "level":
        target = forward
    elif kind == "change":
        target = forward - df[column]
    else:
        return None
    return target.rename(f"{column}_fwd_{kind}")