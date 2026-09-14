import pandas as pd

from src.models.features import build_matrix, leakage_columns
from src.models.targets import value_target


def test_leakage_columns_match_target_stem_only():
    columns = ["rate_ff_eff", "rate_ff_eff_ma3", "rate_ff_effective", "sprd_5y_ff", "cpi"]

    assert leakage_columns(columns, "rate_ff_eff") == ["rate_ff_eff", "rate_ff_eff_ma3"]


def test_build_matrix_drops_target_family_and_keeps_live_anchor():
    index = pd.date_range("2020-01-31", periods=30, freq="ME")
    values = pd.Series(range(30), index=index, dtype=float)
    monthly = pd.DataFrame(
        {"gdp_real": values, "gdp_real_ma3": values, "cpi": values * 2, "rates": values / 2},
        index=index,
    )
    y = value_target(monthly, "gdp_real", horizon=2)

    result = build_matrix(monthly, y, target_column="gdp_real")

    assert list(result["X"].columns) == ["cpi", "rates"]
    assert len(result["X"]) == 28  # the last two months have no realised target
    assert len(result["X_all"]) == 30
    assert result["X_latest"].index[0] > result["X"].index[-1]