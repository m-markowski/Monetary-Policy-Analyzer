import pandas as pd

from src.models.targets import direction_target, momentum_baseline, value_target

INDEX = pd.date_range("2024-01-31", periods=5, freq="ME")
RATES = pd.DataFrame({"rate_ff_eff": [5.00, 5.00, 5.25, 5.00, 5.00]}, index=INDEX)


def test_direction_target_looks_forward_and_applies_deadband():
    target = direction_target(RATES, economy="usa", horizon=1, deadband=0.125)

    assert target.iloc[:4].tolist() == ["Hold", "Hike", "Cut", "Hold"]
    assert pd.isna(target.iloc[4])  # the last month has no realised future rate


def test_momentum_baseline_uses_only_past_information():
    baseline = momentum_baseline(RATES, economy="usa", horizon=1, deadband=0.125)

    # The same moves as the target, seen one month later: the feasible naive forecast.
    assert pd.isna(baseline.iloc[0])
    assert baseline.iloc[1:].tolist() == ["Hold", "Hike", "Cut", "Hold"]


def test_direction_target_is_none_without_policy_rate_column():
    frame = pd.DataFrame({"cpi": [1.0, 2.0]}, index=INDEX[:2])

    assert direction_target(frame, economy="usa", horizon=1) is None


def test_value_target_is_forward_change_not_future_level():
    frame = pd.DataFrame({"gdp_real": [100.0, 103.0, 108.0, 110.0]}, index=INDEX[:4])

    target = value_target(frame, "gdp_real", horizon=2)

    assert target.iloc[:2].tolist() == [8.0, 7.0]
    assert target.iloc[2:].isna().all()
    assert target.name == "gdp_real_fwd_change"