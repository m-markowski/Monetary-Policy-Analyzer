import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    naive_baseline_rows,
    normalize_probabilities,
    predict_with_thresholds,
    probability_metrics,
    skill_vs_naive,
)


def test_skill_vs_naive_is_one_minus_mse_ratio():
    board = pd.DataFrame({"Test RMSE": [2.0, 4.0]}, index=["Ridge", "Baseline: no change"])

    skill = skill_vs_naive(board)

    assert skill.loc["Ridge", "Test Skill vs naive"] == pytest.approx(0.75)
    assert skill.loc["Baseline: no change", "Test Skill vs naive"] == pytest.approx(0.0)


def test_skill_vs_naive_is_none_without_baseline_row():
    board = pd.DataFrame({"Test RMSE": [2.0]}, index=["Ridge"])

    assert skill_vs_naive(board) is None


def test_no_change_baseline_predicts_zero_forward_change():
    y = pd.Series([1.0, -1.0, 2.0])

    rows = naive_baseline_rows({"Test": (None, y)}, task="regression")

    assert rows.loc["Baseline: no change", "Test RMSE"] == pytest.approx(np.sqrt(2.0))
    assert rows.loc["Baseline: no change", "Test MAE"] == pytest.approx(4.0 / 3.0)


def test_thresholds_rescale_probabilities_before_argmax():
    proba = np.array([[0.6, 0.4], [0.3, 0.7]])
    labels = ["Hold", "Hike"]

    assert predict_with_thresholds(proba, {"Hold": 0.5, "Hike": 0.5}, labels).tolist() == ["Hold", "Hike"]
    assert predict_with_thresholds(proba, {"Hold": 0.9, "Hike": 0.3}, labels).tolist() == ["Hike", "Hike"]


def test_probability_rows_must_sum_to_one():
    with pytest.raises(ValueError):
        normalize_probabilities(np.array([[0.7, 0.7]]))


def test_brier_and_log_loss_of_an_uninformative_forecast():
    y_true = np.array([0, 1])
    uniform = np.full((2, 2), 0.5)

    metrics = probability_metrics(y_true, uniform, labels=[0, 1])

    assert metrics["Brier"] == pytest.approx(0.5)
    assert metrics["Log loss"] == pytest.approx(np.log(2))