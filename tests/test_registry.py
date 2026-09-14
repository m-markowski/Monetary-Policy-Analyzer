from src.models.registry import build_metadata, cache_key, is_stale, load_artifacts, save_artifacts, slugify

SIGNATURE = {"rows": 100, "cols": 5, "columns": ["a"], "x_hash": 1, "y_hash": 2, "last_date": "2025-12-31"}


def test_cache_key_is_a_stable_filesystem_safe_slug():
    assert slugify("Policy rate (USA) / h3") == "policy-rate-usa-h3"
    assert cache_key("usa", "classification", "Policy rate", 3, "70/15/15") == (
        "usa_classification_policy-rate_h3_70-15-15"
    )


def test_is_stale_detects_changed_data_signature():
    metadata = {"signature": SIGNATURE}

    assert not is_stale(metadata, SIGNATURE)
    assert is_stale(metadata, {**SIGNATURE, "rows": 101})


def test_artifacts_round_trip_and_reject_stale_data(tmp_path):
    metadata = build_metadata(
        economy="usa",
        task="regression",
        target="Real GDP",
        horizon=1,
        split="70/15/15",
        scoring="RMSE",
        best="Ridge",
        cv_scores={"Ridge": -0.5},
        params={"Ridge": {"model__alpha": 1.0}},
        leaderboard=None,
        signature=SIGNATURE,
    )
    key = cache_key("usa", "regression", "Real GDP", 1, "70/15/15")

    save_artifacts({"Ridge": {"coef": [1.0, 2.0]}}, metadata, key, directory=tmp_path)

    loaded = load_artifacts(key, tmp_path, signature=SIGNATURE)
    assert loaded is not None
    assert loaded["models"]["Ridge"] == {"coef": [1.0, 2.0]}
    assert load_artifacts(key, tmp_path, signature={**SIGNATURE, "rows": 101}) is None