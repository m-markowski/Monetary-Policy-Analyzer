import logging
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.dataset_builder import ECONOMIES, cache_exists, load_master, master_mtime
from src.models import econometrics, ensemble, evaluate, explain, features, pipeline, registry, targets
from src.models import neural as neural_mod
from utils import interpret, plots

st.set_page_config(page_title="Modelling and prediction", layout="wide")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Page-switching in Streamlit clears the session state, so we reassign any model artifacts to keep them alive
for k in list(st.session_state):
    if str(k).startswith("mdl_") and k not in ("mdl_train", "mdl_scn_revert"):
        st.session_state[k] = st.session_state[k]


@st.cache_data(show_spinner=False)
def get_master(economy: str, mtime: float) -> pd.DataFrame:
    """Cached read of a master dataset, date-indexed; invalidated when the file changes."""

    return load_master(economy).set_index("date").sort_index()


@st.cache_data(show_spinner=False)
def get_monthly(economy: str, mtime: float) -> pd.DataFrame:
    """Month-end derived modelling view of the daily master (the single source of truth)."""
    return features.to_monthly(get_master(economy, mtime))


@st.cache_data(show_spinner=False)
def get_monthly_complete(economy: str, mtime: float) -> pd.DataFrame:
    """Monthly view with a partial final month dropped, for the forecast models."""
    monthly = get_monthly(economy, mtime)
    last = get_master(economy, mtime).index[-1]
    if last != last + pd.offsets.BMonthEnd(0):
        monthly = monthly.iloc[:-1]
    return monthly


# Months held out of the order search and used to score the forecast models.
FORECAST_HOLDOUT = 12


@st.cache_data(show_spinner=False)
def arima_search(economy: str, series_col: str, mtime: float) -> dict | None:
    """Cached ARIMA order search on the history before the holdout, so the holdout never informs the order."""
    series = get_monthly_complete(economy, mtime)[series_col].dropna()
    return econometrics.arima_order_search(series.iloc[:-FORECAST_HOLDOUT])


@st.cache_data(show_spinner=False)
def garch_search(economy: str, series_col: str, mtime: float) -> dict | None:
    """Cached GARCH order search on the pre-holdout monthly changes."""
    change = get_monthly_complete(economy, mtime)[series_col].diff().dropna()
    return econometrics.garch_order_search(change.iloc[:-FORECAST_HOLDOUT])


@st.cache_data(show_spinner=False)
def arima_accuracy(economy: str, series_col: str, order: tuple, seasonal_order: tuple, mtime: float) -> dict | None:
    """Cached holdout accuracy: fit on the pre-holdout history, forecast the held-out months."""
    series = get_monthly_complete(economy, mtime)[series_col].dropna()
    return econometrics.arima_backtest(series, order=order, seasonal_order=seasonal_order, holdout=FORECAST_HOLDOUT)


@st.cache_data(show_spinner=False)
def garch_accuracy(economy: str, series_col: str, order: tuple, mtime: float) -> dict | None:
    """Cached holdout volatility accuracy for one GARCH specification."""
    change = get_monthly_complete(economy, mtime)[series_col].diff().dropna()
    return econometrics.garch_backtest(change, p=order[0], q=order[1], holdout=FORECAST_HOLDOUT)


@st.cache_resource(show_spinner=False)
def arima_fit(economy: str, series_col: str, order: tuple, seasonal_order: tuple, mtime: float):
    """Cached full-history ARIMA fit for the displayed forecast."""
    series = get_monthly_complete(economy, mtime)[series_col].dropna()
    return econometrics.fit_arima(series, order=order, seasonal_order=seasonal_order)


@st.cache_resource(show_spinner=False)
def garch_fit(economy: str, series_col: str, order: tuple, mtime: float):
    """Cached full-history GARCH fit for the displayed volatility forecast."""
    change = get_monthly_complete(economy, mtime)[series_col].diff().dropna()
    return econometrics.fit_garch(change, p=order[0], q=order[1])


def build_task_data(monthly: pd.DataFrame, economy: str, task: str, spec: dict | None, horizon: int) -> dict | None:
    """Assemble the leakage-guarded feature matrix and target for one run, or None."""
    if task == "classification":
        spec = targets.CURATED_TARGETS.get(economy, {}).get("Policy rate")
        y = targets.direction_target(monthly, economy, horizon)
    else:
        y = targets.value_target(monthly, spec["column"], horizon) if spec else None
    if y is None or spec is None:
        return None
    data = features.build_matrix(
        monthly, y, spec["column"], extra_exclude=spec["extra_exclude"], target_lags=TARGET_LAGS
    )
    if data is None:
        return None
    data["target_column"] = spec["column"]
    return data


MACRO_RATE_COLUMNS = ("rate_unemployment", "rate_participation", "rate_savings", "rate_homeownership")
ENGINEERED_SUFFIX = ("_ret_", "_ma_", "_vol_", "_chg_")

# Lagged target levels observed at or before t. They are legitimate
# autoregressive inputs for a forward-change target.
TARGET_LAGS = (1, 3, 6)


def group_of(feature: str, target_column: str) -> str:
    """Map a modelling feature (including engineered children) to a business block."""
    if feature.startswith(f"{target_column}_lag"):
        return "Target lags"
    if feature.startswith(MACRO_RATE_COLUMNS):
        return "Macro"
    if feature.startswith(("rate_", "yld_", "sprd_")):
        return "Rates"
    if feature.startswith(("eq_", "fx_", "cmd_", "idx_")):
        return "Market"
    return "Macro"


def is_base_feature(feature: str) -> bool:
    """True for base columns and engineered spreads; False for lags, returns, MAs and vols."""
    return not any(s in feature for s in ENGINEERED_SUFFIX) and "_lag" not in feature


def significance_colour(p: float) -> str:
    """Green text for a coefficient significant at p < 0.05, red otherwise (for Styler.map)."""
    if pd.isna(p):
        return ""
    return "color: #2ca02c" if p < 0.05 else "color: #d62728"


def split_label(preset: str) -> str:
    """Human-readable train/dev/test percentages for a split-preset key."""
    train_frac, valid_frac = pipeline.SPLIT_PRESETS[preset]
    return f"{train_frac:.0%} train / {valid_frac:.0%} dev / {1 - train_frac - valid_frac:.0%} test"


def format_param(value) -> str:
    """Compact display of one tuned hyperparameter (floats to 4 significant figures)."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def show_figure(fig, caption: str | None = None, caption_help: str | None = None) -> None:
    """Render a Plotly figure with an optional caption, skipping None figures."""
    if fig is None:
        st.caption("Not enough data to draw this chart for the current selection.")
        return
    st.plotly_chart(fig, width="stretch")
    if caption:
        st.caption(caption, help=caption_help)


def assemble_bundle(
    models,
    X,
    y,
    task,
    cfg,
    *,
    monthly_data: pd.DataFrame,
    X_latest: pd.DataFrame,
    cv_scores,
    params,
    classes,
    neural_names,
    sig,
    ensemble_info=None,
):
    """
    Score already-fitted models across the chronological splits and package the
    session bundle the tabs consume. No training happens here, so it serves both a
    fresh run and a reload of saved artifacts.
    """
    split_preset, metric = cfg["split_preset"], cfg["metric"]
    splits = pipeline.chronological_split(X, y, preset=split_preset, gap=cfg["horizon"])
    if task == "classification":
        labels = list(range(len(classes)))
        code = {label: i for i, label in enumerate(classes)}
        split_eval = {name: (Xs, np.array([code[v] for v in ys])) for name, (Xs, ys) in splits.items()}
    else:
        labels = None
        split_eval = splits
    board = evaluate.build_leaderboard(models, split_eval, task, labels=labels)
    # Blend/Stack are fit on the dev split, so their dev score is in-sample and would
    # unfairly win a dev-based pick; the winner badge is chosen among base models only.
    selectable = board.drop(index=[n for n in ("Blend", "Stack") if n in board.index])
    valid = selectable[f"Valid {metric}"].dropna()
    if valid.empty:
        best = None
    elif metric in ("RMSE", "MAE"):
        best = valid.idxmin()
    else:
        best = valid.idxmax()
    # Naive baselines join the board after the winner is picked, so they can never take
    # the badge; they are the honest-skill yardstick the real models must beat.
    momentum = None
    if task == "classification":
        momentum = targets.momentum_baseline(
            monthly_data,
            cfg["economy"],
            cfg["horizon"],
        )
        if momentum is not None:
            momentum = momentum.map(code)
    board = pd.concat([board, evaluate.naive_baseline_rows(split_eval, task, labels=labels, momentum=momentum)])
    skill = evaluate.skill_vs_naive(board) if task == "regression" else None
    return {
        "sig": sig,
        "X": X,
        "y": y,
        "X_latest": X_latest,
        "splits": splits,
        "split_eval": split_eval,
        "labels": labels,
        "classes": classes,
        "models": models,
        "neural_names": set(neural_names),
        "cv_scores": cv_scores,
        "params": params,
        "board": board,
        "best": best,
        "skill": skill,
        "ensemble_info": ensemble_info,
        **cfg,
    }


def save_bundle_artifacts(bundle: dict) -> None:
    """Persist the fitted models and metadata sidecar so the bundle auto-reloads."""
    signature = registry.data_signature(bundle["X"], bundle["y"])
    meta = registry.build_metadata(
        economy=bundle["economy"],
        task=bundle["task"],
        target=bundle["target_name"],
        horizon=bundle["horizon"],
        split=bundle["split_preset"],
        scoring=bundle["metric"],
        best=bundle["best"],
        cv_scores=bundle["cv_scores"],
        params=bundle["params"],
        leaderboard=bundle["board"],
        signature=signature,
        extra={
            "classes": bundle["classes"],
            "ensemble_info": bundle.get("ensemble_info"),
            "neural_names": sorted(bundle["neural_names"]),
        },
    )
    key = registry.cache_key(
        bundle["economy"], bundle["task"], bundle["target_name"], bundle["horizon"], bundle["split_preset"]
    )
    registry.save_artifacts(bundle["models"], meta, key)


st.title("Modelling and prediction")

if not cache_exists():
    st.info("No cached data yet. Build it on the main page first.")
    st.stop()

tab_setup, tab_train, tab_diag, tab_scenario, tab_forecast = st.tabs(
    [
        "Setup",
        "Leaderboard & training",
        "Diagnostics & explainability",
        "Scenario",
        "Forecast (time series)",
    ]
)

with tab_setup:
    economy = st.selectbox("Economy", ECONOMIES, format_func=str.upper, key="mdl_economy")
    mtime = master_mtime(economy)
    monthly = get_monthly(economy, mtime)
    value_targets = targets.available_value_targets(monthly, economy)

    if value_targets and st.session_state.get("mdl_val_target") not in value_targets:
        st.session_state["mdl_val_target"] = next(iter(value_targets))

    st.caption("Training and diagnostics always use the full available data range for this economy. ")
    cfg = st.columns([2, 3])
    task_label = cfg[0].radio("Task", ["Direction (Hike/Hold/Cut)", "Value (regression)"], key="mdl_task")
    task = "classification" if task_label.startswith("Direction") else "regression"

    spec = None
    if task == "classification":
        cfg[1].selectbox("Target", ["Forward policy-rate decision"], disabled=True, key="mdl_dir_target")
        target_name = "Policy direction"
    else:
        if not value_targets:
            st.warning("No curated regression targets are available for this economy.")
            target_name = None
        else:
            target_name = cfg[1].selectbox(
                "Target",
                list(value_targets),
                format_func=lambda name: f"{name} ({value_targets[name]['column']})",
                key="mdl_val_target",
                help=interpret.TARGET_HELP,
            )
            spec = value_targets[target_name]
            cfg[1].caption(spec["purpose"])

    st.session_state.setdefault("mdl_horizon", 3)
    horizon = st.slider("Prediction horizon (months ahead)", 1, 12, key="mdl_horizon", help=interpret.HORIZON_HELP)

    st.markdown("**Split, scoring and search**")
    row = st.columns(3)
    st.session_state.setdefault("mdl_split", list(pipeline.SPLIT_PRESETS)[1])
    split_preset = row[0].selectbox(
        "Split preset",
        list(pipeline.SPLIT_PRESETS),
        format_func=split_label,
        key="mdl_split",
        help=interpret.SPLIT_HELP,
    )
    metrics = evaluate.CLASSIFICATION_METRICS if task == "classification" else evaluate.REGRESSION_METRICS
    if st.session_state.get("mdl_metric") not in metrics:
        st.session_state["mdl_metric"] = next(iter(metrics))
    metric = row[1].selectbox(
        "Scoring metric",
        list(metrics),
        key="mdl_metric",
        help=interpret.METRIC_HELP,
    )
    budget = row[2].selectbox(
        "Training budget", list(pipeline.BUDGET_TIERS), key="mdl_budget", help=interpret.BUDGET_HELP
    )

    st.session_state.setdefault("mdl_neural", False)
    include_neural = st.toggle(
        "Include neural nets (LSTM / MLP - slower, opt-in)",
        key="mdl_neural",
        help=interpret.NEURAL_HELP,
    )

    with st.expander("Why monthly, and how splitting / CV work?"):
        st.markdown(interpret.MONTHLY_RATIONALE)
        st.markdown(interpret.CV_HELP)

    data = build_task_data(monthly, economy, task, spec, horizon)
    if data is None:
        st.warning(
            "Could not build a modelling matrix for this configuration (missing target column "
            "or too few complete months after the horizon shift). Adjust the target or horizon."
        )
        X = y = target_column = None
    else:
        X, y, target_column = data["X"], data["y"], data["target_column"]
        st.markdown("**The modelling data**")
        span = f"{X.index.min():%Y-%m} to {X.index.max():%Y-%m}"
        st.caption(f"{X.shape[0]} monthly rows, {X.shape[1]} features, {span}.")
        n_months, lag_cost = len(monthly), max(TARGET_LAGS)
        row_gap = n_months - lag_cost - horizon - X.shape[0]
        st.caption(
            f"The row count and date range move with the horizon: the last {horizon} month(s) have "
            "no observed outcome yet, so they cannot be training rows (they are what the fitted "
            f"model predicts from), and the first {lag_cost} months are consumed by the target's "
            f"lagged values used as features. A total of {lag_cost + horizon} months are thus removed from {n_months} "
            f"months in the data resulting in {n_months - lag_cost - horizon} usable rows."
            + (f" A further {row_gap} month(s) drop out for missing feature values." if row_gap > 0 else "")
        )

        n_train = int(X.shape[0] * pipeline.SPLIT_PRESETS[split_preset][0]) - horizon
        k_auto = pipeline.auto_k_features(n_train, X.shape[1])
        st.caption(
            f"Each row's outcome is only known {horizon} month(s) later, so the last {horizon} training "
            "row(s) before the dev split and the last dev row(s) before the test split are dropped: their "
            "labels would otherwise already contain the next split's outcomes. The same gap separates the "
            "training and validation blocks inside cross-validation."
        )
        
        if task == "classification":
            st.caption(interpret.class_balance_note(y))
            st.caption(
                f"The class mix also shifts with the horizon: the label compares the rate {horizon} "
                "month(s) ahead with today, and cumulative moves grow over a longer window, so fewer "
                "months stay inside the Hold deadband."
            )
        else:
            st.caption(interpret.target_change_note(econometrics.stationarity(y), target_name))
        st.caption(
            "Excluded from the inputs: the target's own column, every feature engineered from it "
            "(moving averages, returns, vols, changes) and any spread it is a component of. Its past "
            f"values at lags {', '.join(str(lag) for lag in TARGET_LAGS)} months are added back "
            "deliberately - they are observed before the prediction is made, so they are honest "
            "autoregressive features, not leakage.",
            help=interpret.LEAKAGE_HELP,
        )
        st.caption(
            f"Not all {X.shape[1]} features reach a model. A one-feature-at-a-time F-test ranks them "
            f"by their association with the target, and only the top {k_auto} are kept. During "
            "cross-validation, selection is re-fit on each fold's training months only to avoid "
            "leakage. The final pipeline re-fits it on the full training split; those selected "
            "features are used on dev and test and listed in Diagnostics. k is limited by the "
            "smallest expanding CV fold, at about 4 training rows per feature."
        )

current_sig = (
    economy,
    task,
    target_name,
    horizon,
    split_preset,
    metric,
    include_neural,
    mtime,
)

bundle = st.session_state.get("mdl_bundle")
if (bundle is None or bundle["sig"] != current_sig) and X is not None and y is not None:
    key = registry.cache_key(economy, task, target_name, horizon, split_preset)
    loaded = registry.load_artifacts(key)
    if loaded is not None and not registry.is_stale(loaded["metadata"], registry.data_signature(X, y)):
        meta = loaded["metadata"]
        models_loaded = dict(loaded["models"])
        wanted_neural = {"MLP", "LSTM"} if include_neural else set()
        neural_loaded = set(meta.get("neural_names") or []) & set(models_loaded)
        for name in neural_loaded - wanted_neural:
            models_loaded.pop(name)
        cfg_load = {
            "economy": economy,
            "task": task,
            "target_name": target_name,
            "target_column": target_column,
            "horizon": horizon,
            "split_preset": split_preset,
            "metric": metric,
            "trained_at": meta.get("trained_at"),
        }
        st.session_state["mdl_bundle"] = assemble_bundle(
            models_loaded,
            X,
            y,
            task,
            cfg_load,
            monthly_data=monthly,
            X_latest=data["X_latest"],
            cv_scores=meta.get("cv_scores", {}),
            params=meta.get("params", {}),
            classes=meta.get("classes"),
            neural_names=neural_loaded & wanted_neural,
            ensemble_info=meta.get("ensemble_info"),
            sig=current_sig,
        )

with tab_train:
    st.caption(interpret.LEADERBOARD_HELP)
    st.caption(
        "Trained models auto-save and reload on their own. Click Train / refit only to rerun the search "
        "on the full current data (for example after the dataset grows) or after changing the setup; "
        "it overwrites the saved models."
    )
    if X is None or y is None:
        st.info("Complete the Setup tab first.")
    else:
        if st.button("Train / refit on current data", type="primary", key="mdl_train"):
            splits = pipeline.chronological_split(X, y, preset=split_preset, gap=horizon)
            if task == "classification":
                train_classes = set(pd.Series(splits["Train"][1]).dropna().unique())
                if train_classes != set(pd.Series(y).dropna().unique()):
                    st.warning(
                        "The training split does not contain every class present later in the "
                        "sample, so that class will be unreliable on this small window."
                    )

            sklearn_names = list(pipeline.model_roster(task))
            n_sklearn = len(sklearn_names)
            queue = sklearn_names + (["MLP", "LSTM"] if include_neural else [])
            total = len(queue)
            status = st.empty()
            progress_table = st.empty()
            start = time.perf_counter()
            finished_rows = []
            last_finish = {"t": start}

            def elapsed_mmss(seconds: float) -> str:
                minutes, secs = divmod(int(seconds), 60)
                return f"{minutes:d}:{secs:02d}"

            def record_finished(name, cv_score, done):
                now = time.perf_counter()
                if cv_score is not None and metric in ("RMSE", "MAE"):
                    cv_score = abs(cv_score)  # stored scores stay sign-aligned; display only
                finished_rows.append(
                    {
                        "Model": name,
                        f"CV {metric}": cv_score,
                        "Time": elapsed_mmss(now - last_finish["t"]),
                        "Cumulative": elapsed_mmss(now - start),
                    }
                )
                last_finish["t"] = now
                if done < total:
                    status.caption(
                        f"Currently training {queue[done]}... ({done} of {total} done, {total - done} remaining)"
                    )
                progress_table.dataframe(
                    pd.DataFrame(finished_rows).set_index("Model").style.format({f"CV {metric}": "{:.3f}"}, na_rep="-"),
                    width="stretch",
                )

            status.caption(f"Currently training {queue[0]}... (0 of {total} done, {total} remaining)")

            def sklearn_progress(done, _total, name, cv_score):
                record_finished(name, cv_score, done)

            roster = pipeline.train_roster(
                splits["Train"][0],
                splits["Train"][1],
                task,
                budget=budget,
                gap=horizon,
                scoring=metric,
                progress=sklearn_progress,
            )
            models = dict(roster["models"])
            classes, encoder = roster["classes"], roster["label_encoder"]

            if task == "classification":
                labels = list(range(len(classes)))
                split_eval = {n: (Xs, encoder.transform(ys)) for n, (Xs, ys) in splits.items()}
                y_train_enc = encoder.transform(splits["Train"][1])
            else:
                labels = None
                split_eval = splits
                y_train_enc = splits["Train"][1]

            neural_names = set()
            if include_neural:

                def neural_progress(done, _total, name, _score):
                    record_finished(name, None, n_sklearn + done)

                try:
                    fitted_neural = neural_mod.fit_neural_models(
                        splits["Train"][0], y_train_enc, task, history=data["X_all"], progress=neural_progress
                    )
                    models.update(fitted_neural)
                    neural_names = set(fitted_neural)
                except Exception as exc:
                    st.warning(f"Neural models were skipped: {exc}")

            ensemble_info = None
            with st.spinner("Blending and stacking..."):
                try:
                    ens = ensemble.build_ensembles(roster["models"], split_eval, task, labels=labels)
                    models.update(ens["ensembles"])
                    ensemble_info = {
                        "members": ens["members"],
                        "weights": ens["weights"],
                        "meta_model": type(ens["ensembles"]["Stack"].meta_model).__name__,
                    }
                except Exception as exc:
                    st.warning(f"Ensembles were skipped: {exc}")

            cfg_now = {
                "economy": economy,
                "task": task,
                "target_name": target_name,
                "target_column": target_column,
                "horizon": horizon,
                "split_preset": split_preset,
                "metric": metric,
            }
            bundle = assemble_bundle(
                models,
                X,
                y,
                task,
                cfg_now,
                monthly_data=monthly,
                X_latest=data["X_latest"],
                cv_scores=roster["cv_scores"],
                params=roster["params"],
                classes=classes,
                neural_names=neural_names,
                ensemble_info=ensemble_info,
                sig=current_sig,
            )
            st.session_state["mdl_bundle"] = bundle
            completed = len(finished_rows)
            status.caption(
                f"Finished: completed {completed} of {total} queued model fits in "
                f"{elapsed_mmss(time.perf_counter() - start)}."
            )
            try:
                save_bundle_artifacts(bundle)
                st.caption("Saved these models to disk; they reload automatically on the next visit.")
            except Exception as exc:
                st.warning(f"Models trained but could not be saved to disk: {exc}")

        bundle = st.session_state.get("mdl_bundle")
        if bundle is None or bundle["sig"] != current_sig:
            st.info("Click Train to build the leaderboard for the current setup.")
        else:
            board, best, metric = bundle["board"], bundle["best"], bundle["metric"]

            if bundle.get("trained_at"):
                st.caption(
                    f"Loaded from disk: these models were trained on {bundle['trained_at'][:10]}, on "
                    f"{len(bundle['X'])} monthly rows spanning {bundle['X'].index.min():%Y-%m} to "
                    f"{bundle['X'].index.max():%Y-%m}."
                )

            task_metrics = (
                evaluate.CLASSIFICATION_METRICS if bundle["task"] == "classification" else evaluate.REGRESSION_METRICS
            )
            if best is not None:
                st.success(
                    interpret.best_model_sentence(
                        best, metric, board.loc[best, f"Valid {metric}"], board.loc[best, f"Test {metric}"]
                    )
                )
                if bundle["task"] == "classification" and metric.startswith("ROC-AUC"):
                    st.caption(interpret.ROC_AUC_OVR_NOTE)
                benchmark = None
                if bundle["task"] == "regression" and metric in ("RMSE", "MAE"):
                    benchmark = board.loc["Baseline: no change", f"Test {metric}"]
                verdict = interpret.metric_verdict(metric, board.loc[best, f"Test {metric}"], benchmark=benchmark)
                if verdict:
                    st.caption(f"On the test set - {verdict}")
                if bundle.get("skill") is not None:
                    st.caption(interpret.skill_verdict(bundle["skill"].loc[best, "Test Skill vs naive"]))
                note = interpret.overfit_note(
                    board.loc[best, f"Train {metric}"], board.loc[best, f"Test {metric}"], metric
                )
                if note:
                    st.caption(note)
                st.caption(interpret.COVID_DEV_NOTE)

            metric_order = [metric] + [m for m in task_metrics if m != metric]
            display_cols = [f"{s} {m}" for m in metric_order for s in ("Train", "Valid", "Test")]
            ascending = metric in ("RMSE", "MAE")
            board_view = board.sort_values(f"Valid {metric}", ascending=ascending)[display_cols]
            board_view = board_view.rename(columns=lambda c: c.replace("Valid", "Dev"))
            if bundle.get("skill") is not None:
                skill_cols = bundle["skill"].rename(columns=lambda c: c.replace("Valid", "Dev"))
                board_view = board_view.join(skill_cols)

            def highlight_best(row):
                colour = "background-color: rgba(84,162,75,0.18)"
                return [colour if row.name == best else "" for _ in row]

            ens_rows = [n for n in ("Blend", "Stack") if n in board_view.index]
            if ens_rows:
                train_cols = [c for c in board_view.columns if c.startswith("Train")]
                board_view.loc[ens_rows, train_cols] = np.nan
            st.markdown(f"**Full leaderboard** - sorted by the Dev {metric} column ")
            st.dataframe(
                board_view.style.format(precision=3, na_rep="-").apply(highlight_best, axis=1),
                width="stretch",
            )
            st.caption(
                "Blend and Stack are fit on the dev split, so their Dev scores are partly in-sample "
                "- that is why the winner is picked among the base models only - and their Train "
                "cells are blanked: a train score for a dev-fit combiner is neither an in- nor an "
                "out-of-sample read."
            )
            st.caption(interpret.BASELINE_HELP_REG if bundle["task"] == "regression" else interpret.BASELINE_HELP_CLF)

            with st.expander("Model configuration (CV scores and chosen hyperparameters)"):
                st.caption(interpret.CV_BUDGET_HELP)
                if bundle["task"] == "classification" and metric.startswith("ROC-AUC"):
                    st.caption(interpret.CV_ROC_AUC_NOTE)
                config_rows = []
                for name, params in bundle["params"].items():
                    tuned = ", ".join(f"{k.replace('model__', '')}={format_param(v)}" for k, v in params.items())
                    cv_score = bundle["cv_scores"].get(name)
                    if cv_score is not None and metric in ("RMSE", "MAE"):
                        cv_score = abs(cv_score)
                    config_rows.append(
                        {
                            "Model": name,
                            f"CV {metric}": cv_score,
                            "Tuned hyperparameters": tuned or "-",
                        }
                    )
                for name in sorted(bundle["neural_names"]):
                    config_rows.append(
                        {
                            "Model": name,
                            f"CV {metric}": np.nan,
                            "Tuned hyperparameters": "Fixed configuration (no search)",
                        }
                    )
                config_df = pd.DataFrame(config_rows).set_index("Model")
                st.dataframe(
                    config_df.style.format({f"CV {metric}": "{:.3f}"}, na_rep="-"),
                    width="stretch",
                )
                if bundle["neural_names"]:
                    st.caption(
                        "The neural nets show no CV score by design: they train once on a fixed, "
                        "heavily regularised configuration (dropout, L2, early stopping) instead of "
                        "running a hyperparameter search."
                    )
                st.caption(
                    "Float hyperparameters are rounded to four significant figures for display; "
                    "the fitted models keep the exact values."
                )

            ens_info = bundle.get("ensemble_info")
            if ens_info:
                with st.expander("Ensemble composition (blend weights, stack members, meta-model)"):
                    st.caption(interpret.ENSEMBLE_HELP)
                    error_name = "dev RMSE" if bundle["task"] == "regression" else "dev log loss"
                    weights = pd.Series(ens_info["weights"], name="Blend weight").sort_values(ascending=False)
                    st.markdown("**Blend**")
                    st.dataframe(weights.to_frame().style.format("{:.3f}"), width="stretch")
                    st.caption(
                        f"These weights are always inverse {error_name}, independent of the scoring "
                        "metric chosen in Setup - that choice drives model selection and reporting, "
                        "not how the blend is mixed."
                    )
                    st.markdown(
                        f"**Stack** - a {ens_info['meta_model']} meta-model trained on the "
                        f"dev-split outputs of: {', '.join(ens_info['members'])}."
                    )

with tab_diag:
    bundle = st.session_state.get("mdl_bundle")
    if bundle is None or bundle["sig"] != current_sig:
        st.info("Train the leaderboard for the current setup first, then return here.")
    else:
        diag_task = bundle["task"]
        model_names = list(bundle["models"])
        default_model = bundle["best"] if bundle["best"] in model_names else model_names[0]
        if st.session_state.get("mdl_diag_model_sig") != bundle["sig"]:
            st.session_state["mdl_diag_model_sig"] = bundle["sig"]
            st.session_state["mdl_diag_model"] = default_model
        elif st.session_state.get("mdl_diag_model") not in model_names:
            st.session_state["mdl_diag_model"] = default_model
        model_name = st.selectbox("Model to inspect", model_names, key="mdl_diag_model")
        model = bundle["models"][model_name]

        splits = bundle["splits"]
        X_valid, y_valid = splits["Valid"]
        X_test, y_test = splits["Test"]
        Xp_valid, yp_valid = bundle["split_eval"]["Valid"]

        eval_choice = st.radio(
            "Evaluate diagnostics on",
            ["Test", "Dev", "Train", "Train + Dev + Test"],
            horizontal=True,
            key="mdl_diag_split",
            help=interpret.DIAG_SPLIT_HELP,
        )
        if eval_choice == "Train + Dev + Test":
            X_show = pd.concat([splits["Train"][0], X_valid, X_test])
            y_show = np.concatenate([np.asarray(splits["Train"][1]), np.asarray(y_valid), np.asarray(y_test)])
        else:
            key_map = {"Test": "Test", "Dev": "Valid", "Train": "Train"}
            X_show, y_show = splits[key_map[eval_choice]]
        use_youden = diag_task == "classification" and st.session_state.setdefault("mdl_diag_youden", False)
        threshold_note = " - the Youden thresholds stay tuned on the dev split throughout" if use_youden else ""
        st.caption(f"The split control only changes which months the charts below are measured on{threshold_note}.")

        if diag_task == "classification":
            class_labels = list(bundle["classes"])
            y_show_str = pd.Series(y_show).astype(str).to_numpy()
            proba_show = model.predict_proba(X_show)
            if use_youden:
                y_valid_str = pd.Series(y_valid).astype(str).to_numpy()
                thresholds = evaluate.youden_thresholds(y_valid_str, model.predict_proba(X_valid), class_labels)
            else:
                # Uniform thresholds make predict_with_thresholds a plain argmax - the
                # same operating point the leaderboard scores use.
                thresholds = dict.fromkeys(class_labels, 0.5)
            pred_show = evaluate.predict_with_thresholds(proba_show, thresholds, class_labels)
            metrics = evaluate.classification_metrics(y_show_str, pred_show, class_labels)
            auc = evaluate.roc_auc_macro_ovr(y_show_str, proba_show, class_labels)
            st.caption(interpret.roc_auc_verdict(auc, split=eval_choice))

            row1 = st.columns(2)
            with row1[0]:
                show_figure(
                    plots.roc_curves(evaluate.roc_curve_data(y_show_str, proba_show, class_labels)),
                    interpret.ROC_HELP,
                )
            with row1[1]:
                matrix_slot = st.container()
                st.toggle(
                    "Tune the operating point with Youden's J on the dev split",
                    key="mdl_diag_youden",
                    help=interpret.THRESHOLD_HELP,
                )
                with matrix_slot:
                    show_figure(
                        plots.confusion_heatmap(metrics["confusion"]),
                        interpret.confusion_verdict(metrics["confusion"]),
                        interpret.CONFUSION_HELP,
                    )
            prob = evaluate.probability_metrics(y_show_str, proba_show, class_labels)
            summary_cols = st.columns(3)
            summary_cols[0].metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            summary_cols[1].metric("Log loss", "-" if prob["Log loss"] is None else f"{prob['Log loss']:.3f}")
            summary_cols[2].metric("Brier (multiclass)", f"{prob['Brier']:.3f}")
            st.dataframe(
                metrics["per_class"].style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
                width="stretch",
            )
            show_figure(
                plots.probability_histogram(proba_show, class_labels, y_true=y_show_str),
                "Predicted-probability distributions per class, on the selected months.",
                interpret.PROB_HIST_HELP,
            )
        else:
            pred_show = np.asarray(model.predict(X_show), dtype=float)
            y_show_arr = np.asarray(y_show, dtype=float)
            level_now = monthly[bundle["target_column"]].reindex(X_show.index).to_numpy()
            target_index = X_show.index + pd.offsets.MonthEnd(bundle["horizon"])
            reg_row = st.columns(2)
            with reg_row[0]:
                show_figure(
                    plots.predicted_vs_actual(
                        level_now + y_show_arr,
                        level_now + pred_show,
                        index=target_index,
                        title=f"Predicted vs actual level - {bundle['target_column']}",
                    )
                )
            with reg_row[1]:
                show_figure(plots.residual_plot(y_show, pred_show, index=X_show.index))
            rmse = evaluate.regression_metrics(y_show_arr, pred_show)["RMSE"]
            naive_mse = float(np.mean(y_show_arr**2))
            if naive_mse > 0:
                skill = 1.0 - float(np.mean((y_show_arr - pred_show) ** 2)) / naive_mse
                benchmark = float(np.sqrt(naive_mse))
            else:
                skill = benchmark = None
            st.caption(
                f"Measured on the {eval_choice} months, in change space (the chart on the left is "
                "reconstructed to levels for display only): "
                + interpret.metric_verdict("RMSE", rmse, benchmark=benchmark)
                + " "
                + interpret.skill_verdict(skill)
            )
            st.caption(interpret.residual_verdict(y_show, pred_show), help=interpret.RESIDUAL_HELP)

        scoring = evaluate.CV_SCORING[bundle["metric"]]
        native = explain.native_importance(model, X_test.columns)
        perm_cache = bundle.setdefault("perm_cache", {})
        if model_name not in perm_cache:
            with st.spinner("Computing permutation importance..."):
                try:
                    perm_cache[model_name] = explain.permutation_importance_scores(
                        model, Xp_valid, yp_valid, scoring=scoring
                    )
                except Exception:
                    perm_cache[model_name] = None
        perm = perm_cache[model_name]
        st.markdown("**Feature importance**")
        kept = explain.selected_features(model, X_test.columns)
        if len(kept) < X_test.shape[1]:
            st.caption(
                f"The fitted pipeline's filter kept {len(kept)} of the {X_test.shape[1]} candidate "
                f"features for this model: {', '.join(sorted(kept))}. Only these reach the model, "
                "so they are the features the measures below can meaningfully rank - a dropped "
                "feature cannot move the predictions."
            )
        st.caption(
            "Both charts ignore the split selector above. Permutation importance is always "
            "measured on the dev months: the model never fitted them, so the score drop from "
            "shuffling a feature reflects genuine out-of-sample signal, while the test months "
            "stay reserved for the final scores. Native importance comes from the fitted model "
            "itself, so no split is involved."
        )
        imp_row = st.columns(2)
        with imp_row[0]:
            st.markdown("**Native**")
            if native is None:
                st.caption(
                    "This model type exposes no native importance. "
                    "Read the permutation importance on the right instead."
                )
            elif native.eq(0).all():
                st.caption(
                    "Every coefficient is zero. Cross-validation picked a penalty strong enough to "
                    "shrink them all away, leaving an intercept-only model that always predicts the "
                    "training mean - a legitimate outcome when no feature reliably beats noise on "
                    "this target. Check its Skill column on the leaderboard: an intercept-only "
                    "model adds essentially nothing over the no-change baseline."
                )
            else:
                show_figure(plots.importance_bar(native, title=None), interpret.importance_sentence(native))
        with imp_row[1]:
            st.markdown("**Permutation**")
            if perm is not None:
                show_figure(plots.importance_bar(perm, title=None), interpret.importance_sentence(perm))
            else:
                st.caption("Permutation importance could not be computed for this model on this split.")
        with st.expander("How to read feature importance"):
            st.markdown(interpret.IMPORTANCE_HELP)

        base_imp = native if native is not None and native.gt(0).any() else perm
        if base_imp is not None and base_imp.gt(0).any():
            blocks = sorted({group_of(f, bundle["target_column"]) for f in bundle["X"].columns})
            group_imp = explain.feature_group_importance(
                base_imp, lambda f: group_of(f, bundle["target_column"]), all_groups=blocks
            )
            show_figure(
                plots.importance_bar(
                    group_imp, title="Importance by business block", value_label="Group importance (0-100)"
                ),
                interpret.group_importance_sentence(group_imp),
                interpret.GROUP_IMPORTANCE_HELP,
            )

        st.markdown("**SHAP contributions (tree and boosting models)**")
        st.caption(
            "SHAP is always computed on the test months, whatever the selector above says: these "
            "values explain individual predictions rather than produce a score, so nothing leaks."
        )
        if explain.is_tree_model(model):
            class_ix = 0
            if diag_task == "classification" and len(class_labels) > 2:
                if st.session_state.get("mdl_diag_shap_class") not in class_labels:
                    st.session_state["mdl_diag_shap_class"] = class_labels[0]
                chosen = st.selectbox("SHAP class", class_labels, key="mdl_diag_shap_class")
                class_ix = class_labels.index(chosen)

            shap_cache = bundle.setdefault("shap_cache", {})
            if model_name not in shap_cache:
                shap_cache[model_name] = explain.tree_shap(model, X_test)
            shap_data = shap_cache[model_name]

            summary = explain.shap_summary(shap_data, class_index=class_ix)
            show_figure(
                plots.importance_bar(summary, title="Which features move predictions most (mean |SHAP|)"),
                "Average absolute SHAP contribution per feature, over the test months.",
                interpret.SHAP_HELP,
            )

            default_row = 0
            if diag_task == "classification":
                pred_test = evaluate.predict_with_thresholds(model.predict_proba(X_test), thresholds, class_labels)
                wrong = explain.misclassified_index(pd.Series(y_test).astype(str).to_numpy(), pred_test)
                if len(wrong):
                    default_row = int(wrong[0])
            month_labels = [f"{d:%Y-%m}" for d in X_test.index]
            if st.session_state.get("mdl_diag_shap_row") not in month_labels:
                st.session_state["mdl_diag_shap_row"] = month_labels[default_row]
            chosen_month = st.selectbox("Month to explain", month_labels, key="mdl_diag_shap_row")
            row_pos = month_labels.index(chosen_month)
            X_row = X_test.iloc[[row_pos]]
            local = explain.local_shap_series(model, X_row, class_index=class_ix)
            show_figure(
                plots.shap_local_bar(local, title=f"Why the model predicted {chosen_month}"),
                "Signed contributions for the selected month (green pushes the prediction up, red pulls it down).",
            )
        else:
            st.caption(
                "SHAP here uses the exact TreeExplainer, so it is available for tree and boosting "
                "models only (Decision tree, Random forest, Gradient boosting, XGBoost, LightGBM). "
                "For this model read the permutation importance above, which is model-agnostic."
            )

        st.markdown("**Partial dependence**")
        ranked = list(base_imp.index) if base_imp is not None else list(X_test.columns)
        pdp_options = [f for f in ranked if is_base_feature(f)][:15]
        if not pdp_options:
            st.caption("No eligible base or spread feature is available for partial dependence.")
        else:
            if st.session_state.get("mdl_diag_pdp") not in pdp_options:
                st.session_state["mdl_diag_pdp"] = pdp_options[0]

            pdp_feature = st.selectbox(
                "Feature to sweep",
                pdp_options,
                key="mdl_diag_pdp",
            )
            st.caption(
                "The list holds the top-15 base features and engineered spreads by importance "
                "(native, falling back to permutation), most important first. Target lags, "
                "returns, moving averages and vols are excluded: sweeping a derived column "
                "while its parent stays fixed is a combination the data can never produce, "
                "so the curve would not be interpretable."
            )

            pdp_labels = class_labels if diag_task == "classification" else None
            try:
                pdp = explain.partial_dependence_data(
                    model,
                    X_test,
                    pdp_feature,
                    task=diag_task,
                )
                show_figure(
                    plots.partial_dependence_plot(
                        pdp,
                        labels=pdp_labels,
                        feature=pdp_feature,
                    ),
                    "Every other feature is held at its observed value while this one is swept.",
                    interpret.PDP_HELP,
                )
            except Exception:
                st.caption("Partial dependence is unavailable for this model.")

        st.markdown("**Econometric baseline**")
        st.caption(interpret.ECON_BASELINE_HELP_REG if diag_task == "regression" else interpret.ECON_BASELINE_HELP_CLF)
        st.caption(
            "The baseline is judged on the very months it was estimated from: it is "
            "fit once on the full monthly frame - including the months the ML models keep as dev and "
            "test - so its fit numbers are optimistic and not comparable with the ML test scores, "
            "and the split selector at the top does not affect this panel. To stay well-conditioned "
            "it enters only the 10 features most correlated with the target."
        )
        if diag_task == "regression":
            ols = econometrics.ols_baseline(bundle["X"], bundle["y"])
            if ols is None:
                st.info("Not enough complete rows to fit the OLS baseline.")
            else:
                fit_cols = st.columns(3)
                fit_cols[0].metric("R2", f"{ols['rsquared']:.3f}")
                fit_cols[1].metric("Adj. R2", f"{ols['rsquared_adj']:.3f}")
                fit_cols[2].metric("F p-value", interpret.format_pvalue(ols["f_pvalue"]))
                st.caption(
                    f"R2 = {ols['rsquared']:.2f}: the baseline explains {ols['rsquared']:.0%} of the "
                    "in-sample variation of the forward change. A low value is the honest, expected "
                    "reading here - monthly changes in macro series are mostly noise, and the change "
                    "target was chosen deliberately because a level regression would fake a near-perfect "
                    "R2 through shared trend alone. Judge the baseline on the F p-value (small = the "
                    "features jointly beat an intercept-only model) and on which coefficients are "
                    "significant, not on R2 alone."
                )
                st.dataframe(
                    ols["coefficients"].style.format(precision=4).map(significance_colour, subset=["p-value"]),
                    width="stretch",
                )
                st.caption("Green coefficients are significant at p < 0.05, red are not.")
                assume = st.columns(3)
                assume[0].metric("Durbin-Watson", f"{ols['durbin_watson']:.2f}")
                assume[1].metric("Normality (JB) p", interpret.format_pvalue(ols["jarque_bera_p"]))
                assume[2].metric("Condition no.", f"{ols['condition_number']:,.0f}")
                st.caption(interpret.ols_assumptions_note(ols))
                st.caption(
                    "Cook's distance asks, month by month: how much would the whole fitted line move "
                    "if this month were dropped and the baseline refit? The usual flag is the 4/n "
                    f"rule of thumb - n is the {ols['nobs']} months in this fit, so the threshold is "
                    f"4/{ols['nobs']} = {ols['cooks_threshold']:.3f} - and {len(ols['influential'])} "
                    "month(s) exceed it, typically shock months that pull the coefficients hardest."
                )
                show_figure(plots.cooks_distance_plot(ols["cooks"], ols["cooks_threshold"]))
        else:
            logit = econometrics.logit_baseline(bundle["X"], bundle["y"])
            if logit is None:
                st.info("The multinomial-logit baseline did not converge on this configuration.")
            else:
                fit_cols = st.columns(4)
                fit_cols[0].metric("Pseudo R2", f"{logit['prsquared']:.3f}")
                fit_cols[1].metric("AIC", f"{logit['aic']:,.0f}")
                fit_cols[2].metric("BIC", f"{logit['bic']:,.0f}")
                fit_cols[3].metric("LLR p-value", interpret.format_pvalue(logit["llr_pvalue"]))
                st.caption(
                    f"Pseudo R2 = {logit['prsquared']:.2f} (McFadden) measures the log-likelihood gain "
                    "over an intercept-only model; the LLR p-value tests that gain jointly (small = the "
                    "features help). The table shows each coefficient's p-value per class versus the "
                    f"baseline class ({logit['baseline_class']})."
                )
                st.dataframe(
                    logit["coefficients"].style.format(precision=4).map(significance_colour),
                    width="stretch",
                )
                st.caption("Green coefficients are significant at p < 0.05, red are not.")

with tab_scenario:
    st.caption(interpret.SCENARIO_HELP)
    bundle = st.session_state.get("mdl_bundle")
    if bundle is None or bundle["sig"] != current_sig:
        st.info("Train the leaderboard for the current setup first, then return here.")
    else:
        scn_task = bundle["task"]
        X_full = bundle["X"]
        model_names = list(bundle["models"])
        default_model = bundle["best"] if bundle["best"] in model_names else model_names[0]
        if st.session_state.get("mdl_scn_model_sig") != bundle["sig"]:
            st.session_state["mdl_scn_model_sig"] = bundle["sig"]
            st.session_state["mdl_scn_model"] = default_model
        elif st.session_state.get("mdl_scn_model") not in model_names:
            st.session_state["mdl_scn_model"] = default_model
        model_name = st.selectbox("Model", model_names, key="mdl_scn_model")
        model = bundle["models"][model_name]

        anchor_row = bundle["X_latest"]
        anchor = anchor_row.iloc[0]
        anchor_month = anchor_row.index[-1]
        target_month = anchor_month + pd.offsets.MonthEnd(bundle["horizon"])

        if scn_task == "classification":
            # Ordered Cut < Hold < Hike so correlation against the codes is meaningful.
            seed_y = pd.Series(
                pd.Categorical(bundle["y"], categories=["Cut", "Hold", "Hike"]).codes,
                index=bundle["y"].index,
            ).astype(float)
        else:
            seed_y = bundle["y"]
        # Only base columns and engineered spreads are offered as levers (same rule as
        # the PDP list): lags/returns/MAs/vols are mechanical transforms of another
        # series and cannot be moved independently.
        base_cols = [c for c in X_full.columns if is_base_feature(c)]
        driver_options = econometrics.top_correlated_features(X_full[base_cols], seed_y, 15)
        default_drivers = driver_options[:5]

        # Driver options and the anchor move with economy/task/target/horizon/data, not with
        # scoring or split choices; reset the controlled scenario widgets only when those change.
        scn_sig = (economy, task, target_name, horizon, mtime)
        if st.session_state.get("mdl_scn_sig") != scn_sig or "mdl_scn_drivers" not in st.session_state:
            st.session_state["mdl_scn_sig"] = scn_sig
            st.session_state["mdl_scn_drivers"] = default_drivers
            for col in driver_options:
                st.session_state[f"mdl_scn_{col}"] = float(anchor[col])

        if st.button("Revert to defaults", key="mdl_scn_revert"):
            st.session_state["mdl_scn_drivers"] = default_drivers
            for col in driver_options:
                st.session_state[f"mdl_scn_{col}"] = float(anchor[col])
        drivers = st.multiselect("Drivers to vary", driver_options, key="mdl_scn_drivers")
        st.caption(
            f"Anchored on {anchor_month:%Y-%m}, the most recent month with a complete feature row, "
            f"so the prediction reads {bundle['horizon']} month(s) ahead - {target_month:%Y-%m}. "
            "Drivers default to the five base features most correlated with the target (top 15 "
            "listed); lags, returns, moving averages and vols are excluded because they cannot move "
            f"independently of their source series. Every feature not shown is held at its "
            f"{anchor_month:%Y-%m} value, so the reading is a ceteris-paribus response of the "
            "chosen drivers."
        )

        scenario = anchor.copy()
        slider_cols = st.columns(2)
        for i, drv in enumerate(drivers):
            lo, hi = float(X_full[drv].min()), float(X_full[drv].max())
            # The anchor month sits after the labelled sample, so its value can fall
            # outside the historical range the bounds are derived from.
            lo, hi = min(lo, float(anchor[drv])), max(hi, float(anchor[drv]))
            if lo == hi:
                continue
            key = f"mdl_scn_{drv}"
            st.session_state.setdefault(key, float(anchor[drv]))
            scenario[drv] = slider_cols[i % 2].slider(drv, lo, hi, step=(hi - lo) / 100, key=key)

        dtypes = X_full.dtypes.to_dict()
        scenario_row = scenario.to_frame().T.astype(dtypes)
        base_row = anchor_row
        try:
            if scn_task == "classification":
                class_labels = list(bundle["classes"])
                proba = model.predict_proba(scenario_row)[0]
                pred = class_labels[int(np.argmax(proba))]
                st.metric(f"Predicted stance for {target_month:%Y-%m}", pred)
                st.caption(
                    f"Read this as the net direction of the policy rate between {anchor_month:%Y-%m} "
                    f"and {target_month:%Y-%m}, not a specific meeting's decision - no Fed/ECB meeting "
                    "calendar is modelled. Hike/Cut means the rate ends at least 12.5 bp (half a "
                    "25 bp step) higher/lower over the window, whatever the number of meetings in "
                    "between; Hold means it stays inside that band."
                )
                show_figure(
                    plots.class_probability_bar(pd.Series(proba, index=class_labels), title="Class probabilities"),
                    "Probability the model assigns to each forward decision under the scenario.",
                )
            else:
                current_level = float(monthly.loc[:anchor_month, bundle["target_column"]].dropna().iloc[-1])
                base_level = current_level + float(model.predict(base_row)[0])
                scn_level = current_level + float(model.predict(scenario_row)[0])
                out = st.columns(2)
                out[0].metric(
                    f"Baseline level for {target_month:%Y-%m}",
                    f"{base_level:.3f}",
                    delta=f"{base_level - current_level:+.3f} vs {anchor_month:%Y-%m}",
                )
                out[1].metric(
                    "Scenario level",
                    f"{scn_level:.3f}",
                    delta=f"{scn_level - base_level:+.3f} vs baseline",
                )
                st.caption(
                    f"Anchor: {bundle['target_column']} was {current_level:.3f} in {anchor_month:%Y-%m}. "
                    f"The model predicts the change from that anchor, and the {target_month:%Y-%m} levels "
                    "shown are the anchor plus that predicted change - the implied change is the delta "
                    "under each number. The baseline holds every driver at the anchor month; the "
                    "scenario moves only the chosen drivers."
                )
        except Exception as exc:
            st.warning(f"This model could not score the scenario row: {exc}")

with tab_forecast:
    st.caption(interpret.FORECAST_HELP)
    monthly_fc = get_monthly_complete(economy, mtime)
    if len(monthly_fc) < len(monthly):
        st.caption(
            f"The current month is still in progress, so it is excluded from these models; "
            f"history ends at {monthly_fc.index[-1]:%B %Y}."
        )
    numeric = monthly_fc.select_dtypes("number")
    series_options = sorted(c for c in numeric.columns if not any(s in c for s in ENGINEERED_SUFFIX))
    policy_col = targets.policy_rate_column(economy)
    default_series = series_options.index(policy_col) if policy_col in series_options else 0
    st.session_state.setdefault("mdl_fc_steps", 12)
    steps = st.slider(
        "Forecast horizon (months)",
        3,
        24,
        key="mdl_fc_steps",
        help="How many months ahead these time-series models project. Independent of the Setup horizon. "
        "Moving it never refits anything: orders are chosen once on the pre-holdout history and the fits "
        "are cached; the slider only changes how many months are projected.",
    )

    arima_tab, garch_tab = st.tabs(["ARIMA / SARIMA", "GARCH volatility"])

    with arima_tab:
        st.caption(interpret.ARIMA_HELP)
        if st.session_state.get("mdl_fc_arima_series") not in series_options:
            st.session_state["mdl_fc_arima_series"] = series_options[default_series]
        series_col = st.selectbox("Series", series_options, key="mdl_fc_arima_series")
        series = monthly_fc[series_col].dropna()
        st.caption(interpret.stationarity_verdict(econometrics.stationarity(series), series_col))
        show_figure(
            plots.acf_pacf_plot(econometrics.acf_pacf(series)),
            "Stems crossing the dashed red lines are statistically significant. These charts are "
            "informational - the automatic search below already weighs this evidence. Most level series "
            "produce this same textbook picture (a slowly decaying ACF, one PACF spike at lag 1) because "
            "nearly all of them are highly persistent, close to random walks; that is a property of the "
            "data, not a plotting error, and differencing (d) is exactly what corrects it.",
        )

        search = arima_search(economy, series_col, mtime)
        if search is None:
            st.info("The ARIMA order search needs at least 20 observations.")
            best_order, best_seasonal = (1, 0, 0), (0, 0, 0, 0)
        else:
            best_order = tuple(int(v) for v in search["best"])
            best_seasonal = tuple(int(v) for v in search["best_seasonal"])
            picked = f"SARIMA{best_order}x{best_seasonal}" if best_seasonal[3] else f"ARIMA{best_order}"
            cutoff = series.index[-FORECAST_HOLDOUT - 1]
            st.caption(
                f"Automatic order (lowest {search['ic'].upper()}, chosen on history to {cutoff:%b %Y}): "
                f"{picked}. Candidates:"
            )
            board = search["leaderboard"].copy()
            board["Order"] = board["Order"].map(str)
            board["Seasonal"] = board["Seasonal"].map(lambda s: str(s) if s[3] else "-")
            board["Ljung-Box p"] = board["Ljung-Box p"].map(interpret.format_pvalue)
            st.dataframe(board, hide_index=True, width="stretch")

        st.session_state.setdefault("mdl_fc_arima_override", False)
        override = st.toggle(
            "Override the order manually",
            key="mdl_fc_arima_override",
        )
        if override:
            st.warning(
                "Manual orders are not validated - an ill-chosen order can produce a misleading or "
                "non-converging fit. The automatic order above is usually the safer choice."
            )
            order = st.columns(3)
            st.session_state.setdefault(f"mdl_fc_p_{series_col}", best_order[0])
            st.session_state.setdefault(f"mdl_fc_d_{series_col}", best_order[1])
            st.session_state.setdefault(f"mdl_fc_q_{series_col}", best_order[2])
            p = order[0].number_input("AR (p)", 0, 5, key=f"mdl_fc_p_{series_col}")
            d = order[1].number_input("I (d)", 0, 2, key=f"mdl_fc_d_{series_col}")
            q = order[2].number_input("MA (q)", 0, 5, key=f"mdl_fc_q_{series_col}")
            chosen_order = (p, d, q)
            seasonal_order = (0, 0, 0, 0)
            st.session_state.setdefault("mdl_fc_seasonal", False)
            if st.toggle("Seasonal (SARIMA)", key="mdl_fc_seasonal"):
                st.caption("The seasonal period is fixed to 12 for the monthly frame (annual seasonality).")
                s = st.columns(3)
                seasonal_order = (
                    s[0].number_input("Seasonal AR (P)", 0, 2, key="mdl_fc_P"),
                    s[1].number_input("Seasonal I (D)", 0, 1, key="mdl_fc_D"),
                    s[2].number_input("Seasonal MA (Q)", 0, 2, key="mdl_fc_Q"),
                    12,
                )
        else:
            chosen_order, seasonal_order = best_order, best_seasonal

        res = arima_fit(economy, series_col, chosen_order, seasonal_order, mtime)
        if res is None:
            st.info("The ARIMA fit needs at least 20 observations and a valid order.")
        else:
            diag = econometrics.arima_diagnostics(res)
            acc = arima_accuracy(economy, series_col, chosen_order, seasonal_order, mtime)
            crit = st.columns(4)
            crit[0].metric("AIC", f"{diag['aic']:,.1f}")
            crit[1].metric("BIC", f"{diag['bic']:,.1f}")
            if acc is not None:
                crit[2].metric(f"Holdout RMSE ({acc['holdout']}m)", f"{acc['rmse']:,.3f}")
                crit[3].metric(f"Holdout MAE ({acc['holdout']}m)", f"{acc['mae']:,.3f}")
            st.caption(interpret.ljung_box_verdict(diag))
            if acc is not None:
                st.caption(
                    f"The order above was chosen, and this specification fit, on data ending {acc['holdout']} "
                    "months before the last observation; forecasting those months gives the holdout RMSE/MAE "
                    f"above (series' own units), versus in-sample one-step errors of {acc['train_rmse']:,.3f} / "
                    f"{acc['train_mae']:,.3f}. The forecast below then refits the same specification on the "
                    "full history - there is no held-out data behind that fit."
                )
            fc_index = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
            fc = {
                k: pd.Series(np.asarray(v), index=fc_index) for k, v in econometrics.arima_forecast(res, steps).items()
            }
            show_figure(
                plots.forecast_fan(series.iloc[-120:], fc, name=series_col),
                "History is truncated to the last 120 months for readability; the dashed line is the "
                "forecast mean and the shaded band its 95% interval, widening with the horizon.",
            )
            show_figure(
                plots.forecast_fan(series.iloc[-6:], fc, name=series_col, title=f"Forecast close-up - {series_col}"),
                "Close-up on the forecast window with the last six observed months for context.",
            )

    with garch_tab:
        st.caption(interpret.GARCH_HELP)
        if st.session_state.get("mdl_fc_garch_series") not in series_options:
            st.session_state["mdl_fc_garch_series"] = series_options[default_series]
        garch_col = st.selectbox("Series (its monthly change is modelled)", series_options, key="mdl_fc_garch_series")
        change = monthly_fc[garch_col].diff().dropna()
        gsearch = garch_search(economy, garch_col, mtime)
        if gsearch is None:
            st.info("GARCH needs at least 50 monthly changes for this series.")
            best_pq = (1, 1)
        else:
            best_pq = tuple(int(v) for v in gsearch["best"])
            st.caption(
                f"Automatic order (lowest {gsearch['ic'].upper()}, chosen on the pre-holdout history): "
                f"GARCH{best_pq}. Candidates:"
            )
            gboard = gsearch["leaderboard"].copy()
            gboard["Order"] = gboard["Order"].map(str)
            st.dataframe(gboard, hide_index=True, width="stretch")
        res = garch_fit(economy, garch_col, best_pq, mtime)
        if res is None:
            st.info("GARCH needs at least 50 monthly changes for this series.")
        else:
            fc = econometrics.garch_forecast(res, steps)
            gacc = garch_accuracy(economy, garch_col, best_pq, mtime)
            fc_index = pd.date_range(change.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
            show_figure(
                plots.garch_volatility_plot(fc, forecast_index=fc_index),
                "Volatility is the conditional standard deviation of the monthly change, in the series' "
                "own units. Blue is that estimate fitted month by month over history (in-sample); red is the "
                "model's projection for months that have not happened yet.",
            )
            st.caption(interpret.garch_persistence_note(fc["persistence"]))
            if gacc is not None:
                st.caption(
                    f"Accuracy check: the order was chosen and the model fit without the last {gacc['holdout']} "
                    f"months; its volatility forecast missed the realized absolute changes by RMSE "
                    f"{gacc['rmse']:,.3f} / MAE {gacc['mae']:,.3f}. The displayed model refits that order on "
                    "the full history."
                )
            close = {**fc, "fitted_volatility": fc["fitted_volatility"].iloc[-12:]}
            show_figure(
                plots.garch_volatility_plot(
                    close,
                    forecast_index=fc_index,
                    title="Conditional volatility - close-up",
                    markers=True,
                ),
                "Close-up on the last 12 months of volatility and the forecast, which now continues from "
                "the last fitted month.",
            )
