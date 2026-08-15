import logging
import time

import numpy as np
import pandas as pd
import streamlit as st
from src.dataset_builder import ECONOMIES, cache_exists, master_mtime, load_master
from src.models import econometrics, ensemble, evaluate, explain, features, pipeline, registry, targets
from src.models import neural as neural_mod
from utils import interpret, plots

st.set_page_config(page_title="Modelling", layout="wide")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


@st.cache_data(show_spinner=False)
def get_master(economy: str, mtime: float) -> pd.DataFrame:
    """Cached read of a master dataset, date-indexed; invalidated when the file changes."""

    return load_master(economy).set_index("date").sort_index()


@st.cache_data(show_spinner=False)
def get_monthly(economy: str, mtime: float) -> pd.DataFrame:
    """Month-end derived modelling view of the daily master (the single source of truth)."""
    return features.to_monthly(get_master(economy, mtime))

@st.cache_data(show_spinner=False)
def arima_search(economy: str, series_col: str, mtime: float) -> dict | None:
    """Cached Box-Jenkins ARIMA order search for one monthly series."""
    series = get_monthly(economy, mtime)[series_col].dropna()
    return econometrics.arima_order_search(series)

@st.cache_data(show_spinner=False)
def garch_search(economy: str, series_col: str, mtime: float) -> dict | None:
    """Cached GARCH order search for one monthly series' first difference."""
    change = get_monthly(economy, mtime)[series_col].diff().dropna()
    return econometrics.garch_order_search(change)

@st.cache_data(show_spinner=False)
def var_orders(economy: str, columns: tuple[str, ...], mtime: float) -> pd.DataFrame | None:
    """Cached VAR information-criteria-by-lag table for a chosen set of series."""
    frame = get_monthly(economy, mtime)[list(columns)].dropna()
    return econometrics.var_order_table(frame)

def build_task_data(
    monthly: pd.DataFrame,
    economy: str,
    task: str,
    spec: dict | None,
    horizon: int,
    kind: str,
) -> dict | None:
    """Assemble the leakage-guarded feature matrix and target for one run, or None."""
    if task == "classification":
        policy = targets.CURATED_TARGETS.get(economy, {}).get("Policy rate", {})
        target_column = policy.get("column")
        extra_exclude = policy.get("extra_exclude", [])
        y = targets.direction_target(monthly, economy, horizon)
    else:
        target_column = spec["column"] if spec else None
        extra_exclude = spec.get("extra_exclude", []) if spec else []
        y = targets.value_target(monthly, target_column, horizon, kind) if target_column else None
    if y is None or target_column is None:
        return None
    data = features.build_matrix(monthly, y, target_column, extra_exclude=extra_exclude, target_lags=TARGET_LAGS)
    if data is None:
        return None
    data["target_column"] = target_column
    return data

MACRO_RATE_COLUMNS = ("rate_unemployment", "rate_participation", "rate_savings", "rate_homeownership")
ENGINEERED_SUFFIX = ("_ret_", "_ma_", "_vol_", "_chg_")

# Autoregressive lags of the target's own series (months). Legitimate for a forward
# target and the main predictor of a slow-moving level.
TARGET_LAGS = (1, 3, 6)

def group_of(feature: str) -> str:
    """Map a modelling feature (including engineered children) to a business block."""
    if feature.startswith(MACRO_RATE_COLUMNS):
        return "Macro"
    if feature.startswith(("rate_", "yld_", "sprd_")):
        return "Rates"
    if feature.startswith(("eq_", "fx_", "cmd_", "idx_")):
        return "Market"
    return "Macro"

def significance_colour(p: float) -> str:
    """Green text for a coefficient significant at p < 0.05, red otherwise (for Styler.map)."""
    if pd.isna(p):
        return ""
    return "color: #2ca02c" if p < 0.05 else "color: #d62728"

def split_label(preset: str) -> str:
    """Human-readable train/dev/test percentages for a split-preset key."""
    train_frac, valid_frac = pipeline.SPLIT_PRESETS[preset]
    return f"{train_frac:.0%} train / {valid_frac:.0%} dev / {1 - train_frac - valid_frac:.0%} test"


def show_figure(fig, caption: str | None = None, caption_help: str | None = None) -> None:
    """Render a Plotly figure with an optional caption, skipping None figures."""
    if fig is None:
        st.caption("Not enough data to draw this chart for the current selection.")
        return
    st.plotly_chart(fig, width="stretch")
    if caption:
        st.caption(caption, help=caption_help)

def assemble_bundle(models, X, y, task, cfg, *, cv_scores, params, classes, neural_names, sig, ensemble_info=None):
    """
    Score already-fitted models across the chronological splits and package the
    session bundle the tabs consume. No training happens here, so it serves both a
    fresh run and a reload of saved artifacts.
    """
    split_preset, metric = cfg["split_preset"], cfg["metric"]
    splits = pipeline.chronological_split(X, y, preset=split_preset)
    if task == "classification":
        labels = list(range(len(classes)))
        code = {label: i for i, label in enumerate(classes)}
        split_eval = {name: (Xs, np.array([code[v] for v in ys])) for name, (Xs, ys) in splits.items()}
    else:
        labels = None
        split_eval = splits
    board = evaluate.build_leaderboard(models, split_eval, task, labels=labels, thresholds=None)
    valid = board[f"Valid {metric}"].dropna()
    if valid.empty:
        best = None
    elif metric in ("RMSE", "MAE"):
        best = valid.idxmin()
    else:
        best = valid.idxmax()
    return {
        "sig": sig,
        "X": X,
        "y": y,
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
        "ensemble_info": ensemble_info,
        **cfg,
    }


def save_bundle_artifacts(bundle: dict) -> None:
    """Persist the picklable models and metadata sidecar so the bundle auto-reloads."""
    picklable = {n: m for n, m in bundle["models"].items() if n not in bundle["neural_names"]}
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
        extra={"classes": bundle["classes"], "ensemble_info": bundle.get("ensemble_info")},
    )
    key = registry.cache_key(
        bundle["economy"], bundle["task"], bundle["target_name"], bundle["horizon"], bundle["split_preset"]
    )
    registry.save_artifacts(picklable, meta, key)

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
    st.caption(
        "Training and diagnostics always use the full available data range for this economy. "
        "There is no evaluation-window control, so the shipped hyperparameters stay valid and "
        "results are reproducible."
    )
    cfg = st.columns([2, 3])
    task_label = cfg[0].radio(
        "Task", ["Direction (Hike/Hold/Cut)", "Value (regression)"], key="mdl_task"
    )
    task = "classification" if task_label.startswith("Direction") else "regression"

    value_targets = targets.available_value_targets(monthly, economy)
    spec = None
    kind = "level"
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
                help=interpret.TARGET_LEVEL_HELP,
            )
            spec = value_targets[target_name]

    horizon = st.slider(
        "Prediction horizon (months ahead)", 1, 12, 3, key="mdl_horizon", help=interpret.HORIZON_HELP
    )

    st.markdown("**Split, scoring and search**")
    row = st.columns(3)
    split_preset = row[0].selectbox(
        "Split preset",
        list(pipeline.SPLIT_PRESETS),
        index=1,
        format_func=split_label,
        key="mdl_split",
        help=interpret.SPLIT_HELP,
    )
    metrics = evaluate.CLASSIFICATION_METRICS if task == "classification" else evaluate.REGRESSION_METRICS
    metric = row[1].selectbox(
        "Scoring metric",
        list(metrics),
        index=0 if task == "classification" else len(metrics) - 1,
        key="mdl_metric",
        help=interpret.METRIC_HELP,
    )
    budget = row[2].selectbox("Training budget", list(pipeline.BUDGET_TIERS), key="mdl_budget", help=interpret.BUDGET_HELP)

    include_neural = st.toggle(
        "Include neural nets (LSTM / MLP - slower, opt-in)",
        value=False,
        key="mdl_neural",
        help=interpret.NEURAL_HELP,
    )
    include_conv = st.toggle("Add Conv1D", value=False, key="mdl_conv", disabled=not include_neural)
    power_transform = False

    with st.expander("Why monthly, and how splitting / CV work?"):
        st.markdown(interpret.MONTHLY_RATIONALE)
        st.markdown(interpret.SPLIT_HELP)
        st.markdown(interpret.CV_HELP)

    data = build_task_data(monthly, economy, task, spec, horizon, kind)
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
        if task == "classification":
            st.caption(interpret.class_balance_note(y))
        else:
            st.caption(interpret.target_level_note(econometrics.stationarity(y), target_name))
        st.caption(
            "The target's own column and its trivially-derived features are excluded from the inputs.",
            help=interpret.LEAKAGE_HELP,
        )

current_sig = (
    economy,
    task,
    target_name,
    horizon,
    kind if task == "regression" else "",
    split_preset,
    metric,
    power_transform,
    include_neural,
    include_conv,
    mtime,
)

bundle = st.session_state.get("mdl_bundle")
if (bundle is None or bundle["sig"] != current_sig) and X is not None and y is not None:
    key = registry.cache_key(economy, task, target_name, horizon, split_preset)
    loaded = registry.load_artifacts(key)
    if loaded is not None and not registry.is_stale(loaded["metadata"], registry.data_signature(X, y)):
        meta = loaded["metadata"]
        cfg_load = {
            "economy": economy,
            "task": task,
            "target_name": target_name,
            "target_column": target_column,
            "horizon": horizon,
            "kind": kind,
            "split_preset": split_preset,
            "metric": metric,
        }
        st.session_state["mdl_bundle"] = assemble_bundle(
            loaded["models"],
            X,
            y,
            task,
            cfg_load,
            cv_scores=meta.get("cv_scores", {}),
            params=meta.get("params", {}),
            classes=meta.get("classes"),
            neural_names=set(),
            ensemble_info=meta.get("ensemble_info"),
            sig=current_sig,
        )

with tab_train:
    st.caption(interpret.LEADERBOARD_HELP)
    st.caption(
        "Trained models auto-save and reload on their own, so this tab, Diagnostics and Scenario "
        "stay populated across tab and page switches. Click Train / refit only to rerun the search "
        "on the full current data (for example after the dataset grows) or after changing the setup; "
        "it overwrites the saved models."
    )
    if X is None or y is None:
        st.info("Complete the Setup tab first.")
    else:
        if st.button("Train / refit on current data", type="primary", key="mdl_train"):
            splits = pipeline.chronological_split(X, y, preset=split_preset)
            if task == "classification":
                train_classes = set(pd.Series(splits["Train"][1]).dropna().unique())
                if train_classes != set(pd.Series(y).dropna().unique()):
                    st.warning(
                        "The training split does not contain every class present later in the "
                        "sample, so that class will be unreliable on this small window."
                    )

            n_sklearn = len(pipeline.model_roster(task))
            n_neural = (2 + (1 if include_conv else 0)) if include_neural else 0
            total = n_sklearn + n_neural
            bar = st.progress(0.0, text="Preparing the training split...")
            start = time.perf_counter()
            counter = {"done": 0}

            def elapsed_mmss():
                minutes, seconds = divmod(int(time.perf_counter() - start), 60)
                return f"{minutes:d}:{seconds:02d}"

            def report(stage):
                bar.progress(
                    min(counter["done"] / total, 1.0),
                    text=f"[{counter['done']}/{total}] {stage} : {elapsed_mmss()} elapsed",
                )

            def sklearn_progress(done, _total, name, cv_score):
                counter["done"] = done
                stage = f"Searched {name}" + (f" (CV {cv_score:.3f})" if cv_score is not None else "")
                report(stage)

            roster = pipeline.train_roster(
                splits["Train"][0],
                splits["Train"][1],
                task,
                budget=budget,
                scoring=metric,
                power_transform=power_transform,
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
                    counter["done"] = n_sklearn + done
                    report(f"Fitted {name} (neural)")

                try:
                    fitted_neural = neural_mod.fit_neural_models(
                        splits["Train"][0],
                        y_train_enc,
                        task,
                        include_conv=include_conv,
                        progress=neural_progress,
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
                "kind": kind,
                "split_preset": split_preset,
                "metric": metric,
            }
            bundle = assemble_bundle(
                models,
                X,
                y,
                task,
                cfg_now,
                cv_scores=roster["cv_scores"],
                params=roster["params"],
                classes=classes,
                neural_names=neural_names,
                ensemble_info=ensemble_info,
                sig=current_sig,
            )
            st.session_state["mdl_bundle"] = bundle
            bar.progress(1.0, text=f"Finished: trained {total} models in {elapsed_mmss()}.")
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
            task_metrics = (
                evaluate.CLASSIFICATION_METRICS
                if bundle["task"] == "classification"
                else evaluate.REGRESSION_METRICS
            )
            if best is not None:
                st.success(interpret.best_model_sentence(best, metric, board.loc[best, f"Test {metric}"]))
                summary = pd.DataFrame(
                    {m: [board.loc[best, f"{s} {m}"] for s in ("Train", "Valid", "Test")] for m in task_metrics},
                    index=("Train", "Dev", "Test"),
                )
                st.markdown(f"**{best} across the splits**")
                st.dataframe(summary.style.format(precision=3, na_rep="-"), width="stretch")
                note = interpret.overfit_note(
                    board.loc[best, f"Train {metric}"], board.loc[best, f"Test {metric}"], metric
                )
                if note:
                    st.caption(note)

            metric_order = [metric] + [m for m in task_metrics if m != metric]
            display_cols = [f"{s} {m}" for m in metric_order for s in ("Train", "Valid", "Test")]
            ascending = metric in ("RMSE", "MAE")
            board_view = board.sort_values(f"Valid {metric}", ascending=ascending)[display_cols]
            board_view = board_view.rename(columns=lambda c: c.replace("Valid", "Dev"))

            def highlight_best(row):
                colour = "background-color: rgba(84,162,75,0.18)"
                return [colour if row.name == best else "" for _ in row]

            st.markdown("**Full leaderboard** (grouped by metric, best model on top)")
            st.dataframe(
                board_view.style.format(precision=3, na_rep="-").apply(highlight_best, axis=1),
                width="stretch",
            )

            with st.expander("Model configuration (CV scores and chosen hyperparameters)"):
                st.caption(interpret.CV_BUDGET_HELP)
                config_rows = []
                for name, params in bundle["params"].items():
                    tuned = ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items())
                    config_rows.append(
                        {
                            "Model": name,
                            f"CV {metric}": bundle["cv_scores"].get(name),
                            "Tuned hyperparameters": tuned or "-",
                        }
                    )
                config_df = pd.DataFrame(config_rows).set_index("Model")
                st.dataframe(
                    config_df.style.format({f"CV {metric}": "{:.3f}"}, na_rep="-"),
                    width="stretch",
                )

            ens_info = bundle.get("ensemble_info")
            if ens_info:
                with st.expander("Ensemble composition (blend weights, stack members, meta-model)"):
                    st.caption(interpret.ENSEMBLE_HELP)
                    weights = pd.Series(ens_info["weights"], name="Blend weight").sort_values(ascending=False)
                    st.markdown("**Blend** - inverse-validation-error weighted average of:")
                    st.dataframe(weights.to_frame().style.format("{:.3f}"), width="stretch")
                    st.markdown(
                        f"**Stack** - a {ens_info['meta_model']} meta-model trained on the "
                        f"validation-set outputs of: {', '.join(ens_info['members'])}."
                    )

with tab_diag:
    bundle = st.session_state.get("mdl_bundle")
    if bundle is None or bundle["sig"] != current_sig:
        st.info("Train the leaderboard for the current setup first, then return here.")
    else:
        diag_task = bundle["task"]
        model_names = list(bundle["models"])
        default_ix = model_names.index(bundle["best"]) if bundle["best"] in model_names else 0
        model_name = st.selectbox("Model to inspect", model_names, index=default_ix, key="mdl_diag_model")
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
            y_show = np.concatenate(
                [np.asarray(splits["Train"][1]), np.asarray(y_valid), np.asarray(y_test)]
            )
        else:
            key_map = {"Test": "Test", "Dev": "Valid", "Train": "Train"}
            X_show, y_show = splits[key_map[eval_choice]]
        st.caption(
            "Class thresholds are always tuned on the dev split; this control only chooses which "
            "split the diagnostics below are measured on. Test is the honest out-of-sample read; "
            "Train and Dev help you spot overfitting. Importance, SHAP and permutation stay on their "
            "principled splits (permutation on dev, SHAP on test)."
        )

        if diag_task == "classification":
            class_labels = list(bundle["classes"])
            y_valid_str = pd.Series(y_valid).astype(str).to_numpy()
            y_show_str = pd.Series(y_show).astype(str).to_numpy()
            proba_valid = model.predict_proba(X_valid)
            proba_show = model.predict_proba(X_show)
            thresholds = evaluate.youden_thresholds(y_valid_str, proba_valid, class_labels)
            pred_show = evaluate.predict_with_thresholds(proba_show, thresholds, class_labels)
            metrics = evaluate.classification_metrics(y_show_str, pred_show, class_labels)
            auc = evaluate.roc_auc_macro_ovr(y_show_str, proba_show, class_labels)
            st.caption(interpret.roc_auc_verdict(auc, split=eval_choice))

            row1 = st.columns(2)
            with row1[0]:
                show_figure(plots.roc_curves(evaluate.roc_curve_data(y_show_str, proba_show, class_labels)), interpret.ROC_HELP)
            with row1[1]:
                show_figure(plots.pr_curves(evaluate.pr_curve_data(y_show_str, proba_show, class_labels)), interpret.PR_HELP)
            row2 = st.columns(2)
            with row2[0]:
                show_figure(plots.lift_curves(evaluate.lift_curve_data(y_show_str, proba_show, class_labels)), interpret.LIFT_HELP)
            with row2[1]:
                st.caption(
                    "The confusion matrix and probabilities use the dev-tuned Youden thresholds, so "
                    "they depend on the operating point (the curves above do not).",
                    help=interpret.THRESHOLD_HELP,
                )
                normalise = st.toggle(
                    "Row-normalise (share of each actual class)", value=False, key="mdl_diag_cm_norm"
                )
                show_figure(
                    plots.confusion_heatmap(metrics["confusion"], normalize=normalise),
                    interpret.confusion_verdict(metrics["confusion"]),
                    interpret.CONFUSION_HELP,
                )
            show_figure(plots.probability_histogram(proba_show, class_labels, y_true=y_show_str), interpret.PROB_HIST_HELP)
        else:
            pred_show = model.predict(X_show)
            r2 = evaluate.regression_metrics(y_show, pred_show)["R2"]
            st.caption(interpret.regression_fit_verdict(r2))
            reg_row = st.columns(2)
            with reg_row[0]:
                show_figure(plots.predicted_vs_actual(y_show, pred_show, index=X_show.index))
            with reg_row[1]:
                show_figure(plots.residual_plot(y_show, pred_show, index=X_show.index))
            st.caption(interpret.residual_verdict(y_show, pred_show), help=interpret.RESIDUAL_HELP)

        scoring = evaluate.CV_SCORING[bundle["metric"]]
        native = explain.native_importance(model, X_test.columns)
        with st.spinner("Computing permutation importance..."):
            try:
                perm = explain.permutation_importance_scores(model, Xp_valid, yp_valid, scoring=scoring)
            except Exception:
                perm = None
        st.markdown("**Feature importance**")
        imp_row = st.columns(2)
        with imp_row[0]:
            st.markdown("**Native**")
            if native is not None:
                show_figure(plots.importance_bar(native, title=None), interpret.importance_sentence(native))
            else:
                st.caption(
                    "This model type exposes no native importance. "
                    "Read the permutation importance on the right instead."
                )
        with imp_row[1]:
            st.markdown("**Permutation**")
            if perm is not None:
                show_figure(plots.importance_bar(perm, title=None), interpret.importance_sentence(perm))
            else:
                st.caption("Permutation importance could not be computed for this model on this split.")
        with st.expander("How to read feature importance"):
            st.markdown(interpret.IMPORTANCE_HELP)

        base_imp = native if native is not None else perm
        if base_imp is not None:
            group_imp = explain.feature_group_importance(base_imp, group_of)
            show_figure(
                plots.importance_bar(group_imp, title="Importance by business block", value_label="Group importance (0-100)"),
                interpret.group_importance_sentence(group_imp),
                interpret.GROUP_IMPORTANCE_HELP,
            )

        st.markdown("**SHAP contributions (tree and boosting models)**")
        if explain.is_tree_model(model):
            class_ix = 0
            if diag_task == "classification" and len(class_labels) > 2:
                chosen = st.selectbox("SHAP class", class_labels, key="mdl_diag_shap_class")
                class_ix = class_labels.index(chosen)
            shap_data = explain.tree_shap(model, X_test)
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
            chosen_month = st.selectbox(
                "Month to explain", month_labels, index=default_row, key="mdl_diag_shap_row"
            )
            row_pos = month_labels.index(chosen_month)
            X_row = X_test.iloc[[row_pos]]
            local = explain.local_shap_series(model, X_row, class_index=class_ix)
            show_figure(
                plots.shap_local_bar(local, title=f"Why the model predicted {chosen_month}"),
                "Signed contributions for the selected month (green pushes the prediction up, red pulls it down).",
                interpret.SHAP_HELP,
            )
        else:
            st.caption(
                "SHAP here uses the exact TreeExplainer, so it is available for tree and boosting "
                "models only (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost). For this "
                "model read the permutation importance above, which is model-agnostic."
            )

        st.markdown("**Partial dependence**")
        pdp_options = list(base_imp.index[:15]) if base_imp is not None else list(X_test.columns)
        pdp_feature = st.selectbox("Feature (importance-ranked)", pdp_options, key="mdl_diag_pdp")
        pdp_labels = class_labels if diag_task == "classification" else None
        try:
            pdp = explain.partial_dependence_data(model, X_test, pdp_feature)
            show_figure(
                plots.partial_dependence_plot(pdp, labels=pdp_labels, feature=pdp_feature),
                "Every other feature is held at its observed value while this one is swept.",
                interpret.PDP_HELP,
            )
        except Exception:
            st.caption("Partial dependence is unavailable for this model.")

        st.markdown("**Econometric baseline**")
        st.caption(interpret.ECON_BASELINE_HELP)
        st.caption(
            "This baseline is fit in-sample on the full monthly frame (not a held-out split): it is an "
            "interpretable reference judged on classical assumptions, not a predictive competitor. To "
            "stay well-conditioned it enters only the 10 features most correlated with the target, "
            "standardised, so the coefficient sizes are comparable."
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
                    "target's in-sample variation; the F p-value tests whether it beats an "
                    "intercept-only model (small = yes)."
                )
                st.dataframe(
                    ols["coefficients"].style.format(precision=4).map(
                        significance_colour, subset=["p-value"]
                    ),
                    width="stretch",
                )
                st.caption("Green coefficients are significant at p < 0.05, red are not.")
                assume = st.columns(3)
                assume[0].metric("Durbin-Watson", f"{ols['durbin_watson']:.2f}")
                assume[1].metric("Normality (JB) p", interpret.format_pvalue(ols["jarque_bera_p"]))
                assume[2].metric("Condition no.", f"{ols['condition_number']:,.0f}")
                st.caption(interpret.ols_assumptions_note(ols))
                st.caption(
                    f"Cook's distance flags {len(ols['influential'])} influential month(s) above the "
                    f"4/n = {ols['cooks_threshold']:.3f} rule of thumb - months whose removal would "
                    "most change the fit."
                )
                st.bar_chart(ols["cooks"])
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
                    f"baseline class ({logit['baseline_class']}, the dominant Hold outcome). Cook's "
                    "distance is an OLS diagnostic and is not shown for the classifier."
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
        default_ix = model_names.index(bundle["best"]) if bundle["best"] in model_names else 0
        model_name = st.selectbox("Model", model_names, index=default_ix, key="mdl_scn_model")
        model = bundle["models"][model_name]

        latest = X_full.iloc[-1]
        if scn_task == "classification":
            seed_y = pd.Series(pd.Categorical(bundle["y"]).codes, index=bundle["y"].index).astype(float)
        else:
            seed_y = bundle["y"]
        driver_options = econometrics.top_correlated_features(X_full, seed_y, 15)
        default_drivers = driver_options[:5]

        if st.button("Revert to defaults", key="mdl_scn_revert"):
            for col in driver_options:
                st.session_state.pop(f"mdl_scn_{col}", None)
            st.session_state.pop("mdl_scn_drivers", None)
            st.rerun()

        drivers = st.multiselect(
            "Drivers to vary", driver_options, default=default_drivers, key="mdl_scn_drivers"
        )
        anchor_month = X_full.index[-1]
        target_month = anchor_month + pd.offsets.MonthEnd(bundle["horizon"])
        st.caption(
            "Drivers default to the five features most correlated with the target; the dropdown lists the "
            f"top 15. Every feature not shown is held at its {anchor_month:%Y-%m} value, so the reading is a "
            "ceteris-paribus response of the chosen drivers."
        )

        scenario = latest.copy()
        slider_cols = st.columns(2)
        for i, drv in enumerate(drivers):
            lo, hi = float(X_full[drv].min()), float(X_full[drv].max())
            if lo == hi:
                continue
            scenario[drv] = slider_cols[i % 2].slider(
                drv, lo, hi, float(latest[drv]), step=(hi - lo) / 100, key=f"mdl_scn_{drv}"
            )

        dtypes = X_full.dtypes.to_dict()
        scenario_row = scenario.to_frame().T.astype(dtypes)
        base_row = latest.to_frame().T.astype(dtypes)
        try:
            if scn_task == "classification":
                class_labels = list(bundle["classes"])
                proba = model.predict_proba(scenario_row)[0]
                pred = class_labels[int(np.argmax(proba))]
                st.metric(
                    f"Predicted decision for {target_month:%Y-%m} "
                    f"({bundle['horizon']} months after {anchor_month:%Y-%m})",
                    pred,
                )
                show_figure(
                    plots.class_probability_bar(
                        pd.Series(proba, index=class_labels), title="Class probabilities"
                    ),
                    "Probability the model assigns to each forward decision under the scenario.",
                )
            else:
                base_pred = float(model.predict(base_row)[0])
                scn_pred = float(model.predict(scenario_row)[0])
                current_level = float(monthly[bundle["target_column"]].dropna().iloc[-1])
                out = st.columns(2)
                out[0].metric(
                    f"Baseline level for {target_month:%Y-%m}",
                    f"{base_pred:.3f}",
                    delta=f"{base_pred - current_level:+.3f} vs {anchor_month:%Y-%m}",
                )
                out[1].metric(
                    "Scenario level",
                    f"{scn_pred:.3f}",
                    delta=f"{scn_pred - base_pred:+.3f} vs baseline",
                )
                st.caption(
                    f"Anchor: {bundle['target_column']} was {current_level:.3f} in {anchor_month:%Y-%m}. "
                    f"The baseline holds every driver at that month and predicts the {target_month:%Y-%m} "
                    "level; the scenario moves only the chosen drivers. Read the deltas as anchor -> "
                    "baseline -> scenario."
                )
        except Exception as exc:
            st.warning(f"This model could not score the scenario row: {exc}")

with tab_forecast:
    st.caption(interpret.FORECAST_HELP)
    st.caption(interpret.FORECAST_INDEPENDENCE)
    numeric = monthly.select_dtypes("number")
    series_options = sorted(c for c in numeric.columns if not any(s in c for s in ENGINEERED_SUFFIX))
    policy_col = targets.CURATED_TARGETS.get(economy, {}).get("Policy rate", {}).get("column")
    default_series = series_options.index(policy_col) if policy_col in series_options else 0
    steps = st.slider(
        "Forecast horizon (months)",
        3,
        24,
        12,
        key="mdl_fc_steps",
        help="How many months ahead these time-series models project. Independent of the Setup horizon.",
    )

    arima_tab, garch_tab, var_tab = st.tabs(["ARIMA / SARIMA", "GARCH volatility", "VAR"])

    with arima_tab:
        st.caption(interpret.ARIMA_HELP)
        series_col = st.selectbox("Series", series_options, index=default_series, key="mdl_fc_arima_series")
        series = monthly[series_col].dropna()
        st.caption(interpret.stationarity_verdict(econometrics.stationarity(series), series_col))
        show_figure(
            plots.acf_pacf_plot(econometrics.acf_pacf(series)),
            "Stems outside the shaded band are significant: the ACF hints at the MA order (q), the PACF "
            "at the AR order (p). The automatic search below considers differencing (d) for you.",
        )

        search = arima_search(economy, series_col, mtime)
        if search is None:
            st.info("The ARIMA order search needs at least 20 observations.")
            best_order = (1, 0, 0)
        else:
            best_order = tuple(int(v) for v in search["best"])
            st.caption(f"Automatic order (lowest {search['ic'].upper()}): ARIMA{best_order}. Candidates:")
            st.dataframe(search["leaderboard"], hide_index=True, width="stretch")

        override = st.toggle("Override the order manually", value=False, key="mdl_fc_arima_override")
        if override:
            st.warning(
                "Manual orders are not validated - an ill-chosen order can produce a misleading or "
                "non-converging fit. The automatic order above is usually the safer choice."
            )
            order = st.columns(3)
            p = order[0].number_input("AR (p)", 0, 5, best_order[0], key=f"mdl_fc_p_{series_col}")
            d = order[1].number_input("I (d)", 0, 2, best_order[1], key=f"mdl_fc_d_{series_col}")
            q = order[2].number_input("MA (q)", 0, 5, best_order[2], key=f"mdl_fc_q_{series_col}")
            chosen_order = (p, d, q)
        else:
            chosen_order = best_order

        seasonal_order = (0, 0, 0, 0)
        if st.toggle("Seasonal (SARIMA)", value=False, key="mdl_fc_seasonal"):
            st.caption("The seasonal period is fixed to 12 for the monthly frame (annual seasonality).")
            s = st.columns(3)
            seasonal_order = (
                s[0].number_input("Seasonal AR (P)", 0, 2, 0, key="mdl_fc_P"),
                s[1].number_input("Seasonal I (D)", 0, 1, 0, key="mdl_fc_D"),
                s[2].number_input("Seasonal MA (Q)", 0, 2, 0, key="mdl_fc_Q"),
                12,
            )

        res = econometrics.fit_arima(series, order=chosen_order, seasonal_order=seasonal_order)
        if res is None:
            st.info("The ARIMA fit needs at least 20 observations and a valid order.")
        else:
            diag = econometrics.arima_diagnostics(res)
            crit = st.columns(2)
            crit[0].metric("AIC", f"{diag['aic']:,.1f}")
            crit[1].metric("BIC", f"{diag['bic']:,.1f}")
            st.caption(interpret.ljung_box_verdict(diag))
            fc_index = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
            fc = {k: pd.Series(np.asarray(v), index=fc_index) for k, v in econometrics.arima_forecast(res, steps).items()}
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
        garch_col = st.selectbox(
            "Series (its monthly change is modelled)", series_options, index=default_series, key="mdl_fc_garch_series"
        )
        st.caption(
            "GARCH is defined on a roughly zero-mean series, so the first difference (monthly change) of "
            "the level is modelled here, never the level itself."
        )
        change = monthly[garch_col].diff().dropna()
        gsearch = garch_search(economy, garch_col, mtime)
        if gsearch is None:
            st.info("GARCH needs at least 50 monthly changes for this series.")
            best_pq = (1, 1)
        else:
            best_pq = tuple(int(v) for v in gsearch["best"])
            st.caption(f"Automatic order (lowest {gsearch['ic'].upper()}): GARCH{best_pq}. Candidates:")
            st.dataframe(gsearch["leaderboard"], hide_index=True, width="stretch")
        res = econometrics.fit_garch(change, p=best_pq[0], q=best_pq[1])
        if res is None:
            st.info("GARCH needs at least 50 monthly changes for this series.")
        else:
            fc = econometrics.garch_forecast(res, steps)
            fc_index = pd.date_range(change.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
            show_figure(
                plots.garch_volatility_plot(fc, forecast_index=fc_index),
                "Blue is the in-sample conditional volatility (how much the monthly change swung, month "
                "by month); red is its forecast. Rising volatility means a more turbulent series ahead, "
                "not a higher level.",
            )
            close = {**fc, "fitted_volatility": fc["fitted_volatility"].iloc[-12:]}
            show_figure(
                plots.garch_volatility_plot(close, forecast_index=fc_index, title="Conditional volatility - close-up"),
                "Close-up on the last 12 months of volatility and the forecast.",
            )

    with var_tab:
        st.caption(interpret.VAR_HELP)
        preferred = [
            policy_col,
            "cpi_sticky_core" if economy == "usa" else "hicp_all",
            "rate_unemployment",
        ]
        default_var = [c for c in preferred if c in series_options][:3]
        chosen = st.multiselect(
            "Series (2-3)", series_options, default=default_var, max_selections=3, key="mdl_fc_var_series"
        )
        if len(chosen) < 2:
            st.info("Pick at least two series for the vector autoregression.")
        else:
            var_frame = monthly[chosen].dropna()
            res = econometrics.fit_var(var_frame)
            if res is None:
                st.info("VAR needs at least 30 aligned rows and two or more series.")
            else:
                st.caption(interpret.var_lag_sentence(res.k_ar, "aic"))
                orders = var_orders(economy, tuple(chosen), mtime)
                if orders is not None:
                    with st.expander("Lag-order information criteria"):
                        st.dataframe(orders, width="stretch")
                vf = econometrics.var_forecast(res, steps)
                fc_index = pd.date_range(var_frame.index[-1] + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
                fan_cols = st.columns(min(len(res.names), 2))
                for i, name in enumerate(res.names):
                    fc = {k: pd.Series(vf[k][name].to_numpy(), index=fc_index) for k in ("mean", "lower", "upper")}
                    with fan_cols[i % len(fan_cols)]:
                        show_figure(plots.forecast_fan(var_frame[name].iloc[-120:], fc, name=name))
                st.caption(
                    "Each fan shows the last 120 months of history and the joint VAR forecast (dashed "
                    "mean, shaded 95% interval) for that series."
                )
                st.markdown("**Impulse responses**")
                show_figure(
                    plots.irf_grid(econometrics.var_irf(res, steps)),
                    "Row = responding series, column = shocked series. Each panel traces one series' "
                    "reaction over the months after a one-off shock to another; the dashed line is zero.",
                )
                st.markdown("**Forecast error variance decomposition**")
                fevd = econometrics.var_fevd(res, steps)
                show_figure(
                    plots.fevd_area(fevd),
                    "Each panel splits a series' forecast uncertainty into the share coming from each "
                    "series' shocks; the shares sum to one.",
                )
                st.caption(interpret.fevd_verdict(fevd))
