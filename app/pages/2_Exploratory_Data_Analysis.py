import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.dataset_builder import ECONOMIES, cache_exists, load_master, master_mtime
from src.regimes import available_regimes, regime_source_columns
from utils import interpret
from utils.plots import (
    box_or_violin,
    category_counts,
    cluster_selection_plot,
    compare_lines,
    distribution_plot,
    explained_variance_plot,
    grouped_histogram,
    levels_with_overlay,
    matrix_heatmap,
    probability_plot,
    projection_scatter,
    scatter_ols,
)
from utils.stats import (
    categorical_association,
    compare_groups,
    correlation_matrix,
    cramers_v_matrix,
    describe_extended,
    mean_ci,
    normality_battery,
    partial_correlation,
    stationarity_and_cointegration,
    variance_inflation_factors,
)
from utils.structure import (
    cluster_agreement,
    hopkins_statistic,
    kmeans_labels,
    kmeans_sweep,
    pca_summary,
    standardize,
)

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

# Columns carrying an engineered suffix; the picker offers base levels only and
# derives returns/changes on the fly via the transform selector.
ENGINEERED_SUFFIX = re.compile(r"_(?:ret|ma|vol)_\d+d$|_chg_\w+$")

# Log return is economically meaningless on rates, yields and spreads,
# so only differencing is offered.
LEVELS_ONLY = {"Rates & yields", "Spreads"}

# Feature shown on first load, per economy.
DEFAULT_VARIABLE = {"usa": "rate_ff_eff", "eurozone": "rate_ecb_dep"}

# Semantic, theme-aligned colours for the categorical regimes.
REGIME_COLORS = {
    "Hiking": "#FF4B4B",
    "Holding": "#9aa0a6",
    "Easing": "#4c78a8",
    "Recession": "#FF4B4B",
    "Expansion": "#54a24b",
    "Inverted": "#FF4B4B",
    "Normal": "#4c78a8",
}

# Diverging scale for correlations: blue (negative) -> white (0) -> Streamlit red (positive).
CORR_COLORSCALE = [[0.0, "#4c78a8"], [0.5, "#f5f5f5"], [1.0, "#FF4B4B"]]


@st.cache_data(show_spinner=False)
def get_master(economy: str, mtime: float) -> pd.DataFrame:
    """Cached read of a master dataset, date-indexed; invalidated when the file changes."""
    return load_master(economy).set_index("date").sort_index()


def family_of(column: str) -> str:
    """Map a base column to its display family for the variable picker."""
    if column.startswith(("rate_", "yld_")):
        return "Rates & yields"
    if column.startswith("sprd_"):
        return "Spreads"
    return "Macro & market levels"


def variables_by_family(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group the base-level columns (engineered features excluded) by family."""
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        if not ENGINEERED_SUFFIX.search(col):
            groups.setdefault(family_of(col), []).append(col)
    return {fam: sorted(cols) for fam, cols in groups.items()}


def allowed_transforms(family: str) -> list[str]:
    """Transforms valid for a family; rates/spreads expose differencing only."""
    if family in LEVELS_ONLY:
        return ["Level", "First difference"]
    return ["Level", "Log return (%)", "First difference"]


def collapse_to_monthly(level: pd.Series) -> pd.Series:
    """
    Down-sample to one value per month so tests see near-independent observations.

    The master data is daily with low-frequency macro series forward-filled and
    policy rates held as step functions, so consecutive daily rows are largely
    redundant and inflate the sample size. Taking the month-end value removes most
    of that redundancy while preserving genuine held periods proportionally.

    Args:
        level (pd.Series): Date-indexed level series.

    Returns:
        pd.Series: Month-end sampled series with gaps dropped.
    """
    return level.resample("ME").last().dropna()


def apply_transform(level: pd.Series, transform: str) -> pd.Series:
    """Apply the chosen transform to a level series (level / log return / difference)."""
    if transform == "Log return (%)":
        return np.log(level.where(level > 0)).diff() * 100
    if transform == "First difference":
        return level.diff()
    return level


def suggest_bins(series: pd.Series, fallback: int = 30, cap: int = 60) -> int:
    """
    Freedman-Diaconis bin count for a series, clamped to a sensible range.

    Bin width = 2 * IQR / n**(1/3); the count is the data range over that width.
    Falls back to a fixed default when the IQR or range is zero.

    Args:
        series (pd.Series): Numeric variable to bin.
        fallback (int): Bin count when the rule is undefined.
        cap (int): Upper limit so heavy-tailed series don't explode the count.

    Returns:
        int: Suggested number of bins.
    """
    values = series.dropna().to_numpy()
    n = values.size
    if n < 2:
        return fallback
    q1, q3 = np.percentile(values, [25, 75])
    width = 2 * (q3 - q1) / np.cbrt(n)
    span = values.max() - values.min()
    if width <= 0 or span <= 0:
        return fallback
    return int(np.clip(round(span / width), 5, cap))


def prepare_series(level: pd.Series, transform: str, monthly: bool, date_range: tuple) -> pd.Series:
    """
    Build the analysis series: down-sample, transform, then slice to the date range.

    Monthly down-sampling runs before the transform so returns/changes are computed
    between month-end observations, not between forward-filled daily duplicates.

    Args:
        level (pd.Series): Date-indexed base-level series.
        transform (str): One of the labels from `allowed_transforms`.
        monthly (bool): Down-sample to month-end before transforming.
        date_range (tuple): (start_date, end_date) to slice to, inclusive.

    Returns:
        pd.Series: Transformed, sliced, missing-dropped series.
    """
    if monthly:
        level = collapse_to_monthly(level)
    series = apply_transform(level, transform).dropna()
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    return series[(series.index >= start) & (series.index <= end)]


def prepare_frame(source: pd.DataFrame, columns: list[str], transform: str, date_range: tuple) -> pd.DataFrame:
    """
    Build an aligned multi-column frame for the relationship views.

    Applies the transform to each column on the shared daily index (no monthly
    down-sampling, which would misalign columns), slices to the date range
    and drops rows with any missing value so every pair is complete.

    Args:
        source (pd.DataFrame): Master dataset, date-indexed.
        columns (list[str]): Base-level columns to include.
        transform (str): 'Level' or 'First difference' (valid for every family).
        date_range (tuple): (start_date, end_date) to slice to, inclusive.

    Returns:
        pd.DataFrame: Transformed, sliced, missing-dropped frame.
    """
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    frame = source.loc[(source.index >= start) & (source.index <= end), columns]
    return frame.apply(lambda s: apply_transform(s, transform)).dropna()


def freeze_corr_selection() -> None:
    """Mark the correlation variable picker as user-edited so it stops auto-updating."""
    st.session_state["corr_touched"] = True


def freeze_struct_selection() -> None:
    """Mark the clustering feature picker as user-edited so it stops auto-updating."""
    st.session_state["struct_touched"] = True


def representative_features(corr: pd.DataFrame, k: int = 8) -> list[str]:
    """
    Pick k features that are as mutually uncorrelated as possible (greedy).

    Seeds with the most central feature (highest mean absolute correlation with
    the rest), then repeatedly adds the feature whose strongest correlation to the
    already-chosen set is smallest. The result spans the data's distinct dimensions,
    so the default matrix shows a spread of relationships rather than a block of
    near-duplicate, all-red cells.

    Args:
        corr (pd.DataFrame): Square correlation matrix over the candidate features.
        k (int): Number of features to return.

    Returns:
        list[str]: The chosen feature names.
    """
    abs_corr = corr.abs()
    for col in abs_corr.columns:
        abs_corr.loc[col, col] = pd.NA
    if abs_corr.shape[1] <= k:
        return list(corr.columns)

    chosen = [abs_corr.mean().idxmax()]
    while len(chosen) < k:
        remaining = [c for c in abs_corr.columns if c not in chosen]
        max_to_chosen = abs_corr.loc[remaining, chosen].max(axis=1)
        chosen.append(max_to_chosen.idxmin())
    return chosen


@st.cache_data(show_spinner="Computing PCA and clustering - this can take a few seconds...")
def run_structure(economy: str, columns: tuple[str, ...], date_range: tuple, mtime: float) -> dict | None:
    """Standardise the chosen level features and precompute the expensive structure views."""
    source = get_master(economy, mtime)
    frame = prepare_frame(source, list(columns), "Level", date_range)
    scaled = standardize(frame)
    if scaled is None or scaled.shape[0] < 10:
        return None
    return {
        "frame": frame[scaled.columns],
        "scaled": scaled,
        "hopkins": hopkins_statistic(scaled.drop_duplicates()),
        "pca": pca_summary(scaled),
        "sweep": kmeans_sweep(scaled),
    }


st.title("Exploratory Data Analysis")

if not cache_exists():
    st.info("No cached data yet. Build it on the main page first.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    economy = st.selectbox("Economy", ECONOMIES, format_func=str.upper)
    if st.session_state.get("last_economy") != economy:
        st.session_state["feat_1"] = DEFAULT_VARIABLE.get(economy, "None")
        for k in ("feat_2", "feat_3", "feat_4"):
            st.session_state[k] = "None"
        for k in (
            "scatter_x",
            "scatter_y",
            "scatter_color",
            "pc_x",
            "pc_y",
            "pc_covars",
            "corr_cols",
            "reg_a",
            "reg_b",
            "struct_cols",
        ):
            st.session_state.pop(k, None)
        st.session_state["corr_touched"] = False
        st.session_state["struct_touched"] = False
        st.session_state["last_economy"] = economy
    df = get_master(economy, master_mtime(economy))
    regimes = available_regimes(df, economy)
    families = variables_by_family(df)
    base_columns = [c for cols in families.values() for c in cols]

    dmin, dmax = df.index.min().date(), df.index.max().date()
    date_range = st.slider("Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))

tab_overview, tab_dist, tab_rel, tab_regime, tab_structure = st.tabs(
    [
        "Overview",
        "Distribution & normality",
        "Relationships",
        "Regime comparison",
        "Structure / regimes",
    ]
)
with tab_overview:
    options = ["None", *base_columns]

    top = st.columns([3, 2, 2])
    chart_transform = top[0].radio(
        "Transform (charts)",
        ["Level", "First difference"],
        horizontal=True,
        key="overview_transform",
    )
    monthly = top[1].toggle(
        "Collapse to monthly",
        value=True,
        key="overview_monthly",
        help=(
            "Sample one value per month instead of daily, so held policy rates and "
            "forward-filled macro series don't flood the chart with flat daily repeats. "
            "Turn off for the full daily series."
        ),
    )
    overlay_returns = False
    if chart_transform == "Level":
        overlay_returns = top[2].toggle(
            "Overlay log return (%)",
            value=True,
            key="overview_overlay",
            help=(
                "Add each feature's log return on a secondary axis, in the same panel, so you "
                "can read how volatile it was against its level. Features already measured in % "
                "(rates, yields, spreads) have no meaningful log return and show the level only."
            ),
        )

    feat_cols = st.columns(4)
    selected = []
    for i, col in enumerate(feat_cols):
        taken = {s for s in selected if s != "None"}
        available = [o for o in options if o not in taken]
        key = f"feat_{i + 1}"
        if key in st.session_state and st.session_state[key] not in available:
            del st.session_state[key]
        selected.append(col.selectbox(f"Feature {i + 1}", available, key=key))

    chart_features = [c for c in selected if c != "None"]

    if chart_features:
        if chart_transform == "Level" and overlay_returns:
            levels_df = pd.DataFrame({c: prepare_series(df[c], "Level", monthly, date_range) for c in chart_features})
            returns_df = pd.DataFrame()
            for c in chart_features:
                if "Log return (%)" in allowed_transforms(family_of(c)):
                    returns_df[c] = prepare_series(df[c], "Log return (%)", monthly, date_range)
                else:
                    st.info(
                        f"**{c}** is measured in %, so a log return is not meaningful for it; "
                        "its panel shows the level only."
                    )
            fig = levels_with_overlay(levels_df, returns_df, chart_features, title="")
        else:
            plot_df = pd.DataFrame(
                {c: prepare_series(df[c], chart_transform, monthly, date_range) for c in chart_features}
            )
            fig = compare_lines(plot_df, chart_features, title="", zero_line=(chart_transform != "Level"))
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

with tab_dist:
    ctrl = st.columns([3, 2, 1])
    default_var = DEFAULT_VARIABLE.get(economy)
    var_index = base_columns.index(default_var) if default_var in base_columns else 0
    variable = ctrl[0].selectbox("Variable", base_columns, index=var_index, key="dist_variable")
    transform = ctrl[1].selectbox("Transform", allowed_transforms(family_of(variable)), key="dist_transform")
    alpha = ctrl[2].selectbox("alpha (significance)", [0.10, 0.05, 0.01], index=1, key="dist_alpha")
    monthly = st.toggle(
        "Collapse to monthly",
        value=True,
        key="dist_monthly",
        help=(
            "Sample one value per month instead of daily. This is the main defence against "
            "inflated significance: daily forward-filled data has a huge, artificial sample "
            "size that pushes every p-value to zero."
        ),
    )
    confidence = 1 - alpha
    series = prepare_series(df[variable], transform, monthly, date_range)
    label = variable if transform == "Level" else f"{variable} - {transform}"

    if series.empty:
        st.info("No data for the working series. Adjust the controls above.")
    else:
        if transform == "Level":
            st.warning(
                "Normality and the mean CI assume independent, identically distributed "
                "observations. Monthly sampling thins out forward-filled repeats, but a "
                "level series still trends and is autocorrelated, so it is not iid. "
                "Use Log return (%) or First difference for a meaningful read."
            )

        desc = describe_extended(series)
        if desc is not None:
            st.dataframe(desc.to_frame(label).T.round(2), width="stretch")
            caption_parts = []
            if transform == "Log return (%)":
                caption_parts.append("Values are in % per observation.")
            if monthly:
                caption_parts.append(
                    'Count is after monthly sampling. Turn off "Collapse to monthly" for the '
                    "full daily count."
                )
            if caption_parts:
                st.caption(" ".join(caption_parts))

        hist_ctrl = st.columns(3)
        hist_sig = (variable, transform, monthly, date_range)
        if st.session_state.get("dist_hist_sig") != hist_sig:
            st.session_state["dist_hist_sig"] = hist_sig
            st.session_state["dist_hist_bins"] = suggest_bins(series)
        bins = hist_ctrl[0].slider("Histogram bins", 5, 100, key="dist_hist_bins")
        show_kde = hist_ctrl[1].toggle("KDE overlay", value=True)
        show_normal = hist_ctrl[2].toggle("Normal fit overlay", value=True)
        fig_dist = distribution_plot(
            series,
            bins=bins,
            show_kde=show_kde,
            show_normal=show_normal,
            title=f"Distribution - {label}",
        )
        if fig_dist is not None:
            st.plotly_chart(fig_dist, width="stretch")

        left, right = st.columns(2)
        with left:
            prob_kind = st.radio("Probability plot", ["qq", "pp"], horizontal=True, format_func=str.upper)
            fig_prob = probability_plot(series, kind=prob_kind, title=f"{prob_kind.upper()} - {label}")
            if fig_prob is not None:
                st.plotly_chart(fig_prob, width="stretch")
            else:
                st.info("Need at least three distinct points for a probability plot.")
        with right:
            spread_kind = st.radio(
                "Spread plot",
                ["box", "violin"],
                horizontal=True,
                format_func=str.capitalize,
                key="dist_spread_kind",
            )
            fig_spread = box_or_violin(series, kind=spread_kind, title=f"{spread_kind.capitalize()} - {label}")
            if fig_spread is not None:
                st.plotly_chart(fig_spread, width="stretch")

        if desc is not None:
            st.markdown("**Shape**")
            st.caption(interpret.skew_verdict(desc["skew"]), help=interpret.SKEW_HELP)
            st.caption(interpret.kurtosis_verdict(desc["excess_kurtosis"]), help=interpret.KURTOSIS_HELP)

        st.markdown("**Normality tests**")
        battery = normality_battery(series, alpha=alpha)
        if battery is None:
            st.info("Need at least 8 observations to run the normality battery.")
        else:
            display = battery.assign(
                statistic=battery["statistic"].round(3),
                p_value=battery["p_value"].map(interpret.format_pvalue),
                normal=battery["normal"].map({True: "Yes", False: "No"}),
            ).rename(columns={"normal": f"normal? (α={alpha:g})"})
            st.dataframe(display, width="stretch", hide_index=True)
            st.caption(interpret.normality_verdict(battery, alpha))
            with st.expander("How to read these tests?"):
                st.markdown(interpret.NORMALITY_GUIDE)

        st.markdown("**Mean confidence interval**")
        ci = mean_ci(series, confidence=confidence)
        if ci is None:
            st.info("Need at least two observations for a confidence interval.")
        else:
            st.caption(
                interpret.mean_ci_sentence(ci, unit="%" if transform == "Log return (%)" else ""),
                help=interpret.MEAN_CI_HELP,
            )

with tab_rel:
    if len(base_columns) < 2:
        st.info("Need at least two base variables to explore relationships.")
    else:
        rel_transform = "Level"
        st.caption(
            "Relationships are shown on **levels** as descriptive co-movement. "
            "First difference is omitted: the macro/rate series are forward-filled "
            "to daily, so differencing them yields mostly zeros on misaligned dates "
            "and collapses correlations toward zero (a fill artefact, not absence of "
            "association). Common trends can still inflate level correlations, so read "
            "these as descriptive, not causal."
        )

        st.markdown("**Pairwise scatter**")
        sc = st.columns(3)
        x_sel = st.session_state.get("scatter_x", "None")
        y_sel = st.session_state.get("scatter_y", "None")
        x_options = ["None", *[c for c in base_columns if c != y_sel]]
        y_options = ["None", *[c for c in base_columns if c != x_sel]]
        x_var = sc[0].selectbox("Feature X", x_options, key="scatter_x")
        y_var = sc[1].selectbox("Feature Y", y_options, key="scatter_y")
        color_choice = sc[2].selectbox(
            "Colour by regime",
            ["None", *regimes],
            key="scatter_color",
            help=interpret.REGIME_AVAILABILITY_HELP,
        )

        if x_var == "None" or y_var == "None":
            st.info("Select Feature X and Feature Y to draw the scatter.")
        else:
            pair = prepare_frame(df, [x_var, y_var], rel_transform, date_range)
            if color_choice != "None":
                pair = pair.join(regimes[color_choice].rename("regime"))
            fig_scatter = scatter_ols(
                pair,
                x=x_var,
                y=y_var,
                color=None if color_choice == "None" else "regime",
                color_map=REGIME_COLORS,
                title=f"{y_var} vs {x_var} - {rel_transform}",
            )
            if fig_scatter is not None:
                st.plotly_chart(fig_scatter, width="stretch")
                plotted = pair.dropna()
                r = plotted[x_var].corr(plotted[y_var])
                st.caption(interpret.scatter_ols_verdict(r), help=interpret.OLS_HELP)
            else:
                st.info("Not enough complete observations for a scatter plot.")

            st.markdown("**Is this relationship real or spurious? (stationarity & cointegration)**")
            coint_pair = prepare_frame(df, [x_var, y_var], rel_transform, date_range)
            coint_res = stationarity_and_cointegration(coint_pair[x_var], coint_pair[y_var])
            if coint_res is None:
                st.info("Not enough aligned observations for a stationarity test.")
            else:
                st.caption(
                    interpret.cointegration_verdict(coint_res, x_var, y_var, 0.05),
                    help=interpret.COINTEGRATION_HELP,
                )

        with st.expander("How are regimes determined?"):
            st.markdown(interpret.regime_guide(economy))

        st.markdown("**Correlation matrix**")
        rel_method = st.radio(
            "Correlation method",
            ["spearman", "pearson"],
            horizontal=True,
            format_func=str.capitalize,
        )

        full_frame = prepare_frame(df, base_columns, rel_transform, date_range)
        full_corr = correlation_matrix(full_frame, method=rel_method)
        if full_corr is not None:
            dynamic_default = representative_features(full_corr, k=8)
        else:
            dynamic_default = base_columns[: min(7, len(base_columns))]
        if not st.session_state.get("corr_touched"):
            st.session_state["corr_cols"] = dynamic_default

        corr_cols = st.multiselect("Variables", base_columns, key="corr_cols", on_change=freeze_corr_selection)
        st.caption(
            "Starts from a representative set spanning the data's distinct dimensions "
            "(each feature roughly uncorrelated with the others), so that the matrix shows "
            "a range of relationships. Add any feature to drill into specific pairs; your "
            "selection is kept when you change the date range. Matrices beyond ~15 stop "
            "showing cell values."
        )
        if len(corr_cols) < 2:
            st.info("Select at least two variables for a correlation matrix.")
        else:
            corr = full_corr.loc[corr_cols, corr_cols] if full_corr is not None else None
            if corr is None or full_frame.shape[0] < 3:
                st.info("Not enough complete observations after transforming.")
            else:
                fig_corr = matrix_heatmap(
                    corr,
                    colorscale=CORR_COLORSCALE,
                    cbar_title="r",
                    value_label="Correlation",
                    title=f"{rel_method.capitalize()} correlation - {rel_transform}",
                )
                if fig_corr is not None:
                    st.plotly_chart(fig_corr, width="stretch")

        st.markdown("**Multicollinearity (VIF)**")
        st.caption(
            "Variance inflation factor across **all** base features (not just the matrix "
            "selection above): the correlation matrix reads collinearity pair by pair, while "
            "VIF ranks how redundant each feature is given every other feature together - the "
            "diagnostic before a linear model. Computed on levels, so a shared trend "
            "lifts every VIF; a term spread is an exact combination of its component yields, so "
            "an infinite VIF there is correct, not a bug."
        )
        vif = variance_inflation_factors(full_frame)
        if vif is None:
            st.info("Not enough complete observations to compute VIF.")
        else:
            vif_table = vif.round(2).rename_axis("Feature").reset_index(name="VIF")
            vif_table["VIF"] = vif_table["VIF"].apply(
                lambda v: "> 1000" if not np.isfinite(v) or v >= 1000 else f"{v:.2f}"
            )
            st.dataframe(vif_table, width="stretch", hide_index=True)
            st.caption(interpret.vif_verdict(vif), help=interpret.VIF_HELP)

        st.markdown("**Partial correlation**")
        st.caption(
            "Association between two variables after removing the linear effect of one or more control variables."
        )
        pc = st.columns(3)
        pc_x_sel = st.session_state.get("pc_x", "None")
        pc_y_sel = st.session_state.get("pc_y", "None")
        pc_x_options = ["None", *[c for c in base_columns if c != pc_y_sel]]
        pc_y_options = ["None", *[c for c in base_columns if c != pc_x_sel]]
        pc_x = pc[0].selectbox("Feature X", pc_x_options, key="pc_x")
        pc_y = pc[1].selectbox("Feature Y", pc_y_options, key="pc_y")
        covar_options = [c for c in base_columns if c not in (pc_x, pc_y)]
        if "pc_covars" in st.session_state:
            st.session_state["pc_covars"] = [c for c in st.session_state["pc_covars"] if c in covar_options]
        covars = pc[2].multiselect("Control for", covar_options, key="pc_covars")

        if pc_x == "None" or pc_y == "None":
            st.info("Select Feature X and Feature Y for a partial correlation.")
        elif not covars:
            st.info("Choose at least one control variable to partial out.")
        else:
            pc_frame = prepare_frame(df, [pc_x, pc_y, *covars], rel_transform, date_range)
            pcorr = partial_correlation(pc_frame, x=pc_x, y=pc_y, covar=covars, method=rel_method)
            if pcorr is None or pc_frame.shape[0] < 3:
                st.info("Not enough complete observations for a partial correlation.")
            else:
                raw = correlation_matrix(pc_frame[[pc_x, pc_y]], method=rel_method)
                st.caption(
                    f"Partial r = {pcorr['r']:+.2f} "
                    f"(p = {interpret.format_pvalue(pcorr['p_value'])}, n = {pcorr['n']:,}), "
                    f"controlling for {', '.join(covars)}. "
                    f"Raw {rel_method} r = {raw.loc[pc_x, pc_y]:+.2f}.",
                    help=interpret.PARTIAL_P_HELP,
                )
                st.caption(
                    "With thousands of daily rows the p-value is near zero regardless of "
                    "strength - read the coefficient magnitude, not significance."
                )

with tab_regime:
    if not regimes:
        st.info(
            "No regime labels are available for this economy and date range. Regimes are "
            "derived from the policy rate, term spread and (USA only) the Sahm recession "
            "indicator; if those are missing, this tab is empty."
        )
    else:
        ctrl = st.columns([3, 2, 2, 1])
        default_var = DEFAULT_VARIABLE.get(economy)
        var_index = base_columns.index(default_var) if default_var in base_columns else 0
        cmp_var = ctrl[0].selectbox("Variable", base_columns, index=var_index, key="regime_variable")
        cmp_transform = ctrl[1].selectbox("Transform", allowed_transforms(family_of(cmp_var)), key="regime_transform")
        group_choice = ctrl[2].selectbox(
            "Group by regime",
            list(regimes),
            key="regime_group",
            help=interpret.REGIME_AVAILABILITY_HELP,
        )
        cmp_alpha = ctrl[3].selectbox("alpha", [0.10, 0.05, 0.01], index=1, key="regime_alpha")
        cmp_monthly = st.toggle(
            "Collapse to monthly",
            value=True,
            key="regime_monthly",
            help=(
                "Sample one value per month instead of daily. This is the main defence against "
                "inflated significance: daily forward-filled data has a huge, artificial sample "
                "size that pushes every p-value to zero."
            ),
        )

        cmp_series = prepare_series(df[cmp_var], cmp_transform, cmp_monthly, date_range)
        cmp_label = cmp_var if cmp_transform == "Level" else f"{cmp_var} - {cmp_transform}"
        groups = regimes[group_choice]

        if cmp_transform == "Level":
            st.caption(
                "On **levels**, this checks whether the usual value of this variable is "
                "different from one regime to another. Because levels move slowly and "
                "trend, nearby days are almost identical, so the test behaves as if it has far "
                "more independent data than it really does and can flag tiny gaps as "
                "significant. Trust a difference only when the box/violin plots clearly "
                "separate."
            )
        else:
            st.caption(
                "On returns/changes, the average move is close to zero in every regime, so a "
                "test on the average almost always reports 'no difference' - that is expected. "
                "What does change between regimes is how *volatile* the moves "
                "are: look at how wide each box/violin is, and at the variability note under "
                "the test."
            )

        with st.expander("How are regimes determined?"):
            st.markdown(interpret.regime_guide(economy))

        working = pd.DataFrame({"value": cmp_series, "group": groups}).dropna()
        if working["group"].nunique() < 2:
            st.info("Need at least two non-empty regime groups in this window.")
        else:
            st.markdown("**Distribution by regime**")
            summary = working.groupby("group", observed=True)["value"].agg(["count", "mean", "median", "std"]).round(2)
            summary.index.name = group_choice
            summary.columns = ["Count", "Mean", "Median", "Std"]
            st.dataframe(summary, width="stretch")

            # One colour per regime, in the same order the box/violin and histogram
            # traces are drawn, so every chart in this block stays visually coherent.
            present = list(working.groupby("group", observed=True).groups)
            palette = px.colors.qualitative.Plotly
            regime_palette = {str(lbl): palette[i % len(palette)] for i, lbl in enumerate(present)}
            group_series = working["group"].astype(str).rename(group_choice)

            freq_cols = st.columns(2)
            fig_bar = category_counts(
                group_series,
                kind="bar",
                color_map=regime_palette,
                title=f"Observations per {group_choice}",
            )
            if fig_bar is not None:
                freq_cols[0].plotly_chart(fig_bar, width="stretch")
            fig_pie = category_counts(
                group_series,
                kind="pie",
                color_map=regime_palette,
                title=f"Share per {group_choice}",
            )
            if fig_pie is not None:
                freq_cols[1].plotly_chart(fig_pie, width="stretch")

            ctrl_left, ctrl_right = st.columns(2)
            with ctrl_left:
                spread_kind = st.radio(
                    "Spread plot",
                    ["box", "violin"],
                    horizontal=True,
                    format_func=str.capitalize,
                    key="regime_spread_kind",
                )
            with ctrl_right:
                hist_ctrl = st.columns([3, 2])
                hist_sig = (cmp_var, cmp_transform, group_choice, cmp_monthly, date_range)
                if st.session_state.get("regime_hist_sig") != hist_sig:
                    st.session_state["regime_hist_sig"] = hist_sig
                    st.session_state["regime_hist_bins"] = suggest_bins(cmp_series)
                hist_bins = hist_ctrl[0].slider("Histogram bins", 5, 80, key="regime_hist_bins")
                as_density = hist_ctrl[1].toggle(
                    "Density",
                    value=False,
                    key="regime_hist_density",
                    help="Normalise each regime to compare shapes when group sizes differ a lot.",
                )

            left, right = st.columns(2)
            with left:
                fig_groups = box_or_violin(
                    cmp_series,
                    groups=groups,
                    kind=spread_kind,
                    title=f"{cmp_label} by {group_choice}",
                )
                if fig_groups is not None:
                    st.plotly_chart(fig_groups, width="stretch")
            with right:
                fig_hist = grouped_histogram(
                    cmp_series,
                    groups,
                    bins=hist_bins,
                    color_map=regime_palette,
                    density=as_density,
                    title=f"{cmp_label} by {group_choice}",
                )
                if fig_hist is not None:
                    st.plotly_chart(fig_hist, width="stretch")
                else:
                    st.info("Need at least two non-empty regimes for overlaid histograms.")

            st.markdown("**Group comparison test**")
            result = compare_groups(cmp_series, groups, alpha=cmp_alpha)
            if result is None:
                st.info("Not enough data per group to run a comparison.")
            else:
                st.markdown(
                    f"**Test:** {result['test']}, "
                    f"**statistic:** {result['statistic']:.3f}, "
                    f"**p-value:** {interpret.format_pvalue(result['p_value'])}"
                )
                st.caption(interpret.compare_groups_sentence(result, cmp_alpha, cmp_label))
                st.caption(interpret.spread_note(result["equal_var"]))
                with st.expander("How is this test chosen, and what does it mean?"):
                    st.markdown(interpret.GROUP_TEST_GUIDE)

                if result["post_hoc"] is not None:
                    st.markdown("**Which regimes differ (Tukey HSD)**")
                    st.caption(
                        "ANOVA only says *some* regime differs. Tukey HSD checks every pair of "
                        "regimes and flags which averages are far enough apart to be a real "
                        "difference, correcting for testing many pairs at once."
                    )
                    tukey = result["post_hoc"]
                    ci_label = f"{1 - cmp_alpha:.0%} CI"
                    tukey_display = pd.DataFrame(
                        {
                            "Regime 1": tukey["group1"].astype(str),
                            "Regime 2": tukey["group2"].astype(str),
                            "Mean difference": tukey["meandiff"].astype(float).round(3),
                            ci_label: [
                                f"[{lo:.3f}, {hi:.3f}]"
                                for lo, hi in zip(tukey["lower"].astype(float), tukey["upper"].astype(float))
                            ],
                            "p-value (adjusted)": tukey["p-adj"].astype(float).map(interpret.format_pvalue),
                            "Different?": tukey["reject"].astype(bool).map({True: "Yes", False: "No"}),
                        }
                    )
                    st.dataframe(tukey_display, width="stretch", hide_index=True)
                    with st.expander("How to read Tukey HSD?"):
                        st.markdown(interpret.TUKEY_GUIDE)
                elif len(result["group_sizes"]) >= 3:
                    st.caption(
                        "A pairwise breakdown (which regime differs from which) appears only "
                        "when the data qualifies for ANOVA. Here it isn't bell-shaped, so a "
                        "rank-based test was used instead - compare the regimes in the "
                        "box/violin above."
                    )

        st.divider()
        st.markdown("**Association between regimes**")
        if len(regimes) < 2:
            st.info("Need at least two regime labels to measure association.")
        else:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            regimes_window = {name: s[(s.index >= start) & (s.index <= end)] for name, s in regimes.items()}

            cv_matrix = cramers_v_matrix(regimes_window)
            fig_cv = matrix_heatmap(
                cv_matrix,
                colorscale=CORR_COLORSCALE,
                cbar_title="V",
                value_label="Cramér's V",
                title="Strength of association (Cramér's V)",
                zmin=0,
            )
            if fig_cv is not None:
                st.plotly_chart(fig_cv, width="stretch")

            cc = st.columns(2)
            reg_names = list(regimes_window)
            default_a = reg_names[0] if reg_names else "None"
            default_b = reg_names[1] if len(reg_names) > 1 else default_a
            reg_a_sel = st.session_state.get("reg_a", default_a)
            reg_b_sel = st.session_state.get("reg_b", default_b)
            reg_a_options = [r for r in reg_names if r != reg_b_sel]
            reg_b_options = [r for r in reg_names if r != reg_a_sel]
            reg_a = cc[0].selectbox("Regime 1", reg_a_options, key="reg_a")
            reg_b = cc[1].selectbox("Regime 2", reg_b_options, key="reg_b")
            assoc = categorical_association(
                regimes_window[reg_a].rename(reg_a),
                regimes_window[reg_b].rename(reg_b),
                alpha=cmp_alpha,
            )
            if assoc is None:
                st.info("These two regimes do not overlap enough to compare here.")
            else:
                st.caption(interpret.association_sentence(assoc, reg_a, reg_b, cmp_alpha))
                with st.expander("How to read this association?"):
                    st.markdown(interpret.ASSOCIATION_GUIDE)

                st.markdown(f"**Day counts: {reg_a} (rows) × {reg_b} (columns)**")
                st.caption(
                    f"Each cell counts the days in the selected range that were in both states "
                    f"at once (total {assoc['n']:,} days). These are daily counts, so they "
                    f"won't match the observation counts in the table at the top, which "
                    f"samples the selected variable monthly (if selected)."
                )
                st.dataframe(assoc["table"], width="stretch")

with tab_structure:
    st.caption(
        "The other tabs use rule-based regimes (policy stance, curve, recession). Here the "
        "data is left to reveal its own structure: collinear features are compressed with PCA, "
        "then K-Means groups the days into data-driven regimes. Everything runs on standardised "
        "levels. Read it as description, not a forecast."
    )

    if len(base_columns) < 3:
        st.info("Need at least three base variables to explore structure.")
    else:
        if not st.session_state.get("struct_touched"):
            seed_frame = prepare_frame(df, base_columns, "Level", date_range)
            seed_corr = correlation_matrix(seed_frame, method="spearman")
            default_cols = regime_source_columns(df, economy)
            if seed_corr is not None:
                for col in representative_features(seed_corr, k=6):
                    if len(default_cols) >= 6:
                        break
                    if col not in default_cols:
                        default_cols.append(col)
            st.session_state["struct_cols"] = default_cols[:6]

        struct_cols = st.multiselect(
            "Features to cluster on",
            base_columns,
            key="struct_cols",
            on_change=freeze_struct_selection,
        )
        st.caption(
            "Default seeds the features the rule-based regimes are built from (policy rate, "
            "curve spread, recession indicator where available), then fills up with distinct "
            "extra dimensions, so the cluster-vs-regime comparison below is meaningful. Note it "
            "is partly circular: clustering on a regime's own inputs will tend to recover it. "
            "Swap in other features to test how far the structure holds."
        )
        with st.expander("How is this built?"):
            st.markdown(interpret.STRUCTURE_GUIDE)

        if len(struct_cols) < 2:
            st.info("Select at least two features to cluster on.")
        else:
            bundle = run_structure(economy, tuple(struct_cols), date_range, master_mtime(economy))
            if bundle is None or bundle["pca"] is None or bundle["sweep"] is None:
                st.info("Not enough overlapping observations for these features and date range.")
            else:
                scaled = bundle["scaled"]
                pca = bundle["pca"]
                sweep = bundle["sweep"]
                scores = pca["scores"]

                if bundle["hopkins"] is not None:
                    st.markdown("**Clustering tendency**")
                    st.caption(interpret.hopkins_verdict(bundle["hopkins"]), help=interpret.HOPKINS_HELP)

                st.markdown("**Principal components**")
                left, right = st.columns(2)
                fig_scree = explained_variance_plot(pca["explained_variance_ratio"])
                if fig_scree is not None:
                    left.plotly_chart(fig_scree, width="stretch")
                n_show = min(3, pca["loadings"].shape[1])
                fig_load = matrix_heatmap(
                    pca["loadings"].iloc[:, :n_show],
                    colorscale=CORR_COLORSCALE,
                    cbar_title="loading",
                    value_label="Loading",
                    title="Component loadings",
                )
                if fig_load is not None:
                    right.plotly_chart(fig_load, width="stretch")
                    right.caption(
                        "Read each PC down its column: features with the largest absolute loadings (toward "
                        "±1) define that component; values near 0 barely contribute."
                    )

                st.markdown("**Choosing the number of clusters**")
                best_k = int(sweep["silhouette"].idxmax())
                fig_sweep = cluster_selection_plot(sweep, best_k=best_k)
                if fig_sweep is not None:
                    st.plotly_chart(fig_sweep, width="stretch")
                st.caption(
                    f"Silhouette is highest at k = {best_k}. Use the elbow in inertia as a "
                    "sanity check, then adjust k below if a different split is more interpretable."
                )
                k = st.slider(
                    "Number of clusters (k)",
                    int(sweep.index.min()),
                    int(sweep.index.max()),
                    best_k,
                )

                km = kmeans_labels(scaled, k)
                if km is None:
                    st.info("K-Means could not fit for this k. Pick a different value.")
                else:
                    st.caption(interpret.silhouette_verdict(km["silhouette"]))

                    st.markdown("**Cluster map (PCA projection)**")
                    proj_choice = st.radio(
                        "Colour projection by",
                        ["Cluster", *regimes],
                        horizontal=True,
                        format_func=str.capitalize,
                    )
                    if proj_choice == "Cluster":
                        color_series, cmap = km["labels"], None
                    else:
                        color_series = regimes[proj_choice].reindex(scores.index)
                        cmap = REGIME_COLORS
                    fig_proj = projection_scatter(
                        scores,
                        color_series,
                        color_map=cmap,
                        title=f"PCA projection coloured by {proj_choice.lower()}",
                    )
                    if fig_proj is not None:
                        st.plotly_chart(fig_proj, width="stretch")
                        evr = pca["explained_variance_ratio"]
                        shown = evr.iloc[0] + evr.iloc[1]
                        st.caption(
                            f"PC1 and PC2 capture {evr.iloc[0]:.0%} and {evr.iloc[1]:.0%} of the "
                            f"total variance ({shown:.0%} together). The remainder is flattened "
                            "out of this 2D view, so the lower this figure the more cautiously "
                            "the layout should be read."
                        )

                    st.markdown("**Cluster profile (average level per cluster)**")
                    profile = bundle["frame"].groupby(km["labels"], observed=True).mean().round(2)
                    profile.index.name = "Cluster"
                    st.dataframe(profile, width="stretch")

                    st.divider()
                    st.markdown("**Do the clusters match a known regime?**")
                    if not regimes:
                        st.info("No rule-based regimes are available for this economy to compare.")
                    else:
                        reg_choice = st.selectbox(
                            "Compare clusters with",
                            list(regimes),
                            key="struct_regime",
                            help=interpret.REGIME_AVAILABILITY_HELP,
                        )
                        agree = cluster_agreement(km["labels"], regimes[reg_choice])
                        pair = pd.DataFrame({"Cluster": km["labels"], reg_choice: regimes[reg_choice]}).dropna()
                        assoc = categorical_association(pair["Cluster"], pair[reg_choice], alpha=0.05)
                        if agree is None or assoc is None:
                            st.info("Not enough overlap between clusters and this regime.")
                        else:
                            st.caption(
                                interpret.cluster_agreement_sentence(agree["ari"], reg_choice),
                                help=interpret.CLUSTER_REGIME_HELP,
                            )
                            st.caption(interpret.cramers_v_verdict(assoc["cramers_v"]))
                            st.markdown(f"**Day counts: cluster (rows) × {reg_choice} (columns)**")
                            st.dataframe(assoc["table"], width="stretch")
