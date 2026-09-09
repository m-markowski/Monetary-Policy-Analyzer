import pandas as pd

SKEW_HELP = (
    "Sample skew rule of thumb: |skew| < 0.5 ≈ symmetric, 0.5-1 moderate, > 1 strong. "
    "Sign gives direction: positive = long right tail, negative = long left tail."
)

KURTOSIS_HELP = (
    "Excess kurtosis (normal = 0): > +1 heavy tails (more outliers than normal), "
    "< -1 light tails (fewer outliers), between ≈ normal-like tails."
)

MEAN_CI_HELP = (
    "The average is the exact mean of the selected observations. The confidence interval "
    "estimates a plausible range for the underlying mean. Because time-series observations "
    "may be correlated, the interval can be too narrow, so treat it as an approximate guide."
)

NORMALITY_GUIDE = (
    "These tests assess whether the sample is consistent with a normal distribution.\n\n"
    "- **p-value** - small values provide evidence against normality; they are not the "
    "probability that the data are normal.\n"
    "- **normal?** - 'Yes' means normality is not rejected at the chosen α; it does not "
    "prove that the distribution is normal.\n"
    "- **Anderson-Darling** and **Shapiro-Wilk** provide complementary normality checks; "
    "Shapiro-Wilk is skipped above 5000 observations because its p-value becomes unreliable.\n\n"
    "With large or time-dependent samples, even small departures can be significant. "
    "Use these tests together with the histogram, QQ/PP plots and skewness/kurtosis."
)

REGIME_AVAILABILITY_HELP = (
    "A regime only appears here if the feature it is derived from is available for this "
    "economy and date range. If that feature was dropped (e.g. as stale) the regime cannot "
    "be computed and is silently unavailable."
)

PARTIAL_P_HELP = (
    "The p-value tests whether the partial correlation could be zero after accounting "
    "for the selected controls. A small value is evidence of an association, not causation. "
    "Because time-series observations may be dependent, treat the p-value as approximate "
    "and focus also on the sign and size of partial r."
)

OLS_HELP = (
    "The dashed line is an ordinary-least-squares (OLS) fit: the straight line that best "
    "summarises how Y moves with X. r is the Pearson correlation (-1 to +1): the sign is "
    "the direction (positive = move together, negative = move in opposite directions) and "
    "the magnitude is the strength. r near 0 means no straight-line relationship; r near "
    "1 means the points hug the line. r measures association, not causation."
)

VIF_HELP = (
    "The variance inflation factor measures multicollinearity: each feature is regressed on "
    "all the others, and VIF = 1 / (1 - R²) of that fit. VIF = 1 means the feature is "
    "uncorrelated with the rest; higher values mean it is increasingly a linear combination "
    "of them. Unlike the correlation matrix (pairwise only), VIF flags a feature that is "
    "redundant given several others at once. Rule of thumb: VIF > 10 is severe collinearity, "
    "a candidate to drop before a linear model. Caveat: on levels the shared trend inflates "
    "VIF across the board, so read it as 'which features are redundant', not as a fault in "
    "the data."
)

GROUP_TEST_GUIDE = (
    "This checks whether a feature has a different typical value from one regime to another.\n\n"
    "- **What is compared** - averages for t-tests/ANOVA; the distribution of values "
    "for rank-based tests. A distribution can differ even when the medians are equal.\n"
    "- **Which test is used (picked automatically):**\n"
    "    - It first checks whether each regime's values are roughly normally distributed and whether "
    "the regimes have a similar spread (variance).\n"
    "    - If yes - a **t-test** (two regimes) or **ANOVA** (three or more).\n"
    "    - If no - a rank-based **Mann-Whitney** (two regimes) or **Kruskal-Wallis** (three "
    "or more), which do not require a normal distribution but still assume independent observations.\n"
    "- **p-value** - the chance of seeing a difference this big if the regimes were truly "
    "identical. **p-value** less than α means the difference is unlikely to be chance alone.\n"
    "- **Reality check:** these tests grow over-sensitive as the sample grows - on the full "
    "daily series (Collapse to monthly off) they flag even trivial gaps, and monthly "
    "sampling eases but does not remove this. Always compare against the box/violin - if the "
    "boxes overlap a lot, the difference is small in practice even when the p-value is tiny."
)

TUKEY_GUIDE = (
    "After ANOVA says *some* regime differs, Tukey HSD (Honestly Significant Difference) "
    "checks every pair of regimes to find which ones actually differ, while correcting for "
    "the fact that comparing many pairs at once would otherwise throw up false positives.\n\n"
    "- **Mean difference** - Regime 2's average minus Regime 1's, in the variable's own units. "
    "Positive means Regime 2 is higher, negative means lower.\n"
    "- **Confidence interval** - the plausible range for that difference. When it straddles 0 "
    "the true gap could be zero, so the difference is not convincing.\n"
    "- **p-value (adjusted)** - the chance of a gap this large if the two regimes were truly "
    "identical, already corrected for testing every pair. Below α it counts as a real difference.\n"
    "- **Different?** - the verdict at your chosen α: 'Yes' when the pair differs (adjusted "
    "p-value below α, equivalently the interval excludes 0), 'No' otherwise.\n"
    "- **Reality check:** 'Yes' means the averages differ, not that the gap is large - read the "
    "mean difference and the box/violin above to judge whether it matters in practice."
)

ASSOCIATION_GUIDE = (
    "This panel shows how often two regime labels occur together.\n\n"
    "- **The table** counts daily observations for each label combination.\n"
    "- **p-value** tests whether the labels are independent, but time dependence can make "
    "the result less reliable.\n"
    "- **Cramer's V** measures association strength from 0 to 1; higher values mean a "
    "stronger relationship. Treat common strength bands as rough guidelines.\n"
    "- **Daily scope**: this analysis uses daily regime labels and is unaffected by the "
    "monthly-collapse option above. Read the results as descriptive rather than definitive."
)

STRUCTURE_GUIDE = (
    "- **PCA** - reduces many standardised features to a few main dimensions, making the "
    "overall structure easier to visualise. K-Means still uses the full selected feature set.\n"
    "- **Loadings** - show which features shape each PCA component most. Larger absolute "
    "values mean a stronger contribution; the sign shows direction.\n"
    "- **K-Means** - groups observations with similar feature levels into data-driven clusters.\n"
    "- **Interpretation** - clusters can highlight broad economic environments or periods, "
    "but they should not be treated as proof of distinct economic regimes.\n"
)

CLUSTER_REGIME_HELP = (
    "ARI and Cramer's V compare the data-driven clusters with the rule-based regime labels. "
    "Higher values mean stronger agreement, but neither measure proves that the clusters are "
    "true economic regimes. Read them together with the contingency table, especially when "
    "the numbers of clusters and regimes differ."
)

COINTEGRATION_HELP = (
    "High correlation between trending series can be misleading. The app first checks "
    "whether both series behave like I(1) processes using ADF tests on levels and first "
    "differences, and runs Engle-Granger only when that condition is met. Evidence of "
    "cointegration suggests a stable long-run relationship in levels, but not causation. "
    "If the test is not run or does not reject, that alone does not prove the relationship is spurious."
)

MONTHLY_RATIONALE = (
    "Modelling uses monthly snapshots to avoid treating forward-filled daily repeats as "
    "independent observations and to match the forecast horizon. Targets are forward changes "
    "or direction labels, while chronological splits and horizon gaps reduce target overlap. "
    "Trend and autocorrelation may remain, and the data are not reconstructed as real-time vintages."
)

SPLIT_HELP = (
    "The data is split in time order (no shuffling) into train, dev and test. Train fits the base models; "
    "dev selects the winner, fits blend weights and the stacking meta-model, and optionally selects "
    "classification thresholds. Neural early stopping uses a separate validation block inside Train, "
    "with its own horizon gap. None of these choices uses Test. Because a row's target is realised "
    "`horizon` months later, the last `horizon` rows before every boundary are purged from the earlier "
    "split. Presets keep dev and test roughly equal in size."
)

CV_HELP = (
    "Five expanding TimeSeriesSplit folds are proposed with a horizon-sized gap. A shared subset is used "
    "by every model: classifier training folds need all outer-training classes, and ROC-AUC validation "
    "needs at least two classes. The app reports omitted folds and refuses fewer than two usable folds. "
    "Feature selection and scaling are fitted within each retained fold."
)

BUDGET_HELP = (
    "The training budget controls how hard the search for good hyperparameters works. Fast and "
    "Balanced use a randomised search over a few / more settings. Thorough uses Optuna (guided "
    "search, more trials, full roster)."
)

METRIC_HELP = (
    "The scoring metric drives both model selection and the reported leaderboard. Regression: RMSE "
    "and MAE are average error sizes in the target's own units (lower is better); RMSE punishes "
    "large misses harder, MAE treats every miss alike. Classification: ROC-AUC (macro, one-vs-rest) "
    "measures class separation, F1-macro balances precision and recall equally across classes, "
    "balanced accuracy averages per-class recall - all robust to the Hold-heavy imbalance."
)

LEAKAGE_HELP = (
    "Leakage means using information that would not be available at prediction time. "
    "The target and its directly derived features are excluded, while past target values "
    "can be used as lags. As with other predictors, real-time validity still depends on "
    "when the data were actually published and revised."
)

THRESHOLD_HELP = (
    "The operating point turns class probabilities into a single Hike/Hold/Cut call. The default "
    "is plain argmax - take the highest-probability class. The toggle switches to per-class Youden's J "
    "thresholds (the point maximising true-positive minus false-positive rate), tuned on the dev split. "
    "Youden favours the rare Hike/Cut classes: it catches more of them at the price of more false "
    "alarms, so try it when missing a move costs more than a wrong call."
)

DIAG_SPLIT_HELP = (
    "Choose which chronological split to use for diagnostics. Test is the default held-out "
    "window, Train shows the fitting period, and Dev is used for model selection, ensemble "
    "fitting and optional class thresholds. Comparing Train with Test can reveal overfitting, "
    "while Train + Dev + Test shows the full sample rather than a single out-of-sample result."
)

LEADERBOARD_HELP = (
    "Base models are scored on Train, Dev and Test. The winner is chosen by the selected "
    "Dev metric, while Test provides the final held-out comparison for this window. "
    "Blend and Stack use Dev to fit their combination layer, so they are excluded from "
    "the winner pick. A large Train-Test gap can indicate overfitting or changing data patterns."
)

HORIZON_HELP = (
    "How many months ahead (h) the target looks. A supervised model cannot see the future, so it "
    "learns from history: each training example pairs the drivers observed in some month t with the "
    "outcome that actually followed h months later - both already in the past at training time. "
    "Applying that learned mapping to the latest month then gives a genuine h-month-ahead "
    "prediction. A longer horizon also moves the usable sample (the last h months have no observed "
    "outcome yet, so they cannot be training rows) and makes the task harder, because more can "
    "happen in between."
)

TARGET_HELP = (
    "Pick the series to forecast. The model predicts its change over the chosen horizon "
    "directly from the current predictors, rather than forecasting month by month. "
    "Displayed levels are reconstructed by adding the predicted change to the latest observed value. "
    "Using changes reduces trend effects but does not guarantee stationarity."
)

NEURAL_HELP = (
    "MLP and LSTM are optional because they take longer to train and can overfit the limited "
    "monthly sample. They use a separate chronological validation block within Train for early "
    "stopping. Their performance should be compared empirically with the other models."
)

CV_BUDGET_HELP = (
    "Hyperparameters are searched with cross-validation on the training split only, in time order "
    "(`TimeSeriesSplit`). Fast and Balanced use a randomised search (15 and 40 settings); Thorough "
    "uses Optuna (60 guided trials). The CV score is the mean fold score in the chosen metric."
)

CV_ROC_AUC_NOTE = (
    "AUC is undefined on a single-class validation period. Such folds are omitted and counted, not "
    "assigned 0.5. Macro OvR AUC averages scoreable classes, so compare scores together with their class "
    "coverage and evaluation dates."
)

ENSEMBLE_HELP = (
    "Both ensembles are built on the dev split. Blend is a "
    "weighted average of the base models' outputs: each weight is proportional to the inverse of "
    "that model's dev error (RMSE for regression, log loss for classification), normalised to sum "
    "to one, so models that erred less on dev count for more. Stack goes further: each base model "
    "produces its dev-split outputs (class probabilities for classification, point predictions for "
    "regression), those outputs become the input columns of a small linear meta-model (logistic "
    "regression / ridge), and the meta-model learns how much to trust each base model. At prediction time "
    "the base models score the new month first and the meta-model combines their outputs. Every sklearn base "
    "model enters both ensembles; the neural nets are excluded."
)

ROC_AUC_OVR_NOTE = (
    "ROC-AUC is calculated separately for Hike, Hold and Cut by comparing each class with all "
    "other outcomes, then averaging the available class scores equally. This means rare Hike/Cut "
    "classes matter just as much as Hold. If a class does not appear together with other outcomes "
    "in the evaluated period, its AUC cannot be calculated."
)

BASELINE_HELP_REG = (
    "The 'Baseline: no change' row is a zero-change random walk: it predicts the series does not "
    "move over the horizon, the classic macro-forecasting yardstick. Read RMSE/MAE - a typical "
    "miss in the target's own units - together with the Skill columns, which rescale each model "
    "against that baseline (1 - MSE_model / MSE_naive: 0 = no better than assuming no change, "
    "1 = perfect, negative = worse than doing nothing)."
)

BASELINE_HELP_CLF = (
    "Two simple baselines provide reference points. Majority class always predicts the most "
    "common Train label, while trailing momentum follows the direction of the previous h-month "
    "rate move. Neither produces probabilities, so ROC-AUC is not available."
)

ROC_HELP = (
    "The ROC curve plots the true-positive rate against the false-positive rate as the threshold "
    "varies, one line per class (one-vs-rest). A curve hugging the top-left is good; the dashed red "
    "diagonal is random guessing (0.5 = chance, AUC 1 = perfect)."
)

CONFUSION_HELP = (
    "The confusion matrix cross-tabulates actual (rows) against predicted (columns) classes. The "
    "diagonal is correct predictions; off-diagonal cells show what gets confused with what."
)

IMPORTANCE_HELP = (
    "Native importance comes from the fitted model. Permutation importance checks whether shuffling "
    "an input makes predictions worse on Dev. The chart shows positive effects, with the largest "
    "scaled to 100. For Blend and Stack, Dev was also used to fit the ensemble."
)

GROUP_IMPORTANCE_HELP = (
    "Importance summed into business blocks instead of individual features, re-scaled to 0-100, so "
    "you can see which kind of information the model leans on overall. Blocks: Rates - the policy "
    "rate, Treasury/benchmark yields and yield-curve spreads (rate_/yld_/sprd_); Macro - real-economy "
    "series such as inflation, unemployment, participation, savings and GDP growth; Market - "
    "equities, FX, commodities and volatility indices (eq_/fx_/cmd_/idx_); Target lags - the "
    "target's own past values added back as autoregressive features. A large Target-lags share "
    "means the model leans on the series' own momentum more than on outside drivers. "
    "The chart aggregates the same scores as the charts above - native importance when the model "
    "exposes one, otherwise permutation importance."
)

SHAP_HELP = (
    "SHAP shows how each feature contributes to a model prediction relative to a reference value. "
    "Positive and negative values indicate which features push the prediction higher or lower. "
    "Shown for supported tree/boosting models; the contributions describe the model, not causal effects."
)

PDP_HELP = (
    "A partial dependence plot shows how the predicted output moves as one feature is varied across "
    "its range while the others are held at their observed values, i.e. the model's average response "
    "to that feature. Nearly flat lines are a genuine model reading, not a plotting bug: they mean "
    "the prediction barely changes as this feature moves, so the model leans on other features "
    "(common with the Hold-dominated direction target)."
)

PROB_HIST_HELP = (
    "Each panel is the distribution of the model's predicted probability for one class, counted "
    "over the months on this split. Bars are stacked, not overlaid: the green segment counts the "
    "months that truly belong to the class, the red segment the rest, and the full bar is every "
    "month in that probability bin. Good separation piles green near 1 and red near 0; mixed bars "
    "in the middle mean the model is unsure about that class."
)

RESIDUAL_HELP = (
    "Residuals are the difference between actual and predicted values. Ideally, they stay close to zero "
    "without a clear pattern. A consistent positive or negative error suggests that the model tends to "
    "under- or over-predict. Lag-1 correlation checks whether errors in neighbouring months tend to move "
    "together. For forecasts longer than one month, some correlation is natural because the forecast "
    "windows overlap, so this is a diagnostic clue rather than a pass/fail test."
)

ECON_BASELINE_HELP_REG = (
    "A statsmodels OLS fit is shown alongside the machine-learning models as an interpretable "
    "baseline. Unlike them it is judged on classical assumptions as well as fit: coefficient "
    "significance, residual autocorrelation (Durbin-Watson), residual normality (Jarque-Bera), "
    "multicollinearity (condition number) and influential months (Cook's distance). The ML/DL "
    "models are judged on prediction only."
)

ECON_BASELINE_HELP_CLF = (
    "A statsmodels multinomial logit is shown alongside the machine-learning models as an "
    "interpretable baseline. It is judged on classical criteria as well as fit: the overall "
    "likelihood-ratio test, McFadden's pseudo-R2 and per-class coefficient significance. The ML/DL "
    "models are judged on prediction only."
)

SCENARIO_HELP = (
    "Move a few key drivers and hold the rest at the anchor month's values to read the model's "
    "prediction under that scenario. The anchor is the most recent month with a complete feature "
    "row, so the baseline is a live forward reading; the sliders are a ceteris-paribus what-if on "
    "top of it. They ignore how the drivers move together in reality, so treat the differences as "
    "sensitivity analysis, not alternative forecasts."
)

FORECAST_HELP = (
    "These univariate models use only the selected series' own history, independently of the "
    "Setup features and trained ML models. ARIMA forecasts future values with a prediction "
    "interval, while GARCH forecasts future volatility. The horizon only controls how far ahead "
    "the forecast extends."
)

ARIMA_HELP = (
    "ARIMA/SARIMA models a series using its own past values (AR), past forecast errors (MA) and "
    "differencing (I) to remove non-stationary trends; the seasonal part applies the same idea at an annual "
    "cycle. The app chooses the automatic order using only the history before the 12-month holdout: an "
    "ADF-based heuristic first selects the differencing order d, then AIC/BIC compare candidate p/q and "
    "seasonal specifications at that fixed d, including a simple random-walk/constant benchmark. The final "
    "12 months are kept out of automatic order selection. ADF and Ljung-Box are diagnostic checks rather "
    "than guarantees that the model is adequate, while the ACF/PACF charts and candidate table are mainly "
    "informational and useful when exploring a manual order."
)

GARCH_HELP = (
    "GARCH models how the volatility of monthly changes evolves over time, capturing periods "
    "when large moves tend to cluster together. The chart shows the expected size of future "
    "fluctuations in the series' own units, not their direction or future level. "
    "The model order is selected before the holdout period and then refitted on the full history for the forecast."
)


def scatter_ols_verdict(r: float) -> str:
    """
    One-sentence reading of an OLS scatter from its correlation coefficient.

    Args:
        r (float): Pearson correlation between the two plotted variables.

    Returns:
        str: Plain-language strength and direction of the linear relationship.
    """
    magnitude = abs(r)
    if magnitude < 0.1:
        return f"r = {r:+.2f}: effectively no linear relationship (the OLS line is flat)."
    if magnitude < 0.3:
        strength = "weak"
    elif magnitude < 0.5:
        strength = "moderate"
    elif magnitude < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    direction = "positive" if r > 0 else "negative"
    slope = "upward" if r > 0 else "downward"
    return (
        f"r = {r:+.2f}: {strength} {direction} linear relationship; the OLS line slopes "
        f"{slope}, and points cluster around it more tightly as |r| approaches 1."
    )


def cointegration_verdict(res: dict, x_name: str, y_name: str, alpha: float) -> str:
    """Plain-language reading of a `stats.stationarity_and_cointegration` result."""

    def word(flag: bool) -> str:
        return "stationary" if flag else "not stationary in levels"

    head = (
        f"ADF: {x_name} is {word(res['x_stationary'])} "
        f"(p = {format_pvalue(res['adf_p_x'])}); "
        f"{y_name} is {word(res['y_stationary'])} "
        f"(p = {format_pvalue(res['adf_p_y'])})."
    )

    if res["x_stationary"] and res["y_stationary"]:
        tail = (
            "Both are already stationary in levels, so an Engle-Granger cointegration "
            "test is not needed; the usual shared-trend concern is reduced."
        )
    elif not res.get("coint_tested", False):
        tail = "Engle-Granger was not run because the pair did not meet the conditions for an I(1) cointegration test."
    elif res["cointegrated"]:
        tail = (
            f"Engle-Granger p = {format_pvalue(res['coint_p'])} < α = {alpha:g}: "
            "there is evidence of cointegration, suggesting a stable long-run "
            "relationship between the series."
        )
    else:
        tail = (
            f"Engle-Granger p = {format_pvalue(res['coint_p'])} "
            f"(not below α = {alpha:g}): no cointegration detected, so a strong "
            "level correlation may reflect shared trends and should be interpreted cautiously."
        )

    return f"{head} {tail}"


def format_number(x: float, decimals: int = 2) -> str:
    """
    Human-readable number: thousands separators, no scientific notation.

    Args:
        x (float): Value to format.
        decimals (int): Decimal places; widened to 4 for sub-unit magnitudes so
            small returns/changes are not rounded away.

    Returns:
        str: Formatted value, or an em dash for missing values.
    """
    if pd.isna(x):
        return "-"
    if x != 0 and abs(x) < 1:
        decimals = max(decimals, 4)
    return f"{x:,.{decimals}f}"


def format_pvalue(p: float, threshold: float = 1e-5) -> str:
    """
    Format a p-value for display, flooring tiny values and flagging missing ones.

    Args:
        p (float): The p-value.
        threshold (float): Values below this are shown as '<threshold' instead of 0.

    Returns:
        str: A readable p-value, '<0.00001' for tiny values, or '-' when absent.
    """
    if pd.isna(p):
        return "-"
    if p < threshold:
        return f"<{threshold:.5f}"
    return f"{p:.5f}"


def skew_verdict(skew: float) -> str:
    """
    Describe distribution asymmetry from a skewness value.

    Args:
        skew (float): Sample skewness (0 = symmetric).

    Returns:
        str: One-sentence reading of direction and strength.
    """
    if pd.isna(skew):
        return "Skewness is unavailable for constant or insufficient data."
    magnitude = abs(skew)
    if magnitude < 0.5:
        shape = "approximately symmetric"
    else:
        direction = "right" if skew > 0 else "left"
        strength = "moderately" if magnitude < 1.0 else "strongly"
        shape = f"{strength} {direction}-skewed"
    return f"Skew {skew:+.2f}: {shape}."


def kurtosis_verdict(excess_kurtosis: float) -> str:
    """
    Describe tail weight from an excess-kurtosis value (normal = 0).

    Args:
        excess_kurtosis (float): Excess kurtosis (Fisher definition; 0 = normal).

    Returns:
        str: One-sentence reading of tail behaviour.
    """
    if pd.isna(excess_kurtosis):
        return "Kurtosis is unavailable for constant or insufficient data."
    if excess_kurtosis > 1.0:
        shape = "heavy-tailed (leptokurtic) - more outliers than a normal"
    elif excess_kurtosis < -1.0:
        shape = "light-tailed (platykurtic) - fewer outliers than a normal"
    else:
        shape = "tails close to normal (mesokurtic)"
    return f"Excess kurtosis {excess_kurtosis:+.2f}: {shape}."


def normality_verdict(battery: pd.DataFrame, alpha: float) -> str:
    """Count individual rejections without treating the battery as a combined test."""
    n_tests = len(battery)
    n_reject = int((~battery["normal"].astype(bool)).sum())
    return (
        f"{n_reject} of {n_tests} tests reject normality at alpha={alpha:g}. "
        "This is a summary of separate diagnostics, not a combined significance test. "
        "Failure to reject does not establish normality, especially for dependent observations."
    )


def compare_groups_sentence(result: dict, alpha: float, label: str) -> str:
    """Plain-language conclusion for a group comparison."""
    comparison = "average" if result["parametric"] else "distribution"
    p = format_pvalue(result["p_value"])
    n_groups = len(result["group_sizes"])
    scope = "between the two regimes" if n_groups == 2 else f"across the {n_groups} regimes"

    if result["differs"]:
        return f"Difference found: the {comparison} of {label} differs {scope} (p = {p}, below α = {alpha:g})."

    return (
        f"No clear difference detected: the {comparison} of {label} does not differ clearly {scope} "
        f"(p = {p}, not below α = {alpha:g})."
    )


def spread_note(equal_var: bool) -> str:
    """One-line reading of the Levene equal-variance precondition."""
    if equal_var:
        return "Variability (Levene's Test): the spread of values is similar across regimes."
    return (
        "Variability (Levene's Test): the spread of values differs across regimes - values swing more in "
        "some regimes than in others (often the more interesting signal for returns)."
    )


def vif_verdict(vif: pd.Series, threshold: float = 10.0) -> str:
    """One-sentence reading of a VIF series (see `stats.variance_inflation_factors`)."""
    flagged = vif[vif > threshold]
    if flagged.empty:
        return (
            f"No feature exceeds VIF {threshold:g}; collinearity is mild "
            f"(highest: {vif.index[0]} at {vif.iloc[0]:.1f})."
        )
    return (
        f"{len(flagged)} of {len(vif)} features exceed VIF {threshold:g} "
        f"(worst: {flagged.index[0]} at {flagged.iloc[0]:.1f}); they are largely redundant "
        "given the others and are candidates to drop before a linear model."
    )


def cramers_v_verdict(v: float) -> str:
    """One-sentence strength reading of a Cramer's V value (0-1)."""
    if v < 0.1:
        strength = "negligible"
    elif v < 0.3:
        strength = "weak"
    elif v < 0.5:
        strength = "moderate"
    else:
        strength = "strong"
    return f"Strength of the link (Cramér's V) = {v:.2f}: {strength} (0 = no association, 1 = maximal association for this table)."


def silhouette_verdict(score: float) -> str:
    """One-sentence interpretation of a mean silhouette score."""
    if score >= 0.5:
        reading = "clusters are fairly well separated"
    elif score >= 0.25:
        reading = "some cluster structure is visible, but groups overlap"
    else:
        reading = "clusters show weak separation and substantial overlap"

    return (
        f"Silhouette = {score:.2f}: {reading}. "
        "Higher values mean clearer separation; these thresholds are only a rule of thumb."
    )


def cluster_agreement_sentence(ari: float, regime_name: str) -> str:
    """Plain-language reading of an adjusted Rand index vs a rule-based regime."""
    if ari >= 0.5:
        reading = f"closely match the {regime_name} labels"
    elif ari >= 0.2:
        reading = f"partly line up with the {regime_name} labels"
    elif ari >= 0.05:
        reading = f"only loosely relate to the {regime_name} labels"
    else:
        reading = f"do not recover the {regime_name} labels (no better than chance)"
    return f"Adjusted Rand index = {ari:.2f}: the data-driven clusters {reading} (1 = identical grouping, 0 = chance)."


def association_sentence(assoc: dict, reg_a: str, reg_b: str, alpha: float) -> str:
    """Plain-language reading of a `stats.categorical_association` result."""
    p = format_pvalue(assoc["p_value"])
    if assoc["associated"]:
        head = (
            f"{reg_a} and {reg_b} are linked - knowing one tells you something about the "
            f"other (p = {p}, below α = {alpha:g})."
        )
    else:
        head = f"No clear link between {reg_a} and {reg_b} (p = {p}, not below α = {alpha:g})."
    return f"{head} {cramers_v_verdict(assoc['cramers_v'])}"


def mean_ci_sentence(ci: dict, unit: str = "") -> str:
    """
    Render a mean confidence interval as one plain-language sentence.

    Distinguishes the exact average of the observed window (a description) from the
    confidence interval, which is inference about the wider data-generating process.

    Args:
        ci (dict): Output of `stats.mean_ci` (mean, lower, upper, confidence, n).
        unit (str): Optional unit label appended after each value (e.g. '%').

    Returns:
        str: The window average and the interval for the process mean.
    """
    pct = ci["confidence"] * 100
    u = f" {unit}" if unit else ""
    return (
        f"Over this period the average is exactly {format_number(ci['mean'])}{u} (n={ci['n']:,}). "
        f"Treating these observations as one sample from the wider data-generating process, "
        f"the {pct:g}% confidence interval for that process's mean is "
        f"{format_number(ci['lower'])}{u} to {format_number(ci['upper'])}{u}."
    )


def regime_guide(economy: str) -> str:
    """
    Plain-language explanation of how each regime label is derived.

    Args:
        economy (str): 'usa' or 'eurozone'; selects the economy-specific source columns.

    Returns:
        str: Markdown describing the policy, curve and (USA-only) recession regimes.
    """
    policy_col = "rate_ff_eff" if economy == "usa" else "rate_ecb_dep"
    curve_col = "sprd_10y_2y" if economy == "usa" else "sprd_10y_ecb"
    text = (
        "Each regime is a categorical label derived from a single source feature. If that "
        "feature is unavailable (e.g. dropped as stale) the regime cannot be computed and "
        "will not appear in the colour selector above.\n\n"
        f"- **Policy regime** (Hiking / Holding / Easing) - from the policy rate "
        f"(`{policy_col}`). Today's rate is compared with its value ~63 trading days earlier "
        "(about two policy meetings). A rise of more than half a 25bps step is *Hiking*, a "
        "fall of more than that is *Easing*, otherwise *Holding*. The two-meeting window "
        "keeps a tightening/easing cycle coherent when a meeting is skipped.\n"
        f"- **Curve state** (Inverted / Normal) - from the term spread (`{curve_col}`, long "
        "yield minus short yield). A negative spread is *Inverted* (long rates below short "
        "rates, a classic recession signal), otherwise *Normal*.\n"
    )
    if economy == "usa":
        text += (
            "- **Recession** (Recession / Expansion) - from the real-time Sahm indicator "
            "(`ind_sahm_realtime`). At or above 0.5 is *Recession*, otherwise *Expansion*. "
            "USA only; the Eurozone has no equivalent series here.\n"
        )
    return text


def best_model_sentence(name: str, metric: str, dev_score: float | None, test_score: float | None) -> str:
    """One-line announcement of the winner: picked on the dev split, reported on test."""
    picked = f"Best model: {name} - picked for the best Dev {metric}"
    if dev_score is not None and not pd.isna(dev_score):
        picked += f" ({dev_score:.3f})"
    if test_score is None or pd.isna(test_score):
        return picked + ". No test score is available for this configuration."
    return picked + f". On the untouched test set it scores {test_score:.3f}."


def overfit_note(train_score: float | None, test_score: float | None, metric: str) -> str:
    """Reading of the train-vs-test gap for the chosen metric (sign-aware)."""
    if train_score is None or test_score is None or pd.isna(train_score) or pd.isna(test_score):
        return ""
    higher_better = metric not in ("RMSE", "MAE")
    gap = train_score - test_score if higher_better else test_score - train_score
    rel = gap / (abs(train_score) if train_score else 1.0)
    head = f"Train {metric} {train_score:.3f} vs test {test_score:.3f}: "
    if rel < -0.1:
        return head + (
            "test scores noticeably better than train - on a chronological split this usually "
            "means the test window was an easier era, not extra generalisation power."
        )
    if rel > 0.25:
        return head + (
            "a large gap - the model fits the training window far better than unseen data, a sign of "
            "overfitting on this sample."
        )
    if rel > 0.1:
        return head + "a moderate gap; some overfitting, read the test column as the honest score."
    return head + "train and test are close, so generalisation looks stable."


def metric_verdict(metric: str, value: float | None, benchmark: float | None = None, unit: str = "") -> str:
    """
    Band reading of one leaderboard metric value, keyed by its display name.

    Covers the curated metric set (`evaluate.REGRESSION_METRICS` /
    `evaluate.CLASSIFICATION_METRICS`). RMSE and MAE have no absolute scale, so
    they are read in the target's own units and, when `benchmark` is given,
    against the same-metric score of the leaderboard's no-change baseline.

    Args:
        metric (str): Display metric name, e.g. 'RMSE' or 'ROC-AUC (macro/OvR)'.
        value (float | None): The metric value to interpret.
        benchmark (float | None): Same-metric score of the no-change (zero
            forward change) baseline, enabling the RMSE/MAE relative reading.
        unit (str): Unit label appended to RMSE/MAE values (e.g. 'pp').

    Returns:
        str: One plain-language sentence, or '' when the value is missing.
    """
    if value is None or pd.isna(value):
        return ""
    if metric.startswith("ROC-AUC"):
        if value >= 0.9:
            band = "excellent"
        elif value >= 0.8:
            band = "strong"
        elif value >= 0.7:
            band = "moderate"
        elif value >= 0.6:
            band = "weak"
        else:
            band = "close to chance"
        return f"ROC-AUC {value:.2f}: {band} class separation (0.5 = coin flip, 1 = perfect)."
    if metric == "F1-macro":
        if value >= 0.75:
            band = "strong"
        elif value >= 0.55:
            band = "moderate"
        elif value >= 0.4:
            band = "weak"
        else:
            band = "poor"
        return (
            f"F1-macro {value:.2f}: {band}. It averages each class's precision-recall balance with "
            "equal weight, so the rare Hike/Cut classes count as much as Hold."
        )
    if metric == "Balanced accuracy":
        if value >= 0.75:
            band = "strong"
        elif value >= 0.55:
            band = "moderate"
        elif value > 0.4:
            band = "modest but above chance"
        else:
            band = "close to chance"
        return (
            f"Balanced accuracy {value:.2f}: {band}. It averages the per-class hit rates, so chance "
            "is about 0.33 for the three classes, not 0.5."
        )
    if metric in ("RMSE", "MAE"):
        u = f" {unit}" if unit else ""
        head = (
            f"{metric} {format_number(value)}{u}: a typical prediction misses by about this much, "
            "in the target's own units."
        )
        if benchmark is None or pd.isna(benchmark) or benchmark <= 0:
            return head
        ratio = value / benchmark
        if ratio >= 1:
            tail = "no better than that baseline"
        elif ratio >= 0.8:
            tail = "a modest improvement on it"
        elif ratio >= 0.5:
            tail = "a solid improvement on it"
        else:
            tail = "a large improvement on it"
        return f"{head} Predicting no change at all would score {format_number(benchmark)}{u}, so this is {tail}."
    return ""


def skill_verdict(skill: float | None, naive_name: str = "the no-change baseline") -> str:
    """
    Band reading of a skill-vs-naive score (1 - MSE_model / MSE_naive).

    Args:
        skill (float | None): Per-split value from `evaluate.skill_vs_naive`.
        naive_name (str): Display name of the naive benchmark being beaten.

    Returns:
        str: One plain-language sentence, or '' when the value is missing.
    """
    if skill is None or pd.isna(skill):
        return ""
    if skill <= 0:
        return (
            f"Skill {skill:+.2f}: no better than {naive_name} - the model adds nothing over assuming nothing changes."
        )
    if skill < 0.1:
        band = "a marginal edge over"
    elif skill < 0.3:
        band = "a modest but genuine edge over"
    elif skill < 0.5:
        band = "a solid edge over"
    else:
        band = "a large edge over"
    return f"Skill {skill:+.2f}: {band} {naive_name} (0 = no better, 1 = perfect)."


def roc_auc_verdict(auc: float | None, split: str | None = None) -> str:
    """Split-aware reading of a macro one-vs-rest ROC-AUC, using the metric_verdict bands."""
    where = f" on the {split.lower()} split" if split else ""
    if auc is None or pd.isna(auc):
        return f"ROC-AUC is undefined{where} (a class may be absent from this split)."
    return (
        f"{metric_verdict('ROC-AUC (macro/OvR)', auc)} Measured{where}; this single number is "
        "the average of the per-class one-vs-rest AUCs shown in the ROC-curve legend below."
    )


def confusion_verdict(cm) -> str:
    """Plain reading of a confusion matrix: overall hit rate and the biggest confusion."""
    if cm is None or getattr(cm, "empty", True):
        return ""
    total = cm.to_numpy().sum()
    if total == 0:
        return ""
    correct = sum(cm.iloc[i, i] for i in range(cm.shape[0]))
    off = cm.copy()
    for i in range(off.shape[0]):
        off.iloc[i, i] = 0
    note = f"The diagonal holds {correct:.0f} of {total:.0f} correct calls ({correct / total:.0%})."
    if off.to_numpy().sum() > 0:
        actual, predicted = off.stack().idxmax()
        worst = off.stack().max()
        note += (
            f" The most common error is predicting {predicted} when the actual outcome is "
            f"{actual} ({worst:.0f} months)."
        )
    return note


def residual_verdict(y_true, y_pred, horizon: int = 1) -> str:
    """Plain-language reading of regression residuals."""
    actual = pd.Series(y_true).to_numpy(dtype=float)
    predicted = pd.Series(y_pred).to_numpy(dtype=float)
    resid = pd.Series(actual - predicted).dropna()

    if len(resid) < 3:
        return ""

    spread = resid.std(ddof=0)
    bias = resid.mean()
    ac1 = resid.autocorr(lag=1) if resid.nunique() > 1 else float("nan")

    parts = []

    if abs(bias) > 0.1 * spread:
        direction = "over-predicts" if bias < 0 else "under-predicts"
        parts.append(f"a systematic bias (it {direction} on average)")

    if pd.notna(ac1) and abs(ac1) > 0.3:
        parts.append(f"a pattern in neighbouring months' errors (lag-1 correlation = {ac1:.2f})")

    if not parts:
        text = "Residuals are centred around zero with no strong month-to-month pattern."
    else:
        text = "The residuals show " + " and ".join(parts) + "."

    if horizon > 1 and pd.notna(ac1) and abs(ac1) > 0.3:
        text += (
            f" Because {horizon}-month forecasts overlap, some correlation between neighbouring "
            "errors is expected and does not necessarily indicate a problem."
        )

    return text


def class_balance_note(y) -> str:
    """Report the class distribution and flag imbalance."""
    counts = pd.Series(y).value_counts()
    total = int(counts.sum())
    if total == 0:
        return ""
    parts = ", ".join(f"{lab}: {n} ({n / total:.0%})" for lab, n in counts.items())
    ratio = counts.max() / counts.min() if counts.min() > 0 else float("inf")
    if ratio >= 3:
        tail = (
            f" The largest class is about {ratio:.0f}x the smallest, so plain accuracy is misleading - "
            "class weights and the Youden threshold address this."
        )
    else:
        tail = " The classes are reasonably balanced."
    return f"Class balance - {parts}.{tail}"


def importance_sentence(importance: pd.Series | None, top_n: int = 3) -> str:
    """Name the top-ranked features by an importance measure."""
    if importance is None or importance.empty:
        return ""
    names = ", ".join(importance.head(top_n).index.astype(str))
    return f"Top drivers: {names} (highest-ranked features by this measure, scaled to 100)."


def group_importance_sentence(group_importance: pd.Series | None) -> str:
    """Name the business block carrying the most predictive signal."""
    if group_importance is None or group_importance.empty:
        return ""
    return f"The {group_importance.index[0]} block contributes the most predictive signal overall."


def stationarity_verdict(res: dict | None, name: str) -> str:
    """Plain reading of an ADF stationarity test."""
    if res is None:
        return f"Not enough observations to test {name} for stationarity."

    p = format_pvalue(res["p_value"])

    if res["stationary"]:
        return f"ADF on {name}: rejects the unit-root hypothesis (p = {p}), which supports stationarity."

    return (
        f"ADF on {name}: does not reject the unit-root hypothesis (p = {p}), "
        "so the series may require differencing. The automatic search selects d separately."
    )


def target_change_note(res: dict | None, name: str) -> str:
    """Plain ADF reading for the forward-change target."""
    if res is None:
        return f"Not enough observations to test the {name} target for stationarity."

    p = format_pvalue(res["p_value"])

    if res["stationary"]:
        return (
            f"ADF on the forward change of {name}: rejects the unit-root hypothesis "
            f"(p = {p}), which supports stationarity."
        )

    return (
        f"ADF on the forward change of {name}: does not reject the unit-root hypothesis "
        f"(p = {p}). Using changes can reduce trend effects, but does not guarantee stationarity."
    )


def ljung_box_verdict(diag: dict, alpha: float = 0.05) -> str:
    """Read the residual autocorrelation check, including an unavailable result."""
    if pd.isna(diag["ljung_box_p"]):
        return "Ljung-Box is unavailable: too few residuals for this model order."
    p = format_pvalue(diag["ljung_box_p"])
    if diag["ljung_box_p"] >= alpha:
        return f"Ljung-Box p = {p}: no clear residual autocorrelation was detected."
    return f"Ljung-Box p = {p}: some autocorrelation remains. The model has not captured all of the time pattern."


def garch_persistence_note(persistence: float) -> str:
    """Plain reading of estimated GARCH persistence."""
    prefix = f"Estimated persistence (α+β) = {persistence:.2f}: "

    if persistence >= 1:
        return prefix + (
            "volatility is extremely persistent and the model does not imply reversion to a finite long-run variance."
        )

    if persistence >= 0.97:
        return prefix + ("volatility shocks are highly persistent and may fade only slowly.")

    return prefix + (
        "volatility shocks tend to fade over time, with variance reverting toward "
        "its long-run level under the fitted model."
    )


def ols_assumptions_note(ols: dict) -> str:
    """Plain summary of OLS residual diagnostics."""
    dw = ols["durbin_watson"]
    if dw < 1.5:
        dw_read = f"Durbin-Watson {dw:.2f} suggests positive residual autocorrelation"
    elif dw > 2.5:
        dw_read = f"Durbin-Watson {dw:.2f} suggests negative residual autocorrelation"
    else:
        dw_read = f"Durbin-Watson {dw:.2f} is near 2, suggesting little first-order autocorrelation"

    jb_read = "rejects residual normality" if ols["jarque_bera_p"] < 0.05 else "does not reject residual normality"

    cond = ols["condition_number"]
    if cond < 30:
        cond_read = "low"
    elif cond < 100:
        cond_read = "moderate"
    else:
        cond_read = "high"

    return (
        f"{dw_read}. Jarque-Bera {jb_read} "
        f"(p = {format_pvalue(ols['jarque_bera_p'])}). "
        f"The condition number is {cond:.1f} ({cond_read}); higher values can indicate "
        "unstable coefficients due to collinearity or scaling."
    )
