import pandas as pd

SKEW_HELP = (
    "Sample skew rule of thumb: |skew| < 0.5 ≈ symmetric, 0.5–1 moderate, > 1 strong. "
    "Sign gives direction: positive = long right tail, negative = long left tail."
)

KURTOSIS_HELP = (
    "Excess kurtosis (normal = 0): > +1 heavy tails (more outliers than normal), "
    "< -1 light tails (fewer outliers), between ≈ normal-like tails."
)

MEAN_CI_HELP = (
    "The average shown is the exact mean of the selected window. The confidence interval "
    "is a different idea: it treats this window as one sample from the broader process that "
    "generates the data and gives a plausible range for that process's mean. '95% confidence' "
    "means ~95% of such intervals would contain the true process mean if sampling were repeated. "
    "Caveat: financial series are autocorrelated, which makes the classic interval too narrow, so "
    "read it as optimistic."
)

NORMALITY_GUIDE = (
    "Each test asks the same question: *could this data plausibly come from a normal "
    "(bell-curve) distribution?*\n\n"
    "- **p-value** — the chance of seeing data this far from normal if it really were "
    "normal. Small p = strong evidence *against* normality.\n"
    "- **normal? (α)** — 'Yes' when p ≥ α (cannot rule out normal), 'No' when p < α "
    "(reject normal).\n"
    "- **Anderson–Darling** reports no p-value by design; its verdict compares the "
    "statistic to a critical value at α (smaller statistic = closer to normal).\n"
    "- **Shapiro–Wilk** is shown only for samples ≤ 5000; beyond that its p-value is "
    "unreliable, so it is skipped.\n\n"
    "*Caveat for this data:* these tests gain power with sample size. On the full daily "
    "series (Collapse to monthly off) the huge, forward-filled sample flags tiny, harmless "
    "departures, so almost everything tests as non-normal. Monthly sampling shrinks the "
    "sample to a few hundred, so a rejection there is more likely a genuine departure - "
    "usually the fat tails of financial returns. Either way, read the QQ plot, histogram "
    "and skew/kurtosis for the practical picture.\n\n"
    "*Note on time-series data:* even first-difference and return series can show "
    "volatility clustering (large moves following large moves), which violates the "
    "independence assumption. The tests remain useful diagnostics but treat the output "
    "as approximate. "
)

REGIME_AVAILABILITY_HELP = (
    "A regime only appears here if the feature it is derived from is available for this "
    "economy and date range. If that feature was dropped (e.g. as stale) the regime cannot "
    "be computed and is silently unavailable."
)

PARTIAL_P_HELP = (
    "The p-value tests the null hypothesis that the partial correlation is zero - that "
    "once the control variables are accounted for, X and Y have no linear association "
    "left. A small p-value is evidence the leftover association is real rather than "
    "chance. The larger the sample the more over-powered the test - on the daily series "
    "(thousands of autocorrelated, forward-filled rows) p is almost always tiny, and "
    "monthly sampling eases this; either way judge the relationship by the size of the "
    "partial r, not by significance."
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
    "- **Typical value** - the average unless the data is skewed or has outliers the "
    "test switches to the median, a more robust middle value.\n"
    "- **Which test is used (picked automatically):**\n"
    "    - It first checks whether each regime's values are roughly normally distributed and whether "
    "the regimes have a similar spread (variance).\n"
    "    - If yes -> a **t-test** (two regimes) or **ANOVA** (three or more).\n"
    "    - If no -> a rank-based **Mann-Whitney** (two regimes) or **Kruskal-Wallis** (three "
    "or more), which need no such assumptions.\n"
    "- **p-value** - the chance of seeing a difference this big if the regimes were truly "
    "identical. **p-value** less than α means the difference is unlikely to be sole luck.\n"
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
    "This measures whether two regime labels tend to occur together.\n\n"
    "- **The table** counts the days falling into each combination.\n"
    "- **Linked vs unrelated** - 'unrelated' means knowing one regime tells you nothing about "
    "the other; 'linked' means some combinations happen far more (or less) often than chance "
    "alone would produce.\n"
    "- **p-value** - the chance of seeing a pattern this strong if the two were truly "
    "unrelated. Below the α = they are linked.\n"
    "- **Cramér's V** - rescales the result to a 0-1 strength score, comparable across "
    "tables: 0 = unrelated, 1 = one regime perfectly predicts the other. Rule of thumb: "
    "under 0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, above 0.5 strong.\n"
    "- **Why this stays daily:** the 'Collapse to monthly' toggle above governs the numeric "
    "variable in the group-comparison test; this block is a separate analysis that compares "
    "two regime labels directly, on their daily overlap. Collapsing to monthly would not "
    "rescue the p-value anyway - a regime persists for months, so the labels are heavily "
    "autocorrelated and the effective sample is far smaller than the row count at any "
    "frequency.\n"
    "- **Caveat:** because of that the p-value is almost always tiny whatever the frequency, "
    "so judge by the strength (Cramér's V), not significance alone."
)

HOPKINS_HELP = (
    "The Hopkins statistic asks whether the data has any clustering tendency before you "
    "cluster it. It compares how close real points sit to their nearest neighbour against "
    "how close uniformly random points (drawn from the same range) sit to the real data. "
    "Near 0.5 the data is spread like noise (clusters would be arbitrary); near 1 points "
    "bunch into dense groups worth clustering. Caveat for this data: on trending level data "
    "Hopkins is biased high and should not be read as evidence of good clusters - judge separation "
    "by silhouette."
)

STRUCTURE_GUIDE = (
    "- **PCA** - rebuilds the features as a few uncorrelated components ordered by how much "
    "variation they capture; many overlapping macro and rate series collapse into a "
    "readable 2D picture.\n"
    "- **Loadings** - each component is a weighted blend of the original features; "
    "the loading is that weight, roughly between -1 and +1. A large positive "
    "loading means the feature rises strongly with the component; a large "
    "negative one means it moves strongly the opposite way; near 0 means the feature "
    "barely shapes that component. Read each component by the features with the biggest "
    "absolute loadings - they name what the axis represents.\n"
    "- **K-Means** - splits the observations into k groups so each observation sits with others most "
    "similar to it; the result is a *data-driven* regime, found without the rule-based labels.\n"
    "- **Levels, not returns** - structure is read on standardised levels, so a cluster is a "
    "persistent state of the economy; neighbouring observations sharing a state is expected.\n"
)

CLUSTER_REGIME_HELP = (
    "Two complementary measures describe how closely the data-driven clusters match a "
    "rule-based regime classification. The Adjusted Rand Index (ARI) compares the two "
    "partitions, correcting for chance (1 = identical, 0 = chance, negative = worse than "
    "chance). Cramér's V measures the strength of association between cluster and regime "
    "labels on a 0-1 scale. High values for both indicate that the clustering has recovered "
    "the known regime structure. Note: ARI is less informative when comparing many clusters "
    "with only a few regimes, as splitting one regime into multiple clusters lowers the "
    "score even if the correspondence is clear. In such cases, Cramér's V is often the more "
    "appropriate measure."
)

COINTEGRATION_HELP = (
    "Two trending series can look strongly correlated in levels even when nothing links "
    "them - the spurious-regression trap. The augmented Dickey-Fuller (ADF) test checks "
    "whether each series is stationary (no persistent trend); most rates and macro levels "
    "are not. The Engle-Granger test then regresses one series on the other and tests "
    "whether the leftover residual is stationary: if it is, the pair is cointegrated - tied "
    "together around a stable long-run equilibrium - and the level relationship is genuine. "
    "If not, a high level correlation is most likely an artefact of shared trends. This is "
    "why differencing is not used here: cointegration is defined on the levels themselves, "
    "and on the forward-filled daily series a naive difference would in any case collapse "
    "to mostly zeros."
)

MONTHLY_RATIONALE = (
    "Modelling runs on a monthly view of the daily data (the month-end snapshot). The targets are "
    "monthly/meeting-cadence, so the daily rows are mostly forward-filled repeats of the same monthly "
    "value - keeping them would inflate the sample, leak across the train/test split and make "
    "cross-validation meaningless. Monthly sampling removes that repetition while leaving the trend "
    "and autocorrelation intact, so the transform and stationarity discipline still applies."
)

SPLIT_HELP = (
    "The data is split in time order (no shuffling) into train, validation and test. Train fits the "
    "models, validation is used to pick the operating threshold, blend weights and early stopping "
    "(kept separate so those choices do not peek at the test set), and test is the untouched final "
    "score. Presets keep validation and test roughly equal in size."
)

CV_HELP = (
    "Cross-validation uses `TimeSeriesSplit`: each fold trains on the past and validates on the next "
    "block."
)

BUDGET_HELP = (
    "The training budget controls how hard the search for good hyperparameters works. Fast and "
    "Balanced use a randomised search over a few / more settings. Thorough uses Optuna (guided "
    "search, more trials, full roster). The shipped models are pre-tuned - a live run refits them on "
    "the current data to fight staleness as the dataset grows."
)

METRIC_HELP = (
    "The scoring metric drives both model selection and the reported leaderboard. Regression: RMSE "
    "and MAE are average error sizes (lower is better), R2 is the share of variance explained (higher "
    "is better). Classification: ROC-AUC (macro, one-vs-rest) measures class separation, F1-macro "
    "balances precision and recall equally across classes, balanced accuracy averages per-class "
    "recall - all robust to the Hold-heavy imbalance."
)

LEAKAGE_HELP = (
    "When the target is a series (e.g. the policy rate), its own current value and the features "
    "trivially derived from it (its moving averages, spreads) are removed from the inputs. Otherwise "
    "the model would 'predict' the future rate from a near-copy of it and score unrealistically well."
)

TARGET_TRANSFORM_HELP = (
    "A target transform (log or Yeo-Johnson) can stabilise a skewed, strictly-positive target such as "
    "inflation or GDP growth. It is never applied to the policy rate or spreads, which pass through "
    "zero and go negative, so the transform would be undefined."
)

THRESHOLD_HELP = (
    "Rather than always taking the highest-probability class, the decision threshold for each class is "
    "set by Youden's J (the point maximising true-positive minus false-positive rate) on the "
    "validation set, then applied unchanged to the test set. Tuning it on validation keeps the test "
    "score honest, and it helps the rare Hike/Cut classes get picked up under the Hold-heavy mix."
)

DIAG_SPLIT_HELP = (
    "Choose which chronological split the diagnostics below are measured on. Test is the honest "
    "out-of-sample read; Dev and Train help you spot overfitting (a large Train-vs-Test gap). The "
    "per-class decision thresholds are always tuned on Dev regardless of this choice."
)

LEADERBOARD_HELP = (
    "Every model is scored on train, validation and test for the chosen metric. Read the test column "
    "for real-world performance; a model that is excellent on train but weak on test is overfitting. "
    "The winner badge marks the best validation score - the pick made without touching the test set."
)

HORIZON_HELP = (
    "How many months ahead the supervised target looks: the model learns to predict the policy "
    "decision / value this many months into the future from today's drivers."
)

TARGET_LEVEL_HELP = (
    "The regression target is always the forward level of the series (its value the chosen number of "
    "months ahead). The implied change versus the latest observed level is shown next to the prediction."
)

NEURAL_HELP = (
    "The Keras nets (an MLP baseline and an LSTM, optionally a Conv1D) are heavy to fit and prone to "
    "overfit on this dataset, and they rarely beat the gradient-boosted trees here, so they are "
    "opt-in. Turn them on to showcase the deep-learning roster; expect the run to take noticeably "
    "longer."
)

CV_BUDGET_HELP = (
    "Hyperparameters are searched with cross-validation on the training split only, in time order "
    "(`TimeSeriesSplit`). Fast and Balanced use a randomised search (15 and 40 settings); Thorough "
    "uses Optuna (60 guided trials). The CV score is the mean fold score in the chosen metric. On the "
    "small monthly sample an early fold can hold a single class, where one-vs-rest ROC-AUC is "
    "undefined and contributes the chance value 0.5 - so a flat-looking CV number reflects those "
    "degenerate folds, not a bug."
)

ENSEMBLE_HELP = (
    "Both ensembles combine the base models on the validation split. Blend is a weighted average "
    "whose weights are proportional to each model's inverse validation error, so stronger models "
    "count more. Stack trains a small meta-model on the base models' validation predictions, learning "
    "how best to combine them."
)

ROC_HELP = (
    "The ROC curve plots the true-positive rate against the false-positive rate as the threshold "
    "varies, one line per class (one-vs-rest). A curve hugging the top-left is good; the dashed red "
    "diagonal is random guessing (0.5 = chance, AUC 1 = perfect)."
)

PR_HELP = (
    "The precision-recall curve trades off precision (how many predicted positives are correct) "
    "against recall (how many actual positives are caught), one line per class. A good curve stays "
    "high and flat toward the top-right; the dotted line is that class's chance level (its base "
    "rate). It is more informative than ROC when a class is rare, as here. Average precision (AP) is "
    "the area under it. Jagged or collapsing shapes usually mean very few positives on this split, so "
    "read that class with caution."
)

LIFT_HELP = (
    "Rank the months from most to least confident for a class, then walk down that ranking. Lift is "
    "how many times more of that class you capture than picking months at random: a lift of 2 across "
    "the top 10% means that slice holds twice the class's base rate. Curves start high and decay "
    "toward the dashed line at 1 (random targeting); the longer a curve stays above 1, the more "
    "useful the model's confidence ranking is."
)

CONFUSION_HELP = (
    "The confusion matrix cross-tabulates actual (rows) against predicted (columns) classes. The "
    "diagonal is correct predictions; off-diagonal cells show what gets confused with what. Normalise "
    "by row to read it as 'of the actual X, what share did we predict as each class'."
)

IMPORTANCE_HELP = (
    "Feature importance ranks the inputs by how much they drive the model, scaled to 0-100. Native "
    "importance comes from the model itself (tree split gains, or the size of linear coefficients); "
    "permutation importance shuffles one feature at a time and measures the drop in score, so it is "
    "model-agnostic and computed on held-out data. They can disagree - native reflects how the model "
    "was built, permutation reflects what actually helps on unseen data."
)

GROUP_IMPORTANCE_HELP = (
    "Importance summed into business blocks instead of individual features, re-scaled to 0-100, so "
    "you can see which kind of information the model leans on overall. Blocks: Rates - the policy "
    "rate, Treasury/benchmark yields and yield-curve spreads (rate_/yld_/sprd_); Macro - real-economy "
    "series such as inflation, unemployment, participation, savings and GDP growth; Market - "
    "equities, FX, commodities and volatility indices (eq_/fx_/cmd_/idx_)."
)

SHAP_HELP = (
    "SHAP values decompose a single prediction into per-feature contributions that add up to the gap "
    "between that prediction and the average prediction, so you can see which features pushed a given "
    "decision up or down. Shown for tree/boosting models via the exact TreeExplainer."
)

PDP_HELP = (
    "A partial dependence plot shows how the predicted output moves as one feature is varied across "
    "its range while the others are held at their observed values, i.e. the model's average response "
    "to that feature. Nearly flat lines are a genuine model reading, not a plotting bug: they mean "
    "the prediction barely changes as this feature moves, so the model leans on other features "
    "(common with the Hold-dominated direction target)."
)

PROB_HIST_HELP = (
    "Each panel is the distribution of the model's predicted probability for one class, counted over "
    "the months on this split. Green bars are months that truly belong to that class, red bars the "
    "rest. Good separation shows green piled near 1 and red near 0; heavy overlap in the middle means "
    "the model is unsure about that class."
)

RESIDUAL_HELP = (
    "A well-specified regression leaves residuals (actual minus predicted) scattered randomly around "
    "zero. A visible trend or a run of same-sign residuals means the model missed structure; a "
    "consistent offset means it is biased high or low."
)

ECON_BASELINE_HELP = (
    "A statsmodels OLS / Logit fit is shown alongside the machine-learning models as an interpretable "
    "baseline. It is judged on classical assumptions (coefficient significance, residual normality, "
    "Durbin-Watson autocorrelation, multicollinearity via the condition number, influential points) "
    "as well as fit; the ML/DL models are judged on prediction only."
)

SCENARIO_HELP = (
    "Move a few key drivers and hold the rest at their latest values to read the model's prediction "
    "under that scenario. This is a ceteris-paribus what-if, not a forecast: it ignores how the "
    "drivers move together in reality, so treat it as sensitivity analysis."
)

FORECAST_HELP = (
    "These are classic time-series models fit on a single series' own past - target-lags only, with "
    "none of the Setup feature matrix or the trained ML models. Pick a series and a horizon; the shaded "
    "band is the confidence interval and it widens further out, because the further ahead the less "
    "certain the forecast."
)

FORECAST_INDEPENDENCE = (
    "This tab is independent of the Setup and Leaderboard tabs: it does not read the trained models, the "
    "task or the target chosen there. Only the economy carries over, to decide which dataset's series "
    "you can forecast. The horizon slider below sets how many months ahead these models project and is "
    "separate from the Setup prediction horizon."
)

ARIMA_HELP = (
    "ARIMA/SARIMA models a series from its own past values (AR), past forecast errors (MA) and "
    "differencing (I) to remove a trend; the seasonal part repeats that at a fixed period. Read the "
    "ACF/PACF to pick the orders, then check that the residuals look like white noise (Ljung-Box) and "
    "compare AIC/BIC across candidates (lower is better)."
)

GARCH_HELP = (
    "GARCH models the variance rather than the level: it captures volatility clustering, where large "
    "moves follow large moves. Fit it on a return or change series, not the level. The plot shows the "
    "estimated conditional volatility over time and its forecast."
)

VAR_HELP = (
    "A vector autoregression models several series jointly, each as a function of the recent past of "
    "all of them, so it captures feedback (e.g. rate <-> inflation <-> unemployment). Impulse "
    "responses trace how a shock to one series propagates to the others; the variance decomposition "
    "shows how much of each series' forecast error each shock explains."
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
        return "stationary" if flag else "non-stationary (trending)"

    head = (
        f"ADF: {x_name} is {word(res['x_stationary'])} (p = {format_pvalue(res['adf_p_x'])}); "
        f"{y_name} is {word(res['y_stationary'])} (p = {format_pvalue(res['adf_p_y'])})."
    )
    if res["x_stationary"] and res["y_stationary"]:
        tail = "Both are already stationary, so the level correlation is not a trend artefact."
    elif res["cointegrated"]:
        tail = (
            f"Engle-Granger p = {format_pvalue(res['coint_p'])} < α = {alpha:g}: the pair is "
            "cointegrated - they share a stable long-run relationship, so the level "
            "association is genuine, not spurious."
        )
    else:
        tail = (
            f"Engle-Granger p = {format_pvalue(res['coint_p'])} (not below α = {alpha:g}): no "
            "cointegration detected, so a strong level correlation here is likely spurious "
            "(shared trends). Read it as co-movement only."
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
        return "—"
    if x != 0 and abs(x) < 1:
        decimals = max(decimals, 4)
    return f"{x:,.{decimals}f}"


def format_pvalue(p: float, threshold: float = 1e-5) -> str:
    """
    Format a p-value for display, flooring tiny values and flagging missing ones.

    Args:
        p (float): The p-value (NaN for tests that report none, e.g. Anderson-Darling).
        threshold (float): Values below this are shown as '<threshold' instead of 0.

    Returns:
        str: A readable p-value, '<0.00001' for tiny values, or '—' when absent.
    """
    if pd.isna(p):
        return "—"
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
    if excess_kurtosis > 1.0:
        shape = "heavy-tailed (leptokurtic) - more outliers than a normal"
    elif excess_kurtosis < -1.0:
        shape = "light-tailed (platykurtic) - fewer outliers than a normal"
    else:
        shape = "tails close to normal (mesokurtic)"
    return f"Excess kurtosis {excess_kurtosis:+.2f}: {shape}."


def normality_verdict(battery: pd.DataFrame, alpha: float) -> str:
    """
    Summarise the normality battery as a single mixed/reject/plausible verdict.

    Args:
        battery (pd.DataFrame): Output of `stats.normality_battery` (has a
            boolean `normal` column, one row per test).
        alpha (float): Significance level used for the verdict.

    Returns:
        str: One sentence stating how many tests reject normality.
    """
    n_tests = len(battery)
    n_normal = int(battery["normal"].sum())
    if n_normal == n_tests:
        return f"All {n_tests} tests fail to reject normality at α={alpha:g}; normal is plausible."
    if n_normal == 0:
        return f"All {n_tests} tests reject normality at α={alpha:g}; treat as non-normal."
    return (
        f"{n_tests - n_normal} of {n_tests} tests reject normality at α={alpha:g}; evidence is mixed, lean non-normal."
    )


def compare_groups_sentence(result: dict, alpha: float, label: str) -> str:
    """
    Plain-language conclusion for a `stats.compare_groups` result.

    Args:
        result (dict): Output of `stats.compare_groups`.
        alpha (float): Significance level used for the verdict.
        label (str): Name of the variable being compared.

    Returns:
        str: Whether the variable's typical value differs across regimes.
    """
    centre = "average" if result["parametric"] else "median"
    p = format_pvalue(result["p_value"])
    n_groups = len(result["group_sizes"])
    scope = "between the two regimes" if n_groups == 2 else f"across the {n_groups} regimes"
    if result["differs"]:
        return (
            f"Difference found: the {centre} of {label} is not the same {scope} "
            f"(p = {p}, below α = {alpha:g}) - unlikely to be down to chance."
        )
    return f"No clear difference: the {centre} of {label} looks the same {scope} (p = {p}, not below α = {alpha:g})."


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
    return f"Strength of the link (Cramér's V) = {v:.2f}: {strength} (0 = unrelated, 1 = lockstep)."


def hopkins_verdict(h: float) -> str:
    """One-sentence reading of a Hopkins statistic (0.5 random, 1 clusterable)."""
    if h >= 0.75:
        reading = "strong clustering tendency, clusters are worth seeking"
    elif h >= 0.6:
        reading = "some clustering tendency"
    else:
        reading = "little structure, close to a uniform cloud, so clusters may be arbitrary"
    return f"Hopkins = {h:.2f}: {reading} (0.5 = random, 1 = highly clusterable)."


def silhouette_verdict(score: float) -> str:
    """One-sentence reading of a mean silhouette score (-1 to 1)."""
    if score >= 0.5:
        reading = "clusters are well separated"
    elif score >= 0.25:
        reading = "clusters are weak and overlap"
    else:
        reading = "little real separation, the grouping is mostly arbitrary"
    return f"Silhouette = {score:.2f}: {reading} (1 = tight and distinct, 0 = overlapping)."


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

def best_model_sentence(name: str, metric: str, test_score: float | None) -> str:
    """One-line announcement of the winning model and its test score."""
    if test_score is None or pd.isna(test_score):
        return f"Best model: {name} (leading on validation {metric})."
    return f"Best model: {name}, scoring {test_score:.3f} on the held-out test set ({metric})."


def overfit_note(train_score: float | None, test_score: float | None, metric: str) -> str:
    """Reading of the train-vs-test gap for the chosen metric (sign-aware)."""
    if train_score is None or test_score is None or pd.isna(train_score) or pd.isna(test_score):
        return ""
    higher_better = metric not in ("RMSE", "MAE")
    gap = train_score - test_score if higher_better else test_score - train_score
    rel = gap / (abs(train_score) if train_score else 1.0)
    head = f"Train {metric} {train_score:.3f} vs test {test_score:.3f}: "
    if rel > 0.25:
        return head + (
            "a large gap - the model fits the training window far better than unseen data, a sign of "
            "overfitting on this small monthly sample."
        )
    if rel > 0.1:
        return head + "a moderate gap; some overfitting, read the test column as the honest score."
    return head + "train and test are close, so generalisation looks stable."


def roc_auc_verdict(auc: float | None, split: str | None = None) -> str:
    """Band reading of a macro one-vs-rest ROC-AUC, averaged over the classes."""
    where = f" on the {split.lower()} split" if split else ""
    if auc is None or pd.isna(auc):
        return f"ROC-AUC is undefined{where} (a class may be absent from this split)."
    if auc >= 0.9:
        band = "excellent"
    elif auc >= 0.8:
        band = "strong"
    elif auc >= 0.7:
        band = "moderate"
    elif auc >= 0.6:
        band = "weak"
    else:
        band = "close to chance"
    return (
        f"ROC-AUC = {auc:.2f}{where}: {band} separation, averaged one-vs-rest over the classes "
        "(0.5 = coin flip, 1 = perfect)."
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

def regression_fit_verdict(r2: float | None) -> str:
    """Band reading of a regression R2."""
    if r2 is None or pd.isna(r2):
        return ""
    if r2 < 0:
        reading = "worse than just predicting the mean - no usable signal on this split"
    elif r2 < 0.3:
        reading = "weak - little of the variation is explained"
    elif r2 < 0.6:
        reading = "moderate"
    elif r2 < 0.8:
        reading = "strong"
    else:
        reading = "very strong - the train-versus-test gap above shows whether it holds out of sample"
    return f"R2 = {r2:.2f}: {reading}."

def residual_verdict(y_true, y_pred) -> str:
    """Plain reading of regression residuals: systematic bias and leftover autocorrelation."""
    actual = pd.Series(y_true).to_numpy(dtype=float)
    predicted = pd.Series(y_pred).to_numpy(dtype=float)
    resid = pd.Series(actual - predicted).dropna()
    if len(resid) < 3:
        return ""
    spread = resid.std(ddof=0) or 1.0
    bias = resid.mean()
    ac1 = resid.autocorr(lag=1)
    parts = []
    if abs(bias) > 0.1 * spread:
        direction = "over-predicts" if bias < 0 else "under-predicts"
        parts.append(f"a systematic bias (it {direction} on average)")
    if ac1 is not None and abs(ac1) > 0.3:
        parts.append(f"leftover autocorrelation (lag-1 = {ac1:.2f}), so they are not white noise")
    if not parts:
        return (
            "Residuals scatter around zero with no strong pattern, which is what a well-specified "
            "model looks like."
        )
    return "The residuals show " + " and ".join(parts) + "."

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
    """Plain reading of an `econometrics.stationarity` ADF result."""
    if res is None:
        return f"Not enough observations to test {name} for stationarity."
    p = format_pvalue(res["p_value"])
    if res["stationary"]:
        return f"ADF on {name}: stationary (p = {p}), so it can be modelled without differencing."
    return (
        f"ADF on {name}: non-stationary (p = {p}), a trend or unit root remains - difference it or "
        "model the change instead of the level."
    )

def target_level_note(res: dict | None, name: str) -> str:
    """
    Stationarity reading tailored to the level-only regression target.

    The forward level is predicted from current drivers, not from the target's own past
    (which is excluded as leakage), so a unit root in the level does not force differencing
    here; it only warns that the level is persistent and the honest read is the test score.

    Args:
        res (dict | None): An `econometrics.stationarity` result, or None.
        name (str): Display name of the target.

    Returns:
        str: One-line reading of the ADF result for a level target.
    """
    if res is None:
        return f"Not enough observations to test {name} for stationarity."
    p = format_pvalue(res["p_value"])
    if res["stationary"]:
        return f"ADF on {name}: stationary (p = {p}); the forward level is well behaved for modelling."
    return (
        f"ADF on {name}: non-stationary (p = {p}). The forward level is predicted from current "
        "drivers with the target's own history excluded, so this persistence is mitigated rather than "
        "removed - judge the fit on the held-out test score."
    )

def ljung_box_verdict(diag: dict, alpha: float = 0.05) -> str:
    """Reading of the Ljung-Box residual white-noise check from `arima_diagnostics`."""
    p = format_pvalue(diag["ljung_box_p"])
    if diag["ljung_box_p"] >= alpha:
        return f"Ljung-Box p = {p}: the residuals look like white noise, so the model has captured the autocorrelation."
    return (
        f"Ljung-Box p = {p} (below α = {alpha:g}): the residuals still carry autocorrelation - the "
        "order is probably too low."
    )

def var_lag_sentence(selected_lag: int, ic: str = "aic") -> str:
    """One line explaining how the VAR lag order was chosen."""
    return (
        f"The lag order was selected automatically as {selected_lag}, by minimising the {ic.upper()} "
        "across the candidate lags shown below."
    )

def fevd_verdict(data: dict | None) -> str:
    """Name, at the final horizon, the dominant driver of each series' forecast error variance."""
    if data is None:
        return ""
    names = list(data["names"])
    decomp = data["decomp"]
    parts = []
    for i, name in enumerate(names):
        shares = list(decomp[i, -1, :])
        j = max(range(len(shares)), key=lambda k: shares[k])
        driver = "its own past shocks" if j == i else names[j]
        parts.append(f"{name} is explained mostly by {driver} ({shares[j]:.0%})")
    return "At the final horizon, " + "; ".join(parts) + "."

def ols_assumptions_note(ols: dict) -> str:
    """Plain reading of the OLS residual diagnostics (Durbin-Watson, Jarque-Bera, condition number)."""
    dw = ols["durbin_watson"]
    if dw < 1.5:
        dw_read = f"Durbin-Watson {dw:.2f} (below 1.5) suggests positively autocorrelated residuals"
    elif dw > 2.5:
        dw_read = f"Durbin-Watson {dw:.2f} (above 2.5) suggests negatively autocorrelated residuals"
    else:
        dw_read = f"Durbin-Watson {dw:.2f} is near 2, so little residual autocorrelation"
    jb_read = (
        "residuals depart from normality"
        if ols["jarque_bera_p"] < 0.05
        else "residuals are consistent with normality"
    )
    cond = ols["condition_number"]
    cond_read = (
        "a high condition number warns of multicollinearity among the entered features"
        if cond > 30
        else "the condition number is moderate, so little multicollinearity"
    )
    return f"{dw_read}; {jb_read} (JB p = {format_pvalue(ols['jarque_bera_p'])}); {cond_read} ({cond:,.0f})."
