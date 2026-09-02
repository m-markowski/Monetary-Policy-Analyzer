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
    "Modelling runs on a monthly view of the daily data (the month-end snapshot). The daily rows are mostly "
    "forward-filled repeats of the same monthly value - keeping them would inflate the sample, leak across "
    "the train/test split and make cross-validation meaningless. Monthly sampling removes that repetition but "
    "leaves the trend and autocorrelation intact; that is handled by the targets themselves, which are forward changes "
    "(regression) or direction labels (classification) rather than trending levels."
)

SPLIT_HELP = (
    "The data is split in time order (no shuffling) into train, dev and test. Train fits the models; "
    "dev picks the class thresholds and blend weights, and steers early stopping for the neural nets "
    "- kept separate so none of those choices peeks at the test set. Presets keep dev and test roughly equal in size."
)

CV_HELP = (
    "Cross-validation uses `TimeSeriesSplit`: each fold trains on the past and validates on the next "
    "block."
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
    "If the inputs contained the target's contemporaneous value - directly, through a feature "
    "engineered from it, or through a spread it is a component of - the model would 'predict' the "
    "future from a near-copy of the present and score unrealistically well, so those columns are "
    "removed. The target's own lagged values are different: they were already observed when the "
    "prediction is made, so including them is standard autoregression, not leakage."
)

THRESHOLD_HELP = (
    "The operating point turns class probabilities into a single Hike/Hold/Cut call. The default "
    "is plain argmax - take the highest-probability class. The toggle switches to per-class Youden's J "
    "thresholds (the point maximising true-positive minus false-positive rate), tuned on the dev split. "
    "Youden favours the rare Hike/Cut classes: it catches more of them at the price of more false "
    "alarms, so try it when missing a move costs more than a wrong call."
)

DIAG_SPLIT_HELP = (
    "Choose which chronological split the diagnostics below are measured on. Test is the honest "
    "out-of-sample read and the default. Train shows the fit on the very months the model was "
    "fitted to - compare it with Test: a much better Train read means the model memorised its "
    "training window (overfitting). Dev is the tuning split (ensemble weights, the winner pick and, when "
    "the toggle is on, the Youden class thresholds); inspect it to see the data those choices were based on, "
    "and note it spans the 2020 COVID shock, so every model reads structurally worse there. "
    "'Train + Dev + Test' shows the whole sample as one path."
)

LEADERBOARD_HELP = (
    "Every model is scored on train, dev and test. Read the test column for real-world performance; "
    "a model that is excellent on train but weak on test is overfitting. The winner badge marks the "
    "base model with the best Dev value of the chosen scoring metric. Blend and Stack are excluded from "
    "the pick: they are fit on the dev split, so their dev scores are partly in-sample."
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
    "Pick the series to forecast. Training uses direct multi-step forecasting: the model maps the "
    "drivers observed in one month straight to this series' movement over the following months (the "
    "chosen horizon) in a single step, rather than iterating month by month. Under the hood it "
    "predicts the forward change - the near-stationary quantity - and the app adds that change back "
    "to the latest observed value wherever a level is displayed."
)

NEURAL_HELP = (
    "The Keras nets (an MLP baseline and an LSTM) are heavy to fit and prone to overfit on this "
    "dataset, and they rarely beat the gradient-boosted trees here, so they are opt-in. Turn them "
    "on to showcase the deep-learning roster; expect the run to take noticeably longer."
)

CV_BUDGET_HELP = (
    "Hyperparameters are searched with cross-validation on the training split only, in time order "
    "(`TimeSeriesSplit`). Fast and Balanced use a randomised search (15 and 40 settings); Thorough "
    "uses Optuna (60 guided trials). The CV score is the mean fold score in the chosen metric; for "
    "the error metrics (RMSE, MAE) it is shown negated - sklearn's sign-aligned convention, so a "
    "higher CV score is always better."
)

CV_ROC_AUC_NOTE = (
    "On the small monthly sample an early fold can hold a single class, where one-vs-rest ROC-AUC "
    "is undefined and contributes the chance value 0.5 - so a flat-looking CV number reflects those "
    "degenerate folds, not a bug."
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
    "How the single ROC-AUC number is formed: the three-class problem is turned into three "
    "one-vs-rest binary problems (Hike vs the rest, Hold vs the rest, Cut vs the rest), an AUC is "
    "computed for each and the three are averaged with equal weight (macro) - so the rare Hike/Cut "
    "classes count as much as the dominant Hold."
)

COVID_DEV_NOTE = (
    "Dev scores may be structurally worse when the 2020 COVID shock falls within the dev window - or "
    "any other split - as the dataset grows and time-based split boundaries shift. This is an era effect, "
    "not a model fault; dev-based ranking remains usable if the shock affects models similarly."
)

BASELINE_HELP_REG = (
    "The 'Baseline: no change' row is a zero-change random walk: it predicts the series does not "
    "move over the horizon, the classic macro-forecasting yardstick. Read RMSE/MAE - a typical "
    "miss in the target's own units - together with the Skill columns, which rescale each model "
    "against that baseline (1 - MSE_model / MSE_naive: 0 = no better than assuming no change, "
    "1 = perfect, negative = worse than doing nothing)."
)

BASELINE_HELP_CLF = (
    "Two naive baselines anchor the board. 'Majority class' always predicts the most common "
    "training label; 'trailing momentum' extrapolates the sign of the rate move over the previous "
    "h months - the strongest baseline that uses only information available at prediction time. The "
    "hard-label baselines output no probabilities, so their ROC-AUC is blank."
)

ROC_HELP = (
    "The ROC curve plots the true-positive rate against the false-positive rate as the threshold "
    "varies, one line per class (one-vs-rest). A curve hugging the top-left is good; the dashed red "
    "diagonal is random guessing (0.5 = chance, AUC 1 = perfect)."
)

CONFUSION_HELP = (
    "The confusion matrix cross-tabulates actual (rows) against predicted (columns) classes. The "
    "diagonal is correct predictions; off-diagonal cells show what gets confused with what. Normalise "
    "by row to read it as 'of the actual X, what share did we predict as each class'."
)

IMPORTANCE_HELP = (
    "Feature importance ranks the inputs by how much they drive the model, scaled to 0-100. Native "
    "importance comes from the model itself (tree split gains, or the size of linear coefficients); "
    "permutation importance shuffles one feature at a time and measures how much the score drops. "
    "The permutation check runs on the dev months - data the model never saw while fitting - and "
    "ignores the split selector at the top, so it measures what genuinely helps on unseen data "
    "rather than what the model memorised. The two can disagree: native reflects how the model "
    "was built, permutation what actually helps out of sample."
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
    "Each panel is the distribution of the model's predicted probability for one class, counted "
    "over the months on this split. Bars are stacked, not overlaid: the green segment counts the "
    "months that truly belong to the class, the red segment the rest, and the full bar is every "
    "month in that probability bin. Good separation piles green near 1 and red near 0; mixed bars "
    "in the middle mean the model is unsure about that class."
)

RESIDUAL_HELP = (
    "A well-specified regression leaves residuals (actual minus predicted) scattered randomly around "
    "zero. A visible trend or a run of same-sign residuals means the model missed structure; a "
    "consistent offset means it is biased high or low. The lag-1 number in the verdict is the "
    "correlation between one month's error and the next month's: near 0 the errors are independent, "
    "as they should be; near 1 the model makes almost the same miss month after month, so "
    "there is predictable structure it failed to use. For the ML models this is a visual health "
    "check, not a pass/fail test - the formal residual assumptions (Durbin-Watson, normality) belong "
    "to the OLS baseline below, the only model actually built on them."
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
    "These are classic time-series models fit on a single series' own past - target-lags only, with "
    "none of the Setup feature matrix or the trained ML models. Only the economy carries over, to decide "
    "which dataset's series you can forecast. Pick a series and a horizon; the shaded band is the confidence "
    "interval and it widens further out, because the further ahead the less certain the forecast."
)

ARIMA_HELP = (
    "ARIMA/SARIMA models a series from its own past values (AR), past forecast errors (MA) and "
    "differencing (I) that removes a trend; the seasonal part repeats that at an annual period. The app "
    "chooses the order for you: it grid-searches candidate orders (seasonal ones included), ranks them "
    "by information criteria (lower AIC/BIC is better) and checks the residuals for leftover "
    "autocorrelation (Ljung-Box). The ACF/PACF charts and the candidate table are informational - "
    "useful mainly if you override the order manually."
)

GARCH_HELP = (
    "GARCH models the variance of a series rather than its level: it captures volatility clustering, "
    "where turbulent months tend to follow turbulent months. The app prepares the input for you - it "
    "always fits the monthly change of the chosen series, because GARCH assumes a roughly zero-mean "
    "input: one that fluctuates around zero with no trend, so all the systematic movement is in the "
    "size of the swings rather than their direction. The chart reads as how large a typical monthly "
    "move is, month by month, and how large the model expects it to be ahead."
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
    against the score a constant mean prediction would get in the same metric
    (the target's standard deviation for RMSE, its mean absolute deviation for
    MAE).

    Args:
        metric (str): Display metric name, e.g. 'RMSE' or 'ROC-AUC (macro/OvR)'.
        value (float | None): The metric value to interpret.
        benchmark (float | None): Same-metric score of a constant mean
            prediction, enabling the RMSE/MAE relative reading.
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
            tail = "no better than guessing the average"
        elif ratio >= 0.8:
            tail = "a modest improvement on guessing the average"
        elif ratio >= 0.5:
            tail = "a solid improvement on guessing the average"
        else:
            tail = "a large improvement on guessing the average"
        return f"{head} A constant mean prediction would score {format_number(benchmark)}{u}, so this is {tail}."
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
            f"Skill {skill:+.2f}: no better than {naive_name} - the model adds nothing over assuming "
            "nothing changes."
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
        parts.append(
            f"leftover autocorrelation (lag-1 = {ac1:.2f}: consecutive months' errors are "
            "correlated, so the model keeps repeating similar misses instead of leaving "
            "unpredictable noise)"
        )
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
    """Plain reading of an `econometrics.stationarity` ADF result, for the forecast tab."""
    if res is None:
        return f"Not enough observations to test {name} for stationarity."
    p = format_pvalue(res["p_value"])
    if res["stationary"]:
        return (
            f"ADF on {name}: stationary (p = {p}), so no differencing is required and the automatic "
            "search can keep d = 0."
        )
    return (
        f"ADF on {name}: non-stationary (p = {p}) - a trend or unit root remains. The automatic order "
        "search handles this for you: the differencing order d it selects removes the trend before the "
        "AR and MA parts are fit."
    )

def target_change_note(res: dict | None, name: str) -> str:
    """
    Stationarity reading for the forward-change regression target.

    The engine models the forward change of the series, not its trending level, and
    the change construction is itself the classic stationarity fix - so the ADF here
    is a confirmation that the modelled target is well behaved, not a decision point.

    Args:
        res (dict | None): An `econometrics.stationarity` result on the change target.
        name (str): Display name of the target.

    Returns:
        str: One-line reading of the ADF result for the change target.
    """
    if res is None:
        return f"Not enough observations to test the {name} target for stationarity."
    p = format_pvalue(res["p_value"])
    if res["stationary"]:
        return (
            f"ADF on the modelled target - the forward change of {name}: stationary (p = {p}). "
            "Modelling the change rather than the trending level is what keeps the target well "
            "behaved; the level is only reconstructed for display."
        )
    return (
        f"ADF on the modelled target - the forward change of {name}: still non-stationary (p = {p}). "
        "The change construction mitigates the level's persistence rather than fully removing it "
        "here, so judge the fit on the held-out test score."
    )

def ljung_box_verdict(diag: dict, alpha: float = 0.05) -> str:
    """Reading of the Ljung-Box residual white-noise check from `arima_diagnostics`."""
    p = format_pvalue(diag["ljung_box_p"])
    if diag["ljung_box_p"] >= alpha:
        return f"Ljung-Box p = {p}: the residuals look like white noise, so the model has captured the autocorrelation."
    return (
        f"Ljung-Box p = {p} (below α = {alpha:g}): some autocorrelation remains that this order does "
        "not capture - even the best candidate can fail this check on a stubborn series, so treat the "
        "forecast interval as approximate."
    )

def garch_persistence_note(persistence: float) -> str:
    """Explain the shape of the volatility forecast from the estimated GARCH persistence."""
    if persistence >= 0.97:
        return (
            f"Estimated persistence (α+β) = {persistence:.2f}: volatility shocks decay very slowly, so "
            "the forecast stays close to the current volatility level - a near-flat line is the genuine "
            "model output here."
        )
    return (
        f"Estimated persistence (α+β) = {persistence:.2f}: volatility shocks fade at this rate month to "
        "month, so the forecast reverts toward the series' long-run volatility over the horizon."
    )

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
    if cond < 30:
        cond_read = "the condition number is low, so multicollinearity is not a concern."
    elif cond < 100:
        cond_read = (
            "the condition number is moderate - some multicollinearity, so individual coefficients "
            "are less precisely pinned down, though the overall fit is unaffected."
        )
    elif cond < 1000:
        cond_read = (
            "the condition number is high - strong multicollinearity, so individual coefficient "
            "sizes and signs are unstable and should be read with caution."
        )
    else:
        cond_read = (
            "the condition number is extreme - the entered features are close to linearly "
            "dependent, so individual coefficients are not trustworthy even where the overall fit "
            "is fine."
        )
    return (
        f"{dw_read}; {jb_read} (JB p = {format_pvalue(ols['jarque_bera_p'])}); {cond_read} "
        f"Rule of thumb: below 30 fine, 30-100 moderate, above 100 problematic)."
    )
