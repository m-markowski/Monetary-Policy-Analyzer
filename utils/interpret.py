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
    "*Caveat for this data:* at large sample sizes these tests are so powerful they flag "
    "tiny, harmless departures, so financial returns almost always test as non-normal. "
    "Read the QQ plot, histogram and skew/kurtosis for the practical picture.\n\n"
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
    "chance. With thousands of daily, autocorrelated rows the test is over-powered, so p "
    "is almost always tiny; judge the relationship by the size of the partial r, not by "
    "significance."
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
    "- **Typical value** - usually the average; if the data is skewed or has outliers the "
    "test switches to the median, a more robust middle value.\n"
    "- **Which test is used (picked automatically):**\n"
    "    - It first checks whether each regime's values are roughly normally distributed and whether "
    "the regimes have a similar spread (variance).\n"
    "    - If yes -> a **t-test** (two regimes) or **ANOVA** (three or more).\n"
    "    - If no -> a rank-based **Mann-Whitney** (two regimes) or **Kruskal-Wallis** (three "
    "or more), which need no such assumptions.\n"
    "- **p-value** - the chance of seeing a difference this big if the regimes were truly "
    "identical. **p-value** less than α means the difference is unlikely to be sole luck.\n"
    "- **Reality check:** with thousands of daily observations the test gets over-sensitive "
    "and flags even trivial gaps. Always compare against the box/violin - if the boxes "
    "overlap a lot, the difference is small in practice even when the p-value is tiny."
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
    "- **Caveat:** with thousands of daily rows the p-value is almost always tiny, so judge "
    "by the strength (Cramér's V), not significance alone."
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
    "- **K-Means** - splits the days into k groups so each day sits with the days most "
    "similar to it; the result is a *data-driven* regime, found without the rule-based labels.\n"
    "- **Levels, not returns** - structure is read on standardised levels, so a cluster is a "
    "persistent state of the economy; neighbouring days sharing a state is expected.\n"
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
    "why differencing is not used here: on forward-filled daily data it collapses to mostly "
    "zeros, whereas cointegration works directly on the levels."
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
