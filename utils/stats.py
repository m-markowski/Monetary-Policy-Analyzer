import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, coint


def describe_extended(series: pd.Series) -> pd.Series | None:
    """
    Summarise a numeric variable with robust descriptive statistics.

    Extends `pandas.describe` with shape measures and IQR.

    Args:
        series (pd.Series): Numeric variable to summarise.

    Returns:
        pd.Series | None: Named statistics, or None if the series is empty after
        dropping missing values.
    """
    values = series.dropna().to_numpy()
    if values.size == 0:
        return None

    mean = values.mean()
    std = values.std(ddof=1)
    q1, q3 = np.percentile(values, [25, 75])
    return pd.Series(
        {
            "count": values.size,
            "mean": mean,
            "median": np.median(values),
            "std": std,
            "min": values.min(),
            "max": values.max(),
            "range": values.max() - values.min(),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "skew": stats.skew(values),
            "excess_kurtosis": stats.kurtosis(values),
        }
    )


def normality_battery(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame | None:
    """
    Run several normality tests and report them side by side.

    Jarque-Bera, Lilliefors and Anderson-Darling always run; Shapiro-Wilk is
    skipped above 5000 points, where its p-value is unreliable. Anderson-Darling
    has no p-value, so its verdict comes from the critical value at `alpha`.

    Args:
        series (pd.Series): Numeric variable to test.
        alpha (float): Significance level for the `normal` verdict.

    Returns:
        pd.DataFrame | None: One row per test (test, statistic, p_value, normal),
        or None if fewer than eight non-missing observations exist.
    """
    values = series.dropna().to_numpy()
    if values.size < 8:
        return None

    jb = stats.jarque_bera(values)
    ll_stat, ll_p = lilliefors(values, dist="norm")
    ad = stats.anderson(values, dist="norm")
    ad_crit = ad.critical_values[int(np.argmin(np.abs(ad.significance_level - alpha * 100)))]

    rows = [
        ("Jarque-Bera", jb.statistic, jb.pvalue, jb.pvalue > alpha),
        ("Lilliefors", ll_stat, ll_p, ll_p > alpha),
        ("Anderson-Darling", ad.statistic, np.nan, ad.statistic < ad_crit),
    ]
    if values.size <= 5000:
        sw = stats.shapiro(values)
        rows.append(("Shapiro-Wilk", sw.statistic, sw.pvalue, sw.pvalue > alpha))

    return pd.DataFrame(rows, columns=["test", "statistic", "p_value", "normal"])


def mean_ci(series: pd.Series, confidence: float = 0.95) -> dict | None:
    """
    Confidence interval for the mean of a variable (Student-t based).

    Args:
        series (pd.Series): Numeric variable.
        confidence (float): Confidence level (e.g. 0.95).

    Returns:
        dict | None: Mean, lower/upper bounds, confidence level and sample size,
        or None if fewer than two non-missing observations exist.
    """
    values = series.dropna().to_numpy()
    n = values.size
    if n < 2:
        return None

    mean = values.mean()
    half_width = stats.sem(values) * stats.t.ppf((1 + confidence) / 2, n - 1)
    return {
        "mean": float(mean),
        "lower": float(mean - half_width),
        "upper": float(mean + half_width),
        "confidence": confidence,
        "n": int(n),
    }


def compare_groups(
    series: pd.Series,
    labels: pd.Series,
    alpha: float = 0.05,
    min_count: int = 2,
) -> dict | None:
    """
    Compare a variable across groups, auto-selecting the appropriate test.

    Checks the parametric preconditions (per-group normality via Shapiro/Jarque-
    Bera, equal variances via Levene), then routes to a t-test/ANOVA when they
    hold and to Mann-Whitney/Kruskal-Wallis otherwise, with a matching post-hoc
    for three or more groups.

    Args:
        series (pd.Series): Numeric variable to compare.
        labels (pd.Series): Categorical group labels, index-aligned with `series`.
        alpha (float): Significance level for preconditions and the verdict.
        min_count (int): Minimum observations for a group to be kept.

    Returns:
        dict | None: Chosen test, statistic, p-value, the preconditions, a plain-language
        `reason`, per-group sizes, an optional post-hoc table and a `differs` flag;
        None if fewer than two groups qualify.
    """
    frame = pd.DataFrame({"value": series, "group": labels}).dropna()
    counts = frame["group"].value_counts()
    frame = frame[frame["group"].isin(counts[counts >= min_count].index)]
    frame["group"] = frame["group"].astype(str)
    arrays = [g["value"].to_numpy() for _, g in frame.groupby("group", observed=True)]
    if len(arrays) < 2:
        return None

    normal = all((stats.shapiro(a).pvalue if a.size <= 5000 else stats.jarque_bera(a).pvalue) > alpha for a in arrays)
    equal_var = stats.levene(*arrays).pvalue > alpha
    parametric = normal and equal_var

    n_groups = len(arrays)
    post_hoc = None
    if n_groups == 2:
        left, right = arrays
        if parametric:
            statistic, p_value = stats.ttest_ind(left, right, equal_var=True)
            test = "Student t-test"
        else:
            statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
            test = "Mann-Whitney U"
    elif parametric:
        statistic, p_value = stats.f_oneway(*arrays)
        test = "One-way ANOVA"
        tukey = pairwise_tukeyhsd(frame["value"], frame["group"], alpha=alpha)
        post_hoc = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    else:
        statistic, p_value = stats.kruskal(*arrays)
        test = "Kruskal-Wallis"
    statistic, p_value = float(statistic), float(p_value)

    normality_note = "all groups ~normal" if normal else "non-normal group(s)"
    variance_note = "equal variances" if equal_var else "unequal variances"
    route = "parametric" if parametric else "nonparametric"
    reason = f"{n_groups} groups; {normality_note}, {variance_note} (Shapiro/JB + Levene) -> {route}: {test}."

    return {
        "test": test,
        "statistic": statistic,
        "p_value": p_value,
        "parametric": parametric,
        "normal": normal,
        "equal_var": equal_var,
        "reason": reason,
        "group_sizes": counts[counts >= min_count].astype(int).to_dict(),
        "post_hoc": post_hoc,
        "differs": bool(p_value < alpha),
    }


def correlation_matrix(
    df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson"
) -> pd.DataFrame | None:
    """
    Correlation matrix for selected numeric columns.

    Args:
        df (pd.DataFrame): Source dataset.
        columns (list[str] | None): Columns to include; None uses all numerics.
        method (str): 'pearson' or 'spearman'.

    Returns:
        pd.DataFrame | None: Square correlation matrix, or None if fewer than two
        usable numeric columns are available.
    """
    numeric = df[columns] if columns is not None else df.select_dtypes("number")
    if numeric.shape[1] < 2:
        return None
    return numeric.corr(method=method)


def stationarity_and_cointegration(
    x: pd.Series, y: pd.Series, alpha: float = 0.05, max_lag: int | None = None
) -> dict | None:
    """
    ADF stationarity of each series plus an Engle-Granger cointegration test.

    Two trending (non-stationary) series can show a strong level correlation even
    when unrelated - the spurious-regression trap. ADF tests whether each series is
    stationary in levels; the Engle-Granger test regresses one on the other and
    checks whether the residual is stationary. A stationary residual means the pair
    moves together around a stable long-run equilibrium (genuine cointegration),
    otherwise a level correlation is likely spurious.

    Args:
        x (pd.Series): First variable (levels), index-aligned with y.
        y (pd.Series): Second variable (levels).
        alpha (float): Significance level for the stationary/cointegrated verdicts.
        max_lag (int | None): Max augmentation lag; None lets statsmodels choose.

    Returns:
        dict | None: ADF p-values and stationary flags for x and y, the Engle-
        Granger statistic/p-value and a cointegrated flag; None if fewer than 20
        aligned observations remain.
    """
    pair = pd.concat([x, y], axis=1).dropna()
    if pair.shape[0] < 20:
        return None
    xv, yv = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()

    adf_x = adfuller(xv, maxlag=max_lag, autolag="aic")
    adf_y = adfuller(yv, maxlag=max_lag, autolag="aic")
    coint_stat, coint_p, _ = coint(xv, yv, maxlag=max_lag, autolag="aic")
    return {
        "n": int(pair.shape[0]),
        "adf_p_x": float(adf_x[1]),
        "adf_p_y": float(adf_y[1]),
        "x_stationary": bool(adf_x[1] < alpha),
        "y_stationary": bool(adf_y[1] < alpha),
        "coint_stat": float(coint_stat),  # statistic from ADF test that is compared against the critical value
        "coint_p": float(coint_p),  # respective p-value for the test
        "cointegrated": bool(coint_p < alpha),
    }


def variance_inflation_factors(df: pd.DataFrame, columns: list[str] | None = None) -> pd.Series | None:
    """
    Variance inflation factor per feature (multicollinearity diagnostic).

    Each feature is regressed on all the others (plus an intercept); its VIF is
    1 / (1 - R^2) of that fit. VIF rises as a feature becomes more linearly
    predictable from the rest, catching multivariate redundancy that a pairwise
    correlation misses. A common flag is VIF > 10.

    Args:
        df (pd.DataFrame): Source dataset.
        columns (list[str] | None): Columns to include; None uses all numerics.

    Returns:
        pd.Series | None: VIF per feature, sorted descending, or None if fewer than
        two usable columns or too few complete rows remain.
    """
    numeric = df[columns] if columns is not None else df.select_dtypes("number")
    numeric = numeric.dropna()
    if numeric.shape[1] < 2 or numeric.shape[0] <= numeric.shape[1] + 1:
        return None
    # np.ones  - constant column at index 0. The loop starts at 1
    design = np.column_stack([np.ones(len(numeric)), numeric.to_numpy()])
    vifs = [variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    return pd.Series(vifs, index=numeric.columns, name="VIF").sort_values(ascending=False)


def partial_correlation(
    df: pd.DataFrame, x: str, y: str, covar: str | list[str], method: str = "pearson"
) -> dict | None:
    """
    Correlation between two variables while controlling for covariates.

    Both variables are residualised on the controls (an intercept plus every
    covariate) by ordinary least squares; the correlation of the residuals is the
    partial correlation. The Spearman variant rank-transforms the columns first,
    then applies the same residualisation. The two-sided p-value comes from the
    Student-t approximation with n - 2 - k degrees of freedom (k = control count).

    Args:
        df (pd.DataFrame): Source dataset.
        x (str): First variable.
        y (str): Second variable.
        covar (str | list[str]): Covariate(s) to partial out.
        method (str): 'pearson' or 'spearman'.

    Returns:
        dict | None: n, r, p_value, dof and the control list, or None if the
        columns are missing or too few complete rows remain.
    """
    covars = [covar] if isinstance(covar, str) else list(covar)
    needed = [x, y, *covars]
    if not set(needed).issubset(df.columns):
        return None

    data = df[needed].dropna()
    dof = len(data) - 2 - len(covars)
    if dof < 1:
        return None
    if method == "spearman":
        data = data.rank()

    controls = np.column_stack([np.ones(len(data)), data[covars].to_numpy()])
    res_x = data[x].to_numpy() - controls @ np.linalg.lstsq(controls, data[x], rcond=None)[0]
    res_y = data[y].to_numpy() - controls @ np.linalg.lstsq(controls, data[y], rcond=None)[0]
    r = float(np.corrcoef(res_x, res_y)[0, 1])

    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_stat = r * np.sqrt(dof / (1 - r**2))
        p_value = float(2 * stats.t.sf(abs(t_stat), dof))
    return {"n": len(data), "r": r, "p_value": p_value, "dof": dof, "controls": covars}


def categorical_association(a: pd.Series, b: pd.Series, alpha: float = 0.05) -> dict | None:
    """
    Test association between two categorical variables.

    Runs a chi-square test of independence and reports Cramer's V as the effect
    size. A 2x2 table is the equivalent of a two-sample proportion z-test, so the
    proportion test is covered here rather than duplicated.

    Args:
        a (pd.Series): First categorical variable (e.g. recession state).
        b (pd.Series): Second categorical variable (e.g. curve state).
        alpha (float): Significance level for the `associated` verdict.

    Returns:
        dict | None: chi-square, p-value, dof, Cramer's V, the contingency table
        and an `associated` flag; None if the table is degenerate (a single row or
        column).
    """
    table = pd.crosstab(a, b)
    table.index.name = None
    if min(table.shape) < 2:
        return None

    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "n": n,
        "table": table,
        "associated": bool(p_value < alpha),
    }


def cramers_v_matrix(categoricals: dict[str, pd.Series]) -> pd.DataFrame | None:
    """
    Cramer's V association matrix for a set of categorical variables.

    Builds a square matrix by running `categorical_association` on every pair,
    with 1.0 on the diagonal. Intended for the regime labels returned by
    `available_regimes` (e.g. Policy regime x Recession x Curve state).

    Args:
        categoricals (dict[str, pd.Series]): Display name -> categorical label
            series, index-aligned with each other.

    Returns:
        pd.DataFrame | None: Square Cramer's V matrix, or None if fewer than two
        variables are supplied.
    """
    names = list(categoricals)
    if len(names) < 2:
        return None

    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        matrix.loc[a, a] = 1.0
        for b in names[i + 1 :]:
            res = categorical_association(categoricals[a], categoricals[b])
            value = res["cramers_v"] if res else np.nan
            matrix.loc[a, b] = value
            matrix.loc[b, a] = value
    return matrix
