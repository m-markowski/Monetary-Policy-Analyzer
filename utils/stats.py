import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors, normal_ad
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
    std = values.std(ddof=1) if values.size > 1 else float("nan")
    has_variation = np.ptp(values) > 0
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
            "skew": stats.skew(values) if has_variation else float("nan"),
            "excess_kurtosis": stats.kurtosis(values) if has_variation else float("nan"),
        }
    )


def normality_battery(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame | None:
    """Run four normality diagnostics; failure to reject is not proof of normality.

    Anderson-Darling uses statsmodels.normal_ad for an unknown mean and variance.
    Its approximate p-value is not clipped to SciPy's interpolation interval.
    Shapiro-Wilk is skipped above 5000 observations because its p-value may be
    inaccurate there. All test readings remain conditional on their assumptions;
    none of these routines corrects for serial dependence in the supplied series.

    Args:
        series (pd.Series): Numeric observations; missing values are dropped.
        alpha (float): Significance level in (0, 1); reject when p < alpha.

    Returns:
        pd.DataFrame | None: Columns test, statistic, p_value and normal, where
        normal means failure to reject. None for fewer than eight usable values,
        non-finite inputs or a constant series.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    values = series.dropna().to_numpy()
    if values.size < 8 or not np.isfinite(values).all() or np.ptp(values) == 0:
        return None

    jb = stats.jarque_bera(values)
    ll_stat, ll_p = lilliefors(values, dist="norm")
    ad_stat, ad_p = normal_ad(values)

    rows = [
        ("Jarque-Bera", jb.statistic, jb.pvalue, jb.pvalue >= alpha),
        ("Lilliefors", ll_stat, ll_p, ll_p >= alpha),
        ("Anderson-Darling", ad_stat, ad_p, ad_p >= alpha),
    ]
    if values.size <= 5000:
        sw = stats.shapiro(values)
        rows.append(("Shapiro-Wilk", sw.statistic, sw.pvalue, sw.pvalue >= alpha))

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
    min_count: int = 3,
) -> dict | None:
    """
    Compare a variable across groups, automatically choosing the test.

    Normality and equal variance are checked first. When these assumptions hold,
    a t-test or ANOVA is used; otherwise Mann-Whitney or Kruskal-Wallis is used.
    For ANOVA with three or more groups, Tukey HSD provides pairwise comparisons.

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
    if len(arrays) < 2 or any(a.size < 3 or not np.isfinite(a).all() or np.ptp(a) == 0 for a in arrays):
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
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        return None

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
    """Screen for I(1)-compatible series before an exploratory Engle-Granger test.

    ADF rejection/non-rejection is evidence, not proof of an integration order.
    The cointegration test is omitted for constant, level-stationary or unresolved
    inputs. None indicates insufficient or numerically unusable data.
    """
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 20 or (pair.nunique() < 2).any():
        return None
    xv, yv = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
    try:
        px = float(adfuller(xv, maxlag=max_lag, autolag="aic")[1])
        py = float(adfuller(yv, maxlag=max_lag, autolag="aic")[1])
        dx, dy = np.diff(xv), np.diff(yv)
        pdx = float(adfuller(dx, maxlag=max_lag, autolag="aic")[1]) if np.ptp(dx) else float("nan")
        pdy = float(adfuller(dy, maxlag=max_lag, autolag="aic")[1]) if np.ptp(dy) else float("nan")
        eligible = px >= alpha and py >= alpha and pdx < alpha and pdy < alpha
        statistic = p_value = float("nan")
        if eligible:
            statistic, p_value, _ = coint(xv, yv, maxlag=max_lag, autolag="aic")
    except (ValueError, np.linalg.LinAlgError):
        return None
    return {
        "n": len(pair),
        "adf_p_x": px,
        "adf_p_y": py,
        "x_stationary": px < alpha,
        "y_stationary": py < alpha,
        "adf_diff_p_x": pdx,
        "adf_diff_p_y": pdy,
        "coint_tested": eligible,
        "coint_stat": float(statistic),
        "coint_p": float(p_value),
        "cointegrated": bool(p_value < alpha) if eligible else None,
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
    with np.errstate(divide="ignore"):  # exact collinearity legitimately yields inf
        vifs = [variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    return pd.Series(vifs, index=numeric.columns, name="VIF").sort_values(ascending=False)


def partial_correlation(
    df: pd.DataFrame, x: str, y: str, covar: str | list[str], method: str = "pearson"
) -> dict | None:
    """
    Correlation between two variables while controlling for covariates.

    Both variables are residualised on the controls by ordinary least squares,
    and their residuals are then correlated. The Spearman variant rank-transforms
    the data first. The two-sided p-value uses a Student-t approximation with
    degrees of freedom based on the effective number of independent controls.
    Serial dependence is not accounted for.

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
    dof = len(data) - np.linalg.matrix_rank(controls) - 1
    if dof < 1:
        return None
    res_x = data[x].to_numpy() - controls @ np.linalg.lstsq(controls, data[x], rcond=None)[0]
    res_y = data[y].to_numpy() - controls @ np.linalg.lstsq(controls, data[y], rcond=None)[0]
    if np.allclose(res_x, 0) or np.allclose(res_y, 0):
        return None
    r = float(np.corrcoef(res_x, res_y)[0, 1])
    if not np.isfinite(r):
        return None

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

    chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
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
