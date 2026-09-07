import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf


def stationarity(series: pd.Series, regression: str = "c") -> dict | None:
    """
    Augmented Dickey-Fuller stationarity test for one series.

    Tested first so the ARIMA differencing choice is honest.

    Args:
        series (pd.Series): Series to test.
        regression (str): ADF deterministic term ('c', 'ct', 'n').

    Returns:
        dict | None: 'adf', 'p_value', 'stationary' (p < 0.05) and 'n', or None if
        fewer than 12 observations remain.
    """
    values = pd.Series(series).dropna()
    if len(values) < 12:
        return None
    result = adfuller(values, regression=regression)
    return {
        "adf": float(result[0]),
        "p_value": float(result[1]),
        "stationary": bool(result[1] < 0.05),
        "n": int(len(values)),
    }


def acf_pacf(series: pd.Series, nlags: int = 24, alpha: float = 0.05) -> dict | None:
    """
    ACF and PACF values (lags 1+) with a fixed white-noise significance bound.

    Lag 0 is dropped (it equals 1 by definition) and the bound is the constant
    +-z/sqrt(n) white-noise band used for order identification.

    Args:
        series (pd.Series): Series to analyse.
        nlags (int): Requested number of lags (capped at n/2 - 1).
        alpha (float): Significance level for the bound (0.05 -> 95%).

    Returns:
        dict | None: 'lags' (1..nlags), 'acf', 'pacf' and the scalar 'conf', or
        None if the series is too short for even one lag.
    """
    values = pd.Series(series).dropna()
    nlags = min(nlags, len(values) // 2 - 1)
    if nlags < 1:
        return None
    acf_vals = acf(values, nlags=nlags, fft=True)
    pacf_vals = pacf(values, nlags=nlags)
    conf = float(norm.ppf(1 - alpha / 2) / np.sqrt(len(values)))
    return {"lags": np.arange(1, nlags + 1), "acf": acf_vals[1:], "pacf": pacf_vals[1:], "conf": conf}


def fit_arima(series: pd.Series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
    """
    Fit an ARIMA / SARIMA model to one series.

    A non-zero seasonal_order turns this into SARIMA; the same entry point serves
    both so the page exposes one order picker plus an optional seasonal block.

    Args:
        series (pd.Series): Series to model (month-end-indexed for the monthly view).
        order (tuple): (p, d, q) non-seasonal order.
        seasonal_order (tuple): (P, D, Q, s) seasonal order (s=0 disables it).

    Returns:
        The fitted statsmodels result, or None if the fit fails or n < 20.
    """
    values = pd.Series(series).dropna()
    if len(values) < 20:
        return None
    try:
        with warnings.catch_warnings():
            # The order search fits many deliberately bad candidates; their expected
            # non-stationary/non-invertible start-parameter and MLE convergence
            # warnings would flood the terminal without changing any result.
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive")
            warnings.filterwarnings("ignore", message="Non-invertible starting MA")
            return ARIMA(values, order=order, seasonal_order=seasonal_order).fit()
    except (ValueError, np.linalg.LinAlgError):
        return None


def arima_diagnostics(result, lb_lags: int = 12) -> dict:
    """
    Extract information criteria and a Ljung-Box residual autocorrelation check.

    Args:
        result: Fitted ARIMA result from `fit_arima`.
        lb_lags (int): Lag at which to report the Ljung-Box statistic.

    Returns:
        dict: 'aic', 'bic', 'ljung_box_stat', 'ljung_box_p' (high p = residuals look
        like white noise) and the residual series.
    """
    resid = pd.Series(result.resid).dropna()
    lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
    return {
        "aic": float(result.aic),
        "bic": float(result.bic),
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
        "resid": resid,
    }


def arima_forecast(result, steps: int = 12, alpha: float = 0.05) -> dict:
    """
    Forecast an ARIMA model with a widening confidence interval.

    Args:
        result: Fitted ARIMA result from `fit_arima`.
        steps (int): Forecast horizon in rows (months on the monthly frame).
        alpha (float): Confidence level for the interval (0.05 -> 95%).

    Returns:
        dict: 'mean', 'lower', 'upper' (each a pd.Series over the forecast index).
    """
    forecast = result.get_forecast(steps=steps)
    ci = forecast.conf_int(alpha=alpha)
    return {
        "mean": forecast.predicted_mean,
        "lower": ci.iloc[:, 0],
        "upper": ci.iloc[:, 1],
    }


def fit_garch(series: pd.Series, p: int = 1, q: int = 1, dist: str = "t"):
    """
    Fit a GARCH(p, q) volatility model to a return/change series.

    Meant for a series where variance clusters (returns, first differences), not the
    level itself.

    Args:
        series (pd.Series): Return or change series (roughly zero-mean).
        p (int): ARCH lag order (squared residuals).
        q (int): GARCH lag order (variance).
        dist (str): Innovation distribution ('t' captures fat tails; 'normal' also ok).

    Returns:
        The fitted arch result, or None if the fit fails or n < 50.
    """
    values = pd.Series(series).dropna()
    if len(values) < 50:
        return None
    try:
        model = arch_model(values, mean="Constant", vol="GARCH", p=p, q=q, dist=dist, rescale=True)
        return model.fit(disp="off")
    except (ValueError, np.linalg.LinAlgError):
        return None


def garch_forecast(result, steps: int = 12) -> dict:
    """
    Forecast conditional volatility from a fitted GARCH model.

    Args:
        result: Fitted arch result from `fit_garch`.
        steps (int): Forecast horizon.

    Returns:
        dict: 'horizon' (1..steps), 'volatility' (forecast conditional std dev),
        'fitted_volatility' (in-sample conditional std dev for context) and 'persistence' (the α+β sum, which sets how
        fast the forecast reverts to the long-run volatility).
    """
    scale = getattr(result, "scale", 1.0)
    forecast = result.forecast(horizon=steps, reindex=False)
    variance = forecast.variance.iloc[-1].to_numpy()
    persistence = float(sum(v for k, v in result.params.items() if k.startswith(("alpha[", "beta["))))
    return {
        "horizon": np.arange(1, steps + 1),
        "volatility": np.sqrt(variance) / scale,
        "fitted_volatility": pd.Series(result.conditional_volatility / scale, index=result.resid.index),
        "persistence": persistence,
    }


def arima_candidate(values: pd.Series, order: tuple, seasonal_order: tuple) -> dict | None:
    """
    Fit one ARIMA/SARIMA candidate and summarise it for the order-search leaderboard.

    Args:
        values (pd.Series): Clean series the search runs on.
        order (tuple): (p, d, q) non-seasonal order.
        seasonal_order (tuple): (P, D, Q, s) seasonal order (s=0 disables it).

    Returns:
        dict | None: 'Order', 'Seasonal', 'AIC', 'BIC' and 'Ljung-Box p', or None
        if the fit fails.
    """
    res = fit_arima(values, order=order, seasonal_order=seasonal_order)
    if res is None:
        return None
    resid = pd.Series(res.resid).dropna()
    try:
        lb_p = float(acorr_ljungbox(resid, lags=[12], return_df=True)["lb_pvalue"].iloc[0])
    except (ValueError, IndexError):
        lb_p = float("nan")
    return {
        "Order": order,
        "Seasonal": seasonal_order,
        "AIC": float(res.aic),
        "BIC": float(res.bic),
        "Ljung-Box p": lb_p,
    }


def arima_order_search(
    series: pd.Series,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    ic: str = "aic",
    top: int = 8,
    seasonal_period: int = 12,
) -> dict | None:
    """
    Small Box-Jenkins grid search over ARIMA orders, with SARIMA candidates.

    Stage one fits every non-seasonal (p, d, q) on the grid; stage two adds a small
    set of seasonal orders on top of the three best non-seasonal candidates, so
    SARIMA competes in the same leaderboard without an exhaustive (and slow) joint
    grid. Differencing is left to the grid rather than forced, so a stationary
    series can still select d = 0.

    Args:
        series (pd.Series): Series to model (month-end-indexed for the monthly view).
        max_p (int): Largest AR order to try.
        max_d (int): Largest differencing order to try.
        max_q (int): Largest MA order to try.
        ic (str): Ranking criterion ('aic' or 'bic').
        top (int): Number of candidate orders to keep in the leaderboard.
        seasonal_period (int): Seasonal period for the stage-two candidates.

    Returns:
        dict | None: 'best' (p, d, q), 'best_seasonal' (P, D, Q, s), 'ic' and a
        'leaderboard' DataFrame (Order/Seasonal/AIC/BIC/Ljung-Box p, best first),
        or None if n < 20 or nothing fits.
    """
    values = pd.Series(series).dropna()
    if len(values) < 20:
        return None
    rows = []
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                row = arima_candidate(values, (p, d, q), (0, 0, 0, 0))
                if row is not None:
                    rows.append(row)
    if not rows:
        return None
    plain = pd.DataFrame(rows).sort_values(ic.upper())
    for order in plain["Order"].head(3):
        for P, D, Q in ((1, 0, 0), (0, 0, 1), (1, 0, 1)):
            row = arima_candidate(values, tuple(order), (P, D, Q, seasonal_period))
            if row is not None:
                rows.append(row)
    leaderboard = pd.DataFrame(rows).sort_values(ic.upper()).reset_index(drop=True)
    return {
        "best": leaderboard.loc[0, "Order"],
        "best_seasonal": leaderboard.loc[0, "Seasonal"],
        "ic": ic,
        "leaderboard": leaderboard.head(top),
    }


def garch_order_search(
    series: pd.Series, max_p: int = 2, max_q: int = 2, dist: str = "t", ic: str = "aic", top: int = 6
) -> dict | None:
    """
    Small grid search over GARCH(p, q) orders on a return/change series.

    Args:
        series (pd.Series): Return or change series (roughly zero-mean).
        max_p (int): Largest ARCH (squared-residual) lag to try.
        max_q (int): Largest GARCH (variance) lag to try.
        dist (str): Innovation distribution passed to `arch_model`.
        ic (str): Ranking criterion ('aic' or 'bic').
        top (int): Number of candidate orders to keep in the leaderboard.

    Returns:
        dict | None: 'best' (p, q), 'ic' and a 'leaderboard' DataFrame (Order/AIC/BIC,
        best first), or None if n < 50 or nothing fits.
    """
    values = pd.Series(series).dropna()
    if len(values) < 50:
        return None
    rows = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                res = arch_model(values, mean="Constant", vol="GARCH", p=p, q=q, dist=dist, rescale=True).fit(
                    disp="off"
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            rows.append({"Order": (p, q), "AIC": float(res.aic), "BIC": float(res.bic)})
    if not rows:
        return None
    leaderboard = pd.DataFrame(rows).sort_values(ic.upper()).reset_index(drop=True)
    return {"best": leaderboard.loc[0, "Order"], "ic": ic, "leaderboard": leaderboard.head(top)}


def arima_backtest(series: pd.Series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), holdout: int = 12) -> dict | None:
    """
    Holdout accuracy check for one ARIMA specification.

    The caller chooses the order on the pre-holdout window; this fits that specification
    without the last holdout observations, forecasts them and reports the errors, so neither
    the order nor the fit saw the scored months.

    Args:
        series (pd.Series): Series to model (month-end-indexed for the monthly view).
        order (tuple): (p, d, q) non-seasonal order.
        seasonal_order (tuple): (P, D, Q, s) seasonal order (s=0 disables it).
        holdout (int): Number of trailing observations held out and forecast.

    Returns:
        dict | None: 'holdout', 'rmse', 'mae' (holdout forecast errors) and
        'train_rmse', 'train_mae' (in-sample one-step errors of the training fit),
        or None if the training window falls under 20 rows or the fit fails.
    """
    values = pd.Series(series).dropna()
    if len(values) - holdout < 20:
        return None
    train, test = values.iloc[:-holdout], values.iloc[-holdout:]
    res = fit_arima(train, order=order, seasonal_order=seasonal_order)
    if res is None:
        return None
    errors = test.to_numpy() - np.asarray(res.get_forecast(steps=holdout).predicted_mean)
    resid = np.asarray(res.resid)
    return {
        "holdout": int(holdout),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "train_rmse": float(np.sqrt(np.mean(resid**2))),
        "train_mae": float(np.mean(np.abs(resid))),
    }


def garch_backtest(series: pd.Series, p: int = 1, q: int = 1, dist: str = "t", holdout: int = 12) -> dict | None:
    """
    Holdout accuracy check for one GARCH specification.

    Refits on all but the last `holdout` changes, forecasts their volatility and
    compares it to the realized absolute demeaned changes - a noisy but standard
    proxy, since true volatility is never directly observed.

    Args:
        series (pd.Series): Return or change series (roughly zero-mean).
        p (int): ARCH lag order (squared residuals).
        q (int): GARCH lag order (variance).
        dist (str): Innovation distribution passed to `arch_model`.
        holdout (int): Number of trailing observations held out and forecast.

    Returns:
        dict | None: 'holdout', 'rmse' and 'mae' in the modelled series' units, or
        None if the training window falls under 50 rows or the fit fails.
    """
    values = pd.Series(series).dropna()
    if len(values) - holdout < 50:
        return None
    train, test = values.iloc[:-holdout], values.iloc[-holdout:]
    try:
        res = arch_model(train, mean="Constant", vol="GARCH", p=p, q=q, dist=dist, rescale=True).fit(disp="off")
    except (ValueError, np.linalg.LinAlgError):
        return None
    scale = getattr(res, "scale", 1.0)
    volatility = np.sqrt(res.forecast(horizon=holdout, reindex=False).variance.iloc[-1].to_numpy()) / scale
    realized = np.abs(test.to_numpy() - float(res.params["mu"]) / scale)
    errors = realized - volatility
    return {
        "holdout": int(holdout),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def top_correlated_features(X: pd.DataFrame, target: pd.Series, k: int) -> list[str]:
    """
    Pick the k features most linearly correlated with the target.

    Keeps the statsmodels econometric baseline readable and well-conditioned when
    the full feature matrix has nearly as many columns as rows.

    Args:
        X (pd.DataFrame): Feature matrix.
        target (pd.Series): Numeric target aligned to X's index.
        k (int): Number of features to keep.

    Returns:
        list[str]: Column names, most correlated first.
    """
    corr = X.corrwith(target).abs().dropna().sort_values(ascending=False)
    return list(corr.head(k).index)


def ols_baseline(X: pd.DataFrame, y: pd.Series, max_features: int = 10) -> dict | None:
    """
    Ordinary-least-squares baseline with the classical assumption diagnostics.

    Fits statsmodels OLS on the most target-correlated features (standardised, with
    an intercept) as an interpretable reference for the machine-learning models. The
    feature count is capped so the fit stays well-conditioned on the monthly frame.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Numeric regression target aligned to X's index.
        max_features (int): Cap on the number of features entered.

    Returns:
        dict | None: Fit summary ('rsquared', 'rsquared_adj', 'fvalue', 'f_pvalue',
        'aic', 'bic', 'nobs'), a 'coefficients' DataFrame on the standardised inputs,
        residual diagnostics ('durbin_watson', 'jarque_bera_p', 'condition_number')
        and Cook's distance ('cooks', 'cooks_threshold', 'influential'), or None if
        fewer than 20 aligned rows remain.
    """
    data = X.join(y.rename("__y__"), how="inner").dropna()
    if data.shape[0] < 20:
        return None
    target = data.pop("__y__")
    cols = top_correlated_features(data, target, max_features)
    if not cols:
        return None
    standardised = (data[cols] - data[cols].mean()) / data[cols].std(ddof=0)
    standardised = standardised.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if standardised.empty:
        return None
    design = sm.add_constant(standardised)
    try:
        res = sm.OLS(target.loc[design.index], design).fit()
    except (ValueError, np.linalg.LinAlgError):
        return None
    coefficients = pd.DataFrame(
        {"Coefficient": res.params, "Std error": res.bse, "t": res.tvalues, "p-value": res.pvalues}
    )
    cooks = pd.Series(res.get_influence().cooks_distance[0], index=design.index)
    threshold = 4.0 / len(cooks)
    return {
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
        "fvalue": float(res.fvalue),
        "f_pvalue": float(res.f_pvalue),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "nobs": int(res.nobs),
        "coefficients": coefficients,
        "durbin_watson": float(durbin_watson(res.resid)),
        "jarque_bera_p": float(jarque_bera(res.resid)[1]),
        "condition_number": float(res.condition_number),
        "cooks": cooks,
        "cooks_threshold": float(threshold),
        "influential": cooks[cooks > threshold].index,
    }


def logit_baseline(X: pd.DataFrame, y: pd.Series, max_features: int = 10) -> dict | None:
    """
    Multinomial-logit baseline for the direction classifier.

    Fits statsmodels MNLogit on the most target-correlated, standardised features as
    an interpretable reference: pseudo-R², information criteria, the overall
    likelihood-ratio test and per-class coefficient significance.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Categorical class labels aligned to X's index.
        max_features (int): Cap on the number of features entered.

    Returns:
        dict | None: 'prsquared', 'aic', 'bic', 'llr_pvalue', 'nobs', a
        'coefficients' DataFrame of p-values per non-baseline class, and
        'baseline_class', or None if the fit fails or fewer than 30 rows remain.
    """
    data = X.join(pd.Series(y, name="__y__"), how="inner").dropna()
    if data.shape[0] < 30:
        return None
    raw = data.pop("__y__")
    order = list(pd.Series(raw).value_counts().index)
    labels = pd.Categorical(raw, categories=order)
    codes = pd.Series(labels.codes, index=data.index)
    cols = top_correlated_features(data, codes.astype(float), max_features)
    if not cols:
        return None
    standardised = (data[cols] - data[cols].mean()) / data[cols].std(ddof=0)
    standardised = standardised.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if standardised.empty:
        return None
    design = sm.add_constant(standardised)
    try:
        res = sm.MNLogit(codes.loc[design.index], design).fit(disp=0, maxiter=200)
    except Exception:
        return None
    categories = list(labels.categories)
    coefficients = pd.DataFrame(np.asarray(res.pvalues), index=design.columns, columns=categories[1:])
    return {
        "prsquared": float(res.prsquared),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "llr_pvalue": float(res.llr_pvalue),
        "nobs": int(res.nobs),
        "coefficients": coefficients,
        "baseline_class": categories[0],
    }
