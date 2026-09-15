# Case study: Forecasting the Federal Funds Rate and Its Volatility

## Question

**Can univariate time-series models improve on a no-change rate forecast and provide
a complementary view of policy-rate volatility?**

> **Result:** The automatically selected SARIMA outperformed the no-change baseline
> on a 12-month holdout, reducing mean squared error by approximately 40%. GARCH
> provided a separate volatility projection, evaluated against a rough volatility
> proxy rather than a competing benchmark.

This case study examines the Forecast (time series) module in Monetary Policy Analyzer.
ARIMA/SARIMA forecasts the level of the Effective Federal Funds Rate, while GARCH
forecasts the conditional volatility of its monthly changes.

## Experimental setup

| Setting | Value |
| --- | --- |
| Economy | United States |
| Series | Effective Federal Funds Rate (`rate_ff_eff`) |
| Modelling frequency | Monthly |
| Available history | Approximately 1991-August 2026 |
| Forward forecast horizon | 12 months |
| Holdout | September 2025-August 2026 |
| Order selection | Lowest AIC on pre-holdout history |
| Selected rate model | SARIMA(3, 1, 2) x (1, 0, 1, 12) |
| Selected volatility model | GARCH(2, 1) |
| Rate forecast baseline | No change |

The dataset was refreshed in September 2026. The incomplete current month is excluded
from these models, so the last observation is August 2026.

Unlike the regression and classification experiments, this module does not use the
70/15/15 split. Model orders are selected using history through August 2025. Each
selected specification is fitted on that history and forecasts the next 12 months
from a single origin, without updating on observations inside the holdout.

For the forward charts, the same specifications are then refitted on all complete
history. The displayed future forecasts are therefore separate from the holdout evaluation.

## Time-series diagnostics

[//]: # (ts2 placeholder)

The rate-level ACF declines slowly, while the PACF is dominated by its first lag.
This indicates strong persistence. The displayed Augmented Dickey-Fuller test does
not reject the unit-root hypothesis (p = 0.24091); this is not proof of a unit root.

These full-history diagnostics are descriptive. The automatic search selects
its differencing order separately on the pre-holdout history and chooses a model
with one non-seasonal difference (d = 1).

## SARIMA results

| Model | Holdout RMSE | Holdout MAE | Holdout skill vs no-change |
| --- | ---: | ---: | ---: |
| No-change baseline | 0.634 | 0.618 | 0.000 |
| **Selected SARIMA** | **0.491** | **0.474** | **Approximately +0.40** |

[//]: # (ts3 placeholder)

The no-change baseline carries the final pre-holdout rate forward for all 12 months.
SARIMA produced lower RMSE and MAE on this window; both errors are measured in
percentage points.

Skill is calculated as `1 - MSE(model) / MSE(no change)`. The reported value of
approximately **+0.40** therefore means about **40% lower mean squared error**,
not 40% lower RMSE.

The selected specification had a pre-holdout AIC of **356.34** and BIC of **388.57**.
The separate headline values of 355.0 and 387.4 refer to the full-history refit,
not the fit used for order selection.

Residual dependence remains: the Ljung-Box p-value is **0.01196** for the pre-holdout
fit and **0.01058** after refitting on the full history. The model therefore improves
on the benchmark in this window without fully capturing the series' time dependence.

## Forward rate forecast

[//]: # (ts5 placeholder)

After refitting through August 2026, SARIMA projects the rate over the following
12 months. The forecast mean suggests a modest decline from around 3.6%, followed
by a broadly flat path near **3.4-3.5%**.

The **95% prediction interval** widens substantially. The relatively stable central path
therefore does not imply a precise forecast.

## GARCH volatility analysis

The automatic search selected **GARCH(2, 1)** with a pre-holdout AIC of **-162.79**.
It models monthly rate changes rather than the rate level or SARIMA residuals.

[//]: # (ts7 placeholder)

The historical fitted volatility is higher during clusters of large rate changes
and lower during stable periods. This is an in-sample description, not evidence
that the model predicted those episodes in advance.

Estimated persistence is displayed as **1.00 after rounding**, indicating highly
persistent volatility dynamics in the fitted model.

The 12-month holdout check reports **RMSE of 0.111** and **MAE of 0.075**. It compares
forecast volatility with absolute demeaned monthly changes, a rough proxy rather
than directly observed volatility. No volatility benchmark is reported, so these
errors alone do not demonstrate superior forecasting performance.

[//]: # (ts8 placeholder)

After refitting, the forward volatility projection starts near **0.02 percentage
points** and gradually rises towards **0.04-0.05 percentage points** over the horizon.
These approximate values are conditional standard deviations of monthly changes,
not predicted rate levels or signals of a hike or cut.

## Limitations

The experiment uses currently available historical data rather than a reconstructed
point-in-time dataset. Excluding the incomplete current month does not remove all
potential differences between historical observations and information available in real time.

Evaluation covers a single 12-month holdout and one series. The positive SARIMA result
should not be generalised to other periods or policy regimes, particularly given the
remaining residual autocorrelation.

The GARCH check uses a noisy volatility proxy and lacks a volatility benchmark.
Its errors cannot be compared directly with SARIMA's rate-level errors.

These results are also not directly comparable with the earlier three-month machine-learning
experiments, which use different targets, forecast origins and evaluation windows.

The analysis is predictive rather than causal.

## Takeaway

SARIMA outperformed the no-change rate forecast on this 12-month holdout, with RMSE
of **0.491 versus 0.634** and approximately **40% lower mean squared error**.
Remaining residual dependence and wide forward intervals nevertheless limit the
strength of the conclusion.

GARCH adds a complementary view of the scale and persistence of monthly rate
fluctuations, but its proxy-based evaluation does not establish an advantage over
a volatility benchmark.
