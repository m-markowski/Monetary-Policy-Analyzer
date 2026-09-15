# Case study: Forecasting the Federal Funds Rate

## Question

**Can macro-financial features improve on a no-change baseline when forecasting the Federal Funds Rate?**

> **Result:** The Dev-selected SVR did not outperform the no-change baseline on the final Test period.

This case study examines one forecasting setup in Monetary Policy Analyzer:
predicting the three-month forward change in the Effective Federal Funds Rate.

The objective is not to find the best result across many horizons or configurations,
but to compare a model selected on the Dev period with a simple and economically
meaningful benchmark: assuming that the policy rate does not change.

## Experimental setup

| Setting | Value |
| --- | --- |
| Economy | United States |
| Task | Regression |
| Target | Effective Federal Funds Rate (`rate_ff_eff`) |
| Forecast horizon | 3 months |
| Target definition | Forward change over the forecast horizon |
| Modelling frequency | Monthly |
| Split | 70% Train / 15% Dev / 15% Test |
| Selection metric | RMSE |
| Training budget | Balanced |
| Neural networks | Disabled |
| Baseline | No change |

*Blend and Stack are excluded from winner selection because their combination
layers are fitted on the Dev set, making their Dev scores in-sample for that layer.*

Models are trained using chronological data rather than shuffled observations.
Expanding-window cross-validation is used during training, with a gap equal to
the forecast horizon. Feature selection and scaling are fitted inside the training
folds.

The final model is selected using Dev performance. The Test period is then used
to evaluate whether this selection generalises to unseen data.

The dataset was refreshed in September 2026 and covered approximately 1991-2026.

## Results

| Model | Dev RMSE | Test RMSE | Test skill vs no-change |
| --- | ---: | ---: | ---: |
| **No change baseline** | 0.412 | **0.570** | **0.000** |
| **SVR** | **0.390** | 0.619 | -0.177 |
| Stack | 0.381 | 0.597 | -0.095 |
| Decision tree | 0.478 | **0.543** | 0.095 |
| XGBoost | 0.412 | 0.650 | -0.298 |

<img width="1462" height="497" alt="leaderboard_case" src="https://github.com/user-attachments/assets/d67e7cea-724b-40a2-a94f-d06ebb65d509" />

SVR was selected among the base models because it achieved the lowest Dev RMSE.
Its Dev RMSE of 0.390 was lower than the no-change benchmark of 0.412.

This advantage did not carry over to the Test period. SVR produced a Test RMSE
of 0.619 compared with 0.570 for the no-change baseline, resulting in a negative
Test skill of -0.177.

The experiment therefore provides no evidence that the Dev-selected model improved
on the no-change benchmark on the final holdout period.

A decision tree achieved a lower Test RMSE of 0.543, but it had performed materially
worse on Dev and therefore was not selected. Choosing it after observing Test results
would effectively use the holdout period for model selection, so its Test result is
reported as an observation rather than treated as the main result.

## Error analysis

<img width="1462" height="677" alt="residuals_case" src="https://github.com/user-attachments/assets/bd60593c-8bf2-4f4b-8ae5-b146129afb5d" />

The largest errors are concentrated around the rapid Federal Reserve tightening cycle.
During 2022 and early 2023, residuals are strongly positive, meaning that the realised
policy rate was substantially higher than the model predicted.

The selected model therefore reacted too slowly to the speed and magnitude of the
rate increases. Errors became considerably smaller once the policy rate stabilised,
while some negative residuals later indicate periods in which the model predicted
a higher rate than was subsequently observed.

Residuals are strongly correlated across neighbouring months. This should be
interpreted cautiously because three-month forecasts overlap, so some dependence
between adjacent forecast errors is mechanically expected.

Overall, the error pattern suggests that the main challenge was not reproducing
stable policy-rate periods, but anticipating rapid changes in the policy regime.

## Model interpretation

<img width="732" height="587" alt="feature_importance_case" src="https://github.com/user-attachments/assets/4ece4d89-e820-4ea5-9328-18862579fbe4" />

Permutation importance for the selected SVR model was calculated on the Dev period.
The strongest feature was the one-year change in median house prices, followed by
changes in industrial production, nonfarm employment, real GDP and initial
unemployment claims.

The ranking suggests that the fitted model relied primarily on information describing
the real economy, labour market and housing conditions when producing its forecasts.

These importance values are predictive rather than causal. They indicate how much
the fitted model depends on individual inputs for predictive performance and should
not be interpreted as evidence that a given variable causes Federal Reserve policy
changes.

## Forward forecast and scenario analysis

The trained model can also produce a forward-looking forecast from the latest complete
observation. In this run, the September 2026 Effective Federal Funds Rate was
**3.630%** and the selected SVR predicts a three-month change of **-0.219 percentage
points**, implying a **December 2026 rate of approximately 3.411%**.

<img width="1467" height="627" alt="scenario_regression" src="https://github.com/user-attachments/assets/e1b5e539-4ceb-425d-a638-8f15c92888d9" />

The **Baseline level** shown above is the SVR forecast with all displayed drivers held
at their September values. It is different from the historical **no-change benchmark**,
which would simply carry the 3.630% rate forward.

Scenario controls allow selected inputs to be changed while the remaining features stay
fixed. The resulting Scenario level therefore shows the model's sensitivity to alternative
input assumptions. In the screenshot, no inputs have been changed, so the scenario and
baseline forecasts are both **3.411%**.

This is a **what-if sensitivity tool, not a causal or macroeconomic simulation**. Scenario
responses show how the fitted model reacts to changed inputs; they do not imply that those
changes would cause Federal Reserve policy to move accordingly.

## Limitations

The experiment uses currently available historical data rather than a complete
point-in-time information set. Macroeconomic series may contain later revisions,
and observation dates do not always correspond to the dates on which the information
became available to market participants.

The experiment covers one economy, one target, one forecast horizon and one
Train/Dev/Test configuration. Its result should therefore not be generalised to
other horizons, targets or monetary-policy environments.

The analysis is predictive rather than causal.

## Takeaway

SVR outperformed the no-change benchmark on Dev but failed to maintain that advantage
on Test. The simple no-change forecast therefore remained the stronger out-of-sample
benchmark for this three-month task.

The largest errors occurred during the rapid tightening cycle, illustrating the difficulty
of anticipating abrupt policy changes from historical macro-financial relationships.

The model can still support forward-looking research through point forecasts and scenario
analysis, but its negative Test skill means these outputs should be treated as model-based
research signals rather than evidence of reliable future policy prediction.
