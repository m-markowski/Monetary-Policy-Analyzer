# Case study: Forecasting the Direction of the Federal Funds Rate

## Question

**Can macro-financial features improve on simple baselines when forecasting the direction of the Federal Funds Rate?**

> **Result:** The Dev-selected logistic regression did not generalise to the final Test period and did not outperform the trailing-momentum baseline on hard-class metrics.

This case study examines one classification setup in Monetary Policy Analyzer:
predicting whether the Effective Federal Funds Rate will be Cut, Held or Hiked over
the next three months.

The objective is not to find the best result across many horizons or configurations,
but to select a model on the Dev period and compare its final Test performance with
simple directional benchmarks.

## Experimental setup

| Setting | Value |
| --- | --- |
| Economy | United States |
| Task | Direction classification |
| Target | Effective Federal Funds Rate (`rate_ff_eff`) |
| Forecast horizon | 3 months |
| Classes | Cut / Hold / Hike |
| Hold band | ±0.125 percentage points |
| Modelling frequency | Monthly |
| Split | 70% Train / 15% Dev / 15% Test |
| Selection metric | ROC-AUC (macro, OvR) |
| Training budget | Balanced |
| Neural networks | Disabled |
| Baselines | Majority class / trailing momentum |

A month is labelled Hike when the rate three months later is more than 12.5 basis
points higher, Cut when it is more than 12.5 basis points lower, and Hold otherwise.

*Blend and Stack are excluded from winner selection because their combination layers
are fitted on the Dev set, making their Dev scores in-sample for that layer.*

Models are trained using chronological data rather than shuffled observations.
Expanding-window cross-validation is used during training, with a gap equal to
the forecast horizon. Feature selection and scaling are fitted inside the training
folds.

The final model is selected using Dev performance. The Test period is then used
to evaluate whether this selection generalises to unseen data.

The dataset was refreshed in September 2026 and covered approximately 1991-2026.

## Results

| Model | Dev ROC-AUC | Test ROC-AUC | Test F1-macro |
| --- | ---: | ---: | ---: |
| **Logistic regression** | **0.743** | 0.531 | 0.227 |
| Random forest | 0.741 | 0.704 | 0.311 |
| Gradient boosting | 0.613 | 0.756 | 0.412 |
| Majority baseline | - | - | 0.227 |
| Trailing-momentum baseline | - | - | **0.690** |

<img width="1457" height="502" alt="leaderboard_case" src="https://github.com/user-attachments/assets/cf66b579-dc61-4737-bdd2-67b2dd46eb34" />

Logistic regression was selected among the base models because it achieved the highest
Dev ROC-AUC of 0.743, narrowly ahead of random forest at 0.741.

This advantage did not carry over to the Test period. Test ROC-AUC fell to **0.531**,
close to the 0.5 level associated with random ranking.

Other models achieved higher Test ROC-AUC values, but they had not been selected on Dev.
Choosing one of them after observing Test results would effectively use the holdout
period for model selection, so these results are treated as observations rather than
the main result.

The majority and trailing-momentum baselines produce hard class labels rather than
probabilities, so ROC-AUC is not available for them. On F1-macro, logistic regression
matched the majority baseline and performed substantially worse than trailing momentum.

## Classification diagnostics

Under the default argmax rule, the selected logistic regression predicted **Hold for
all 64 Test observations**.

<img width="1461" height="650" alt="case_argmax" src="https://github.com/user-attachments/assets/41a91ecf-4e14-43e8-9cc0-ea46d533699a" />

This correctly classified all 33 Hold months but missed all 12 Cut and 19 Hike months.
As a result, accuracy was approximately 52%, while balanced accuracy was only **0.333**
and F1-macro **0.227**.

The ROC curves show that the model retained some ranking ability for Cut
(AUC approximately 0.70), but performance for Hike and Hold was weak.

The application can alternatively tune class thresholds using **Youden's J statistic**
on the Dev split and apply those fixed thresholds to Test.

<img width="1460" height="656" alt="case_youden" src="https://github.com/user-attachments/assets/286efd11-4e1b-4c6a-8f23-4e0c7e7e65fd" />

With Youden tuning, the model correctly classified 12 of 12 Cut months, 11 of 19 Hike
months and 10 of 33 Hold months. Overall accuracy remained approximately 52%, but
balanced accuracy increased to approximately **0.627** and F1-macro to **0.531**.

The improvement therefore came from distributing predictions more evenly across
classes rather than increasing the total number of correct predictions. It also
introduced a trade-off: many actual Hold months were classified as Cut.

## Model interpretation

<img width="1466" height="687" alt="case_feature" src="https://github.com/user-attachments/assets/6e2852e1-17aa-42a6-866d-a267246c988a" />

The selected logistic regression provides two complementary views of feature importance.

Native coefficient importance ranks **housing starts**, **total government debt** and
the **labour-force participation rate** among the strongest inputs.

Permutation importance on Dev instead places the **six-month lag of the Federal Funds
Rate** first, followed by the **five-year Treasury yield** and **M2 money velocity**.

The rankings need not agree: coefficient magnitude describes how strongly the fitted
model uses an input, while permutation importance measures how much predictive
performance deteriorates when that information is disrupted.

These importance values are predictive rather than causal and should not be interpreted
as evidence that individual variables cause Federal Reserve policy changes.

## Forward classification and scenario analysis

The trained model can also produce a forward classification from the latest complete
observation.

For the September-to-December 2026 horizon, the selected logistic regression assigns:

| Class | Probability |
| --- | ---: |
| Cut | 23.3% |
| Hike | 21.4% |
| **Hold** | **55.3%** |

The resulting predicted stance is therefore **Hold**.

<img width="1465" height="717" alt="case_scenario_class" src="https://github.com/user-attachments/assets/71a391e8-ffc2-4a35-a68e-8e6829a0f000" />

This should be interpreted as the net direction of the policy rate over the three-month
window rather than a prediction for a specific Federal Reserve meeting.

In this example, no scenario inputs were changed, so the displayed probabilities
represent the model's baseline forecast from the latest complete feature row.

Scenario controls allow selected inputs to be changed while the remaining features
stay fixed. The resulting probabilities show how the fitted classifier responds
to alternative assumptions.

This is a **what-if sensitivity tool, not a causal or macroeconomic simulation**.
Scenario responses do not imply that changes in individual inputs would cause the
Federal Reserve to adopt the corresponding policy stance.

## Limitations

The experiment uses currently available historical data rather than a complete
point-in-time information set. Macroeconomic series may contain later revisions,
and observation dates do not always correspond to the dates on which information
became available to policymakers or market participants.

The three-class target depends on a ±12.5 basis-point threshold, so observations close
to the boundary may receive different labels despite relatively small economic
differences.

The Test period contains only 64 observations, and the three-month forecast windows
overlap across neighbouring months.

The experiment covers one economy, one horizon and one Train/Dev/Test configuration.
Its result should therefore not be generalised to other targets or monetary-policy
environments.

The analysis is predictive rather than causal.

## Takeaway

Logistic regression achieved the strongest Dev ROC-AUC among the eligible base models,
but this advantage did not generalise. Test ROC-AUC fell from **0.743 to 0.531**.

Under the default argmax rule, the model predicted Hold for every Test observation
and performed no better than the majority-class baseline on F1-macro.

Dev-tuned Youden thresholds produced a more balanced set of predictions, but the
trailing-momentum baseline remained stronger on the final Test period.

As in the regression case study, the result illustrates the difficulty of translating
patterns found in historical macro-financial data into reliable out-of-sample monetary
policy forecasts. Forward class probabilities and scenario analysis can still be useful
as research signals, but they should not be interpreted as evidence of reliable future
policy prediction.
