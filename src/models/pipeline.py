import numpy as np
import optuna
import pandas as pd
from joblib import parallel_config
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from config.settings import SEED
from src.models.evaluate import CV_SCORING

# Regression selects on RMSE, not R2: near-constant target windows (ZIRP-era CV folds,
# the COVID dev window) make fold R2 explode while RMSE stays comparable across models.
DEFAULT_SCORING = {"regression": "RMSE", "classification": "ROC-AUC (macro/OvR)"}

# Chronological split presets (train, valid); test takes the remainder so valid ~= test.
SPLIT_PRESETS = {"60/20/20": (0.60, 0.20), "70/15/15": (0.70, 0.15), "80/10/10": (0.80, 0.10)}

# Training-budget tiers: cheap random search up front, Optuna for the deep search.
BUDGET_TIERS = {
    "Fast": {"backend": "random", "n_iter": 15},
    "Balanced": {"backend": "random", "n_iter": 40},
    "Thorough": {"backend": "optuna", "n_trials": 60},
}


def to_sklearn_distributions(space: dict) -> dict:
    """
    Translate a parameter space into scipy/list distributions for RandomizedSearchCV.

    The space vocabulary is ('int', low, high), ('float', low, high[, log]) and
    ('cat', choices); keys are already pipeline-prefixed (e.g. 'model__C').

    Args:
        space (dict): Parameter name -> spec tuple.

    Returns:
        dict: Parameter name -> scipy distribution or list of choices.
    """
    dists = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            dists[name] = randint(spec[1], spec[2] + 1)
        elif kind == "float":
            log = len(spec) > 3 and spec[3]
            dists[name] = loguniform(spec[1], spec[2]) if log else uniform(spec[1], spec[2] - spec[1])
        elif kind == "cat":
            dists[name] = spec[1]
    return dists


def suggest_from_space(trial, space: dict) -> dict:
    """
    Sample one parameter set from the space using an Optuna trial.

    Args:
        trial: Optuna trial object.
        space (dict): Parameter name -> spec tuple (see `to_sklearn_distributions`).

    Returns:
        dict: Parameter name -> sampled value.
    """
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=len(spec) > 3 and spec[3])
        elif kind == "cat":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def model_roster(task: str, random_state: int = SEED, class_weight: bool = True) -> dict:
    """
    Build the per-task model roster with curated tunable hyperparameters.

    Only a handful of impactful parameters is tuned per model. `needs_scaling`
    marks the scale-sensitive models (linear and SVM).
    Class weights are applied where the estimator supports them natively; the
    remaining imbalance is handled downstream by threshold tuning.

    Args:
        task (str): 'regression' or 'classification'.
        random_state (int): Seed for the stochastic estimators.
        class_weight (bool): Apply 'balanced' weighting where supported (clf only).

    Returns:
        dict: Model name -> {'estimator', 'space', 'needs_scaling'}.
    """
    cw = "balanced" if class_weight else None
    if task == "regression":
        return {
            "Linear regression": {"estimator": LinearRegression(), "space": {}, "needs_scaling": True},
            "Ridge": {
                "estimator": Ridge(random_state=random_state),
                "space": {"model__alpha": ("float", 1e-3, 1e3, True)},
                "needs_scaling": True,
            },
            "Lasso": {
                "estimator": Lasso(random_state=random_state, max_iter=50000),
                "space": {"model__alpha": ("float", 1e-4, 1e2, True)},
                "needs_scaling": True,
            },
            "ElasticNet": {
                "estimator": ElasticNet(random_state=random_state, max_iter=50000),
                "space": {
                    "model__alpha": ("float", 1e-4, 1e2, True),
                    "model__l1_ratio": ("float", 0.05, 0.95),
                },
                "needs_scaling": True,
            },
            "SVR": {
                "estimator": TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
                "space": {
                    "model__regressor__C": ("float", 1e-1, 1e3, True),
                    "model__regressor__epsilon": ("float", 1e-2, 3e-1, True),
                    "model__regressor__gamma": ("cat", ["scale", "auto"]),
                    "model__regressor__kernel": ("cat", ["rbf", "linear"]),
                },
                "needs_scaling": True,
            },
            "Decision tree": {
                "estimator": DecisionTreeRegressor(random_state=random_state),
                "space": {
                    "model__max_depth": ("int", 2, 12),
                    "model__min_samples_leaf": ("int", 1, 20),
                },
                "needs_scaling": False,
            },
            "Random forest": {
                "estimator": RandomForestRegressor(random_state=random_state, n_jobs=1),
                "space": {
                    "model__n_estimators": ("int", 100, 600),
                    "model__max_depth": ("int", 3, 16),
                    "model__min_samples_leaf": ("int", 1, 10),
                    "model__max_features": ("cat", ["sqrt", "log2", 1.0]),
                },
                "needs_scaling": False,
            },
            "Gradient boosting": {
                "estimator": GradientBoostingRegressor(random_state=random_state),
                "space": {
                    "model__n_estimators": ("int", 100, 500),
                    "model__learning_rate": ("float", 1e-2, 3e-1, True),
                    "model__max_depth": ("int", 2, 5),
                },
                "needs_scaling": False,
            },
            "XGBoost": {
                "estimator": XGBRegressor(random_state=random_state, tree_method="hist", verbosity=0, n_jobs=1),
                "space": {
                    "model__n_estimators": ("int", 100, 600),
                    "model__max_depth": ("int", 2, 8),
                    "model__learning_rate": ("float", 1e-2, 3e-1, True),
                    "model__subsample": ("float", 0.6, 1.0),
                    "model__colsample_bytree": ("float", 0.6, 1.0),
                },
                "needs_scaling": False,
            },
            "LightGBM": {
                "estimator": LGBMRegressor(random_state=random_state, verbose=-1, n_jobs=1),
                "space": {
                    "model__n_estimators": ("int", 100, 600),
                    "model__num_leaves": ("int", 15, 63),
                    "model__learning_rate": ("float", 1e-2, 3e-1, True),
                    "model__min_child_samples": ("int", 5, 30),
                },
                "needs_scaling": False,
            },
        }

    return {
        "Logistic regression": {
            "estimator": LogisticRegression(solver="saga", max_iter=5000, class_weight=cw, random_state=random_state),
            "space": {
                "model__C": ("float", 1e-3, 1e2, True),
                "model__penalty": ("cat", ["l1", "l2"]),
            },
            "needs_scaling": True,
        },
        "SVM": {
            "estimator": SVC(probability=True, class_weight=cw, random_state=random_state),
            "space": {
                "model__C": ("float", 1e-1, 1e3, True),
                "model__gamma": ("cat", ["scale", "auto"]),
                "model__kernel": ("cat", ["rbf", "linear"]),
            },
            "needs_scaling": True,
        },
        "Decision tree": {
            "estimator": DecisionTreeClassifier(class_weight=cw, random_state=random_state),
            "space": {
                "model__max_depth": ("int", 2, 12),
                "model__min_samples_leaf": ("int", 1, 20),
            },
            "needs_scaling": False,
        },
        "Random forest": {
            "estimator": RandomForestClassifier(class_weight=cw, random_state=random_state, n_jobs=1),
            "space": {
                "model__n_estimators": ("int", 100, 600),
                "model__max_depth": ("int", 3, 16),
                "model__min_samples_leaf": ("int", 1, 10),
                "model__max_features": ("cat", ["sqrt", "log2"]),
            },
            "needs_scaling": False,
        },
        "Gradient boosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "space": {
                "model__n_estimators": ("int", 100, 500),
                "model__learning_rate": ("float", 1e-2, 3e-1, True),
                "model__max_depth": ("int", 2, 5),
            },
            "needs_scaling": False,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=random_state, tree_method="hist", verbosity=0, eval_metric="mlogloss", n_jobs=1
            ),
            "space": {
                "model__n_estimators": ("int", 100, 600),
                "model__max_depth": ("int", 2, 8),
                "model__learning_rate": ("float", 1e-2, 3e-1, True),
                "model__subsample": ("float", 0.6, 1.0),
                "model__colsample_bytree": ("float", 0.6, 1.0),
            },
            "needs_scaling": False,
        },
        "LightGBM": {
            "estimator": LGBMClassifier(class_weight=cw, random_state=random_state, verbose=-1, n_jobs=1),
            "space": {
                "model__n_estimators": ("int", 100, 600),
                "model__num_leaves": ("int", 15, 63),
                "model__learning_rate": ("float", 1e-2, 3e-1, True),
                "model__min_child_samples": ("int", 5, 30),
            },
            "needs_scaling": False,
        },
    }


def build_pipeline(
    estimator,
    needs_scaling: bool,
    task: str,
    k_features: int,
) -> Pipeline:
    """
    Wrap an estimator in a Pipeline so preprocessing is fit on train folds only.

    A univariate SelectKBest filter is prepended so feature selection is re-fit
    inside every CV fold (leakage-safe) and curbs the p >> n overfit on the monthly
    frame.

    Args:
        estimator: The final sklearn-compatible estimator.
        needs_scaling (bool): Prepend a StandardScaler for scale-sensitive models.
        task (str): 'classification' or 'regression'; picks the SelectKBest
            score function.
        k_features (int): Number of features to keep.

    Returns:
        Pipeline: The assembled pipeline with the estimator as step 'model'.
    """
    steps = []
    score_func = f_classif if task == "classification" else f_regression
    steps.append(("select", SelectKBest(score_func=score_func, k=k_features)))
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe


def auto_k_features(n_rows: int, n_features: int, n_splits: int = 5, gap: int = 0) -> int:
    """Bound feature count by the smallest planned expanding training fold.

    The five-feature floor is a heuristic, capped by the available columns.
    The gap is excluded from the fold's training rows.
    """
    if n_features < 1 or n_splits < 2 or gap < 0:
        raise ValueError("Expected features > 0, n_splits >= 2 and gap >= 0.")
    test_size = n_rows // (n_splits + 1)
    smallest_fold = n_rows - n_splits * test_size - gap
    if test_size < 1 or smallest_fold < 1:
        raise ValueError("Not enough rows for the requested folds and gap.")
    return min(n_features, max(5, smallest_fold // 4))


def chronological_split(X: pd.DataFrame, y: pd.Series, preset: str = "70/15/15", gap: int = 0) -> dict:
    """
    Split X/y chronologically into train/valid/test without shuffling.

    The target of row t is realised at t + horizon, so the last `gap` rows before
    each boundary are dropped from the earlier split: their labels would otherwise
    already contain outcomes from the period the next split is evaluated on.

    Args:
        X (pd.DataFrame): Feature matrix, already ordered by date.
        y (pd.Series): Aligned target.
        preset (str): One of SPLIT_PRESETS.
        gap (int): Rows purged before each boundary (the target horizon in months).

    Returns:
        dict: 'Train'/'Valid'/'Test' -> (X_slice, y_slice), preserving order.
    """
    if not isinstance(gap, int) or isinstance(gap, bool) or gap < 0:
        raise ValueError("gap must be a non-negative integer.")
    if not X.index.equals(y.index) or not X.index.is_monotonic_increasing or not X.index.is_unique:
        raise ValueError("X and y must have the same unique, increasing date index.")
    train_frac, valid_frac = SPLIT_PRESETS[preset]
    n = len(X)
    valid_start = int(n * train_frac)
    test_start = valid_start + int(n * valid_frac)
    bounds = {"Train": (0, valid_start - gap), "Valid": (valid_start, test_start - gap), "Test": (test_start, n)}
    if any(hi <= lo for lo, hi in bounds.values()):
        raise ValueError("The split and horizon leave an empty training, validation or test set.")
    return {name: (X.iloc[lo:hi], y.iloc[lo:hi]) for name, (lo, hi) in bounds.items()}


def search_estimator(pipe, space, X, y, scoring, cv, budget, random_state=SEED):
    """
    Tune one pipeline's hyperparameters under the chosen budget backend.

    Fast/Balanced use RandomizedSearchCV; Thorough uses Optuna (TPE). A model
    with no tunable space is simply cross-validated and refit as-is.

    Args:
        pipe (Pipeline): Pipeline to tune.
        space (dict): Parameter space (see `to_sklearn_distributions`).
        X, y: Training data (the train split only, never valid/test).
        scoring: sklearn scoring string or callable from CV_SCORING.
        cv: The shared list of usable expanding train/validation index pairs.
        budget (str): One of BUDGET_TIERS.
        random_state (int): Seed for the search.

    Returns:
        tuple: (fitted_estimator, best_params, cv_score). cv_score is sign-aligned
        so higher is always better.
    """
    if not space:
        score = float(np.mean(cross_val_score(pipe, X, y, scoring=scoring, cv=cv)))
        return clone(pipe).fit(X, y), {}, score

    cfg = BUDGET_TIERS[budget]
    if cfg["backend"] == "random":
        search = RandomizedSearchCV(
            pipe,
            to_sklearn_distributions(space),
            n_iter=cfg["n_iter"],
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=4,
            refit=True,
        )
        with parallel_config(backend="threading"):
            search.fit(X, y)
        return search.best_estimator_, search.best_params_, float(search.best_score_)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        candidate = clone(pipe).set_params(**suggest_from_space(trial, space))
        return float(np.mean(cross_val_score(candidate, X, y, scoring=scoring, cv=cv)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=cfg["n_trials"])
    best = clone(pipe).set_params(**study.best_params).fit(X, y)
    return best, study.best_params, float(study.best_value)


def usable_time_folds(X, y, task: str, scoring: str, n_splits: int = 5, gap: int = 0) -> tuple[list, int]:
    """Return expanding folds usable by the whole roster, without moving dates.

    Classification folds must train on every class present in the outer training
    data. ROC-AUC additionally requires at least two validation classes. Skipped
    folds are reported; undefined scores are never replaced with chance scores.
    """
    classes = set(np.asarray(y))
    folds = []
    for train, valid in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        if task == "classification":
            if set(np.asarray(y)[train]) != classes or len(classes) < 2:
                continue
            if scoring == "ROC-AUC (macro/OvR)" and len(np.unique(np.asarray(y)[valid])) < 2:
                continue
        folds.append((train, valid))
    if len(folds) < 2:
        raise ValueError(
            "Fewer than two usable expanding CV folds remain. Choose another horizon, "
            "split preset or metric; no random splitting or synthetic AUC is used."
        )
    return folds, n_splits - len(folds)


def train_roster(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    *,
    budget: str = "Fast",
    n_splits: int = 5,
    gap: int = 0,
    scoring: str | None = None,
    class_weight: bool = True,
    k_features: int | str = "auto",
    random_state: int = SEED,
    progress=None,
) -> dict:
    """
    Search and fit the whole roster on the training split, returning fitted models.

    For classification the target is label-encoded (so XGBoost accepts it); the
    original class names are returned for display. CV uses TimeSeriesSplit, so the
    temporal order is respected and no future information leaks across folds.

    Args:
        X (pd.DataFrame): Training-split features (do not pass valid/test here).
        y (pd.Series): Training-split target.
        task (str): 'regression' or 'classification'.
        budget (str): One of BUDGET_TIERS.
        n_splits (int): Number of TimeSeriesSplit folds.
        scoring (str | None): Display metric driving selection; defaults per task.
        gap (int): Rows skipped between each fold's training and validation blocks
            (the target horizon), so no training label is realised inside its own
            validation block.
        class_weight (bool): Pass 'balanced' weighting where supported (clf).
        k_features (int | str): 'auto' sizes the in-CV SelectKBest to the smallest CV
            fold (see `auto_k_features`), or an explicit count.
        random_state (int): Seed.
        progress (callable | None): Called as progress(done, total, name, cv_score)
            after each model, for a UI progress bar.

    Returns:
        dict: 'models' (name -> fitted estimator), 'cv_scores', 'params', 'scoring',
        'best' (top model name), 'classes' (clf only) and 'label_encoder' (clf only).
    """
    scoring = scoring or DEFAULT_SCORING[task]
    scorer = CV_SCORING[scoring]
    cv, skipped_folds = usable_time_folds(X, y, task, scoring, n_splits, gap)

    if k_features == "auto":
        k_features = auto_k_features(len(X), X.shape[1], n_splits, gap=gap)

    classes, encoder = None, None
    if task == "classification":
        encoder = LabelEncoder().fit(y)
        classes = list(encoder.classes_)
        y = pd.Series(encoder.transform(y), index=y.index)

    roster = model_roster(task, random_state=random_state, class_weight=class_weight)

    fitted, cv_scores, params = {}, {}, {}
    total = len(roster)
    for done, (name, cfg) in enumerate(roster.items(), start=1):
        pipe = build_pipeline(cfg["estimator"], cfg["needs_scaling"], task=task, k_features=k_features)
        model, best_params, cv_score = search_estimator(pipe, cfg["space"], X, y, scorer, cv, budget, random_state)
        fitted[name] = model
        cv_scores[name] = cv_score
        params[name] = best_params
        if progress:
            progress(done, total, name, cv_score)

    best = max(cv_scores, key=cv_scores.get) if cv_scores else None
    return {
        "models": fitted,
        "cv_folds": len(cv),
        "skipped_folds": skipped_folds,
        "cv_scores": cv_scores,
        "params": params,
        "scoring": scoring,
        "best": best,
        "classes": classes,
        "label_encoder": encoder,
    }
