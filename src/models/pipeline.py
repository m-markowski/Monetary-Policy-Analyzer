import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.models.evaluate import CV_SCORING
from config.settings import SEED

# Higher-is-better metrics; only these support the "stop when good enough" hook.
MAXIMISE_METRICS = ("ROC-AUC (macro/OvR)", "F1-macro", "Balanced accuracy")
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
                "estimator": SVR(),
                "space": {
                    "model__C": ("float", 1e-1, 1e3, True),
                    "model__gamma": ("cat", ["scale", "auto"]),
                    "model__kernel": ("cat", ["rbf", "linear"]),
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
                "estimator": RandomForestRegressor(random_state=random_state, n_jobs=-1),
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
                "estimator": XGBRegressor(random_state=random_state, tree_method="hist", verbosity=0),
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
                "estimator": LGBMRegressor(random_state=random_state, verbose=-1),
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
            "estimator": LogisticRegression(
                solver="saga", max_iter=5000, class_weight=cw, random_state=random_state
            ),
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
            "estimator": RandomForestClassifier(
                class_weight=cw, random_state=random_state, n_jobs=-1
            ),
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
                random_state=random_state, tree_method="hist", verbosity=0, eval_metric="mlogloss"
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
            "estimator": LGBMClassifier(class_weight=cw, random_state=random_state, verbose=-1),
            "space": {
                "model__n_estimators": ("int", 100, 600),
                "model__num_leaves": ("int", 15, 63),
                "model__learning_rate": ("float", 1e-2, 3e-1, True),
                "model__min_child_samples": ("int", 5, 30),
            },
            "needs_scaling": False,
        }
    }


def build_pipeline(
    estimator,
    needs_scaling: bool,
    task: str | None = None,
    k_features: int | None = None,
) -> Pipeline:
    """
    Wrap an estimator in a Pipeline so preprocessing is fit on train folds only.

    A univariate SelectKBest filter is prepended when `k_features` is given, so the
    selection is re-fit inside every CV fold (leakage-safe) and curbs the p >> n
    overfit on the monthly frame.

    Args:
        estimator: The final sklearn-compatible estimator.
        needs_scaling (bool): Prepend a StandardScaler for scale-sensitive models.
        task (str | None): 'classification' or 'regression'; picks the SelectKBest
            score function. Selection is skipped when None.
        k_features (int | None): Number of features to keep; no selection when None.

    Returns:
        Pipeline: The assembled pipeline with the estimator as step 'model'.
    """
    steps = []
    if k_features is not None and task is not None:
        score_func = f_classif if task == "classification" else f_regression
        steps.append(("select", SelectKBest(score_func=score_func, k=k_features)))
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe

def auto_k_features(n_rows: int, n_features: int, n_splits: int = 5) -> int:
    """
    Size the SelectKBest k to the smallest expanding CV fold.

    The earliest TimeSeriesSplit fold trains on ~n_rows/(n_splits+1) rows; keeping
    about 4 training rows per feature keeps even that fold well-conditioned and the
    CV score stable.

    Args:
        n_rows (int): Number of training rows.
        n_features (int): Number of candidate features.
        n_splits (int): Number of TimeSeriesSplit folds.

    Returns:
        int: Number of features to keep (at least 5, at most n_features).
    """
    smallest_fold = n_rows // (n_splits + 1)
    return max(5, min(n_features, smallest_fold // 4))


def chronological_split(X: pd.DataFrame, y: pd.Series, preset: str = "70/15/15") -> dict:
    """
    Split X/y chronologically into train/valid/test without shuffling.

    Args:
        X (pd.DataFrame): Feature matrix, already ordered by date.
        y (pd.Series): Aligned target.
        preset (str): One of SPLIT_PRESETS.

    Returns:
        dict: 'Train'/'Valid'/'Test' -> (X_slice, y_slice), preserving order.
    """
    train_frac, valid_frac = SPLIT_PRESETS[preset]
    n = len(X)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    return {
        "Train": (X.iloc[:n_train], y.iloc[:n_train]),
        "Valid": (X.iloc[n_train : n_train + n_valid], y.iloc[n_train : n_train + n_valid]),
        "Test": (X.iloc[n_train + n_valid :], y.iloc[n_train + n_valid :]),
    }


def search_estimator(pipe, space, X, y, scoring, cv, budget, random_state=SEED):
    """
    Tune one pipeline's hyperparameters under the chosen budget backend.

    Fast/Balanced use RandomizedSearchCV; Thorough uses Optuna (TPE). A model
    with no tunable space is simply cross-validated and refit as-is.

    Args:
        pipe (Pipeline): Pipeline to tune.
        space (dict): Parameter space (see `to_sklearn_distributions`).
        X, y: Training data (the train split only, never valid/test).
        scoring (str): sklearn scoring string (from CV_SCORING).
        cv: Cross-validation splitter (TimeSeriesSplit).
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
            n_jobs=-1,
            refit=True,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, float(search.best_score_)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        candidate = clone(pipe).set_params(**suggest_from_space(trial, space))
        return float(np.mean(cross_val_score(candidate, X, y, scoring=scoring, cv=cv)))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=cfg["n_trials"])
    best = clone(pipe).set_params(**study.best_params).fit(X, y)
    return best, study.best_params, float(study.best_value)


def train_roster(
    X,
    y,
    task,
    *,
    budget="Fast",
    n_splits=5,
    scoring=None,
    models=None,
    class_weight=True,
    k_features="auto",
    early_stop=None,
    random_state=SEED,
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
        models (list | None): Subset of roster names to train (all if None).
        class_weight (bool): Pass 'balanced' weighting where supported (clf).
        k_features (int | str): 'auto' sizes the in-CV SelectKBest to the smallest CV
            fold (see `auto_k_features`), or an explicit count.
        early_stop (float | None): Stop once a model's CV score reaches this value
            (only for higher-is-better metrics).
        random_state (int): Seed.
        progress (callable | None): Called as progress(done, total, name, cv_score)
            after each model, for a UI progress bar.

    Returns:
        dict: 'models' (name -> fitted estimator), 'cv_scores', 'params', 'scoring',
        'best' (top model name), 'classes' (clf only) and 'label_encoder' (clf only).
    """
    scoring = scoring or DEFAULT_SCORING[task]
    scorer = CV_SCORING[scoring]
    cv = TimeSeriesSplit(n_splits=n_splits)

    if k_features == "auto":
        k_features = auto_k_features(len(X), X.shape[1], n_splits)

    classes, encoder = None, None
    if task == "classification":
        encoder = LabelEncoder().fit(y)
        classes = list(encoder.classes_)
        y = pd.Series(encoder.transform(y), index=y.index)

    roster = model_roster(task, random_state=random_state, class_weight=class_weight)
    if models:
        roster = {name: cfg for name, cfg in roster.items() if name in models}

    fitted, cv_scores, params = {}, {}, {}
    total = len(roster)
    can_early_stop = early_stop is not None and scoring in MAXIMISE_METRICS
    for done, (name, cfg) in enumerate(roster.items(), start=1):
        pipe = build_pipeline(cfg["estimator"], cfg["needs_scaling"], task=task, k_features=k_features)
        model, best_params, cv_score = search_estimator(
            pipe, cfg["space"], X, y, scorer, cv, budget, random_state
        )
        fitted[name] = model
        cv_scores[name] = cv_score
        params[name] = best_params
        if progress:
            progress(done, total, name, cv_score)
        if can_early_stop and cv_score >= early_stop:
            break

    best = max(cv_scores, key=cv_scores.get) if cv_scores else None
    return {
        "models": fitted,
        "cv_scores": cv_scores,
        "params": params,
        "scoring": scoring,
        "best": best,
        "classes": classes,
        "label_encoder": encoder,
    }
