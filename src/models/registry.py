import json
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from config.settings import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODEL_SCHEMA_VERSION = 3


def slugify(text: str) -> str:
    """
    Reduce a label to a filesystem-safe slug.

    Args:
        text (str): Arbitrary label.

    Returns:
        str: Lowercased, non-alphanumerics collapsed to single hyphens.
    """
    slug = "".join(c if c.isalnum() else "-" for c in str(text).lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def cache_key(economy: str, task: str, target: str, horizon: int, split: str) -> str:
    """
    Deterministic artifact key for one modelling configuration.

    Args:
        economy (str): Economy identifier.
        task (str): 'regression' or 'classification'.
        target (str): Target display name.
        horizon (int): Forecast horizon in months.
        split (str): Split preset (e.g. '70/15/15').

    Returns:
        str: A stable key used for both the .pkl and .json filenames.
    """
    parts = [economy, task, target, f"h{horizon}", split]
    return "_".join(slugify(p) for p in parts)


def data_signature(X: pd.DataFrame, y) -> dict:
    """
    Fingerprint the modelling data so a saved artifact can detect staleness.

    Args:
        X (pd.DataFrame): Feature matrix.
        y: Aligned target.

    Returns:
        dict: Row/column counts, column names, content hashes and the last date.
    """
    x_hash = int(pd.util.hash_pandas_object(X, index=True).sum())
    y_hash = int(pd.util.hash_pandas_object(pd.Series(y), index=True).sum())
    return {
        "rows": int(len(X)),
        "cols": int(X.shape[1]),
        "columns": [str(c) for c in X.columns],
        "x_hash": x_hash,
        "y_hash": y_hash,
        "last_date": str(X.index[-1]),
    }


def build_metadata(
    *,
    economy: str,
    task: str,
    target: str,
    horizon: int,
    split: str,
    scoring: str,
    best: str | None,
    cv_scores: dict,
    params: dict,
    leaderboard: pd.DataFrame | None,
    signature: dict,
    extra: dict | None = None,
) -> dict:
    """
    Assemble the JSON-serialisable metadata sidecar for a training run.

    Args:
        economy, task, target, horizon, split, scoring: The run configuration.
        best (str | None): Winning model name.
        cv_scores (dict): Model name -> CV score.
        params (dict): Model name -> chosen hyperparameters.
        leaderboard (pd.DataFrame | None): The train/valid/test leaderboard.
        signature (dict): Output of `data_signature`.
        extra (dict | None): Any additional fields to store.

    Returns:
        dict: Metadata including a 'trained_at' timestamp.
    """
    metadata = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "economy": economy,
        "task": task,
        "target": target,
        "horizon": horizon,
        "split": split,
        "scoring": scoring,
        "best_model": best,
        "cv_scores": cv_scores,
        "params": params,
        "leaderboard": (leaderboard.reset_index().to_dict(orient="records") if leaderboard is not None else None),
        "signature": signature,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_artifacts(models: dict, metadata: dict, key: str, directory: Path = MODELS_DIR) -> dict:
    """
    Serialise the fitted models and their metadata sidecar.

    Args:
        models (dict): Model name -> fitted estimator. Keras wrappers are safe to
            include: `KerasEstimator` pickles by serialising its network with
            Keras' native `.keras` format, so the whole roster round-trips.
        metadata (dict): Output of `build_metadata`.
        key (str): Artifact key from `cache_key`.
        directory (Path): Destination directory (created if absent).

    Returns:
        dict: 'model_path' and 'meta_path'.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{key}.pkl"
    meta_path = directory / f"{key}.json"
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        staged_model = Path(temporary) / model_path.name
        staged_meta = Path(temporary) / meta_path.name
        joblib.dump(models, staged_model)
        staged_meta.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        # No stale sidecar may describe a newly written model if the second move fails.
        meta_path.unlink(missing_ok=True)
        staged_model.replace(model_path)
        staged_meta.replace(meta_path)
    return {"model_path": model_path, "meta_path": meta_path}


def load_artifacts(
    key: str,
    directory: Path = MODELS_DIR,
    *,
    signature: dict | None = None,
    training_options: dict | None = None,
) -> dict | None:
    """Load trusted local artifacts only after checking metadata and compatibility.

    Pickle is executable input, not an interchange format for untrusted models.
    Corrupt or incompatible local cache files are ignored and must be retrained.
    """
    directory = Path(directory)
    model_path, meta_path = directory / f"{key}.pkl", directory / f"{key}.json"
    if not model_path.is_file() or not meta_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
            return None
        if signature is not None and is_stale(metadata, signature):
            return None
        if training_options is not None and metadata.get("training_options") != training_options:
            return None
        models = joblib.load(model_path)
    except Exception:
        # Cache incompatibilities must not prevent training a replacement.
        return None
    return {"models": models, "metadata": metadata}


def is_stale(metadata: dict, signature: dict) -> bool:
    """Reject changed values, row counts, feature names or feature ordering."""
    stored = metadata.get("signature", {})
    fields = ("x_hash", "y_hash", "rows", "cols", "columns", "last_date")
    return any(stored.get(field) != signature.get(field) for field in fields)
