import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import PROJECT_ROOT
from src.data_loader import EconomyDataLoader

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
METADATA_PATH = CACHE_DIR / "cache_metadata.json"
ECONOMIES = ("usa", "eurozone")
TAIL_BUFFER_DAYS = 400  # > longest feature lookback (252d) + buffer for FRED revisions


def raw_path(economy: str) -> Path:
    """Path to the cached raw levels snapshot for one economy."""
    return CACHE_DIR / f"raw_{economy}.parquet"


def master_path(economy: str) -> Path:
    """Path to the engineered master CSV for one economy."""
    return CACHE_DIR / f"master_dataset_{economy}.csv"


def build_and_save(economy: str, full_refresh: bool = False) -> dict:
    """
    Build the engineered dataset for one economy and persist it.

    On the first run (or full_refresh), the entire history is fetched. Otherwise only
    a trailing window is refetched and spliced into the cached raw levels, which
    also absorbs FRED revisions to recent observations.

    Args:
        economy (str): Economy identifier ('usa' or 'eurozone').
        full_refresh (bool): Ignore the cached raw snapshot and refetch in full.

    Returns:
        dict: Metadata describing the built dataset.
    """
    rp = raw_path(economy)
    loader = EconomyDataLoader(economy=economy)
    raw = None

    if rp.exists() and not full_refresh:
        stored = pd.read_parquet(rp)
        tail_start = (stored["date"].max() - pd.Timedelta(days=TAIL_BUFFER_DAYS)).strftime("%Y-%m-%d")
        tail = loader.build_raw_dataset(start_date=tail_start)
        if set(tail.columns) == set(stored.columns):
            raw = pd.concat([stored[stored["date"] < tail["date"].min()], tail], ignore_index=True)
            raw = raw.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

    if raw is None:
        # First run, forced refresh, or schema drift -> rebuild fully with a clean loader.
        loader = EconomyDataLoader(economy=economy)
        raw = loader.build_raw_dataset()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(rp, index=False)

    master = loader.engineer_all_features(raw)
    master.to_csv(master_path(economy), index=False)

    return {
        "economy": economy,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "raw_start": raw["date"].min().date().isoformat(),
        "raw_end": raw["date"].max().date().isoformat(),
        "working_start": master["date"].min().date().isoformat(),
        "working_end": master["date"].max().date().isoformat(),
        "n_rows": int(len(master)),
        "n_features": int(master.shape[1] - 1),
        "kept_fred": [name for _, name in loader.fred_config["rates"] + loader.fred_config["other"]],
        "kept_tickers": [name for _, name in loader.market_tickers],
        "dropped_fred": loader.dropped_fred,
        "dropped_tickers": loader.dropped_tickers,
        "events": loader.events,
        "skipped_features": loader.skipped_features,
        "feature_manifest": loader.feature_manifest,
    }


def compute_overlap(metas: dict) -> dict:
    """
    Compute the common working date window across all economies.

    Args:
        metas (dict): Per-economy metadata keyed by economy name.

    Returns:
        dict: Keys 'start', 'end', 'has_overlap' describing the shared window.
    """
    starts = [metas[e]["working_start"] for e in ECONOMIES]
    ends = [metas[e]["working_end"] for e in ECONOMIES]
    o_start, o_end = max(starts), min(ends)
    has_overlap = o_start <= o_end
    return {
        "start": o_start if has_overlap else None,
        "end": o_end if has_overlap else None,
        "has_overlap": has_overlap,
    }


def save_metadata(metas: dict) -> dict:
    """
    Attach cross-economy overlap and persist the metadata sidecar.

    Args:
        metas (dict): Per-economy metadata to enrich and persist.

    Returns:
        dict: The metadata dict with the added 'overlap' entry.
    """
    metas = dict(metas)
    metas["overlap"] = compute_overlap(metas)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metas, indent=2))
    return metas


def cache_exists() -> bool:
    """True when the metadata sidecar and both master datasets exist."""
    return METADATA_PATH.exists() and all(master_path(e).exists() for e in ECONOMIES)


def load_metadata() -> dict | None:
    """Load the metadata sidecar, or None if it does not exist."""
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None


def load_master(economy: str) -> pd.DataFrame:
    """Read one economy's engineered master dataset from CSV."""
    return pd.read_csv(master_path(economy), parse_dates=["date"])


def master_mtime(economy: str) -> float:
    """File mtime used as a cache-busting key for Streamlit."""
    p = master_path(economy)
    return p.stat().st_mtime if p.exists() else 0.0
