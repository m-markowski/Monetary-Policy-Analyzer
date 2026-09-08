import pandas as pd
import streamlit as st

from src.dataset_builder import (
    ECONOMIES,
    build_and_save,
    cache_exists,
    load_master,
    load_metadata,
    master_mtime,
    save_metadata,
)

st.set_page_config(page_title="Monetary Policy Analyzer", layout="wide")

FAMILY_LABELS = {
    "ret": "Return",
    "ma": "Moving average",
    "vol": "Volatility",
    "chg": "Period-over-period change",
    "spread": "Spread",
}


@st.cache_data(show_spinner=False)
def get_master(economy: str, mtime: float) -> pd.DataFrame:
    """Cached read of a master dataset; invalidated when the file changes."""
    return load_master(economy)


def render_economy_card(meta: dict) -> None:
    """
    Render a Streamlit card summarising one economy's cached dataset.

    Args:
        meta (dict): Per-economy metadata produced by build_and_save.

    Returns:
        None. Writes directly to the Streamlit page.
    """
    eco = meta["economy"].upper()
    st.subheader(eco)
    c1, c2, c3 = st.columns(3)
    c1.metric("Working range", f"{meta['working_start']} - {meta['working_end']}")
    c2.metric("Rows", f"{meta['n_rows']:,}")
    c3.metric("Features", meta["n_features"])
    st.caption(f"Raw (levels) range: {meta['raw_start']} - {meta['raw_end']}. Built {meta['built_at']}")

    dropped = meta["dropped_fred"] + meta["dropped_tickers"]
    if dropped:
        with st.expander(f"Dropped {len(dropped)} stale series/tickers"):
            st.dataframe(
                pd.DataFrame(dropped, columns=["id", "name", "reason"]),
                width="stretch",
                hide_index=True,
            )
    else:
        st.caption("No series dropped for staleness.")

    skipped = meta.get("skipped_features", [])
    if skipped:
        with st.expander(f"{len(skipped)} feature(s) skipped"):
            st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)

    with st.expander("Engineered features by source"):
        rows = [
            {
                "source": m["base"],
                "frequency": m["frequency"],
                "families": ", ".join(FAMILY_LABELS.get(k, k) for k in m["families"]),
                "n_features": sum(len(v) for v in m["families"].values()),
            }
            for m in meta.get("feature_manifest", [])
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    events = meta.get("events", [])
    levels = {"error": st.error, "warning": st.warning, "info": st.info}
    if events:
        with st.expander(
            f"Load log ({len(events)} messages)",
            expanded=any(e["level"] == "error" for e in events),
        ):
            for e in events:
                levels.get(e["level"], st.write)(f"[{e['stage']}] {e['message']}")


st.title("Monetary Policy Analyzer")
st.write(
    "Load the base datasets (USA + Eurozone) from FRED and yfinance. "
    "The longest available history is fetched and cached locally; the rest "
    "of the app works on these cached datasets."
)

label = "Reload data" if cache_exists() else "Load data"
col_a, col_b = st.columns([1, 2])
reload_clicked = col_a.button(label, type="primary")
full_refresh = col_b.checkbox("Force full refresh (ignore cache, repull all history)")

if reload_clicked:
    metas = load_metadata() or {}
    with st.status("Fetching from FRED + yfinance...", expanded=True) as status:
        for eco in ECONOMIES:
            st.write(f"Building {eco.upper()} dataset...")
            try:
                metas[eco] = build_and_save(eco, full_refresh=full_refresh)
                if all(name in metas for name in ECONOMIES):
                    save_metadata(metas)
            except (RuntimeError, ValueError, OSError) as exc:
                status.update(label="Data refresh stopped", state="error")
                st.error(str(exc))
                st.stop()
            m = metas[eco]
            st.write(
                f"{eco.upper()} ready: {m['working_start']} - {m['working_end']} "
                f"({m['n_rows']:,} rows, {m['n_features']} features)"
            )
        status.update(label="Data loaded and cached", state="complete")
    get_master.clear()
    st.rerun()

st.divider()

if not cache_exists():
    st.info("No cached data yet. Click **Load data** to build the datasets.")
    st.stop()

meta = load_metadata()
for eco in ECONOMIES:
    render_economy_card(meta[eco])
    st.divider()

ov = meta["overlap"]
if ov["has_overlap"]:
    st.success(f"Common comparison window (USA & Eurozone): **{ov['start']} - {ov['end']}**")
else:
    st.warning("The two economies do not overlap - cross-economy comparison will be limited.")

with st.expander("Preview cached data"):
    eco = st.selectbox("Economy", ECONOMIES, format_func=str.upper)
    df = get_master(eco, master_mtime(eco))
    st.dataframe(df.tail(10), width="stretch")
