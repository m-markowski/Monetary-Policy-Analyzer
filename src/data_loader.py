import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import yfinance as yf
from config.settings import EconomyConfig
from utils.logger import exception_logger


class EconomyDataLoader:
    """
    Economy-specific data loader for FRED and market data.

    Attributes:
        economy (str): Normalized economy identifier (lowercase).
        fred (Fred): FRED API client for economic data retrieval.
        fred_config (dict): FRED feature IDs grouped into 'rates' and 'other'.
        market_tickers (list[list[str]]): [ticker, friendly_name] pairs from yfinance.
        start_date (str): Fallback start date for data retrieval.
        features (dict): Periods/windows for feature engineering.
        staleness_tolerance (dict): Max observation age (days) per frequency.
        dropped_fred (list): Dropped stale series as (id, name, reason).
        dropped_tickers (list): Dropped stale tickers as (id, name, reason).
        events (list[dict]): User-facing log events ({'level', 'stage', 'message'}).
        skipped_features (list[dict]): Engineered features skipped, with reasons.
        feature_manifest (list[dict]): Per-source description of engineered features.
    """

    FREQ_MAP = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly"}

    def __init__(self, economy: str = "USA"):
        """
        Initialize the data loader with economy-specific configurations.

        Args:
            economy (str): Economy identifier aligned with YAML config keys.
        """
        economy_cfg = EconomyConfig(economy)
        self.economy = economy_cfg.economy
        self.fred = economy_cfg.fred
        self.fred_config = economy_cfg.fred_config
        self.market_tickers = economy_cfg.market_tickers
        self.start_date = economy_cfg.start_date
        self.features = economy_cfg.features
        self.staleness_tolerance = economy_cfg.staleness_tolerance
        self.metadata_cache = {}
        self.ticker_cache = {}
        self.dropped_fred = []
        self.dropped_tickers = []
        self.events = []  # {'level': 'warning'|'error'|'info', 'stage': str, 'message': str}
        self.skipped_features = []
        self.feature_manifest = []

    def record(self, level: str, message: str, stage: str = "") -> None:
        """
        Collect a user-facing event instead of printing/logging to a file.

        Args:
            level (str): Severity ('info', 'warning' or 'error').
            message (str): Human-readable event message.
            stage (str): Pipeline stage that produced the event.

        Returns:
            None. Appends an entry to self.events.
        """
        self.events.append({"level": level, "stage": stage, "message": message})

    def get_series_metadata(self, series_id: str) -> dict | None:
        """
        Fetch and cache FRED series metadata.

        Args:
            series_id (str): FRED series identifier.

        Returns:
            dict: Metadata dictionary for the series, or None if fetch fails.
        """
        if series_id in self.metadata_cache:
            return self.metadata_cache[series_id]
        try:
            info = self.fred.get_series_info(series_id).to_dict()
        except Exception as e:
            exception_logger.warning(f"Metadata fetch failed for {series_id}: {e}")
            info = None
        self.metadata_cache[series_id] = info
        return info

    def is_series_stale(self, series_id: str) -> tuple[bool, str]:
        """
        Return (is_stale, reason). A series is stale if its last observation
        is older than its frequency-specific tolerance.

        Args:
            series_id (str): FRED series identifier.

        Returns:
            tuple[bool, str]: (is_stale, reason). Reason is empty when the series is fresh.
        """
        info = self.get_series_metadata(series_id)
        if info is None:
            return True, "metadata unavailable"
        obs_end = info.get("observation_end")
        if obs_end is None or pd.isna(obs_end):
            return True, "no observation_end"
        freq_short = (info.get("frequency_short") or "").upper()
        freq = self.FREQ_MAP.get(freq_short, "monthly")
        tolerance = self.staleness_tolerance.get(freq, 90)
        age_days = (pd.Timestamp.today().normalize() - pd.Timestamp(obs_end)).days
        if age_days > tolerance:
            return (
                True,
                f"last obs {pd.Timestamp(obs_end).date()} is {age_days}d old (>{tolerance}d for {freq})",
            )
        return False, ""

    def get_ticker_last_quote(self, ticker: str) -> pd.Timestamp | None:
        """
        Fetch and cache the last available trading day for a yfinance ticker.

        Args:
            ticker (str): yfinance ticker symbol.

        Returns:
            pd.Timestamp | None: Date of the most recent quote or None if no data was returned.
        """
        if ticker in self.ticker_cache:
            return self.ticker_cache[ticker]
        last = None
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=30)
        try:
            df = yf.download(
                tickers=ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                ts = pd.Timestamp(df.index.max())
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(None)
                last = ts.normalize()
        except Exception as e:
            exception_logger.warning(f"yfinance metadata fetch failed for {ticker}: {e}")
        self.ticker_cache[ticker] = last
        return last

    def is_ticker_stale(self, ticker: str) -> tuple[bool, str]:
        """
        Mirror of is_series_stale for yfinance market tickers.

        Args:
            ticker (str): yfinance ticker symbol.

        Returns:
            tuple[bool, str]: (is_stale, reason). Reason is empty when the ticker is fresh.
        """
        last = self.get_ticker_last_quote(ticker)
        if last is None:
            return True, "no quotes returned by yfinance"
        tolerance = self.staleness_tolerance.get("daily", 10)
        age_days = (pd.Timestamp.today().normalize() - last).days
        if age_days > tolerance:
            return True, f"last quote {last.date()} is {age_days}d old (>{tolerance}d for daily)"
        return False, ""

    def filter_stale_features(self) -> None:
        """
        Drop stale FRED series and stale market tickers in place.

        Returns:
            None. Mutates self.fred_config and self.market_tickers in place.
        """
        for bucket in ("rates", "other"):
            kept, dropped = [], []
            for sid, name in self.fred_config[bucket]:
                stale, reason = self.is_series_stale(sid)
                if stale:
                    dropped.append((sid, name, reason))
                else:
                    kept.append([sid, name])
            self.fred_config[bucket] = kept
            self.dropped_fred.extend(dropped)
            for sid, name, reason in dropped:
                exception_logger.warning(f"Dropped stale series {sid} ({name}): {reason}")

        kept, dropped = [], []
        for ticker, name in self.market_tickers:
            stale, reason = self.is_ticker_stale(ticker)
            if stale:
                dropped.append((ticker, name, reason))
            else:
                kept.append([ticker, name])
        self.market_tickers = kept
        self.dropped_tickers = dropped
        for ticker, name, reason in dropped:
            exception_logger.warning(f"Dropped stale ticker {ticker} ({name}): {reason}")

    def resolve_date_range(self, start_date: str | None, end_date: str | None) -> tuple[str, str]:
        """
        Resolve the date range for data retrieval.

        Args:
            start_date (str | None): Start date (YYYY-MM-DD); resolved from series metadata when None.
            end_date (str | None): End date (YYYY-MM-DD); defaults to today when None.

        Returns:
            tuple[str, str]: A tuple containing the resolved start and end dates.
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        if start_date is None:
            starts = []
            for sid, _ in self.fred_config["rates"] + self.fred_config["other"]:
                info = self.get_series_metadata(sid)
                if info and info.get("observation_start"):
                    starts.append(pd.Timestamp(info["observation_start"]))
            start_date = max(starts).strftime("%Y-%m-%d") if starts else self.start_date
        return start_date, end_date

    def fetch_fred_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch a single FRED series.

        Args:
            series_id (str): FRED series identifier.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.Series: Time-indexed series of observations.

        Raises:
            Exception: If data retrieval fails after retries.
        """
        max_retries = 5
        delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                data = self.fred.get_series(series_id=series_id, observation_start=start_date, observation_end=end_date)
                return data
            except Exception as e:
                if attempt == max_retries:
                    exception_logger.error(f"Failed to fetch {series_id} after {max_retries} attempts: {e}")
                    self.record(
                        "error",
                        f"Failed to fetch {series_id} after {max_retries} attempts: {e}",
                        stage="fetch",
                    )
                    raise
                exception_logger.warning(f"Attempt {attempt}/{max_retries} failed for {series_id}: {e}")
                self.record(
                    "warning",
                    f"Attempt {attempt}/{max_retries} failed for {series_id}: {e}",
                    stage="fetch",
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def fetch_all_fred_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch all configured FRED series and merge into one DataFrame.

        Args:
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with all FRED series as columns.
        """
        all_series = [tick[0] for tick in (self.fred_config["rates"] + self.fred_config["other"])]

        def fetch_job(series_id: str) -> tuple[str, pd.Series | None]:
            try:
                return series_id, self.fetch_fred_series(series_id=series_id, start_date=start_date, end_date=end_date)
            except Exception:
                return series_id, None

        results = {}
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_series = {executor.submit(fetch_job, series_id): series_id for series_id in all_series}
            for future in as_completed(future_to_series):
                expected_series_id = future_to_series[future]
                try:
                    series_id, data = future.result()
                    if data is not None:
                        results[series_id] = data
                except Exception:
                    exception_logger.error(f"Thread crashed for {expected_series_id}", exc_info=True)
                    self.record("error", f"Thread crashed for {expected_series_id}", stage="fetch")
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.index.name = "date"
        return df.reset_index()

    def fetch_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch market data for specified tickers using yfinance.

        Args:
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with market closing prices for each ticker.
        """
        data_dict = {}

        for ticker, _ in self.market_tickers:
            try:
                df = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False,
                )
                if not df.empty:
                    data_dict[ticker] = df["Close"]
            except Exception as e:
                exception_logger.error(f"Failed to fetch {ticker}: {e}")
                self.record("error", f"Failed to fetch {ticker}: {e}", stage="fetch")

        if data_dict:
            df = pd.concat(data_dict.values(), axis=1)
            df.columns = data_dict.keys()
            df.index.name = "date"
            return df.reset_index()
        return pd.DataFrame()

    def build_raw_dataset(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """
        Build a complete raw dataset (before feature engineering).

        Args:
            start_date (str, optional): Start date (YYYY-MM-DD).
            end_date (str, optional): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with all raw data.
        """
        self.filter_stale_features()  # updates self.fred_config and self.market_tickers with data to keep
        start_date, end_date = self.resolve_date_range(start_date, end_date)

        fred_df = self.fetch_all_fred_data(start_date=start_date, end_date=end_date)
        market_df = self.fetch_market_data(start_date=start_date, end_date=end_date)

        if not fred_df.empty and not market_df.empty:
            df = pd.merge(fred_df, market_df, on="date", how="outer")
        elif not fred_df.empty:
            df = fred_df
        elif not market_df.empty:
            df = market_df
        else:
            exception_logger.error("No data could be fetched from any source.")
            raise ValueError("No data could be fetched from any source")

        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date").resample("D").asfreq()

        fred_cols = [c for c in fred_df.columns if c != "date"]
        market_cols = [c for c in market_df.columns if c != "date"]

        # Forward fill macro data if there are gaps
        df[fred_cols] = df[fred_cols].ffill()
        # Drop rows where market data is missing (non-trading days)
        df = df.dropna(subset=market_cols, how="all")
        # Forward fill market data if there are gaps on days such as July 4th
        df[market_cols] = df[market_cols].ffill()
        # Drop any remaining NaNs
        df = df.dropna()
        df = df.reset_index()
        df.rename(
            columns=dict(self.fred_config["rates"] + self.fred_config["other"] + self.market_tickers),
            inplace=True,
        )
        return df

    # Feature Engineering
    def detect_frequency(self, col: str) -> str | None:
        """
        Detect the frequency of a given column based on FRED metadata or market tickers.

        Args:
            col (str): Column name to check.

        Returns:
            str | None: Detected frequency category ('daily', 'weekly', 'monthly', 'quarterly'), or None if
            undetectable.
        """
        if col in [tick[0] for tick in self.market_tickers]:
            return "daily"
        info = self.get_series_metadata(col)
        if info is None:
            exception_logger.error(f"No frequency detected for {col} (no metadata)")
            return None
        return self.FREQ_MAP.get((info.get("frequency_short") or "").upper())

    def classify_all_frequencies(self, cols: list[str]) -> dict[str, list[str]]:
        """
        Classify all columns into frequency categories.

        Args:
            cols (list[str]): List of column names to classify.

        Returns:
            dict[str, list[str]]: Dictionary with frequency categories as keys and lists of column names as values.
        """
        frequencies = {"daily": [], "weekly": [], "monthly": [], "quarterly": []}
        name_to_ticker = {name: ticker for ticker, name in (self.fred_config["other"] + self.market_tickers)}
        for col in cols:
            raw_id = name_to_ticker.get(col, col)
            freq = self.detect_frequency(col=raw_id)
            if freq is not None and freq in frequencies:
                frequencies[freq].append(col)
        return frequencies

    def create_features(self, df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
        """
        Create frequency-aware features for the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.
            all_cols (list[str]): List of all columns to process.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        all_cols = list(set(all_cols))
        freq_classification = self.classify_all_frequencies(cols=all_cols)

        # Process daily data with returns, volatility, and moving averages
        for col in freq_classification["daily"]:
            if col not in df.columns:
                continue
            families = {
                "ret": [f"{p}d" for p in self.features["daily_return_periods"]],
                "ma": [f"{w}d" for w in self.features["daily_ma_windows"]],
            }
            for period in self.features["daily_return_periods"]:
                df[f"{col}_ret_{period}d"] = df[col].pct_change(period)
            for window in self.features["daily_ma_windows"]:
                df[f"{col}_ma_{window}d"] = df[col].rolling(window).mean()

            returns = df[col].pct_change()
            if returns.std() > 0.0005:  # Threshold to avoid computing on near-constant series
                families["vol"] = [f"{w}d" for w in self.features["daily_volatility_windows"]]
                for window in self.features["daily_volatility_windows"]:
                    df[f"{col}_vol_{window}d"] = returns.rolling(window).std()
            else:
                self.skipped_features.append({"feature": f"{col}_vol_*", "reason": "near-constant series (std ~ 0)"})
            self.feature_manifest.append({"base": col, "frequency": "daily", "families": families})

        # Process weekly, monthly, and quarterly data with period-based changes
        frequencies = {"week": "W", "month": "M", "quarter": "Q"}
        for freq_name, freq_code in frequencies.items():
            df[f"year_{freq_name}"] = df["date"].dt.to_period(freq_code)
            for col in freq_classification[freq_name + "ly"]:
                if col not in df.columns:
                    continue
                # Get last non-null observation per period
                period_data = df.groupby(f"year_{freq_name}")[col].last()
                for periods, label in [tuple(period) for period in self.features[f"{freq_name}ly_periods"]]:
                    change_series = period_data.pct_change(periods)
                    df[f"{col}_chg_{label}"] = df[f"year_{freq_name}"].map(change_series)
                labels = [label for _, label in self.features[f"{freq_name}ly_periods"]]
                self.feature_manifest.append({"base": col, "frequency": f"{freq_name}ly", "families": {"chg": labels}})
            df.drop(columns=[f"year_{freq_name}"], axis=1, inplace=True, errors="ignore")
        return df

    def create_yield_curve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create yield curve related features.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.

        Returns:
            pd.DataFrame: DataFrame with yield curve features added.
        """
        specs = {
            "usa": [
                ("sprd_yld_30y10y", "yld_ust_30y", "yld_ust_10y", "Long-end term premium"),
                ("sprd_yld_5y2y", "yld_ust_5y", "yld_ust_2y", "Mid-curve steepness"),
                ("sprd_yld_10y5y", "yld_ust_10y", "yld_ust_5y", "Back-end steepness"),
                ("sprd_5y_ff", "yld_ust_5y", "rate_ff_eff", "Policy stance vs medium term"),
                ("sprd_2y_ff", "yld_ust_2y", "rate_ff_eff", "Expected hikes vs cuts"),
            ],
            "eurozone": [
                ("sprd_10y_ib3m", "yld_10y_gov", "rate_ib_3m", "Long vs short-term expectations"),
                (
                    "sprd_10y_ecb",
                    "yld_10y_gov",
                    "rate_ecb_dep",
                    "Policy restrictiveness vs long end",
                ),
                (
                    "psprd_ib_3m_ecb",
                    "rate_ib_3m",
                    "rate_ecb_dep",
                    "Market expectations vs ECB stance",
                ),
            ],
        }
        for out, a, b, desc in specs.get(self.economy, []):
            if a in df.columns and b in df.columns:
                df[out] = df[a] - df[b]
                self.feature_manifest.append(
                    {
                        "base": out,
                        "frequency": "derived",
                        "families": {"spread": [f"{a} - {b}"]},
                        "desc": desc,
                    }
                )
            else:
                missing = [c for c in (a, b) if c not in df.columns]
                self.skipped_features.append(
                    {
                        "feature": out,
                        "reason": f"missing input(s): {', '.join(missing)} (dropped as stale)",
                        "desc": desc,
                    }
                )
        return df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.

        Returns:
            pd.DataFrame: DataFrame with all engineered features.
        """
        # No rates here as these are already transformed interest rates.
        # Calculating returns/volatility on rates doesn't make economic sense.
        all_cols = [tick[1] for tick in (self.fred_config["other"] + self.market_tickers)]
        df = self.create_features(df=df, all_cols=all_cols)
        df = self.create_yield_curve_features(df=df)
        df = df.dropna()
        return df
