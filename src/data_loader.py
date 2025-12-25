import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))
import pandas as pd
from pandas.errors import PerformanceWarning
import yfinance as yf
import warnings
warnings.simplefilter(action = 'ignore', category = (FutureWarning, PerformanceWarning))
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from fredapi import Fred
from config.settings import EconomyConfig
from utils.logger import exception_logger

class EconomyDataLoader:
    """
    Economy-specific data loader for FRED and market data.

    Attributes:
        fred (Fred): FRED API client for economic data retrieval.
        fred_config (dict): FRED features tickers for the economy.
        market_tickers (list[str]): Market tickers associated with the economy. Obtained from yfinance.
        start_date (str): Default start date for data retrieval.
        features (dict): Periods for feature engineering.
    """
    def __init__(self, economy: str = 'USA'):
        """
        Initialize the data loader with economy-specific configurations.

        Args:
            economy (str): Economy identifier aligned with YAML config keys.
        """
        economy_cfg = EconomyConfig(economy = economy.lower())
        self.fred = economy_cfg.fred
        self.fred_config = economy_cfg.fred_config
        self.market_tickers = economy_cfg.market_tickers
        self.start_date = economy_cfg.start_date
        self.features = economy_cfg.features

    def fetch_fred_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch a single FRED series.

        Args:
            series_id (str): FRED series identifier.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.Series: Time-indexed series of observations. Returns an empty series if the data cannot be retrieved.
        """
        try:
            data = self.fred.get_series(series_id = series_id,
                                        observation_start = start_date,
                                        observation_end = end_date)
            return data
        except Exception as e:
            exception_logger.error(
                f"Failed to fetch FRED series '{series_id}': {type(e).__name__} - {e}", exc_info=True
            )
            print(f"Could not fetch {series_id}")
            return pd.Series(dtype=float)

    def fetch_all_fred_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch FRED data for a given series and merge into one DataFrame.

        Args:
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: DataFrame with all FRED series as columns.
        """
        all_series = [tick[0] for tick in (self.fred_config['rates'] + self.fred_config['other'])]

        def fetch_job(series_id: str) -> tuple[str, pd.Series] | tuple[str, None]:
            try:
                return series_id, self.fetch_fred_series(series_id = series_id,
                                                         start_date = start_date,
                                                         end_date = end_date)
            except Exception as e:
                exception_logger.error(f"Failed to fetch {series_id}: {e}", exc_info=True)
                print(f"Failed to fetch {series_id}")
                return series_id, None

        results = {}
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_series = {executor.submit(fetch_job, series_id): series_id
                                for series_id in all_series}
            for future in as_completed(future_to_series):
                series_id, data = future.result()
                if data is not None:
                    results[series_id] = data
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.index.name = 'date'
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
                df = yf.download(tickers = ticker, start = start_date, end = end_date, progress=False)
                if not df.empty:
                    data_dict[ticker] = df['Close']
            except Exception as e:
                exception_logger.error(
                    f"Failed to fetch ticker '{ticker}': {type(e).__name__} - {e}", exc_info=True
                )
                print(f"Could not fetch {ticker}")

        if data_dict:
            df = pd.concat(data_dict.values(), axis=1)
            df.columns = data_dict.keys()
            df.index.name = 'date'
            return df.reset_index()
        return pd.DataFrame()

    def build_raw_dataset(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Build a complete raw dataset (before feature engineering).

        Args:
            start_date (str): Start date (YYYY-MM-DD). Defaults to config value.
            end_date (str): End date (YYYY-MM-DD). Defaults to today.

        Returns:
            pd.DataFrame: Merged DataFrame with FRED and market data (yfinance).
        """
        if start_date is None:
            start_date = self.start_date
        elif isinstance(start_date, str) and datetime.strptime(start_date, "%Y-%m-%d"):
            start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days = 365*5)
            start_date = datetime.strftime(start_date, "%Y-%m-%d")
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        fred_df = self.fetch_all_fred_data(start_date = start_date, end_date = end_date)
        market_df = self.fetch_market_data(start_date = start_date, end_date = end_date)

        if not fred_df.empty and not market_df.empty:
            df = pd.merge(left = fred_df, right = market_df, on = 'date', how='outer')
        elif not fred_df.empty:
            df = fred_df
        elif not market_df.empty:
            df = market_df
        else:
            exception_logger.error("No data could be fetched from any source")
            raise ValueError("No data could be fetched from any source")

        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date').resample('D').asfreq()
        # Forward fill macro data if there are gaps
        df[[col for col in fred_df.columns if col != 'date']] = df[[col for col in fred_df.columns if col != 'date']].ffill()
        # Drop rows where market data is missing (non-trading days)
        df = df.dropna(subset = [col for col in market_df.columns if col != 'date'], how='all')
        # Forward fill market data if there are gaps on days such as July 4th
        df[[col for col in market_df.columns if col != 'date']] = df[[col for col in market_df.columns if col != 'date']].ffill()
        # Drop any remaining NaNs
        df = df.dropna()
        # Replace incorrect negative values with mean for a given column (does not apply to metrics that can be negative)
        for col in [col for col in df.columns if col not in ['T10Y2Y', 'FEDFUNDS', 'DGS2', 'DGS5', 'DGS10', 'DGS30',
                                                         'IRLTLT01EZM156N', 'IR3TIB01EZM156N', 'ECBDFR', 'CSCICP02EZM460S', 'QXMR368BIS']]:
            mean_val = df.loc[df[col] >= 0, col].mean()
            df.loc[df[col] < 0, col] = mean_val
        df = df.reset_index()
        df.rename(columns=dict(self.fred_config['rates'] + self.fred_config['other'] + self.market_tickers), inplace=True)
        return df

    # Feature Engineering
    def detect_frequency(self, col: str, fred_client: Fred) -> str | None:
        """
        Detect the frequency of a given column based on FRED metadata or market tickers.

        Args:
            col (str): Column name to check.
            fred_client (Fred): FRED API client for metadata retrieval.

        Returns:
            str: Detected frequency ('daily', 'weekly', 'monthly', 'quarterly').
        """
        if col in [tick[0] for tick in self.market_tickers]:
            return 'daily'
        try:
            series_info = fred_client.get_series_info(col)
            freq_short = series_info.get('frequency_short', '').upper()

            freq_map = {
                'D': 'daily',
                'W': 'weekly',
                'M': 'monthly',
                'Q': 'quarterly',
            }

            if freq_short in freq_map:
                return freq_map[freq_short]
        except Exception as e:
            exception_logger.error(f"No frequency detected for {col}: {e}", exc_info=True)
            print(f"No frequency detected for {col}")

    def classify_all_frequencies(self, cols: list[str]) -> dict[str, list[str]]:
        """
        Classify all columns into frequency categories.

        Args:
            cols (list[str]): List of column names to classify.

        Returns:
            dict[str, list[str]]: Dictionary with frequency categories as keys and lists of column names as values.
        """
        frequencies = {'daily': [], 'weekly': [], 'monthly': [], 'quarterly': []}
        for col in cols:
            freq = self.detect_frequency(col = col, fred_client = self.fred)
            for ticker, name in (self.fred_config['other'] + self.market_tickers):
                if col == ticker:
                    frequencies[freq].append(name)
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
        freq_classification = self.classify_all_frequencies(cols = all_cols)

        # Process daily data with returns, volatility, and moving averages
        for col in freq_classification['daily']:
            if col not in df.columns:
                continue
            for period in self.features['daily_return_periods']:
                df[f'{col}_ret_{period}d'] = df[col].pct_change(period)

            returns = df[col].pct_change()
            if returns.std() > 0.001:  # Threshold to avoid computing on near-constant series
                for window in self.features['daily_volatility_windows']:
                    df[f'{col}_vol_{window}d'] = returns.rolling(window).std()
                for window in self.features['daily_ma_windows']:
                    df[f'{col}_ma_{window}d'] = df[col].rolling(window).mean()

        # Process weekly, monthly, and quarterly data with period-based changes
        frequencies = {
            'week': 'W',
            'month': 'M',
            'quarter': 'Q'}
        for freq_name, freq_code in frequencies.items():
            df[f'year_{freq_name}'] = df['date'].dt.to_period(freq_code)
            for col in freq_classification[freq_name + 'ly']:
                if col not in df.columns:
                    continue
                # Get last non-null observation per period
                period_data = df.groupby(f'year_{freq_name}')[col].last()
                for periods, label in [tuple(period) for period in self.features[f'{freq_name}ly_periods']]:
                    change_series = period_data.pct_change(periods)
                    df[f'{col}_chg_{label}'] = df[f'year_{freq_name}'].map(change_series)
        df.drop(['year_month', 'year_quarter', 'year_week'], axis=1, inplace=True, errors='ignore')
        return df

    def create_yield_curve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create yield curve related features.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.

        Returns:
            pd.DataFrame: DataFrame with yield curve features added.
        """
        # USA
        if 'Fed_Funds_Rate' in df.columns:
            treasury_rates = [
                ('yield_30y', 'yield_10y'), # Long-end term premium
                ('yield_5y', 'yield_2y'), # Mid-curve steepness
                ('yield_10y', 'yield_5y') # Back-end steepness
            ]
            for rate1, rate2 in treasury_rates:
                if rate1 in df.columns and rate2 in df.columns:
                    df[f'yield_curve_{rate1[6:]}{rate2[6:]}'] = df[rate1] - df[rate2]

            # Policy stance vs medium term
            if 'yield_5y' in df.columns and 'Fed_Funds_Rate' in df.columns:
                df['policy_spread_5y'] = df['yield_5y'] - df['Fed_Funds_Rate']
            # Distinguishes “expected hikes” vs “cuts”
            if 'yield_2y' in df.columns and 'Fed_Funds_Rate' in df.columns:
                df['policy_expectation_2y'] = df['yield_2y'] - df['Fed_Funds_Rate']
        # Euro Area
        elif 'ECB_Deposit_Rate' in df.columns:
            # Long vs short term expectations
            if 'yield_10y' in df.columns and 'interbank_3m' in df.columns:
                df['yield_curve_10y_3m'] = (df['yield_10y'] - df['interbank_3m'])
            # Policy restrictiveness vs long end
            if 'yield_10y' in df.columns and 'ECB_Deposit_Rate' in df.columns:
                df['policy_spread_10y'] = (df['yield_10y'] - df['ECB_Deposit_Rate'])
            # Market expectations vs ECB stance
            if 'interbank_3m' in df.columns and 'ECB_Deposit_Rate' in df.columns:
                df['policy_expectation_3m'] = (df['interbank_3m'] - df['ECB_Deposit_Rate'])
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
        all_cols = [tick[0] for tick in (self.fred_config['other'] + self.market_tickers)]
        df = self.create_features(df = df, all_cols = all_cols)
        df = self.create_yield_curve_features(df = df)
        df = df.dropna()
        return df