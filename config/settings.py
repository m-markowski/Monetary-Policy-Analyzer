import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fredapi import Fred

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ~\Monetary_Policy_Analyzer
YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

load_dotenv(dotenv_path=ENV_PATH)


class EconomyConfig:
    """
    Load and store economy-specific configuration settings.

    Attributes:
        economy (str): Normalized economy identifier (lowercase).
        features (dict): Periods/windows for feature engineering.
        staleness_tolerance (dict): Max observation age (days) per frequency.
        fred (Fred): Initialized FRED API client.
        fred_config (dict): FRED feature IDs grouped into 'rates' and 'other'.
        market_tickers (list[list[str]]): [ticker, friendly_name] pairs from yfinance.
        start_date (str): Default start date for modeling.
    """

    def __init__(self, economy: str = "USA"):
        """
        Initialize economy configuration and FRED client.

        Args:
            economy (str): Economy identifier aligned with YAML config keys.

        Raises:
            KeyError: If the economy is not defined in the YAML configuration.
            FileNotFoundError: If the YAML configuration file cannot be found.
        """
        self.economy = economy.lower()
        with open(YAML_PATH, "r") as f:
            config = yaml.safe_load(f)
        self.features = config["features"]
        self.staleness_tolerance = config.get("staleness_tolerance_days", {})
        self.fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        self.fred_config = config["economy"][self.economy]["fred_features"]
        self.market_tickers = config["economy"][self.economy]["market_tickers"]
        self.start_date = config["modeling"]["default_start_date"]
