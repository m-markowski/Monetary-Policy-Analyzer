import yaml
import os
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ~\Monetary_Policy_Analyzer
YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

load_dotenv(dotenv_path = ENV_PATH)

class EconomyConfig:
    """
    Load and store economy-specific configuration settings.

    Attributes:
        economy (str): Normalized economy identifier (lowercase).
        features (dict): Periods for feature engineering.
        fred (Fred): Initialized FRED API client.
        fred_config (dict): FRED features tickers for the economy.
        market_tickers (list[str]): Market tickers associated with the economy. Obtained from yfinance.
        start_date (str): Default start date for modeling, adjusted by subtracting 5 years.
    """
    def __init__(self, economy: str = 'USA'):
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
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        self.fred_config = config["economy"][self.economy]["fred_features"]
        self.market_tickers = config["economy"][self.economy]["market_tickers"]
        self.start_date = datetime.strftime((datetime.strptime(config['modeling']['default_start_date'],
                                                               "%Y-%m-%d") - timedelta(days=365*5)), "%Y-%m-%d")