"""
Brazilian REH Analyzer - A comprehensive econometric analysis tool for Brazilian inflation forecasts.

This package provides automated, reproducible analysis of Brazilian inflation forecast
rationality using real-time data from the Central Bank of Brazil (BCB).
"""

__version__ = "1.0.0"
__author__ = "KoscheiiB"
__email__ = "KoscheiiB@users.noreply.github.com"
__description__ = "Econometric analysis tool for assessing Brazilian Focus Bulletin inflation forecast rationality"

# Import main analyzer class
from .analyzer import BrazilianREHAnalyzer

# Import data handling components
from .data_fetcher import RespectfulBCBClient, DataCache

# Import utility functions
from .utils import rate_limit_decorator

__all__ = [
    "BrazilianREHAnalyzer",
    "RespectfulBCBClient",
    "DataCache",
    "rate_limit_decorator",
]
