"""
Production-Ready REH Analyzer with Data Caching and Bug Fixes
Fixed version with proper data alignment, caching, and scalar value handling
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import warnings
from datetime import datetime, timedelta
import logging
import time
import random
import requests
from functools import wraps
import os
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from bcb import sgs, Expectativas

    BCB_AVAILABLE = True
except ImportError:
    logger.warning("python-bcb not installed. Install with: pip install python-bcb")
    BCB_AVAILABLE = False


def rate_limit_decorator(
    min_delay=1.0,
    max_delay=3.0,
    requests_per_batch=10,
    batch_delay_min=10,
    batch_delay_max=20,
):
    """
    Decorator to add respectful rate limiting to API calls
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Random delay between requests
            delay = random.uniform(min_delay, max_delay)
            logger.debug(f"Rate limiting: waiting {delay:.2f} seconds...")
            time.sleep(delay)

            # Check if we need a longer break (every N requests)
            if not hasattr(wrapper, "request_count"):
                wrapper.request_count = 0

            wrapper.request_count += 1

            if wrapper.request_count % requests_per_batch == 0:
                batch_delay = random.uniform(batch_delay_min, batch_delay_max)
                logger.info(
                    f"Batch break: waiting {batch_delay:.1f} seconds after {wrapper.request_count} requests..."
                )
                time.sleep(batch_delay)

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Add delay even on errors to be respectful
                error_delay = random.uniform(2.0, 5.0)
                logger.warning(
                    f"Error occurred, waiting {error_delay:.1f} seconds before continuing..."
                )
                time.sleep(error_delay)
                raise

        return wrapper

    return decorator


class DataCache:
    """
    Handles caching of fetched data to avoid repeated API calls
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_filename(
        self, data_type: str, start_date: str, end_date: str
    ) -> Path:
        """Generate cache filename based on data type and date range"""
        safe_start = start_date.replace("-", "")
        safe_end = end_date.replace("-", "")
        timestamp = datetime.now().strftime("%Y%m%d")
        return (
            self.cache_dir / f"{data_type}_{safe_start}_to_{safe_end}_{timestamp}.pkl"
        )

    def _find_existing_cache(
        self, data_type: str, start_date: str, end_date: str, max_age_hours: int = 24
    ) -> Optional[Path]:
        """Find existing cache file that covers the date range and is recent enough"""
        pattern = f"{data_type}_*_to_*.pkl"

        for cache_file in self.cache_dir.glob(pattern):
            # Check if file is recent enough
            file_age = datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime
            )
            if file_age.total_seconds() > max_age_hours * 3600:
                continue

            # Parse filename to check date coverage
            try:
                parts = cache_file.stem.split("_")
                if len(parts) >= 6:
                    cached_start = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
                    cached_end = f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]}"

                    # Check if cached range covers requested range
                    if cached_start <= start_date and cached_end >= end_date:
                        return cache_file
            except (ValueError, IndexError):
                continue

        return None

    def load_data(
        self, data_type: str, start_date: str, end_date: str, max_age_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if available and recent"""
        cache_file = self._find_existing_cache(
            data_type, start_date, end_date, max_age_hours
        )

        if cache_file is None:
            logger.info(f"No valid cache found for {data_type}")
            return None

        try:
            logger.info(f"Loading {data_type} from cache: {cache_file.name}")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)

            # Filter to exact date range requested
            if isinstance(data, pd.DataFrame) and "Data" in data.columns:
                data = data[(data["Data"] >= start_date) & (data["Data"] <= end_date)]
            elif isinstance(data, pd.Series):
                data = data[(data.index >= start_date) & (data.index <= end_date)]

            logger.info(f"Loaded {len(data)} records from cache")
            return data

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
            return None

    def save_data(
        self, data: pd.DataFrame, data_type: str, start_date: str, end_date: str
    ):
        """Save data to cache"""
        cache_file = self._get_cache_filename(data_type, start_date, end_date)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved {data_type} to cache: {cache_file.name}")

            # Clean up old cache files for this data type
            self._cleanup_old_cache(data_type, keep_latest=3)

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _cleanup_old_cache(self, data_type: str, keep_latest: int = 3):
        """Remove old cache files, keeping only the most recent ones"""
        pattern = f"{data_type}_*.pkl"
        cache_files = list(self.cache_dir.glob(pattern))

        if len(cache_files) > keep_latest:
            # Sort by modification time, newest first
            cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for old_file in cache_files[keep_latest:]:
                try:
                    old_file.unlink()
                    logger.debug(f"Removed old cache file: {old_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove old cache {old_file}: {e}")


class RespectfulBCBClient:
    """
    Wrapper for BCB API calls with built-in rate limiting and error handling
    """

    def __init__(self):
        self.session = requests.Session()
        # Add browser-like headers to avoid blocks
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "cross-site",
            }
        )

    @rate_limit_decorator(min_delay=0.8, max_delay=2.5, requests_per_batch=8)
    def get_sgs_data(self, series_code, start_date, end_date):
        """Rate-limited SGS data fetching"""
        return sgs.get(series_code, start=start_date, end=end_date)

    @rate_limit_decorator(min_delay=1.5, max_delay=4.0, requests_per_batch=5)
    def get_expectations_data(self, endpoint_name, query_params):
        """Rate-limited Expectations data fetching"""
        em = Expectativas()
        ep = em.get_endpoint(endpoint_name)

        # Build query step by step
        query = ep.query()

        # Apply filters
        if "filters" in query_params:
            for filter_condition in query_params["filters"]:
                query = query.filter(filter_condition)

        # Apply select
        if "select" in query_params:
            query = query.select(*query_params["select"])

        # Apply orderby
        if "orderby" in query_params:
            query = query.orderby(query_params["orderby"])

        # Apply limit
        if "limit" in query_params:
            query = query.limit(query_params["limit"])

        return query.collect()


class BrazilianREHAnalyzer:
    """
    Enhanced REH Analyzer with data caching, proper temporal alignment,
    and respectful API usage
    """

    def __init__(
        self,
        start_date: str = "2017-01-01",
        end_date: str = "2024-12-31",
        cache_dir: str = "data_cache",
    ):
        """
        Initialize analyzer with date range and caching

        Parameters:
        - start_date: Analysis start date (YYYY-MM-DD)
        - end_date: Analysis end date (YYYY-MM-DD)
        - cache_dir: Directory for data caching
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.ipca_data = None
        self.focus_data = None
        self.aligned_data = None
        self.forecast_errors = None
        self.results = {}
        self.bcb_client = RespectfulBCBClient()
        self.cache = DataCache(cache_dir)

    def _ensure_scalar(self, value) -> float:
        """
        Ensure a value is a scalar float, not a Series or other pandas object
        """
        if isinstance(value, pd.Series):
            if len(value) == 1:
                return float(value.iloc[0])
            else:
                raise ValueError(
                    f"Expected single value, got Series with {len(value)} values"
                )
        elif isinstance(value, (np.ndarray, list)) and len(value) == 1:
            return float(value[0])
        else:
            return float(value)

    def fetch_ipca_data(self, force_refresh: bool = False) -> pd.Series:
        """
        Fetch IPCA 12-month accumulated data from BCB with caching
        """
        if not BCB_AVAILABLE:
            raise ImportError(
                "python-bcb library required. Install with: pip install python-bcb"
            )

        start_str = self.start_date.strftime("%Y-%m-%d")
        end_str = self.end_date.strftime("%Y-%m-%d")

        # Try to load from cache first
        if not force_refresh:
            cached_data = self.cache.load_data(
                "ipca", start_str, end_str, max_age_hours=168
            )  # 1 week
            if cached_data is not None:
                self.ipca_data = cached_data
                return cached_data

        logger.info("Fetching IPCA 12-month accumulated data from BCB...")

        try:
            # Use rate-limited client
            ipca_raw = self.bcb_client.get_sgs_data(433, start_str, end_str)

            # Clean and format data
            ipca_clean = ipca_raw.dropna()
            ipca_clean.name = "IPCA_12m"

            # Convert to monthly frequency using 'ME' instead of deprecated 'M'
            ipca_monthly = ipca_clean.resample("ME").last()

            # Save to cache
            self.cache.save_data(ipca_monthly, "ipca", start_str, end_str)

            self.ipca_data = ipca_monthly
            logger.info(f"Fetched {len(ipca_monthly)} IPCA observations")

            return ipca_monthly

        except Exception as e:
            logger.error(f"Error fetching IPCA data: {e}")
            raise

    def fetch_focus_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch Focus Bulletin IPCA expectations data with caching
        """
        if not BCB_AVAILABLE:
            raise ImportError(
                "python-bcb library required. Install with: pip install python-bcb"
            )

        start_str = self.start_date.strftime("%Y-%m-%d")
        end_str = self.end_date.strftime("%Y-%m-%d")

        # Try to load from cache first
        if not force_refresh:
            cached_data = self.cache.load_data(
                "focus", start_str, end_str, max_age_hours=24
            )  # 1 day
            if cached_data is not None:
                self.focus_data = cached_data
                return cached_data

        logger.info("Fetching Focus Bulletin expectations data from BCB...")

        try:
            # First, let's check available endpoints
            em = Expectativas()
            logger.info("Available endpoints:")
            em.describe()

            # Use the correct endpoint for 12-month inflation expectations
            ep = em.get_endpoint("ExpectativasMercadoInflacao12Meses")

            logger.info("Endpoint properties:")
            em.describe("ExpectativasMercadoInflacao12Meses")

            # Query with proper filters using the rate-limited client
            logger.info("Fetching data in batches to respect API limits...")

            # Split date range into chunks to avoid overwhelming the API
            date_ranges = self._split_date_range(
                self.start_date, self.end_date, months=12
            )
            all_data = []

            for start_chunk, end_chunk in date_ranges:
                logger.info(
                    f"Fetching data for {start_chunk.date()} to {end_chunk.date()}"
                )

                try:
                    # Build query for this chunk
                    chunk_data = (
                        ep.query()
                        .filter(ep.Indicador == "IPCA")
                        .filter(ep.Data >= start_chunk.strftime("%Y-%m-%d"))
                        .filter(ep.Data <= end_chunk.strftime("%Y-%m-%d"))
                        .filter(ep.Suavizada == "N")  # Non-smoothed
                        .select(ep.Data, ep.Mediana, ep.Media, ep.numeroRespondentes)
                        .orderby(ep.Data.desc())
                        .collect()
                    )

                    if len(chunk_data) > 0:
                        all_data.append(chunk_data)
                        logger.info(f"  -> Got {len(chunk_data)} records")
                    else:
                        logger.warning(f"  -> No data for this period")

                    # Rate limiting between chunks
                    time.sleep(random.uniform(2.0, 4.0))

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch chunk {start_chunk.date()}-{end_chunk.date()}: {e}"
                    )
                    time.sleep(random.uniform(5.0, 10.0))  # Longer delay on error
                    continue

            if not all_data:
                raise ValueError("No Focus data could be retrieved")

            # Combine all chunks
            focus_raw = pd.concat(all_data, ignore_index=True)

            # Convert dates and clean data
            focus_raw["Data"] = pd.to_datetime(focus_raw["Data"])

            # Filter for quality (minimum number of respondents)
            focus_filtered = focus_raw[focus_raw["numeroRespondentes"] >= 10]

            # Remove duplicates and take latest for each date
            focus_filtered = focus_filtered.drop_duplicates(
                subset=["Data"], keep="last"
            )

            # Sort by date
            focus_filtered = focus_filtered.sort_values("Data").set_index("Data")

            # Save to cache
            self.cache.save_data(focus_filtered, "focus", start_str, end_str)

            self.focus_data = focus_filtered
            logger.info(
                f"Successfully fetched {len(focus_filtered)} Focus observations"
            )

            return focus_filtered

        except Exception as e:
            logger.error(f"Error fetching Focus data: {e}")
            raise

    def _split_date_range(self, start_date, end_date, months=12):
        """Split date range into smaller chunks to be respectful to the API"""
        ranges = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + pd.DateOffset(months=months), end_date)
            ranges.append((current_start, current_end))
            current_start = current_end + pd.DateOffset(days=1)

        return ranges

    def align_forecast_realization_data(self) -> pd.DataFrame:
        """
        Properly align forecasts with their corresponding realizations
        FIXED: Now properly extracts scalar values from Series objects
        """
        if self.ipca_data is None or self.focus_data is None:
            raise ValueError("Must fetch both IPCA and Focus data first")

        logger.info("Aligning forecast and realization data...")

        # Data validation
        logger.info(
            f"IPCA data type: {type(self.ipca_data)}, shape: {self.ipca_data.shape}"
        )
        logger.info(
            f"Focus data type: {type(self.focus_data)}, shape: {self.focus_data.shape}"
        )

        aligned_pairs = []
        skipped_pairs = 0

        for forecast_date, forecast_row in self.focus_data.iterrows():
            try:
                # For 12-month ahead forecasts, the realization date is 12 months later
                realization_date = forecast_date + pd.DateOffset(months=12)

                # Find the corresponding IPCA realization (closest month-end)
                realization_month = realization_date.to_period("M").to_timestamp("M")

                # Look for IPCA data in that month (allowing some flexibility for release dates)
                ipca_candidates = self.ipca_data[
                    (self.ipca_data.index >= realization_month)
                    & (
                        self.ipca_data.index
                        <= realization_month + pd.DateOffset(days=15)
                    )
                ]

                if not ipca_candidates.empty:
                    # FIXED: Properly extract scalar value from Series
                    realized_ipca_value = self._ensure_scalar(ipca_candidates.iloc[0])

                    # FIXED: Ensure all values are scalars
                    aligned_pairs.append(
                        {
                            "forecast_date": forecast_date,
                            "realization_date": realization_month,
                            "forecast": self._ensure_scalar(forecast_row["Mediana"]),
                            "realized": realized_ipca_value,
                            "respondents": self._ensure_scalar(
                                forecast_row["numeroRespondentes"]
                            ),
                            "forecast_mean": self._ensure_scalar(
                                forecast_row.get("Media", np.nan)
                            ),
                        }
                    )
                else:
                    skipped_pairs += 1

            except Exception as e:
                logger.warning(f"Failed to align data for {forecast_date}: {e}")
                skipped_pairs += 1
                continue

        if not aligned_pairs:
            raise ValueError("No forecast-realization pairs could be aligned")

        aligned_df = pd.DataFrame(aligned_pairs)
        aligned_df = aligned_df.set_index("forecast_date").sort_index()

        # Calculate forecast errors - now with proper scalar values
        aligned_df["forecast_error"] = aligned_df["realized"] - aligned_df["forecast"]

        # Validate that all values are indeed scalars
        for col in ["forecast", "realized", "forecast_error"]:
            if not pd.api.types.is_numeric_dtype(aligned_df[col]):
                raise ValueError(f"Column {col} contains non-numeric values")

        self.aligned_data = aligned_df
        self.forecast_errors = aligned_df["forecast_error"]

        logger.info(
            f"Successfully aligned {len(aligned_df)} forecast-realization pairs"
        )
        logger.info(f"Skipped {skipped_pairs} pairs due to missing IPCA data")
        logger.info(
            f"Date range: {aligned_df.index.min().date()} to {aligned_df.index.max().date()}"
        )

        return aligned_df

    def comprehensive_analysis(
        self,
        fetch_data: bool = True,
        force_refresh: bool = False,
        external_vars: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run complete enhanced analysis with real Brazilian data and caching
        """
        if fetch_data:
            logger.info("Starting comprehensive analysis with real Brazilian data...")
            if not force_refresh:
                logger.info("Checking cache for existing data...")
            else:
                logger.info("Force refresh enabled - fetching fresh data...")

            self.fetch_ipca_data(force_refresh=force_refresh)
            self.fetch_focus_data(force_refresh=force_refresh)
            self.align_forecast_realization_data()

        if self.aligned_data is None:
            raise ValueError(
                "No aligned data available. Set fetch_data=True or load data manually."
            )

        # Run all analyses
        results = {}

        # Basic descriptive statistics - FIXED: Using .item() to get scalars
        results["descriptive_stats"] = {
            "forecast_mean": self.aligned_data["forecast"].mean(),
            "realized_mean": self.aligned_data["realized"].mean(),
            "error_mean": self.forecast_errors.mean(),
            "error_std": self.forecast_errors.std(),
            "error_min": self.forecast_errors.min(),
            "error_max": self.forecast_errors.max(),
            "n_observations": len(self.aligned_data),
            "date_range": f"{self.aligned_data.index.min().date()} to {self.aligned_data.index.max().date()}",
            "mean_respondents": self.aligned_data["respondents"].mean(),
        }

        logger.info("Running Mincer-Zarnowitz test...")
        # Mincer-Zarnowitz test
        try:
            X = sm.add_constant(self.aligned_data["forecast"])
            mz_model = sm.OLS(self.aligned_data["realized"], X).fit()

            # Joint test: (α, β) = (0, 1)
            restrictions = "const = 0, forecast = 1"
            joint_test = mz_model.f_test(restrictions)

            results["mincer_zarnowitz"] = {
                "alpha": float(mz_model.params["const"]),
                "beta": float(mz_model.params["forecast"]),
                "alpha_pvalue": float(mz_model.pvalues["const"]),
                "beta_pvalue": float(mz_model.pvalues["forecast"]),
                "joint_test_fstat": float(joint_test.fvalue[0][0]),
                "joint_test_pvalue": float(joint_test.pvalue),
                "r_squared": float(mz_model.rsquared),
                "durbin_watson": float(
                    sm.stats.stattools.durbin_watson(mz_model.resid)
                ),
            }
        except Exception as e:
            logger.error(f"Mincer-Zarnowitz test failed: {e}")
            results["mincer_zarnowitz"] = {"error": str(e)}

        logger.info("Running autocorrelation tests...")
        # Autocorrelation test
        try:
            errors_clean = self.forecast_errors.dropna()
            max_lags = min(10, len(errors_clean) // 4)

            if max_lags > 0:
                lb_test = acorr_ljungbox(errors_clean, lags=max_lags, return_df=True)

                results["autocorrelation"] = {
                    "ljung_box_stat": (
                        float(lb_test["lb_stat"].iloc[-1])
                        if not lb_test.empty
                        else np.nan
                    ),
                    "ljung_box_pvalue": (
                        float(lb_test["lb_pvalue"].iloc[-1])
                        if not lb_test.empty
                        else np.nan
                    ),
                    "significant_autocorr": (
                        bool((lb_test["lb_pvalue"] < 0.05).any())
                        if not lb_test.empty
                        else False
                    ),
                    "max_lags_tested": int(max_lags),
                }
            else:
                results["autocorrelation"] = {
                    "error": "Insufficient data for autocorrelation test"
                }
        except Exception as e:
            logger.error(f"Autocorrelation test failed: {e}")
            results["autocorrelation"] = {"error": str(e)}

        logger.info("Running bias tests...")
        # Simple bias test (Holden-Peel)
        try:
            errors_clean = self.forecast_errors.dropna()

            if len(errors_clean) > 5:
                # Test if mean error is significantly different from zero
                t_stat, p_value = stats.ttest_1samp(errors_clean, 0)

                results["bias_test"] = {
                    "mean_error": float(errors_clean.mean()),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "is_biased": bool(p_value < 0.05),
                    "bias_direction": (
                        "overestimation"
                        if errors_clean.mean() < 0
                        else "underestimation"
                    ),
                }
            else:
                results["bias_test"] = {"error": "Insufficient data for bias test"}
        except Exception as e:
            logger.error(f"Bias test failed: {e}")
            results["bias_test"] = {"error": str(e)}

        # Overall rationality assessment
        try:
            mz_rational = (
                results.get("mincer_zarnowitz", {}).get("joint_test_pvalue", 0) > 0.05
            )
            no_autocorr = not results.get("autocorrelation", {}).get(
                "significant_autocorr", True
            )
            unbiased = not results.get("bias_test", {}).get("is_biased", True)

            results["rationality_assessment"] = {
                "unbiased": unbiased,
                "mz_rational": mz_rational,
                "efficient": no_autocorr,
                "overall_rational": unbiased and mz_rational and no_autocorr,
                "assessment_date": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Rationality assessment failed: {e}")
            results["rationality_assessment"] = {"error": str(e)}

        self.results = results
        logger.info("Comprehensive analysis completed!")

        return results

    def plot_enhanced_diagnostics(self):
        """
        Enhanced diagnostic plots for Brazilian data analysis
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Forecast vs Realization scatter
        axes[0, 0].scatter(
            self.aligned_data["forecast"],
            self.aligned_data["realized"],
            alpha=0.7,
            s=50,
            color="blue",
            edgecolors="darkblue",
        )
        min_val = min(
            self.aligned_data["forecast"].min(), self.aligned_data["realized"].min()
        )
        max_val = max(
            self.aligned_data["forecast"].max(), self.aligned_data["realized"].max()
        )
        axes[0, 0].plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Forecast Line",
        )
        axes[0, 0].set_xlabel("Focus Median Forecast (%)")
        axes[0, 0].set_ylabel("Realized IPCA 12m (%)")
        axes[0, 0].set_title("Focus Forecasts vs Realized IPCA")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = self.aligned_data["forecast"].corr(self.aligned_data["realized"])
        axes[0, 0].text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=axes[0, 0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Plot 2: Forecast errors over time
        axes[0, 1].plot(
            self.forecast_errors.index,
            self.forecast_errors.values,
            linewidth=2,
            color="blue",
            marker="o",
            markersize=4,
        )
        axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        axes[0, 1].fill_between(
            self.forecast_errors.index,
            self.forecast_errors.values,
            0,
            alpha=0.3,
            color="lightblue",
        )
        axes[0, 1].set_xlabel("Forecast Date")
        axes[0, 1].set_ylabel("Forecast Error (p.p.)")
        axes[0, 1].set_title("Forecast Errors Over Time")
        axes[0, 1].grid(True, alpha=0.3)

        # Add mean error line
        mean_error = self.forecast_errors.mean()
        axes[0, 1].axhline(
            y=mean_error,
            color="green",
            linestyle=":",
            label=f"Mean Error: {mean_error:.2f} p.p.",
        )
        axes[0, 1].legend()

        # Plot 3: Error distribution
        axes[0, 2].hist(
            self.forecast_errors.dropna(),
            bins=20,
            alpha=0.7,
            density=True,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 2].axvline(
            x=0, color="red", linestyle="--", linewidth=2, label="Zero Error"
        )
        axes[0, 2].axvline(
            x=self.forecast_errors.mean(),
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {self.forecast_errors.mean():.2f}",
        )

        # Add normal distribution overlay
        x = np.linspace(self.forecast_errors.min(), self.forecast_errors.max(), 100)
        normal_dist = stats.norm.pdf(
            x, self.forecast_errors.mean(), self.forecast_errors.std()
        )
        axes[0, 2].plot(x, normal_dist, "g-", linewidth=2, label="Normal Distribution")

        axes[0, 2].set_xlabel("Forecast Error (p.p.)")
        axes[0, 2].set_ylabel("Density")
        axes[0, 2].set_title("Distribution of Forecast Errors")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Rolling statistics
        window = min(12, len(self.forecast_errors) // 3)  # 1-year rolling window
        if window >= 3:
            rolling_mean = self.forecast_errors.rolling(
                window=window, center=True
            ).mean()
            rolling_std = self.forecast_errors.rolling(window=window, center=True).std()

            axes[1, 0].plot(
                rolling_mean.index,
                rolling_mean.values,
                label="Rolling Mean",
                linewidth=2,
                color="blue",
            )
            axes[1, 0].fill_between(
                rolling_mean.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.3,
                color="lightblue",
                label="±1 Std Dev",
            )
            axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
            axes[1, 0].set_xlabel("Date")
            axes[1, 0].set_ylabel("Rolling Error Statistics")
            axes[1, 0].set_title(f"Rolling Statistics ({window}-month window)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Q-Q plot for normality check
        from scipy import stats

        stats.probplot(self.forecast_errors.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot: Forecast Errors vs Normal Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Summary statistics
        axes[1, 2].axis("off")

        # Create summary text
        summary_stats = self.results.get("descriptive_stats", {})
        rationality = self.results.get("rationality_assessment", {})

        summary_text = f"""
SUMMARY STATISTICS
──────────────────
Observations: {summary_stats.get('n_observations', 'N/A')}
Date Range: {summary_stats.get('date_range', 'N/A')}

Mean Forecast Error: {summary_stats.get('error_mean', 0):.3f} p.p.
Error Std Dev: {summary_stats.get('error_std', 0):.3f} p.p.
Error Min/Max: {summary_stats.get('error_min', 0):.2f} / {summary_stats.get('error_max', 0):.2f} p.p.

RATIONALITY TESTS
─────────────────
Unbiased: {'✓' if rationality.get('unbiased', False) else '✗'}
MZ Test Passed: {'✓' if rationality.get('mz_rational', False) else '✗'}
Efficient: {'✓' if rationality.get('efficient', False) else '✗'}
Overall Rational: {'✓' if rationality.get('overall_rational', False) else '✗'}

Average Respondents: {summary_stats.get('mean_respondents', 0):.0f}
        """

        axes[1, 2].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def export_results_summary(self, filename: str = "reh_analysis_summary.txt"):
        """Export a human-readable summary of results"""
        if not self.results:
            raise ValueError(
                "No results to export. Run comprehensive_analysis() first."
            )

        with open(filename, "w") as f:
            f.write("RATIONAL EXPECTATIONS HYPOTHESIS ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            desc_stats = self.results.get("descriptive_stats", {})
            f.write(f"Analysis Period: {desc_stats.get('date_range', 'N/A')}\n")
            f.write(
                f"Number of Observations: {desc_stats.get('n_observations', 'N/A')}\n"
            )
            f.write(
                f"Mean Forecast Error: {desc_stats.get('error_mean', 0):.4f} p.p.\n"
            )
            f.write(
                f"Error Standard Deviation: {desc_stats.get('error_std', 0):.4f} p.p.\n\n"
            )

            if "rationality_assessment" in self.results:
                ra = self.results["rationality_assessment"]
                f.write("RATIONALITY TEST RESULTS:\n")
                f.write(
                    f"Unbiased: {'PASS' if ra.get('unbiased', False) else 'FAIL'}\n"
                )
                f.write(
                    f"MZ Test: {'PASS' if ra.get('mz_rational', False) else 'FAIL'}\n"
                )
                f.write(
                    f"Efficient: {'PASS' if ra.get('efficient', False) else 'FAIL'}\n"
                )
                f.write(
                    f"Overall Rational: {'PASS' if ra.get('overall_rational', False) else 'FAIL'}\n\n"
                )

            # Add detailed test results
            if (
                "mincer_zarnowitz" in self.results
                and "error" not in self.results["mincer_zarnowitz"]
            ):
                mz = self.results["mincer_zarnowitz"]
                f.write("MINCER-ZARNOWITZ TEST DETAILS:\n")
                f.write(
                    f"Alpha (intercept): {mz['alpha']:.6f} (p-value: {mz['alpha_pvalue']:.6f})\n"
                )
                f.write(
                    f"Beta (slope): {mz['beta']:.6f} (p-value: {mz['beta_pvalue']:.6f})\n"
                )
                f.write(f"Joint test p-value: {mz['joint_test_pvalue']:.6f}\n")
                f.write(f"R-squared: {mz['r_squared']:.6f}\n\n")

        logger.info(f"Results summary exported to {filename}")


# Example usage with fixed data handling and caching
if __name__ == "__main__":
    # Initialize analyzer with caching
    analyzer = BrazilianREHAnalyzer(
        start_date="2017-01-01", end_date="2024-12-31", cache_dir="data_cache"
    )

    try:
        logger.info("=" * 70)
        logger.info("RATIONAL EXPECTATIONS HYPOTHESIS ANALYSIS - REAL DATA")
        logger.info("=" * 70)
        logger.info("Note: Data will be cached to avoid repeated API calls")

        # Run comprehensive analysis with caching (set force_refresh=True to fetch fresh data)
        results = analyzer.comprehensive_analysis(fetch_data=True, force_refresh=False)

        # Display key results
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)

        desc_stats = results.get("descriptive_stats", {})
        print(f"Analysis Period: {desc_stats.get('date_range', 'N/A')}")
        print(f"Number of Observations: {desc_stats.get('n_observations', 'N/A')}")
        print(f"Mean Forecast Error: {desc_stats.get('error_mean', 0):.3f} p.p.")
        print(f"Error Standard Deviation: {desc_stats.get('error_std', 0):.3f} p.p.")
        print(f"Average Respondents: {desc_stats.get('mean_respondents', 0):.1f}")

        if "rationality_assessment" in results:
            ra = results["rationality_assessment"]
            print(f"\nRationality Tests:")
            print(f"Unbiased: {'✓ PASS' if ra.get('unbiased', False) else '✗ FAIL'}")
            print(
                f"MZ Test Passed: {'✓ PASS' if ra.get('mz_rational', False) else '✗ FAIL'}"
            )
            print(f"Efficient: {'✓ PASS' if ra.get('efficient', False) else '✗ FAIL'}")
            print(
                f"Overall Rational: {'✓ PASS' if ra.get('overall_rational', False) else '✗ FAIL'}"
            )

        # Display specific test results
        if "mincer_zarnowitz" in results and "error" not in results["mincer_zarnowitz"]:
            mz = results["mincer_zarnowitz"]
            print(f"\nMincer-Zarnowitz Test:")
            print(
                f"  α (intercept): {mz['alpha']:.4f} (p-value: {mz['alpha_pvalue']:.4f})"
            )
            print(f"  β (slope): {mz['beta']:.4f} (p-value: {mz['beta_pvalue']:.4f})")
            print(f"  Joint test p-value: {mz['joint_test_pvalue']:.4f}")
            print(f"  R-squared: {mz['r_squared']:.4f}")

        if "bias_test" in results and "error" not in results["bias_test"]:
            bias = results["bias_test"]
            print(f"\nBias Test:")
            print(f"  Mean error: {bias['mean_error']:.4f} p.p.")
            print(f"  t-statistic: {bias['t_statistic']:.4f}")
            print(f"  p-value: {bias['p_value']:.4f}")
            print(f"  Bias direction: {bias['bias_direction']}")

        # Generate enhanced diagnostic plots
        analyzer.plot_enhanced_diagnostics()

        # Export summary
        analyzer.export_results_summary("reh_analysis_results.txt")

        print(f"\nAnalysis completed successfully!")
        print(f"Data cached in ./data_cache/ directory for future runs")
        print(f"Results summary exported to reh_analysis_results.txt")

    except ImportError:
        print("python-bcb library required for real data access.")
        print("Install with: pip install python-bcb")
        print("Then run this script again.")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Analysis failed: {e}")
        print("Check your internet connection and try again.")
        print("If the error persists, try running with force_refresh=True")
