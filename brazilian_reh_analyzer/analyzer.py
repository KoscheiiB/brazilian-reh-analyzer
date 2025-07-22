"""
Main Brazilian REH Analyzer class for comprehensive econometric analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
from scipy import stats
import warnings

from .data_fetcher import RespectfulBCBClient, DataCache
from .tests import REHTests
from .visualizations import REHVisualizations
from .utils import ensure_scalar, validate_data_types, format_results_summary, ProgressTracker

# Configure logging
logger = logging.getLogger(__name__)


class BrazilianREHAnalyzer:
    """
    Enhanced REH Analyzer with data caching, proper temporal alignment,
    and respectful API usage for Brazilian inflation forecast analysis.
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
        
        # Validate date range
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        
        # Data storage
        self.ipca_data = None
        self.focus_data = None
        self.aligned_data = None
        self.forecast_errors = None
        self.results = {}
        
        # Components
        self.bcb_client = RespectfulBCBClient()
        self.cache = DataCache(cache_dir)
        self.reh_tests = REHTests()
        self.visualizations = REHVisualizations()
        
        logger.info(f"Initialized Brazilian REH Analyzer for period {start_date} to {end_date}")

    def fetch_ipca_data(self, force_refresh: bool = False) -> pd.Series:
        """
        Fetch IPCA 12-month accumulated data from BCB with caching
        
        Parameters:
        - force_refresh: If True, ignore cache and fetch fresh data
        
        Returns:
        - pd.Series: IPCA 12-month accumulated data
        """
        start_str = self.start_date.strftime("%Y-%m-%d")
        end_str = self.end_date.strftime("%Y-%m-%d")

        # Try to load from cache first
        if not force_refresh:
            cached_data = self.cache.load_data(
                "ipca", start_str, end_str, max_age_hours=168  # 1 week
            )
            if cached_data is not None:
                self.ipca_data = cached_data
                return cached_data

        logger.info("Fetching IPCA data from BCB...")
        
        try:
            # Fetch using the BCB client
            ipca_monthly = self.bcb_client.fetch_ipca_data(start_str, end_str)
            
            # Save to cache
            self.cache.save_data(ipca_monthly, "ipca", start_str, end_str)
            
            self.ipca_data = ipca_monthly
            logger.info(f"Successfully fetched {len(ipca_monthly)} IPCA observations")
            
            return ipca_monthly

        except Exception as e:
            logger.error(f"Error fetching IPCA data: {e}")
            raise

    def fetch_focus_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch Focus Bulletin IPCA expectations data with caching
        
        Parameters:
        - force_refresh: If True, ignore cache and fetch fresh data
        
        Returns:
        - pd.DataFrame: Focus Bulletin expectations data
        """
        start_str = self.start_date.strftime("%Y-%m-%d")
        end_str = self.end_date.strftime("%Y-%m-%d")

        # Try to load from cache first
        if not force_refresh:
            cached_data = self.cache.load_data(
                "focus", start_str, end_str, max_age_hours=24  # 1 day
            )
            if cached_data is not None:
                self.focus_data = cached_data
                return cached_data

        logger.info("Fetching Focus data from BCB...")
        
        try:
            # Fetch using the BCB client
            focus_filtered = self.bcb_client.fetch_focus_data(self.start_date, self.end_date)
            
            # Save to cache
            self.cache.save_data(focus_filtered, "focus", start_str, end_str)
            
            self.focus_data = focus_filtered
            logger.info(f"Successfully fetched {len(focus_filtered)} Focus observations")
            
            return focus_filtered

        except Exception as e:
            logger.error(f"Error fetching Focus data: {e}")
            raise

    def align_forecast_realization_data(self) -> pd.DataFrame:
        """
        Properly align forecasts with their corresponding realizations
        
        Returns:
        - pd.DataFrame: Aligned forecast-realization pairs
        """
        if self.ipca_data is None or self.focus_data is None:
            raise ValueError("Must fetch both IPCA and Focus data first")

        logger.info("Aligning forecast and realization data...")
        
        # Progress tracking
        progress = ProgressTracker(len(self.focus_data), "Aligning data")

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
                    & (self.ipca_data.index <= realization_month + pd.DateOffset(days=15))
                ]

                if not ipca_candidates.empty:
                    # Extract scalar values properly
                    realized_ipca_value = ensure_scalar(ipca_candidates.iloc[0])

                    aligned_pairs.append(
                        {
                            "forecast_date": forecast_date,
                            "realization_date": realization_month,
                            "forecast": ensure_scalar(forecast_row["Mediana"]),
                            "realized": realized_ipca_value,
                            "respondents": ensure_scalar(forecast_row["numeroRespondentes"]),
                            "forecast_mean": ensure_scalar(forecast_row.get("Media", np.nan)),
                        }
                    )
                else:
                    skipped_pairs += 1

                progress.update(f"Processed {forecast_date.date()}")

            except Exception as e:
                logger.warning(f"Failed to align data for {forecast_date}: {e}")
                skipped_pairs += 1
                continue

        progress.finish()

        if not aligned_pairs:
            raise ValueError("No forecast-realization pairs could be aligned")

        aligned_df = pd.DataFrame(aligned_pairs)
        aligned_df = aligned_df.set_index("forecast_date").sort_index()

        # Calculate forecast errors
        aligned_df["forecast_error"] = aligned_df["realized"] - aligned_df["forecast"]

        # Validate data types
        validate_data_types(aligned_df, ["forecast", "realized", "forecast_error"])

        self.aligned_data = aligned_df
        self.forecast_errors = aligned_df["forecast_error"]

        logger.info(f"Successfully aligned {len(aligned_df)} forecast-realization pairs")
        logger.info(f"Skipped {skipped_pairs} pairs due to missing IPCA data")
        logger.info(f"Date range: {aligned_df.index.min().date()} to {aligned_df.index.max().date()}")

        return aligned_df

    def comprehensive_analysis(
        self,
        fetch_data: bool = True,
        force_refresh: bool = False,
        external_vars: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run complete enhanced analysis with real Brazilian data and caching
        
        Parameters:
        - fetch_data: Whether to fetch data from APIs
        - force_refresh: Force refresh of cached data
        - external_vars: Optional external variables for orthogonality testing
        
        Returns:
        - Dict: Comprehensive analysis results
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE REH ANALYSIS")
        logger.info("=" * 60)
        
        if fetch_data:
            if not force_refresh:
                logger.info("Checking cache for existing data...")
            else:
                logger.info("Force refresh enabled - fetching fresh data...")

            # Fetch data with progress tracking
            self.fetch_ipca_data(force_refresh=force_refresh)
            self.fetch_focus_data(force_refresh=force_refresh)
            self.align_forecast_realization_data()

        if self.aligned_data is None:
            raise ValueError(
                "No aligned data available. Set fetch_data=True or load data manually."
            )

        # Calculate rich descriptive statistics
        logger.info("Calculating comprehensive descriptive statistics...")
        self.results["rich_descriptive_stats"] = self.calculate_rich_descriptive_stats()
        
        # Keep basic stats for backward compatibility
        self.results["descriptive_stats"] = {
            "forecast_mean": float(self.aligned_data["forecast"].mean()),
            "realized_mean": float(self.aligned_data["realized"].mean()),
            "error_mean": float(self.forecast_errors.mean()),
            "error_std": float(self.forecast_errors.std()),
            "error_min": float(self.forecast_errors.min()),
            "error_max": float(self.forecast_errors.max()),
            "n_observations": int(len(self.aligned_data)),
            "date_range": f"{self.aligned_data.index.min().date()} to {self.aligned_data.index.max().date()}",
            "mean_respondents": float(self.aligned_data["respondents"].mean()),
            "period_info": self.results["rich_descriptive_stats"]["period_info"]
        }

        # Run comprehensive REH tests
        logger.info("Running comprehensive REH test suite...")
        test_results = self.reh_tests.comprehensive_reh_assessment(
            self.aligned_data["forecast"],
            self.aligned_data["realized"],
            external_vars=external_vars,
            max_autocorr_lags=10
        )
        
        # Merge test results into main results
        self.results.update(test_results)
        
        # Run enhanced analyses
        logger.info("Running detailed Mincer-Zarnowitz analysis...")
        self.results["detailed_mincer_zarnowitz"] = self.detailed_mincer_zarnowitz_analysis()
        
        logger.info("Analyzing sub-periods with structural break detection...")
        self.results["sub_period_analysis"] = self.analyze_sub_periods()
        
        logger.info("Running rolling window analysis...")
        self.results["rolling_window_analysis"] = self.rolling_window_analysis()
        
        logger.info("Generating economic interpretation...")
        self.results["economic_interpretation"] = self.generate_economic_interpretation()

        logger.info("=" * 60)
        logger.info("ENHANCED ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Log summary
        ra = self.results.get("rationality_assessment", {})
        logger.info(f"Overall Rational: {'✓' if ra.get('overall_rational', False) else '✗'}")
        logger.info(f"Mean Forecast Error: {self.results['descriptive_stats']['error_mean']:.3f} p.p.")

        return self.results

    def plot_enhanced_diagnostics(self, show_plots: bool = True) -> None:
        """
        Generate enhanced diagnostic plots for Brazilian data analysis
        
        Parameters:
        - show_plots: Whether to display plots immediately
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")

        logger.info("Generating diagnostic plots...")
        
        fig = self.visualizations.create_comprehensive_diagnostics(
            self.aligned_data["forecast"],
            self.aligned_data["realized"],
            self.results
        )
        
        if show_plots:
            fig.show()
        
        return fig

    def calculate_rich_descriptive_stats(self) -> Dict:
        """
        Calculate comprehensive descriptive statistics for any date range
        
        Returns:
        - Dict: Rich descriptive statistics
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")
        
        forecast = self.aligned_data["forecast"]
        realized = self.aligned_data["realized"] 
        errors = self.aligned_data["forecast_error"]
        
        stats_dict = {}
        
        # For each series, calculate comprehensive statistics
        for series_name, series_data in [("forecast", forecast), ("realized", realized), ("errors", errors)]:
            stats_dict[series_name] = {
                "mean": float(series_data.mean()),
                "median": float(series_data.median()),
                "std": float(series_data.std()),
                "min": float(series_data.min()),
                "max": float(series_data.max()),
                "skewness": float(stats.skew(series_data.dropna())),
                "kurtosis": float(stats.kurtosis(series_data.dropna())),
                "q25": float(series_data.quantile(0.25)),
                "q75": float(series_data.quantile(0.75)),
                "n_obs": int(len(series_data.dropna()))
            }
        
        # Additional period information
        stats_dict["period_info"] = {
            "start_date": self.aligned_data.index.min().strftime("%Y-%m-%d"),
            "end_date": self.aligned_data.index.max().strftime("%Y-%m-%d"),
            "period_length_years": float((self.aligned_data.index.max() - self.aligned_data.index.min()).days / 365.25),
            "avg_respondents": float(self.aligned_data["respondents"].mean()),
            "min_respondents": int(self.aligned_data["respondents"].min()),
            "max_respondents": int(self.aligned_data["respondents"].max())
        }
        
        return stats_dict

    def detect_structural_breaks(self, min_segment_size: int = None) -> List[Tuple[str, str]]:
        """
        Automatically detect structural breaks in forecast errors using statistical tests
        Returns periods for flexible sub-period analysis
        
        Parameters:
        - min_segment_size: Minimum observations per segment (auto-calculated if None)
        
        Returns:
        - List[Tuple]: List of (start_date, end_date) for identified periods
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")
        
        data = self.aligned_data.copy()
        errors = data["forecast_error"].dropna()
        
        # Auto-calculate minimum segment size based on data length
        if min_segment_size is None:
            min_segment_size = max(50, len(errors) // 5)  # At least 50 obs or 1/5 of total
        
        # If insufficient data for multiple periods, return single period
        if len(errors) < min_segment_size * 2:
            return [(data.index.min().strftime("%Y-%m-%d"), 
                    data.index.max().strftime("%Y-%m-%d"))]
        
        # Simple structural break detection using rolling variance changes
        window_size = max(min_segment_size // 2, 20)
        rolling_std = errors.rolling(window=window_size, center=True).std()
        rolling_mean = errors.rolling(window=window_size, center=True).mean()
        
        # Detect significant changes in volatility or mean
        std_changes = np.abs(rolling_std.diff()) > rolling_std.std()
        mean_changes = np.abs(rolling_mean.diff()) > errors.std() * 0.5
        
        # Combine change indicators
        change_points = std_changes | mean_changes
        significant_changes = change_points.index[change_points].tolist()
        
        # Filter change points to ensure minimum segment size
        filtered_changes = []
        last_change = data.index.min()
        
        for change_point in significant_changes:
            if (change_point - last_change).days >= min_segment_size * 7:  # Assuming weekly data
                filtered_changes.append(change_point)
                last_change = change_point
        
        # Create periods
        periods = []
        start_date = data.index.min()
        
        for change_point in filtered_changes:
            periods.append((start_date.strftime("%Y-%m-%d"), 
                          change_point.strftime("%Y-%m-%d")))
            start_date = change_point
        
        # Add final period
        periods.append((start_date.strftime("%Y-%m-%d"), 
                       data.index.max().strftime("%Y-%m-%d")))
        
        # If no breaks detected, create equal-length periods
        if len(periods) == 1 and len(errors) > min_segment_size * 3:
            n_periods = min(3, len(errors) // min_segment_size)
            period_length = len(errors) // n_periods
            periods = []
            
            for i in range(n_periods):
                start_idx = i * period_length
                end_idx = (i + 1) * period_length if i < n_periods - 1 else len(errors)
                periods.append((errors.index[start_idx].strftime("%Y-%m-%d"),
                              errors.index[end_idx - 1].strftime("%Y-%m-%d")))
        
        return periods

    def analyze_sub_periods(self) -> Dict:
        """
        Perform flexible sub-period analysis based on detected structural breaks
        
        Returns:
        - Dict: Sub-period analysis results
        """
        periods = self.detect_structural_breaks()
        sub_period_results = {}
        
        for i, (start_date, end_date) in enumerate(periods):
            period_name = f"Period_{i+1}_({start_date}_to_{end_date})"
            
            # Filter data for this period
            period_mask = ((self.aligned_data.index >= pd.to_datetime(start_date)) & 
                          (self.aligned_data.index <= pd.to_datetime(end_date)))
            period_data = self.aligned_data[period_mask]
            
            if len(period_data) < 10:  # Skip periods with insufficient data
                continue
            
            # Calculate period-specific statistics
            forecast = period_data["forecast"]
            realized = period_data["realized"]
            errors = period_data["forecast_error"]
            
            # Basic statistics
            period_stats = {
                "n_observations": len(period_data),
                "start_date": start_date,
                "end_date": end_date,
                "mean_error": float(errors.mean()),
                "std_error": float(errors.std()),
                "bias_direction": "overestimation" if errors.mean() < 0 else "underestimation",
                "mean_forecast": float(forecast.mean()),
                "mean_realized": float(realized.mean())
            }
            
            # Run period-specific REH tests
            try:
                period_reh_results = self.reh_tests.comprehensive_reh_assessment(
                    forecast, realized, max_autocorr_lags=min(10, len(period_data)//4)
                )
                period_stats["reh_tests"] = period_reh_results
            except Exception as e:
                logger.warning(f"Could not run REH tests for {period_name}: {e}")
                period_stats["reh_tests"] = {"error": str(e)}
            
            sub_period_results[period_name] = period_stats
        
        return sub_period_results

    def rolling_window_analysis(self, window_size: int = None) -> Dict:
        """
        Perform rolling window analysis to detect time-varying bias patterns
        
        Parameters:
        - window_size: Size of rolling window (auto-calculated if None)
        
        Returns:
        - Dict: Rolling window analysis results
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")
        
        errors = self.aligned_data["forecast_error"]
        n_obs = len(errors)
        
        # Auto-calculate window size
        if window_size is None:
            window_size = max(50, n_obs // 10)  # At least 50 obs or 1/10 of total
        
        # Calculate rolling statistics
        rolling_mean = errors.rolling(window=window_size, center=True).mean()
        rolling_std = errors.rolling(window=window_size, center=True).std()
        rolling_corr = errors.rolling(window=window_size).apply(
            lambda x: stats.pearsonr(range(len(x)), x)[0] if len(x) == window_size else np.nan
        )
        
        # Detect significant changes
        mean_changes = np.abs(rolling_mean.diff()) > errors.std() * 0.3
        volatility_changes = np.abs(rolling_std.diff()) > errors.std() * 0.2
        
        results = {
            "window_size": window_size,
            "rolling_mean_error": rolling_mean.dropna().to_dict(),
            "rolling_std_error": rolling_std.dropna().to_dict(),
            "rolling_trend_correlation": rolling_corr.dropna().to_dict(),
            "significant_mean_changes": mean_changes.index[mean_changes].strftime("%Y-%m-%d").tolist(),
            "significant_volatility_changes": volatility_changes.index[volatility_changes].strftime("%Y-%m-%d").tolist(),
            "max_abs_bias": float(rolling_mean.abs().max()),
            "min_abs_bias": float(rolling_mean.abs().min()),
            "bias_range": float(rolling_mean.max() - rolling_mean.min())
        }
        
        return results

    def detailed_mincer_zarnowitz_analysis(self) -> Dict:
        """
        Perform detailed Mincer-Zarnowitz regression with rich output
        
        Returns:
        - Dict: Detailed MZ regression results
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")
        
        forecast = self.aligned_data["forecast"].values
        realized = self.aligned_data["realized"].values
        
        # Remove any NaN values
        mask = ~(np.isnan(forecast) | np.isnan(realized))
        forecast_clean = forecast[mask]
        realized_clean = realized[mask]
        
        if len(forecast_clean) < 10:
            return {"error": "Insufficient clean data for MZ regression"}
        
        # Perform regression: realized = alpha + beta * forecast + error
        X = np.column_stack([np.ones(len(forecast_clean)), forecast_clean])
        y = realized_clean
        
        # Manual calculation for more control
        XtX_inv = np.linalg.inv(X.T @ X)
        coeffs = XtX_inv @ X.T @ y
        alpha, beta = coeffs
        
        # Calculate residuals and statistics
        y_pred = X @ coeffs
        residuals = y - y_pred
        n, k = X.shape
        
        # Standard errors
        mse = np.sum(residuals**2) / (n - k)
        std_errors = np.sqrt(np.diag(XtX_inv * mse))
        
        # T-statistics and p-values
        t_stats = coeffs / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # R-squared
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(residuals**2)
        r_squared = 1 - (rss / tss)
        adjusted_r_squared = 1 - ((rss / (n - k)) / (tss / (n - 1)))
        
        # Joint test: H0: alpha=0, beta=1
        # Test statistic: R = [alpha, beta-1]
        R = np.array([[1, 0], [0, 1]])  # Coefficient selection matrix
        r = np.array([0, 1])  # Restriction values
        restriction_vector = R @ coeffs - r
        
        # F-statistic for joint test
        f_stat = (restriction_vector.T @ np.linalg.inv(R @ XtX_inv @ R.T * mse) @ restriction_vector) / k
        joint_p_value = 1 - stats.f.cdf(f_stat, k, n - k)
        
        # Confidence intervals (95%)
        t_critical = stats.t.ppf(0.975, n - k)
        alpha_ci = [alpha - t_critical * std_errors[0], alpha + t_critical * std_errors[0]]
        beta_ci = [beta - t_critical * std_errors[1], beta + t_critical * std_errors[1]]
        
        results = {
            "alpha": float(alpha),
            "beta": float(beta),
            "alpha_std_error": float(std_errors[0]),
            "beta_std_error": float(std_errors[1]),
            "alpha_t_stat": float(t_stats[0]),
            "beta_t_stat": float(t_stats[1]),
            "alpha_p_value": float(p_values[0]),
            "beta_p_value": float(p_values[1]),
            "alpha_95_ci": [float(alpha_ci[0]), float(alpha_ci[1])],
            "beta_95_ci": [float(beta_ci[0]), float(beta_ci[1])],
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adjusted_r_squared),
            "joint_f_statistic": float(f_stat),
            "joint_p_value": float(joint_p_value),
            "mse": float(mse),
            "n_observations": int(n),
            "passes_unbiasedness": joint_p_value > 0.05,
            "alpha_significant": p_values[0] < 0.05,
            "beta_significantly_different_from_1": abs(beta - 1) / std_errors[1] > t_critical
        }
        
        return results

    def generate_economic_interpretation(self) -> Dict:
        """
        Generate flexible economic interpretation that adapts to any time period
        
        Returns:
        - Dict: Economic interpretation and policy implications
        """
        if not self.results or self.aligned_data is None:
            return {"error": "No results available for interpretation"}
        
        stats = self.results.get("descriptive_stats", {})
        ra = self.results.get("rationality_assessment", {})
        mean_error = stats.get("error_mean", 0)
        error_std = stats.get("error_std", 0)
        period_length = stats.get("period_info", {}).get("period_length_years", 0)
        
        # Classify bias magnitude
        if abs(mean_error) < 0.5:
            bias_severity = "minimal"
        elif abs(mean_error) < 1.5:
            bias_severity = "moderate"
        elif abs(mean_error) < 3.0:
            bias_severity = "substantial"
        else:
            bias_severity = "severe"
        
        # Classify autocorrelation
        autocorr_result = self.results.get("autocorrelation", {})
        ljung_box_stat = autocorr_result.get("ljung_box_stat", 0)
        
        if ljung_box_stat < 10:
            autocorr_severity = "minimal"
        elif ljung_box_stat < 100:
            autocorr_severity = "moderate" 
        elif ljung_box_stat < 1000:
            autocorr_severity = "substantial"
        else:
            autocorr_severity = "extreme"
        
        # Generate interpretation
        interpretation = {
            "bias_analysis": {
                "direction": "overestimation" if mean_error < 0 else "underestimation",
                "magnitude_pp": abs(mean_error),
                "severity": bias_severity,
                "economic_significance": "high" if abs(mean_error) > 1.0 else "moderate" if abs(mean_error) > 0.5 else "low"
            },
            "efficiency_analysis": {
                "autocorr_severity": autocorr_severity,
                "ljung_box_statistic": ljung_box_stat,
                "learning_failure": ljung_box_stat > 100,
                "information_processing": "poor" if ljung_box_stat > 100 else "adequate"
            },
            "overall_assessment": {
                "reh_compatibility": "rejected" if not ra.get("overall_rational", False) else "accepted",
                "forecast_quality": self._assess_forecast_quality(mean_error, error_std),
                "period_challenges": self._identify_period_challenges(period_length),
                "primary_failures": self._identify_primary_failures(ra)
            },
            "policy_implications": {
                "central_bank": [
                    f"Focus forecasts show {bias_severity} systematic {'overestimation' if mean_error < 0 else 'underestimation'}",
                    f"Market expectations exhibit {autocorr_severity} autocorrelation, suggesting poor information processing",
                    "Consider enhanced communication strategies to improve expectation formation"
                ],
                "market_participants": [
                    "Systematic biases present opportunities for contrarian strategies",
                    f"Forecast errors are predictable, violating market efficiency",
                    "Consider alternative forecasting models to exploit identified biases"
                ],
                "researchers": [
                    f"REH violations persistent over {period_length:.1f}-year period",
                    "Results suggest adaptive or sticky information models more appropriate",
                    "Structural breaks and learning dynamics warrant further investigation"
                ]
            }
        }
        
        return interpretation

    def _assess_forecast_quality(self, mean_error: float, error_std: float) -> str:
        """Helper function to assess overall forecast quality"""
        rmse = np.sqrt(mean_error**2 + error_std**2)
        if rmse < 0.5:
            return "excellent"
        elif rmse < 1.0:
            return "good"
        elif rmse < 2.0:
            return "moderate"
        else:
            return "poor"

    def _identify_period_challenges(self, period_length: float) -> List[str]:
        """Helper function to identify challenges based on period length"""
        challenges = []
        if period_length >= 8:
            challenges.append("Long analysis period spans multiple economic cycles")
        if period_length >= 5:
            challenges.append("Period likely includes major economic/policy regime changes")
        if period_length <= 2:
            challenges.append("Relatively short period may limit generalizability")
        return challenges

    def _identify_primary_failures(self, ra: Dict) -> List[str]:
        """Helper function to identify main REH failure modes"""
        failures = []
        if not ra.get("unbiased", True):
            failures.append("systematic bias")
        if not ra.get("mz_rational", True):
            failures.append("Mincer-Zarnowitz joint test failure")
        if not ra.get("efficient", True):
            failures.append("autocorrelated forecast errors")
        return failures

    def export_results_summary(self, filename: str = "reh_analysis_summary.txt") -> None:
        """
        Export comprehensive academic-style analysis summary with rich interpretation
        
        Parameters:
        - filename: Output filename for the summary
        """
        if not self.results:
            raise ValueError("No results to export. Run comprehensive_analysis() first.")

        logger.info(f"Exporting enhanced results summary to {filename}...")
        
        with open(filename, "w") as f:
            # Header
            f.write("╔" + "═" * 68 + "╗\n")
            f.write("║" + " " * 68 + "║\n")
            f.write("║" + " BRAZILIAN REH ANALYZER v2.0.0 ".center(68) + "║\n")
            f.write("║" + " ENHANCED ACADEMIC FRAMEWORK ".center(68) + "║\n")
            f.write("║" + " " * 68 + "║\n")
            f.write("╚" + "═" * 68 + "╝\n\n")
            
            # 1. COMPREHENSIVE DESCRIPTIVE STATISTICS
            self._write_descriptive_stats(f)
            
            # 2. RATIONALITY TEST SUMMARY
            self._write_rationality_summary(f)
            
            # 3. DETAILED MINCER-ZARNOWITZ REGRESSION
            self._write_detailed_mz_analysis(f)
            
            # 4. SUB-PERIOD ANALYSIS
            self._write_sub_period_analysis(f)
            
            # 5. ROLLING WINDOW BIAS ANALYSIS
            self._write_rolling_window_analysis(f)
            
            # 6. ECONOMIC INTERPRETATION
            self._write_economic_interpretation(f)
            
            # 7. POLICY IMPLICATIONS
            self._write_policy_implications(f)
            
            # Footer
            f.write("\n" + "═" * 70 + "\n")
            f.write("Generated by Brazilian REH Analyzer - Advanced Econometric Analysis Framework\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═" * 70 + "\n")

        logger.info(f"Enhanced results summary exported to {filename}")

    def _write_descriptive_stats(self, f) -> None:
        """Write comprehensive descriptive statistics section"""
        f.write("1. COMPREHENSIVE DESCRIPTIVE STATISTICS\n")
        f.write("═" * 70 + "\n\n")
        
        rich_stats = self.results.get("rich_descriptive_stats", {})
        period_info = rich_stats.get("period_info", {})
        
        # Period Information
        f.write(f"Analysis Period: {period_info.get('start_date', 'N/A')} to {period_info.get('end_date', 'N/A')}\n")
        f.write(f"Period Length: {period_info.get('period_length_years', 0):.1f} years\n")
        f.write(f"Survey Respondents: {period_info.get('avg_respondents', 0):.0f} average ({period_info.get('min_respondents', 0)} - {period_info.get('max_respondents', 0)} range)\n\n")
        
        # Detailed Statistics Table
        f.write("                   ┌───────────────┬────────────────┬────────────────┐\n")
        f.write("                   │ Observed IPCA │ Focus Forecast │ Forecast Error │\n")
        f.write("                   │      (%)      │       (%)      │     (p.p.)     │\n")
        f.write("───────────────────┼───────────────┼────────────────┼────────────────┤\n")
        
        forecast_stats = rich_stats.get("forecast", {})
        realized_stats = rich_stats.get("realized", {})
        error_stats = rich_stats.get("errors", {})
        
        f.write(f"Mean               │    {realized_stats.get('mean', 0):7.3f}    │     {forecast_stats.get('mean', 0):7.3f}    │     {error_stats.get('mean', 0):7.3f}    │\n")
        f.write(f"Median             │    {realized_stats.get('median', 0):7.3f}    │     {forecast_stats.get('median', 0):7.3f}    │     {error_stats.get('median', 0):7.3f}    │\n")
        f.write(f"Standard Deviation │    {realized_stats.get('std', 0):7.3f}    │     {forecast_stats.get('std', 0):7.3f}    │     {error_stats.get('std', 0):7.3f}    │\n")
        f.write(f"Minimum            │    {realized_stats.get('min', 0):7.3f}    │     {forecast_stats.get('min', 0):7.3f}    │     {error_stats.get('min', 0):7.3f}    │\n")
        f.write(f"Maximum            │    {realized_stats.get('max', 0):7.3f}    │     {forecast_stats.get('max', 0):7.3f}    │     {error_stats.get('max', 0):7.3f}    │\n")
        f.write(f"Skewness           │    {realized_stats.get('skewness', 0):7.3f}    │     {forecast_stats.get('skewness', 0):7.3f}    │     {error_stats.get('skewness', 0):7.3f}    │\n")
        f.write(f"Kurtosis           │    {realized_stats.get('kurtosis', 0):7.3f}    │     {forecast_stats.get('kurtosis', 0):7.3f}    │     {error_stats.get('kurtosis', 0):7.3f}    │\n")
        f.write(f"Observations       │    {realized_stats.get('n_obs', 0):7.0f}    │     {forecast_stats.get('n_obs', 0):7.0f}    │     {error_stats.get('n_obs', 0):7.0f}    │\n")
        f.write("───────────────────┴───────────────┴────────────────┴────────────────┘\n\n")
        
        # Statistical Interpretation
        f.write("STATISTICAL INTERPRETATION:\n")
        mean_error = error_stats.get('mean', 0)
        if mean_error < -1.0:
            f.write("• Systematic OVERESTIMATION bias detected (negative mean error)\n")
        elif mean_error > 1.0:
            f.write("• Systematic UNDERESTIMATION bias detected (positive mean error)\n")
        else:
            f.write("• Minimal systematic bias in forecast errors\n")
        
        error_skew = error_stats.get('skewness', 0)
        if abs(error_skew) > 0.5:
            f.write(f"• Error distribution is {'negatively' if error_skew < 0 else 'positively'} skewed\n")
        
        error_kurtosis = error_stats.get('kurtosis', 0)
        if error_kurtosis > 3:
            f.write("• Fat tails detected - extreme forecast errors more common than normal distribution\n")
        f.write("\n")

    def _write_rationality_summary(self, f) -> None:
        """Write rationality test summary"""
        f.write("2. RATIONALITY TEST SUMMARY\n")
        f.write("═" * 70 + "\n\n")
        
        ra = self.results.get("rationality_assessment", {})
        
        f.write("                   ┌────────────────┬────────────────┬──────────────┐\n")
        f.write("                   │      Test      │     Result     │  Implication │\n")
        f.write("───────────────────┼────────────────┼────────────────┼──────────────┤\n")
        f.write(f"Unbiasedness       │ {'     PASS     ' if ra.get('unbiased', False) else '     FAIL     '} │ {'No systematic' if ra.get('unbiased', False) else 'Systematic  '} │\n")
        f.write(f"Mincer-Zarnowitz   │ {'     PASS     ' if ra.get('mz_rational', False) else '     FAIL     '} │ {'   bias       ' if ra.get('unbiased', False) else 'forecast bias'} │\n")
        f.write(f"Efficiency         │ {'     PASS     ' if ra.get('efficient', False) else '     FAIL     '} │ {'Optimal info ' if ra.get('efficient', False) else 'Poor learning'} │\n")
        f.write(f"Overall REH        │ {'     PASS     ' if ra.get('overall_rational', False) else '     FAIL     '} │ {'   usage      ' if ra.get('efficient', False) else '   patterns  '} │\n")
        f.write("───────────────────┴────────────────┴────────────────┴──────────────┘\n\n")
        
        # Overall Assessment
        if ra.get('overall_rational', False):
            f.write("✅ OVERALL ASSESSMENT: Forecasts are compatible with Rational Expectations Hypothesis\n\n")
        else:
            f.write("❌ OVERALL ASSESSMENT: Forecasts VIOLATE Rational Expectations Hypothesis\n\n")

    def _write_detailed_mz_analysis(self, f) -> None:
        """Write detailed Mincer-Zarnowitz regression results"""
        f.write("3. DETAILED MINCER-ZARNOWITZ REGRESSION ANALYSIS\n")
        f.write("═" * 70 + "\n\n")
        
        detailed_mz = self.results.get("detailed_mincer_zarnowitz", {})
        
        if "error" in detailed_mz:
            f.write(f"ERROR: {detailed_mz['error']}\n\n")
            return
        
        f.write("Regression: Realized = α + β × Forecast + ε\n")
        f.write("Null Hypothesis: H₀: α = 0, β = 1 (rational expectations)\n\n")
        
        f.write("Coefficient    │ Estimate │ Std Error │ t-stat │ p-value │ 95% Confidence Interval\n")
        f.write("───────────────┼──────────┼───────────┼────────┼─────────┼──────────────────────────\n")
        f.write(f"α (Intercept)  │  {detailed_mz.get('alpha', 0):6.3f}  │   {detailed_mz.get('alpha_std_error', 0):6.3f}  │ {detailed_mz.get('alpha_t_stat', 0):6.2f} │ {detailed_mz.get('alpha_p_value', 1):7.4f} │ [{detailed_mz.get('alpha_95_ci', [0,0])[0]:6.3f}, {detailed_mz.get('alpha_95_ci', [0,0])[1]:6.3f}]\n")
        f.write(f"β (Slope)      │  {detailed_mz.get('beta', 0):6.3f}  │   {detailed_mz.get('beta_std_error', 0):6.3f}  │ {detailed_mz.get('beta_t_stat', 0):6.2f} │ {detailed_mz.get('beta_p_value', 1):7.4f} │ [{detailed_mz.get('beta_95_ci', [0,0])[0]:6.3f}, {detailed_mz.get('beta_95_ci', [0,0])[1]:6.3f}]\n")
        f.write("───────────────┴──────────┴───────────┴────────┴─────────┴──────────────────────────\n\n")
        
        f.write(f"R² = {detailed_mz.get('r_squared', 0):.4f}    │    Adjusted R² = {detailed_mz.get('adjusted_r_squared', 0):.4f}    │    Observations = {detailed_mz.get('n_observations', 0):,}\n\n")
        
        f.write("JOINT TEST: H₀: α=0, β=1\n")
        f.write(f"F-statistic = {detailed_mz.get('joint_f_statistic', 0):.4f}    │    p-value = {detailed_mz.get('joint_p_value', 1):.6f}    │    {'ACCEPT H₀' if detailed_mz.get('passes_unbiasedness', False) else 'REJECT H₀'}\n\n")
        
        f.write("ECONOMIC INTERPRETATION:\n")
        beta = detailed_mz.get('beta', 1)
        alpha = detailed_mz.get('alpha', 0)
        
        if detailed_mz.get('alpha_significant', False):
            f.write(f"• α = {alpha:.3f} ≠ 0: Systematic forecast bias present\n")
        if detailed_mz.get('beta_significantly_different_from_1', False):
            f.write(f"• β = {beta:.3f} ≠ 1: Forecasters {'under' if beta < 1 else 'over'}-respond to their own predictions\n")
        if not detailed_mz.get('passes_unbiasedness', False):
            f.write("• Joint test rejection indicates violations of both unbiasedness AND efficiency\n")
        f.write("\n")

    def _write_sub_period_analysis(self, f) -> None:
        """Write sub-period analysis results"""
        f.write("4. SUB-PERIOD ANALYSIS (STRUCTURAL BREAK DETECTION)\n")
        f.write("═" * 70 + "\n\n")
        
        sub_periods = self.results.get("sub_period_analysis", {})
        
        if not sub_periods:
            f.write("No sub-periods detected or analysis failed.\n\n")
            return
        
        f.write("                   │  Period  │   Period   │ Mean Error │ REH Status │\n")
        f.write("                   │  Start   │    End     │    (p.p.)  │  Overall   │\n")
        f.write("───────────────────┼──────────┼────────────┼────────────┼────────────┤\n")
        
        for period_name, period_data in sub_periods.items():
            start_short = period_data.get('start_date', '')[:7]  # YYYY-MM
            end_short = period_data.get('end_date', '')[:7]      # YYYY-MM
            mean_error = period_data.get('mean_error', 0)
            reh_tests = period_data.get('reh_tests', {})
            reh_overall = reh_tests.get('rationality_assessment', {}).get('overall_rational', False)
            
            f.write(f"Period {period_name.split('_')[1]:<11}│ {start_short:>8} │ {end_short:>10} │ {mean_error:>10.3f} │ {'   PASS   ' if reh_overall else '   FAIL   '} │\n")
        
        f.write("───────────────────┴──────────┴────────────┴────────────┴────────────┘\n\n")
        
        f.write("STRUCTURAL BREAK INTERPRETATION:\n")
        periods_list = list(sub_periods.values())
        if len(periods_list) > 1:
            errors = [p.get('mean_error', 0) for p in periods_list]
            max_error = max(errors)
            min_error = min(errors)
            f.write(f"• Bias ranges from {min_error:.3f} to {max_error:.3f} p.p. across sub-periods\n")
            f.write(f"• Total bias variation: {abs(max_error - min_error):.3f} p.p.\n")
            if abs(max_error - min_error) > 1.0:
                f.write("• SUBSTANTIAL time-variation in forecast bias detected\n")
        f.write("\n")

    def _write_rolling_window_analysis(self, f) -> None:
        """Write rolling window analysis results"""
        f.write("5. ROLLING WINDOW BIAS ANALYSIS\n")
        f.write("═" * 70 + "\n\n")
        
        rolling_analysis = self.results.get("rolling_window_analysis", {})
        
        if "error" in rolling_analysis or not rolling_analysis:
            f.write("Rolling window analysis not available.\n\n")
            return
        
        f.write(f"Window Size: {rolling_analysis.get('window_size', 0)} observations\n")
        f.write(f"Maximum Absolute Bias: {rolling_analysis.get('max_abs_bias', 0):.3f} p.p.\n")
        f.write(f"Minimum Absolute Bias: {rolling_analysis.get('min_abs_bias', 0):.3f} p.p.\n")
        f.write(f"Bias Range: {rolling_analysis.get('bias_range', 0):.3f} p.p.\n\n")
        
        mean_changes = rolling_analysis.get('significant_mean_changes', [])
        volatility_changes = rolling_analysis.get('significant_volatility_changes', [])
        
        f.write("SIGNIFICANT STRUCTURAL CHANGES DETECTED:\n")
        if mean_changes:
            f.write(f"• Mean bias changes: {len(mean_changes)} detected\n")
            if len(mean_changes) <= 5:
                f.write(f"  Dates: {', '.join(mean_changes)}\n")
        if volatility_changes:
            f.write(f"• Volatility changes: {len(volatility_changes)} detected\n")
            if len(volatility_changes) <= 5:
                f.write(f"  Dates: {', '.join(volatility_changes)}\n")
        
        if not mean_changes and not volatility_changes:
            f.write("• No significant structural changes detected in rolling analysis\n")
        f.write("\n")

    def _write_economic_interpretation(self, f) -> None:
        """Write economic interpretation section"""
        f.write("6. ECONOMIC INTERPRETATION\n")
        f.write("═" * 70 + "\n\n")
        
        econ_interp = self.results.get("economic_interpretation", {})
        
        if "error" in econ_interp:
            f.write("Economic interpretation not available.\n\n")
            return
        
        bias_analysis = econ_interp.get("bias_analysis", {})
        efficiency_analysis = econ_interp.get("efficiency_analysis", {})
        overall_assessment = econ_interp.get("overall_assessment", {})
        
        f.write("BIAS ANALYSIS:\n")
        f.write(f"• Direction: {bias_analysis.get('direction', 'unknown').upper()}\n")
        f.write(f"• Magnitude: {bias_analysis.get('magnitude_pp', 0):.3f} percentage points\n")
        f.write(f"• Severity: {bias_analysis.get('severity', 'unknown').upper()}\n")
        f.write(f"• Economic Significance: {bias_analysis.get('economic_significance', 'unknown').upper()}\n\n")
        
        f.write("EFFICIENCY ANALYSIS:\n")
        f.write(f"• Autocorrelation Severity: {efficiency_analysis.get('autocorr_severity', 'unknown').upper()}\n")
        f.write(f"• Learning Failure: {'YES' if efficiency_analysis.get('learning_failure', False) else 'NO'}\n")
        f.write(f"• Information Processing: {efficiency_analysis.get('information_processing', 'unknown').upper()}\n\n")
        
        f.write("OVERALL ASSESSMENT:\n")
        f.write(f"• REH Compatibility: {overall_assessment.get('reh_compatibility', 'unknown').upper()}\n")
        f.write(f"• Forecast Quality: {overall_assessment.get('forecast_quality', 'unknown').upper()}\n")
        
        primary_failures = overall_assessment.get('primary_failures', [])
        if primary_failures:
            f.write(f"• Primary Failure Modes: {', '.join(primary_failures)}\n")
        
        challenges = overall_assessment.get('period_challenges', [])
        if challenges:
            f.write("• Period-Specific Challenges:\n")
            for challenge in challenges:
                f.write(f"  - {challenge}\n")
        f.write("\n")

    def _write_policy_implications(self, f) -> None:
        """Write policy implications section"""
        f.write("7. POLICY IMPLICATIONS\n")
        f.write("═" * 70 + "\n\n")
        
        econ_interp = self.results.get("economic_interpretation", {})
        policy_impl = econ_interp.get("policy_implications", {})
        
        if not policy_impl:
            f.write("Policy implications not available.\n\n")
            return
        
        f.write("FOR CENTRAL BANK POLICYMAKERS:\n")
        for implication in policy_impl.get("central_bank", []):
            f.write(f"• {implication}\n")
        f.write("\n")
        
        f.write("FOR MARKET PARTICIPANTS:\n")
        for implication in policy_impl.get("market_participants", []):
            f.write(f"• {implication}\n")
        f.write("\n")
        
        f.write("FOR RESEARCHERS:\n")
        for implication in policy_impl.get("researchers", []):
            f.write(f"• {implication}\n")
        f.write("\n")

    def export_plots(self, output_dir: str = "plots", dpi: int = 300) -> None:
        """
        Export all diagnostic plots to individual files
        
        Parameters:
        - output_dir: Directory to save plots
        - dpi: Resolution for saved plots
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data available")

        logger.info(f"Exporting plots to {output_dir}/...")
        
        self.visualizations.export_plots_to_files(
            self.aligned_data["forecast"],
            self.aligned_data["realized"],
            self.results,
            output_dir=output_dir,
            dpi=dpi
        )
        
        logger.info("Plot export completed")

    def get_results_summary(self) -> str:
        """
        Get a formatted string summary of results
        
        Returns:
        - str: Formatted summary text
        """
        if not self.results:
            return "No results available. Run comprehensive_analysis() first."
        
        return format_results_summary(self.results)

    def save_data(self, filename: str = "aligned_data.csv") -> None:
        """
        Save aligned data to CSV file
        
        Parameters:
        - filename: Output CSV filename
        """
        if self.aligned_data is None:
            raise ValueError("No aligned data to save")
        
        self.aligned_data.to_csv(filename)
        logger.info(f"Aligned data saved to {filename}")

    def load_data(self, filename: str) -> None:
        """
        Load previously saved aligned data from CSV
        
        Parameters:
        - filename: Input CSV filename
        """
        self.aligned_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        self.forecast_errors = self.aligned_data["forecast_error"]
        logger.info(f"Loaded aligned data from {filename}")
    
    def export_latex_report(
        self, 
        output_file: str = "reh_analysis_report.tex",
        title: str = "Brazilian REH Analyzer - Academic Report",
        author: str = "Brazilian REH Analyzer Framework"
    ) -> str:
        """
        Export comprehensive analysis results in LaTeX format for academic presentation
        
        Parameters:
        - output_file: Output LaTeX file path
        - title: Document title  
        - author: Document author
        
        Returns:
        - str: Path to generated LaTeX file
        """
        if not self.results:
            raise ValueError("No results available. Run comprehensive_analysis() first.")
        
        logger.info(f"Exporting LaTeX report to {output_file}...")
        
        latex_file = self.visualizations.export_latex_report(
            results=self.results,
            output_file=output_file,
            title=title,
            author=author
        )
        
        logger.info(f"LaTeX report exported to {latex_file}")
        return latex_file