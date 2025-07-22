"""
Main Brazilian REH Analyzer class for comprehensive econometric analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

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

        # Calculate descriptive statistics
        logger.info("Calculating descriptive statistics...")
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

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
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

    def export_results_summary(self, filename: str = "reh_analysis_summary.txt") -> None:
        """
        Export a human-readable summary of results
        
        Parameters:
        - filename: Output filename for the summary
        """
        if not self.results:
            raise ValueError("No results to export. Run comprehensive_analysis() first.")

        logger.info(f"Exporting results summary to {filename}...")
        
        with open(filename, "w") as f:
            f.write("RATIONAL EXPECTATIONS HYPOTHESIS ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            desc_stats = self.results.get("descriptive_stats", {})
            f.write(f"Analysis Period: {desc_stats.get('date_range', 'N/A')}\n")
            f.write(f"Number of Observations: {desc_stats.get('n_observations', 'N/A')}\n")
            f.write(f"Mean Forecast Error: {desc_stats.get('error_mean', 0):.4f} p.p.\n")
            f.write(f"Error Standard Deviation: {desc_stats.get('error_std', 0):.4f} p.p.\n\n")

            if "rationality_assessment" in self.results:
                ra = self.results["rationality_assessment"]
                f.write("RATIONALITY TEST RESULTS:\n")
                f.write(f"Unbiased: {'PASS' if ra.get('unbiased', False) else 'FAIL'}\n")
                f.write(f"MZ Test: {'PASS' if ra.get('mz_rational', False) else 'FAIL'}\n")
                f.write(f"Efficient: {'PASS' if ra.get('efficient', False) else 'FAIL'}\n")
                f.write(f"Overall Rational: {'PASS' if ra.get('overall_rational', False) else 'FAIL'}\n\n")

            # Add detailed test results
            if "mincer_zarnowitz" in self.results and "error" not in self.results["mincer_zarnowitz"]:
                mz = self.results["mincer_zarnowitz"]
                f.write("MINCER-ZARNOWITZ TEST DETAILS:\n")
                f.write(f"Alpha (intercept): {mz['alpha']:.6f} (p-value: {mz['alpha_pvalue']:.6f})\n")
                f.write(f"Beta (slope): {mz['beta']:.6f} (p-value: {mz['beta_pvalue']:.6f})\n")
                f.write(f"Joint test p-value: {mz['joint_test_pvalue']:.6f}\n")
                f.write(f"R-squared: {mz['r_squared']:.6f}\n\n")

            if "bias_test" in self.results and "error" not in self.results["bias_test"]:
                bias = self.results["bias_test"]
                f.write("BIAS TEST DETAILS:\n")
                f.write(f"Mean error: {bias['mean_error']:.6f} p.p.\n")
                f.write(f"t-statistic: {bias['t_statistic']:.6f}\n")
                f.write(f"p-value: {bias['p_value']:.6f}\n")
                f.write(f"Bias direction: {bias['bias_direction']}\n\n")

            if "autocorrelation" in self.results and "error" not in self.results["autocorrelation"]:
                autocorr = self.results["autocorrelation"]
                f.write("AUTOCORRELATION TEST DETAILS:\n")
                f.write(f"Ljung-Box statistic: {autocorr.get('ljung_box_stat', 'N/A'):.6f}\n")
                f.write(f"Ljung-Box p-value: {autocorr.get('ljung_box_pvalue', 'N/A'):.6f}\n")
                f.write(f"Significant autocorrelation: {autocorr.get('significant_autocorr', 'N/A')}\n")

        logger.info(f"Results summary exported to {filename}")

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