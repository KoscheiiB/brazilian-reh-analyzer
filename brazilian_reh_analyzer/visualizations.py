"""
Visualization functions for Brazilian REH Analyzer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class REHVisualizations:
    """
    Visualization components for REH analysis results
    """
    
    @staticmethod
    def setup_plot_style():
        """Configure matplotlib and seaborn for publication-quality plots"""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
    
    @staticmethod
    def plot_forecast_vs_realization(
        forecast: pd.Series,
        realized: pd.Series,
        ax: Optional[plt.Axes] = None,
        title: str = "Focus Forecasts vs Realized IPCA"
    ) -> plt.Figure:
        """
        Create scatter plot of forecasts vs realizations
        
        Parameters:
        - forecast: Series of forecasts
        - realized: Series of realized values
        - ax: Optional matplotlib axis
        - title: Plot title
        
        Returns:
        - matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.get_figure()
        
        # Align data
        aligned_data = pd.DataFrame({'forecast': forecast, 'realized': realized}).dropna()
        
        # Scatter plot
        ax.scatter(
            aligned_data['forecast'],
            aligned_data['realized'],
            alpha=0.7,
            s=50,
            color="blue",
            edgecolors="darkblue",
            label="Observations"
        )
        
        # Perfect forecast line
        min_val = min(aligned_data['forecast'].min(), aligned_data['realized'].min())
        max_val = max(aligned_data['forecast'].max(), aligned_data['realized'].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Forecast Line"
        )
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned_data['forecast'], aligned_data['realized']
        )
        regression_line = slope * aligned_data['forecast'] + intercept
        ax.plot(
            aligned_data['forecast'],
            regression_line,
            "g-",
            linewidth=2,
            alpha=0.7,
            label=f"Regression Line (R²={r_value**2:.3f})"
        )
        
        ax.set_xlabel("Focus Median Forecast (%)")
        ax.set_ylabel("Realized IPCA 12m (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = aligned_data['forecast'].corr(aligned_data['realized'])
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}\nN: {len(aligned_data)}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_forecast_errors_timeseries(
        forecast_errors: pd.Series,
        ax: Optional[plt.Axes] = None,
        title: str = "Forecast Errors Over Time"
    ) -> plt.Figure:
        """
        Plot forecast errors as time series
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - ax: Optional matplotlib axis
        - title: Plot title
        
        Returns:
        - matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()
        
        errors_clean = forecast_errors.dropna()
        
        # Time series plot
        ax.plot(
            errors_clean.index,
            errors_clean.values,
            linewidth=2,
            color="blue",
            marker="o",
            markersize=4,
            label="Forecast Errors"
        )
        
        # Zero line
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Zero Error")
        
        # Mean error line
        mean_error = errors_clean.mean()
        ax.axhline(
            y=mean_error,
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Mean Error: {mean_error:.3f} p.p."
        )
        
        # Confidence bands (±2 std dev)
        std_error = errors_clean.std()
        ax.fill_between(
            errors_clean.index,
            mean_error - 2*std_error,
            mean_error + 2*std_error,
            alpha=0.2,
            color="gray",
            label="±2σ Band"
        )
        
        ax.set_xlabel("Forecast Date")
        ax.set_ylabel("Forecast Error (p.p.)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_error_distribution(
        forecast_errors: pd.Series,
        ax: Optional[plt.Axes] = None,
        title: str = "Distribution of Forecast Errors"
    ) -> plt.Figure:
        """
        Plot distribution of forecast errors with normality test
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - ax: Optional matplotlib axis
        - title: Plot title
        
        Returns:
        - matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
        
        errors_clean = forecast_errors.dropna()
        
        # Histogram
        n_bins = min(20, len(errors_clean) // 5)
        ax.hist(
            errors_clean,
            bins=n_bins,
            alpha=0.7,
            density=True,
            color="lightcoral",
            edgecolor="black",
            label="Actual Distribution"
        )
        
        # Zero line
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
        
        # Mean line
        ax.axvline(
            x=errors_clean.mean(),
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {errors_clean.mean():.3f}"
        )
        
        # Normal distribution overlay
        x = np.linspace(errors_clean.min(), errors_clean.max(), 100)
        normal_dist = stats.norm.pdf(x, errors_clean.mean(), errors_clean.std())
        ax.plot(x, normal_dist, "g-", linewidth=2, label="Normal Distribution")
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(errors_clean)
        jarque_stat, jarque_p = stats.jarque_bera(errors_clean)
        
        ax.set_xlabel("Forecast Error (p.p.)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add test statistics
        test_text = f"Shapiro-Wilk p-value: {shapiro_p:.4f}\nJarque-Bera p-value: {jarque_p:.4f}"
        ax.text(
            0.72,
            0.95,
            test_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            fontsize=9
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rolling_statistics(
        forecast_errors: pd.Series,
        window: int = 12,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rolling mean and standard deviation of forecast errors
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - window: Rolling window size
        - ax: Optional matplotlib axis
        - title: Plot title
        
        Returns:
        - matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.get_figure()
        
        if title is None:
            title = f"Rolling Statistics ({window}-observation window)"
        
        errors_clean = forecast_errors.dropna()
        window = min(window, len(errors_clean) // 3)
        
        if window < 3:
            ax.text(0.5, 0.5, "Insufficient data for rolling statistics", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return fig
        
        # Rolling statistics
        rolling_mean = errors_clean.rolling(window=window, center=True).mean()
        rolling_std = errors_clean.rolling(window=window, center=True).std()
        
        # Plot rolling mean
        ax.plot(
            rolling_mean.index,
            rolling_mean.values,
            label="Rolling Mean",
            linewidth=2,
            color="blue"
        )
        
        # Confidence bands
        ax.fill_between(
            rolling_mean.index,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.3,
            color="lightblue",
            label="±1 Std Dev"
        )
        
        # Zero line
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Zero")
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Error Statistics")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_qq_normality(
        forecast_errors: pd.Series,
        ax: Optional[plt.Axes] = None,
        title: str = "Q-Q Plot: Forecast Errors vs Normal Distribution"
    ) -> plt.Figure:
        """
        Create Q-Q plot for normality assessment
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - ax: Optional matplotlib axis
        - title: Plot title
        
        Returns:
        - matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()
        
        errors_clean = forecast_errors.dropna()
        
        # Q-Q plot
        stats.probplot(errors_clean, dist="norm", plot=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_comprehensive_diagnostics(
        forecast: pd.Series,
        realized: pd.Series,
        results: Dict,
        figsize: Tuple[int, int] = (18, 12)
    ) -> plt.Figure:
        """
        Create comprehensive diagnostic plot with all key visualizations
        
        Parameters:
        - forecast: Series of forecasts
        - realized: Series of realized values
        - results: Dictionary containing analysis results
        - figsize: Figure size tuple
        
        Returns:
        - matplotlib Figure with multiple subplots
        """
        REHVisualizations.setup_plot_style()
        
        # Calculate forecast errors
        aligned_data = pd.DataFrame({'forecast': forecast, 'realized': realized}).dropna()
        forecast_errors = aligned_data['realized'] - aligned_data['forecast']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Forecast vs Realization scatter
        REHVisualizations.plot_forecast_vs_realization(
            aligned_data['forecast'], aligned_data['realized'], ax=axes[0, 0]
        )
        
        # Plot 2: Forecast errors over time
        REHVisualizations.plot_forecast_errors_timeseries(
            forecast_errors, ax=axes[0, 1]
        )
        
        # Plot 3: Error distribution
        REHVisualizations.plot_error_distribution(
            forecast_errors, ax=axes[0, 2]
        )
        
        # Plot 4: Rolling statistics
        window = min(12, len(forecast_errors) // 3)
        REHVisualizations.plot_rolling_statistics(
            forecast_errors, window=window, ax=axes[1, 0]
        )
        
        # Plot 5: Q-Q plot
        REHVisualizations.plot_qq_normality(
            forecast_errors, ax=axes[1, 1]
        )
        
        # Plot 6: Summary statistics
        axes[1, 2].axis('off')
        
        # Create summary text
        desc_stats = results.get("descriptive_stats", {})
        rationality = results.get("rationality_assessment", {})
        
        summary_text = f"""
SUMMARY STATISTICS
──────────────────
Observations: {desc_stats.get('n_observations', 'N/A')}
Date Range: {desc_stats.get('date_range', 'N/A')}

Mean Forecast Error: {desc_stats.get('error_mean', 0):.3f} p.p.
Error Std Dev: {desc_stats.get('error_std', 0):.3f} p.p.
Error Min/Max: {desc_stats.get('error_min', 0):.2f} / {desc_stats.get('error_max', 0):.2f} p.p.

RATIONALITY TESTS
─────────────────
Unbiased: {'✓' if rationality.get('unbiased', False) else '✗'}
MZ Test Passed: {'✓' if rationality.get('mz_rational', False) else '✗'}
Efficient: {'✓' if rationality.get('efficient', False) else '✗'}
Overall Rational: {'✓' if rationality.get('overall_rational', False) else '✗'}

Average Respondents: {desc_stats.get('mean_respondents', 0):.0f}
        """
        
        axes[1, 2].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )
        
        plt.suptitle("Brazilian REH Analyzer - Comprehensive Diagnostic Report", fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def export_plots_to_files(
        forecast: pd.Series,
        realized: pd.Series,
        results: Dict,
        output_dir: str = "plots",
        dpi: int = 300
    ):
        """
        Export all plots to individual files for publication
        
        Parameters:
        - forecast: Series of forecasts
        - realized: Series of realized values
        - results: Dictionary containing analysis results
        - output_dir: Directory to save plots
        - dpi: Resolution for saved plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        aligned_data = pd.DataFrame({'forecast': forecast, 'realized': realized}).dropna()
        forecast_errors = aligned_data['realized'] - aligned_data['forecast']
        
        plots = {
            "forecast_vs_realization": lambda: REHVisualizations.plot_forecast_vs_realization(
                aligned_data['forecast'], aligned_data['realized']
            ),
            "forecast_errors_timeseries": lambda: REHVisualizations.plot_forecast_errors_timeseries(
                forecast_errors
            ),
            "error_distribution": lambda: REHVisualizations.plot_error_distribution(
                forecast_errors
            ),
            "rolling_statistics": lambda: REHVisualizations.plot_rolling_statistics(
                forecast_errors
            ),
            "qq_plot": lambda: REHVisualizations.plot_qq_normality(
                forecast_errors
            ),
            "comprehensive_diagnostics": lambda: REHVisualizations.create_comprehensive_diagnostics(
                aligned_data['forecast'], aligned_data['realized'], results
            )
        }
        
        for plot_name, plot_func in plots.items():
            try:
                fig = plot_func()
                output_path = os.path.join(output_dir, f"{plot_name}.png")
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                logger.info(f"Saved plot: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save plot {plot_name}: {e}")