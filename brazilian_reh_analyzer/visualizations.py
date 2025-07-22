"""
Enhanced Visualization functions for Brazilian REH Analyzer.
Academic-grade plots with professional styling and comprehensive diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Professional Academic Color Palette (colorblind-friendly)
ACADEMIC_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Burgundy accent  
    'error': '#F18F01',        # Warning orange
    'success': '#C73E1D',      # Dark red
    'neutral': '#6B7280',      # Professional gray
    'light_primary': '#7FC4E8', # Light blue
    'light_secondary': '#D4A5C2', # Light burgundy
    'background': '#F8FAFC'     # Light background
}


class REHVisualizations:
    """
    Enhanced visualization components for REH analysis results with academic-grade styling
    """
    
    @staticmethod
    def setup_academic_style():
        """Configure matplotlib and seaborn for publication-quality academic plots"""
        # Set academic style
        plt.style.use('default')  # Start with clean default style
        
        # Academic publication standards
        plt.rcParams.update({
            # Figure settings
            'figure.figsize': (10, 6),
            'figure.facecolor': 'white',
            'figure.edgecolor': 'none',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            
            # Font settings
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            
            # Grid and axes
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.6,
            'axes.edgecolor': ACADEMIC_COLORS['neutral'],
            'axes.facecolor': 'white',
            
            # Lines and markers
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            
            # Legend
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.framealpha': 0.9,
            'legend.edgecolor': ACADEMIC_COLORS['neutral'],
            
            # Colors
            'axes.prop_cycle': plt.cycler('color', [
                ACADEMIC_COLORS['primary'],
                ACADEMIC_COLORS['secondary'], 
                ACADEMIC_COLORS['error'],
                ACADEMIC_COLORS['success'],
                ACADEMIC_COLORS['neutral']
            ])
        })
        
        # Set seaborn style for additional enhancements
        sns.set_palette([
            ACADEMIC_COLORS['primary'],
            ACADEMIC_COLORS['secondary'], 
            ACADEMIC_COLORS['error'],
            ACADEMIC_COLORS['success']
        ])
    
    @staticmethod
    def setup_plot_style():
        """Backward compatibility - calls setup_academic_style"""
        REHVisualizations.setup_academic_style()
    
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
        
        # Scatter plot with academic colors
        ax.scatter(
            aligned_data['forecast'],
            aligned_data['realized'],
            alpha=0.7,
            s=50,
            color=ACADEMIC_COLORS['primary'],
            edgecolors=ACADEMIC_COLORS['neutral'],
            linewidth=0.5,
            label="Observations"
        )
        
        # Perfect forecast line
        min_val = min(aligned_data['forecast'].min(), aligned_data['realized'].min())
        max_val = max(aligned_data['forecast'].max(), aligned_data['realized'].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color=ACADEMIC_COLORS['success'],
            linestyle="--",
            linewidth=2,
            label="Perfect Forecast (45° line)"
        )
        
        # Add regression line with confidence interval
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned_data['forecast'], aligned_data['realized']
        )
        regression_line = slope * aligned_data['forecast'] + intercept
        ax.plot(
            aligned_data['forecast'],
            regression_line,
            color=ACADEMIC_COLORS['secondary'],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            label=f"Regression Line (R²={r_value**2:.3f})"
        )
        
        # Add confidence intervals (95%)
        n = len(aligned_data)
        t_val = stats.t.ppf(0.975, n-2)  # 95% confidence
        residuals = aligned_data['realized'] - regression_line
        mse = np.sum(residuals**2) / (n-2)
        se_line = np.sqrt(mse * (1/n + (aligned_data['forecast'] - aligned_data['forecast'].mean())**2 / 
                                np.sum((aligned_data['forecast'] - aligned_data['forecast'].mean())**2)))
        ci_upper = regression_line + t_val * se_line
        ci_lower = regression_line - t_val * se_line
        
        ax.fill_between(
            aligned_data['forecast'],
            ci_lower, ci_upper,
            alpha=0.2,
            color=ACADEMIC_COLORS['secondary'],
            label="95% Confidence Interval"
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
        
        # Time series plot with academic colors
        ax.plot(
            errors_clean.index,
            errors_clean.values,
            linewidth=1.5,
            color=ACADEMIC_COLORS['primary'],
            marker="o",
            markersize=3,
            alpha=0.8,
            label="Forecast Errors"
        )
        
        # Zero line (rational expectations benchmark)
        ax.axhline(
            y=0, 
            color=ACADEMIC_COLORS['success'], 
            linestyle="--", 
            linewidth=2, 
            alpha=0.8, 
            label="Rational Expectations (Zero Error)"
        )
        
        # Mean error line
        mean_error = errors_clean.mean()
        ax.axhline(
            y=mean_error,
            color=ACADEMIC_COLORS['error'],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            label=f"Mean Bias: {mean_error:.3f} p.p."
        )
        
        # Confidence bands (±2 std dev) - academic gray
        std_error = errors_clean.std()
        ax.fill_between(
            errors_clean.index,
            mean_error - 2*std_error,
            mean_error + 2*std_error,
            alpha=0.15,
            color=ACADEMIC_COLORS['neutral'],
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
        
        # Histogram with academic colors
        n_bins = min(20, len(errors_clean) // 5)
        ax.hist(
            errors_clean,
            bins=n_bins,
            alpha=0.6,
            density=True,
            color=ACADEMIC_COLORS['light_primary'],
            edgecolor=ACADEMIC_COLORS['primary'],
            linewidth=0.8,
            label="Forecast Error Distribution"
        )
        
        # Zero line (rational expectations benchmark)
        ax.axvline(
            x=0, 
            color=ACADEMIC_COLORS['success'], 
            linestyle="--", 
            linewidth=2, 
            alpha=0.8,
            label="Rational Expectations (Zero)"
        )
        
        # Mean line
        mean_error = errors_clean.mean()
        ax.axvline(
            x=mean_error,
            color=ACADEMIC_COLORS['error'],
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            label=f"Sample Mean: {mean_error:.3f} p.p."
        )
        
        # Normal distribution overlay
        x = np.linspace(errors_clean.min(), errors_clean.max(), 100)
        normal_dist = stats.norm.pdf(x, errors_clean.mean(), errors_clean.std())
        ax.plot(
            x, normal_dist, 
            color=ACADEMIC_COLORS['secondary'], 
            linestyle="-",
            linewidth=2.5, 
            alpha=0.8,
            label="Theoretical Normal Distribution"
        )
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(errors_clean)
        jarque_stat, jarque_p = stats.jarque_bera(errors_clean)
        
        ax.set_xlabel("Forecast Error (p.p.)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add test statistics with academic styling
        test_text = f"Normality Tests:\nShapiro-Wilk: p={shapiro_p:.4f}\nJarque-Bera: p={jarque_p:.4f}"
        if shapiro_p < 0.05 or jarque_p < 0.05:
            test_text += "\nReject normality"
        else:
            test_text += "\nNormal distribution"
        
        ax.text(
            0.72,
            0.95,
            test_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=ACADEMIC_COLORS['neutral']),
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
        
        # Plot rolling mean with academic colors
        ax.plot(
            rolling_mean.index,
            rolling_mean.values,
            label="Rolling Mean Bias",
            linewidth=2.5,
            color=ACADEMIC_COLORS['primary']
        )
        
        # Confidence bands - fix the hard-to-see light blue issue
        ax.fill_between(
            rolling_mean.index,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.25,
            color=ACADEMIC_COLORS['primary'],
            label="±1 Std Dev"
        )
        
        # Zero line (rational expectations benchmark)
        ax.axhline(
            y=0, 
            color=ACADEMIC_COLORS['success'], 
            linestyle="--", 
            linewidth=2,
            alpha=0.8, 
            label="Rational Expectations"
        )
        
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
    def plot_acf_pacf_analysis(
        forecast_errors: pd.Series,
        max_lags: int = 24,
        ax: Optional[plt.Axes] = None,
        title: str = "Autocorrelation Analysis of Forecast Errors"
    ) -> plt.Figure:
        """
        Create ACF/PACF plots for autocorrelation analysis (replaces summary statistics table)
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - max_lags: Maximum number of lags to show
        - ax: Optional matplotlib axis
        - title: Plot title
        
        Returns:
        - matplotlib Figure with ACF and PACF subplots
        """
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            # If single axis provided, create subplots within that area
            fig = ax.get_figure()
            ax.axis('off')
            ax1 = fig.add_subplot(2, 2, 3)  # Top subplot for ACF
            ax2 = fig.add_subplot(2, 2, 4)  # Bottom subplot for PACF
        
        errors_clean = forecast_errors.dropna()
        n_obs = len(errors_clean)
        max_lags = min(max_lags, n_obs // 4)  # Ensure reasonable lag count
        
        if max_lags < 5:
            ax1.text(0.5, 0.5, "Insufficient data\nfor ACF/PACF analysis", 
                   transform=ax1.transAxes, ha='center', va='center')
            ax2.text(0.5, 0.5, "Need at least\n20 observations", 
                   transform=ax2.transAxes, ha='center', va='center')
            return fig
        
        # Calculate ACF and PACF
        acf_vals, acf_confint = acf(errors_clean, nlags=max_lags, alpha=0.05)
        pacf_vals, pacf_confint = pacf(errors_clean, nlags=max_lags, alpha=0.05)
        
        # Theoretical bounds for white noise
        theoretical_bound = 1.96 / np.sqrt(n_obs)
        
        # Plot ACF
        lags = np.arange(max_lags + 1)
        ax1.bar(lags, acf_vals, width=0.6, color=ACADEMIC_COLORS['primary'], alpha=0.7)
        ax1.axhline(y=0, color=ACADEMIC_COLORS['neutral'], linewidth=1)
        ax1.axhline(y=theoretical_bound, color=ACADEMIC_COLORS['success'], 
                   linestyle='--', alpha=0.8, label=f'95% CI (±{theoretical_bound:.3f})')
        ax1.axhline(y=-theoretical_bound, color=ACADEMIC_COLORS['success'], 
                   linestyle='--', alpha=0.8)
        ax1.fill_between(lags, -theoretical_bound, theoretical_bound, 
                        alpha=0.1, color=ACADEMIC_COLORS['success'])
        
        ax1.set_title("Autocorrelation Function (ACF)", fontsize=11)
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot PACF (skip lag 0 which is always 1)
        ax2.bar(lags[1:], pacf_vals[1:], width=0.6, color=ACADEMIC_COLORS['secondary'], alpha=0.7)
        ax2.axhline(y=0, color=ACADEMIC_COLORS['neutral'], linewidth=1)
        ax2.axhline(y=theoretical_bound, color=ACADEMIC_COLORS['success'], 
                   linestyle='--', alpha=0.8, label=f'95% CI (±{theoretical_bound:.3f})')
        ax2.axhline(y=-theoretical_bound, color=ACADEMIC_COLORS['success'], 
                   linestyle='--', alpha=0.8)
        ax2.fill_between(lags[1:], -theoretical_bound, theoretical_bound, 
                        alpha=0.1, color=ACADEMIC_COLORS['success'])
        
        ax2.set_title("Partial Autocorrelation Function (PACF)", fontsize=11)
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Add interpretation text
        n_significant_acf = np.sum(np.abs(acf_vals[1:]) > theoretical_bound)
        n_significant_pacf = np.sum(np.abs(pacf_vals[1:]) > theoretical_bound)
        
        interpretation = f"Significant ACF lags: {n_significant_acf}/{max_lags}\n"
        interpretation += f"Significant PACF lags: {n_significant_pacf}/{max_lags}\n"
        if n_significant_acf > max_lags * 0.1:  # More than 10% significant
            interpretation += "Strong autocorrelation detected"
        else:
            interpretation += "Weak autocorrelation"
        
        fig.suptitle(f"{title}\n{interpretation}", fontsize=12, y=0.95)
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
        
        # Enhanced Q-Q plot with confidence bands
        osm, osr = stats.probplot(errors_clean, dist="norm", plot=ax)
        
        # Add confidence bands (95%)
        n = len(errors_clean)
        # Theoretical quantiles from probplot
        theoretical_quantiles = osm[0]
        sample_quantiles = osm[1]
        
        # Calculate confidence bands using order statistics
        # Approximate confidence intervals for Q-Q plot
        alpha = 0.05
        p = np.arange(1, n+1) / (n+1)
        lower_ci = stats.norm.ppf(stats.beta.ppf(alpha/2, np.arange(1, n+1), n - np.arange(1, n+1) + 1))
        upper_ci = stats.norm.ppf(stats.beta.ppf(1-alpha/2, np.arange(1, n+1), n - np.arange(1, n+1) + 1))
        
        # Sort for plotting
        sort_idx = np.argsort(theoretical_quantiles)
        ax.fill_between(
            theoretical_quantiles[sort_idx], 
            lower_ci[sort_idx], 
            upper_ci[sort_idx],
            alpha=0.2, 
            color=ACADEMIC_COLORS['primary'], 
            label='95% Confidence Band'
        )
        
        # Improve plot styling
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Theoretical Quantiles (Normal Distribution)", fontsize=11)
        ax.set_ylabel("Sample Quantiles (Forecast Errors)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add normality test results
        shapiro_stat, shapiro_p = stats.shapiro(errors_clean)
        jarque_stat, jarque_p = stats.jarque_bera(errors_clean)
        
        test_text = f"Shapiro-Wilk: p={shapiro_p:.4f}\nJarque-Bera: p={jarque_p:.4f}"
        if shapiro_p < 0.05 or jarque_p < 0.05:
            test_text += "\nNon-normal distribution"
        else:
            test_text += "\nNormal distribution"
        
        ax.text(
            0.05, 0.95, test_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            fontsize=9
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_comprehensive_diagnostics(
        forecast: pd.Series,
        realized: pd.Series,
        results: Dict,
        figsize: Tuple[int, int] = (20, 12)
    ) -> plt.Figure:
        """
        Create comprehensive diagnostic plot with improved layout
        
        Layout:
        Row 1: Forecast vs Realized | Forecast Errors | Distribution  
        Row 2: Rolling Stats | Q-Q Plot | ACF & PACF (stacked)
        
        Parameters:
        - forecast: Series of forecasts
        - realized: Series of realized values
        - results: Dictionary containing analysis results
        - figsize: Figure size tuple
        
        Returns:
        - matplotlib Figure with multiple subplots
        """
        REHVisualizations.setup_academic_style()
        
        # Calculate forecast errors
        aligned_data = pd.DataFrame({'forecast': forecast, 'realized': realized}).dropna()
        forecast_errors = aligned_data['realized'] - aligned_data['forecast']
        
        # Create figure with custom GridSpec for better control
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25,
                         top=0.92, bottom=0.08, left=0.06, right=0.98)
        
        # Row 1: Three equal columns
        # Plot 1: Forecast vs Realized (R1C1)
        ax1 = fig.add_subplot(gs[0, 0])
        REHVisualizations.plot_forecast_vs_realization(
            aligned_data['forecast'], aligned_data['realized'], ax=ax1
        )
        
        # Plot 2: Forecast Errors Over Time (R1C2)
        ax2 = fig.add_subplot(gs[0, 1])
        REHVisualizations.plot_forecast_errors_timeseries(
            forecast_errors, ax=ax2
        )
        
        # Plot 3: Error Distribution (R1C3)
        ax3 = fig.add_subplot(gs[0, 2])
        REHVisualizations.plot_error_distribution(
            forecast_errors, ax=ax3
        )
        
        # Row 2: Three columns
        # Plot 4: Rolling Statistics (R2C1)
        ax4 = fig.add_subplot(gs[1, 0])
        window = min(12, len(forecast_errors) // 3)
        REHVisualizations.plot_rolling_statistics(
            forecast_errors, window=window, ax=ax4
        )
        
        # Plot 5: Q-Q Plot (R2C2)
        ax5 = fig.add_subplot(gs[1, 1])
        REHVisualizations.plot_qq_normality(
            forecast_errors, ax=ax5
        )
        
        # Plot 6: ACF and PACF stacked (R2C3)
        # Create nested GridSpec for ACF/PACF column
        gs_acf = gs[1, 2].subgridspec(2, 1, hspace=0.4)
        
        # ACF plot (top half of R2C3)
        ax6a = fig.add_subplot(gs_acf[0])
        REHVisualizations.plot_single_acf(forecast_errors, ax=ax6a, max_lags=20)
        
        # PACF plot (bottom half of R2C3)
        ax6b = fig.add_subplot(gs_acf[1])
        REHVisualizations.plot_single_pacf(forecast_errors, ax=ax6b, max_lags=20)
        
        plt.suptitle("Brazilian REH Analyzer - Comprehensive Diagnostic Report", 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    @staticmethod
    def plot_single_acf(
        forecast_errors: pd.Series,
        ax: Optional[plt.Axes] = None,
        max_lags: int = 20,
        title: str = "Autocorrelation Function (ACF)"
    ) -> plt.Figure:
        """Create standalone ACF plot"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            fig = ax.get_figure()
            
        errors_clean = forecast_errors.dropna()
        max_lags = min(max_lags, len(errors_clean) // 4)
        
        try:
            from statsmodels.tsa.stattools import acf
            
            # Calculate ACF
            acf_values = acf(errors_clean, nlags=max_lags, fft=True)
            lags = np.arange(len(acf_values))
            
            # Confidence intervals
            n = len(errors_clean)
            confidence_interval = 1.96 / np.sqrt(n)
            
            # Plot ACF
            ax.stem(lags, acf_values, basefmt=" ", linefmt='#1f77b4')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Lag', fontsize=10)
            ax.set_ylabel('ACF', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'ACF calculation failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        return fig
    
    @staticmethod 
    def plot_single_pacf(
        forecast_errors: pd.Series,
        ax: Optional[plt.Axes] = None,
        max_lags: int = 20,
        title: str = "Partial Autocorrelation Function (PACF)"
    ) -> plt.Figure:
        """Create standalone PACF plot"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            fig = ax.get_figure()
            
        errors_clean = forecast_errors.dropna()
        max_lags = min(max_lags, len(errors_clean) // 4)
        
        try:
            from statsmodels.tsa.stattools import pacf
            
            # Calculate PACF
            pacf_values = pacf(errors_clean, nlags=max_lags)
            lags = np.arange(len(pacf_values))
            
            # Confidence intervals
            n = len(errors_clean)
            confidence_interval = 1.96 / np.sqrt(n)
            
            # Plot PACF
            ax.stem(lags, pacf_values, basefmt=" ", linefmt='#ff7f0e')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Lag', fontsize=10)
            ax.set_ylabel('PACF', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'PACF calculation failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
        
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
            ),
            "acf_analysis": lambda: REHVisualizations.plot_single_acf(
                forecast_errors
            ),
            "pacf_analysis": lambda: REHVisualizations.plot_single_pacf(
                forecast_errors
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
        
        logger.info(f"Plot export completed. Files saved to {output_dir}/")
    
    @staticmethod
    def export_latex_report(
        results: Dict,
        output_file: str = "reh_analysis_report.tex",
        title: str = "Brazilian REH Analyzer - Academic Report",
        author: str = "Brazilian REH Analyzer Framework"
    ) -> str:
        """
        Export comprehensive analysis results in LaTeX format for academic presentation
        
        Parameters:
        - results: Dictionary containing analysis results from comprehensive_analysis()
        - output_file: Output LaTeX file path
        - title: Document title
        - author: Document author
        
        Returns:
        - Path to generated LaTeX file
        """
        import os
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # LaTeX document preamble
            f.write("\\documentclass[11pt,a4paper]{article}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage[T1]{fontenc}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\usepackage{amsmath,amssymb}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{array}\n")
            f.write("\\usepackage{longtable}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{float}\n")
            f.write("\\usepackage{xcolor}\n")
            f.write("\\usepackage{hyperref}\n")
            f.write("\\usepackage{bookmark}\n")
            f.write("\\usepackage{fancyhdr}\n")
            f.write("\n")
            f.write("\\geometry{margin=1in}\n")
            f.write("\\pagestyle{fancy}\n")
            f.write("\\fancyhf{}\n")
            f.write("\\rhead{Brazilian REH Analysis}\n")
            f.write("\\lfoot{\\today}\n")
            f.write("\\rfoot{\\thepage}\n")
            f.write("\n")
            f.write("\\definecolor{academicblue}{HTML}{2E86AB}\n")
            f.write("\\definecolor{academicred}{HTML}{C73E1D}\n")
            f.write("\n")
            f.write(f"\\title{{{title}}}\n")
            f.write(f"\\author{{{author}}}\n")
            f.write("\\date{\\today}\n")
            f.write("\n")
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n")
            f.write("\n")
            
            # Executive Summary
            REHVisualizations._write_latex_executive_summary(f, results)
            
            # Descriptive Statistics
            REHVisualizations._write_latex_descriptive_stats(f, results)
            
            # Rationality Tests
            REHVisualizations._write_latex_rationality_tests(f, results)
            
            # Mincer-Zarnowitz Analysis
            REHVisualizations._write_latex_mincer_zarnowitz(f, results)
            
            # Sub-period Analysis
            REHVisualizations._write_latex_sub_period_analysis(f, results)
            
            # Economic Interpretation
            REHVisualizations._write_latex_economic_interpretation(f, results)
            
            # Policy Implications
            REHVisualizations._write_latex_policy_implications(f, results)
            
            f.write("\\end{document}\n")
        
        logger.info(f"LaTeX report generated: {output_file}")
        return output_file
    
    @staticmethod
    def _write_latex_executive_summary(f, results: Dict) -> None:
        """Write LaTeX executive summary section"""
        f.write("\\section{Executive Summary}\n\n")
        
        desc_stats = results.get("descriptive_stats", {})
        rationality = results.get("rationality_assessment", {})
        econ_interp = results.get("economic_interpretation", {})
        
        # Create summary box
        f.write("\\begin{center}\n")
        f.write("\\fbox{\\begin{minipage}{0.9\\textwidth}\n")
        f.write("\\textbf{\\large Analysis Overview}\\\\[0.5em]\n")
        
        # Overall assessment
        overall_rational = rationality.get('overall_rational', False)
        status_color = "green" if overall_rational else "academicred"
        status_text = "PASS" if overall_rational else "FAIL"
        
        f.write(f"\\textcolor{{{status_color}}}{{\\textbf{{Rational Expectations Hypothesis: {status_text}}}}}\\\\[0.3em]\n")
        
        # Key metrics
        date_range = desc_stats.get('date_range', 'N/A')
        n_obs = desc_stats.get('n_observations', 0)
        error_mean = desc_stats.get('error_mean', 0)
        
        f.write(f"\\textbf{{Analysis Period:}} {date_range}\\\\[0.2em]\n")
        f.write(f"\\textbf{{Observations:}} {n_obs:,}\\\\[0.2em]\n")
        f.write(f"\\textbf{{Mean Forecast Bias:}} {error_mean:.3f} p.p.\\\\[0.2em]\n")
        
        if econ_interp and "bias_analysis" in econ_interp:
            bias_analysis = econ_interp["bias_analysis"]
            severity = bias_analysis.get("severity", "unknown")
            direction = bias_analysis.get("direction", "unknown")
            f.write(f"\\textbf{{Bias Severity:}} {severity.title()} ({direction.title()})\\\\[0.2em]\n")
        
        f.write("\\end{minipage}}\n")
        f.write("\\end{center}\n\n")
    
    @staticmethod
    def _write_latex_descriptive_stats(f, results: Dict) -> None:
        """Write LaTeX descriptive statistics table"""
        f.write("\\section{Comprehensive Descriptive Statistics}\n\n")
        
        rich_stats = results.get("rich_descriptive_stats", {})
        
        if not rich_stats:
            f.write("Descriptive statistics not available.\n\n")
            return
        
        # Create professional statistics table
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comprehensive Statistical Summary}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Statistic} & \\textbf{Forecast (\\%)} & \\textbf{Realized (\\%)} & \\textbf{Error (p.p.)} \\\\\n")
        f.write("\\midrule\n")
        
        # Extract statistics
        forecast_stats = rich_stats.get("forecast", {})
        realized_stats = rich_stats.get("realized", {})
        error_stats = rich_stats.get("errors", {})
        
        stats_rows = [
            ("Mean", "mean"),
            ("Median", "median"),
            ("Std. Deviation", "std"),
            ("Minimum", "min"),
            ("Maximum", "max"),
            ("Skewness", "skewness"),
            ("Kurtosis", "kurtosis"),
            ("Observations", "n_obs")
        ]
        
        for stat_name, stat_key in stats_rows:
            forecast_val = forecast_stats.get(stat_key, 0)
            realized_val = realized_stats.get(stat_key, 0)
            error_val = error_stats.get(stat_key, 0)
            
            if stat_key == "n_obs":
                f.write(f"{stat_name} & {forecast_val:,.0f} & {realized_val:,.0f} & {error_val:,.0f} \\\\\n")
            else:
                f.write(f"{stat_name} & {forecast_val:.3f} & {realized_val:.3f} & {error_val:.3f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
    
    @staticmethod  
    def _write_latex_rationality_tests(f, results: Dict) -> None:
        """Write LaTeX rationality test results"""
        f.write("\\section{Rationality Test Results}\n\n")
        
        rationality = results.get("rationality_assessment", {})
        
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{REH Test Results Summary}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Test} & \\textbf{Result} & \\textbf{Implication} \\\\\n")
        f.write("\\midrule\n")
        
        tests = [
            ("Unbiasedness", rationality.get('unbiased', False), "Systematic bias"),
            ("Mincer-Zarnowitz", rationality.get('mz_rational', False), "Forecast efficiency"), 
            ("Efficiency", rationality.get('efficient', False), "Information usage"),
            ("Overall REH", rationality.get('overall_rational', False), "Rational expectations")
        ]
        
        for test_name, passed, implication in tests:
            status = "\\textcolor{green}{PASS}" if passed else "\\textcolor{academicred}{FAIL}"
            f.write(f"{test_name} & {status} & {implication} \\\\\n")
        
        f.write("\\bottomrule\n") 
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
    
    @staticmethod
    def _write_latex_mincer_zarnowitz(f, results: Dict) -> None:
        """Write LaTeX Mincer-Zarnowitz regression analysis"""
        f.write("\\section{Mincer-Zarnowitz Regression Analysis}\n\n")
        
        detailed_mz = results.get("detailed_mincer_zarnowitz", {})
        
        if not detailed_mz:
            f.write("Mincer-Zarnowitz analysis not available.\n\n")
            return
        
        # Regression equation
        f.write("The Mincer-Zarnowitz regression tests the null hypothesis of rational expectations:\n")
        f.write("\\begin{equation}\n")
        f.write("P_t = \\alpha + \\beta \\cdot E_{t-12}[P_t] + \\varepsilon_t\n")
        f.write("\\end{equation}\n")
        f.write("where $H_0: (\\alpha, \\beta) = (0, 1)$ under rational expectations.\n\n")
        
        # Results table
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Mincer-Zarnowitz Regression Results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Parameter} & \\textbf{Estimate} & \\textbf{Std. Error} & \\textbf{t-stat} & \\textbf{p-value} & \\textbf{95\\% CI} \\\\\n")
        f.write("\\midrule\n")
        
        # Alpha row
        alpha = detailed_mz.get('alpha', 0)
        alpha_se = detailed_mz.get('alpha_std_err', 0)  
        alpha_t = detailed_mz.get('alpha_t_stat', 0)
        alpha_p = detailed_mz.get('alpha_p_value', 1)
        alpha_ci = detailed_mz.get('alpha_95_ci', [0, 0])
        
        f.write(f"$\\alpha$ (Intercept) & {alpha:.3f} & {alpha_se:.3f} & {alpha_t:.2f} & {alpha_p:.4f} & [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}] \\\\\n")
        
        # Beta row  
        beta = detailed_mz.get('beta', 0)
        beta_se = detailed_mz.get('beta_std_err', 0)
        beta_t = detailed_mz.get('beta_t_stat', 0) 
        beta_p = detailed_mz.get('beta_p_value', 1)
        beta_ci = detailed_mz.get('beta_95_ci', [0, 0])
        
        f.write(f"$\\beta$ (Slope) & {beta:.3f} & {beta_se:.3f} & {beta_t:.2f} & {beta_p:.4f} & [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}] \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Additional statistics
        r_squared = detailed_mz.get('r_squared', 0)
        f_stat = detailed_mz.get('joint_f_statistic', 0)
        f_p = detailed_mz.get('joint_p_value', 1)
        
        f.write(f"\\textbf{{Model Diagnostics:}} $R^2 = {r_squared:.4f}$, ")
        f.write(f"Joint F-statistic = {f_stat:.2f} (p = {f_p:.6f})\n\n")
        
        # Economic interpretation
        f.write("\\subsection{Economic Interpretation}\n")
        f.write("\\begin{itemize}\n")
        if detailed_mz.get('alpha_significant', False):
            f.write(f"\\item $\\alpha = {alpha:.3f} \\neq 0$: Systematic forecast bias detected\n")
        if detailed_mz.get('beta_significantly_different_from_1', False):
            response_type = "under" if beta < 1 else "over"
            f.write(f"\\item $\\beta = {beta:.3f} \\neq 1$: Forecasters {response_type}-respond to their predictions\n")
        if f_p < 0.05:
            f.write("\\item Joint test rejection indicates violations of both unbiasedness and efficiency\n")
        f.write("\\end{itemize}\n\n")
    
    @staticmethod
    def _write_latex_sub_period_analysis(f, results: Dict) -> None:
        """Write LaTeX sub-period analysis"""
        f.write("\\section{Structural Break Analysis}\n\n")
        
        sub_periods = results.get("sub_period_analysis", {})
        
        if not sub_periods:
            f.write("No sub-periods detected or analysis failed.\n\n")
            return
        
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Sub-period Analysis Results}\n")
        f.write("\\begin{tabular}{lcccc}\n") 
        f.write("\\toprule\n")
        f.write("\\textbf{Period} & \\textbf{Start} & \\textbf{End} & \\textbf{Mean Error} & \\textbf{REH Status} \\\\\n")
        f.write("\\midrule\n")
        
        for period_name, period_data in sub_periods.items():
            start_date = period_data.get('start_date', '')[:10]  # YYYY-MM-DD
            end_date = period_data.get('end_date', '')[:10]
            mean_error = period_data.get('mean_error', 0)
            reh_tests = period_data.get('reh_tests', {})
            reh_overall = reh_tests.get('rationality_assessment', {}).get('overall_rational', False)
            
            period_num = period_name.split('_')[1] if '_' in period_name else period_name
            status = "\\textcolor{green}{PASS}" if reh_overall else "\\textcolor{academicred}{FAIL}"
            
            f.write(f"Period {period_num} & {start_date} & {end_date} & {mean_error:.3f} & {status} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Interpretation
        f.write("\\subsection{Structural Break Interpretation}\n")
        periods_list = list(sub_periods.values())
        if len(periods_list) > 1:
            errors = [p.get('mean_error', 0) for p in periods_list]
            max_error = max(errors)
            min_error = min(errors)
            f.write(f"\\begin{{itemize}}\n")
            f.write(f"\\item Forecast bias ranges from {min_error:.3f} to {max_error:.3f} p.p. across sub-periods\n")
            f.write(f"\\item Total bias variation: {abs(max_error - min_error):.3f} p.p.\n")
            if abs(max_error - min_error) > 1.0:
                f.write("\\item \\textbf{Substantial} time-variation in forecast bias detected\n")
            f.write("\\end{itemize}\n\n")
    
    @staticmethod
    def _write_latex_economic_interpretation(f, results: Dict) -> None:
        """Write enhanced LaTeX economic interpretation section with quantitative assessments"""
        f.write("\\section{Economic Interpretation}\n\n")
        
        econ_interp = results.get("economic_interpretation", {})
        
        if "error" in econ_interp or not econ_interp:
            f.write("Economic interpretation not available.\n\n")
            return
        
        bias_analysis = econ_interp.get("bias_analysis", {})
        efficiency_analysis = econ_interp.get("efficiency_analysis", {})
        overall_assessment = econ_interp.get("overall_assessment", {})
        coeff_interp = econ_interp.get("coefficient_interpretation", {})
        scenario_analysis = econ_interp.get("scenario_analysis", {})
        
        # Enhanced Bias Analysis with Quantitative Metrics
        f.write("\\subsection{Quantitative Bias Assessment}\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Enhanced Bias Analysis}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{Value} & \\textbf{Assessment} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Direction & {bias_analysis.get('direction', 'unknown').title()} & -- \\\\\n")
        f.write(f"Magnitude & {bias_analysis.get('magnitude_pp', 0):.3f} p.p. & {bias_analysis.get('severity', 'unknown').title()} \\\\\n")
        f.write(f"Grade Category & {bias_analysis.get('category', 'N/A')} & {bias_analysis.get('economic_significance', 'unknown').title()} Impact \\\\\n")
        f.write(f"Bias Ratio & {bias_analysis.get('bias_ratio', 0):.2f} & {'High' if bias_analysis.get('bias_ratio', 0) > 2 else 'Moderate' if bias_analysis.get('bias_ratio', 0) > 1 else 'Low'} Dominance \\\\\n")
        f.write(f"Systematic Component & {bias_analysis.get('systematic_component_pct', 0):.1f}\\% & of Total Error \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Enhanced Efficiency Analysis with Quantitative Metrics
        f.write("\\subsection{Quantitative Efficiency Assessment}\n")
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Enhanced Efficiency Analysis}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{Value} & \\textbf{Assessment} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Ljung-Box Statistic & {efficiency_analysis.get('ljung_box_statistic', 0):.1f} & {efficiency_analysis.get('autocorr_severity', 'unknown').title()} \\\\\n")
        f.write(f"LB p-value & {efficiency_analysis.get('ljung_box_p_value', 1.0):.4f} & {'Significant' if efficiency_analysis.get('ljung_box_p_value', 1.0) < 0.05 else 'Not Significant'} \\\\\n")
        f.write(f"Efficiency Score & {efficiency_analysis.get('efficiency_score', 0):.1f}/100 & {'Excellent' if efficiency_analysis.get('efficiency_score', 0) > 85 else 'Good' if efficiency_analysis.get('efficiency_score', 0) > 70 else 'Poor'} \\\\\n")
        f.write(f"Predictability Index & {efficiency_analysis.get('predictability_index', 0):.2f} & {'High' if efficiency_analysis.get('predictability_index', 0) > 5 else 'Low'} Predictability \\\\\n")
        f.write(f"Information Processing & {efficiency_analysis.get('information_processing_quality', 'unknown').title()} & Quality Assessment \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Enhanced Coefficient Interpretation
        if coeff_interp:
            f.write("\\subsection{Enhanced Mincer-Zarnowitz Coefficient Analysis}\n")
            alpha_val = coeff_interp.get('alpha_value', 0)
            beta_val = coeff_interp.get('beta_value', 0)
            alpha_ci = coeff_interp.get('alpha_95_ci', [0, 0])
            beta_ci = coeff_interp.get('beta_95_ci', [0, 0])
            
            f.write("\\textbf{Alpha Coefficient Interpretation:}\\\\\n")
            f.write(f"$\\alpha = {alpha_val:.3f}$ (95\\% CI: [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}])\\\\\n")
            f.write(f"\\textit{{{coeff_interp.get('alpha_interpretation', 'No interpretation available')}}}\n\n")
            
            f.write("\\textbf{Beta Coefficient Interpretation:}\\\\\n")
            f.write(f"$\\beta = {beta_val:.3f}$ (95\\% CI: [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}])\\\\\n")
            f.write(f"\\textit{{{coeff_interp.get('beta_interpretation', 'No interpretation available')}}}\n\n")
            
            f.write("\\textbf{Rationality Plausibility Assessment:}\\\\\n")
            f.write(f"$\\alpha = 0$ plausible: {'Yes' if coeff_interp.get('alpha_zero_plausible', False) else 'No'}\\\\\n")
            f.write(f"$\\beta = 1$ plausible: {'Yes' if coeff_interp.get('beta_one_plausible', False) else 'No'}\\\\\n")
            f.write(f"Joint rationality plausible: {'Yes' if coeff_interp.get('joint_rationality_plausible', False) else 'No'}\n\n")
        
        # Enhanced Overall Assessment with Quality Score
        f.write("\\subsection{Comprehensive Assessment Dashboard}\n")
        quality_score = overall_assessment.get('forecast_quality_score', 0)
        quality_cat = overall_assessment.get('forecast_quality', 'unknown')
        rmse = overall_assessment.get('rmse', 0)
        mae = overall_assessment.get('mae', 0)
        r_squared = overall_assessment.get('r_squared', 0)
        
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comprehensive Quality Assessment}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Assessment Dimension} & \\textbf{Value} & \\textbf{Category} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Overall Quality Score & {quality_score:.1f}/100 & {quality_cat.title()} \\\\\n")
        f.write(f"Root Mean Square Error & {rmse:.3f} p.p. & Accuracy Measure \\\\\n")
        f.write(f"Mean Absolute Error & {mae:.3f} p.p. & Precision Measure \\\\\n")
        f.write(f"R-Squared & {r_squared:.3f} & {(r_squared * 100):.1f}\\% Explained \\\\\n")
        
        reh_compatible = overall_assessment.get('reh_compatibility', 'unknown')
        joint_strength = overall_assessment.get('joint_test_strength', 'unknown')
        f.write(f"REH Compatibility & \\textcolor{{academicred}}{{{reh_compatible.upper()}}} & {joint_strength.title()} Evidence \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Scenario Analysis Section
        if scenario_analysis:
            f.write("\\subsection{Policy Scenario Analysis}\n")
            f.write("Following 2024 central bank forecasting standards (Bernanke Review), we present scenario-based assessments:\n\n")
            
            for scenario_name, scenario_data in scenario_analysis.items():
                if isinstance(scenario_data, dict):
                    scenario_title = scenario_name.replace('_', ' ').title()
                    probability = scenario_data.get('probability', 0) * 100
                    expected_mae = scenario_data.get('expected_mae', 0)
                    priority = scenario_data.get('policy_priority', 'unknown')
                    
                    f.write(f"\\textbf{{{scenario_title}}} (Probability: {probability:.0f}\\%):\\\\\n")
                    f.write(f"\\textit{{{scenario_data.get('description', 'No description available')}}}\\\\\n")
                    f.write(f"Expected MAE: {expected_mae:.2f} p.p., Priority: {priority.title()}\n\n")
        
        # Quantitative Summary
        quant_summary = overall_assessment.get('quantitative_summary', {})
        if quant_summary:
            f.write("\\subsection{Key Quantitative Insights}\n")
            f.write("\\begin{itemize}\n")
            f.write(f"\\item Bias magnitude: {quant_summary.get('bias_magnitude', 0):.2f} percentage points\n")
            f.write(f"\\item Efficiency loss: {quant_summary.get('efficiency_loss', 0):.1f}\\% of variation unexplained\n")
            f.write(f"\\item Predictable error component: {quant_summary.get('predictable_error_pct', 0):.1f}\\% of total error\n")
            f.write("\\end{itemize}\n\n")
    
    @staticmethod
    def _write_latex_policy_implications(f, results: Dict) -> None:
        """Write enhanced LaTeX policy implications section with quantitative backing"""
        f.write("\\section{Enhanced Policy Implications}\n\n")
        f.write("Following 2024 forecast evaluation standards with quantitative evidence-based recommendations.\n\n")
        
        econ_interp = results.get("economic_interpretation", {})
        policy_impl = econ_interp.get("policy_implications", {})
        scenario_analysis = econ_interp.get("scenario_analysis", {})
        
        if not policy_impl:
            f.write("Policy implications not available.\n\n")
            return
        
        # Central Bank with Quantitative Targets
        if "central_bank" in policy_impl:
            f.write("\\subsection{For Central Bank Policymakers}\n")
            f.write("\\textbf{Quantitative Evidence-Based Recommendations:}\n\n")
            f.write("\\begin{itemize}\n")
            for implication in policy_impl["central_bank"]:
                f.write(f"\\item {implication}\n")
            f.write("\\end{itemize}\n\n")
            
            # Add specific targets if available
            bias_analysis = econ_interp.get("bias_analysis", {})
            efficiency_analysis = econ_interp.get("efficiency_analysis", {})
            if bias_analysis or efficiency_analysis:
                f.write("\\textbf{Specific Performance Targets:}\n")
                f.write("\\begin{itemize}\n")
                if bias_analysis:
                    magnitude = bias_analysis.get('magnitude_pp', 0)
                    target_reduction = max(magnitude * 0.7, 0.5)  # 70% reduction or 0.5 p.p. minimum
                    f.write(f"\\item Reduce systematic bias from {magnitude:.2f} to <{target_reduction:.2f} percentage points within 24 months\n")
                
                if efficiency_analysis:
                    ljung_box = efficiency_analysis.get('ljung_box_statistic', 0)
                    if ljung_box > 20:
                        f.write(f"\\item Improve efficiency from current LB statistic of {ljung_box:.0f} to <20 within 18 months\n")
                f.write("\\end{itemize}\n\n")
        
        # Market Participants with Quantitative Opportunities
        if "market_participants" in policy_impl:
            f.write("\\subsection{For Market Participants}\n")
            f.write("\\textbf{Quantified Market Opportunities:}\n\n")
            f.write("\\begin{itemize}\n")
            for implication in policy_impl["market_participants"]:
                f.write(f"\\item {implication}\n")
            f.write("\\end{itemize}\n\n")
            
            # Add risk assessment
            overall_assessment = econ_interp.get("overall_assessment", {})
            quality_score = overall_assessment.get('forecast_quality_score', 0)
            rmse = overall_assessment.get('rmse', 0)
            
            f.write("\\textbf{Risk-Return Assessment:}\n")
            f.write("\\begin{itemize}\n")
            f.write(f"\\item Strategy Risk Level: {'High' if quality_score < 50 else 'Moderate' if quality_score < 70 else 'Low'} (Quality Score: {quality_score:.1f}/100)\n")
            f.write(f"\\item Expected Volatility: {rmse:.2f} percentage points RMSE\n")
            if quality_score < 50:
                f.write("\\item \\textcolor{academicred}{WARNING: Very poor forecast quality increases strategy risk}\n")
            f.write("\\end{itemize}\n\n")
        
        # Researchers with Model Specifications
        if "researchers" in policy_impl:
            f.write("\\subsection{For Researchers}\n")
            f.write("\\textbf{Research Priorities with Statistical Evidence:}\n\n")
            f.write("\\begin{itemize}\n")
            for implication in policy_impl["researchers"]:
                f.write(f"\\item {implication}\n")
            f.write("\\end{itemize}\n\n")
            
            # Add model recommendations
            coeff_interp = econ_interp.get("coefficient_interpretation", {})
            if coeff_interp:
                alpha_val = coeff_interp.get('alpha_value', 0)
                beta_val = coeff_interp.get('beta_value', 0)
                
                f.write("\\textbf{Model Development Priorities:}\n")
                f.write("\\begin{itemize}\n")
                
                if beta_val < 0:
                    f.write("\\item \\textbf{URGENT:} Investigate counter-intuitive negative $\\beta$ coefficient - suggests fundamental model misspecification\n")
                elif beta_val < 0.5:
                    f.write("\\item Develop models explaining severe under-response to information ($\\beta$ = {:.3f})\n".format(beta_val))
                
                if abs(alpha_val) > 1.0:
                    f.write("\\item Model systematic bias component ({:.2f} p.p.) - consider regime-switching or time-varying parameter models\n".format(abs(alpha_val)))
                
                overall_r2 = overall_assessment.get('r_squared', 0)
                if overall_r2 < 0.1:
                    f.write("\\item Low explanatory power (R² = {:.3f}) suggests need for alternative theoretical frameworks\n".format(overall_r2))
                
                f.write("\\end{itemize}\n\n")
        
        # Scenario-Based Implementation Strategy
        if scenario_analysis:
            f.write("\\subsection{Scenario-Based Implementation Strategy}\n")
            f.write("\\textbf{Recommended approach based on probabilistic scenarios:}\n\n")
            
            # Sort scenarios by probability
            scenario_items = list(scenario_analysis.items())
            scenario_items.sort(key=lambda x: x[1].get('probability', 0), reverse=True)
            
            f.write("\\begin{enumerate}\n")
            for scenario_name, scenario_data in scenario_items:
                if isinstance(scenario_data, dict):
                    scenario_title = scenario_name.replace('_', ' ').title()
                    probability = scenario_data.get('probability', 0) * 100
                    priority = scenario_data.get('policy_priority', 'unknown')
                    
                    f.write(f"\\item \\textbf{{{scenario_title}}} ({probability:.0f}\\% probability):\n")
                    f.write(f"Priority Level: {priority.title()}\n")
                    
                    actions = scenario_data.get('specific_actions', [])
                    if actions:
                        f.write("\\begin{itemize}\n")
                        for action in actions:
                            f.write(f"\\item {action}\n")
                        f.write("\\end{itemize}\n")
            f.write("\\end{enumerate}\n\n")
            
        # Implementation Timeline
        f.write("\\subsection{Recommended Implementation Timeline}\n")
        f.write("\\textbf{Evidence-based priority sequence:}\n\n")
        f.write("\\begin{description}\n")
        f.write("\\item[Immediate (0-6 months):] Address most severe biases and communication failures\n")
        f.write("\\item[Short-term (6-18 months):] Implement efficiency improvements and forecaster training\n")
        f.write("\\item[Medium-term (18-36 months):] Monitor improvements and adjust strategies based on scenario outcomes\n")
        f.write("\\item[Long-term (36+ months):] Evaluate fundamental model changes if improvements insufficient\n")
        f.write("\\end{description}\n\n")