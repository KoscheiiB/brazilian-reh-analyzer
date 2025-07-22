"""
Tests for enhanced visualization features in v2.0.0 Enhanced Academic Framework
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from unittest.mock import patch
from brazilian_reh_analyzer.visualizations import REHVisualizations


class TestREHVisualizationsV2:
    """Test cases for v2.0.0 enhanced visualization features"""
    
    def setup_method(self):
        """Set up test data for each test"""
        np.random.seed(42)
        
        # Create realistic mock data
        n_obs = 200
        dates = pd.date_range("2020-01-01", periods=n_obs, freq="D")
        
        self.forecast = pd.Series(
            np.random.normal(4.5, 1.0, n_obs),
            index=dates,
            name="forecast"
        )
        
        # Add some systematic bias and autocorrelation for realistic testing
        noise = np.random.normal(0, 0.5, n_obs)
        trend = np.linspace(0, 0.5, n_obs)  # Small trend
        self.realized = pd.Series(
            self.forecast * 0.95 + noise + trend,  # Systematic underestimation
            index=dates,
            name="realized"
        )
        
        self.forecast_errors = self.realized - self.forecast
        
        # Mock comprehensive results for testing
        self.mock_results = {
            "descriptive_stats": {
                "error_mean": -0.25,
                "error_std": 0.8,
                "n_observations": n_obs
            },
            "mincer_zarnowitz": {
                "alpha": 0.1,
                "beta": 0.95,
                "passes_joint_test": False
            },
            "rationality_assessment": {
                "overall_rational": False,
                "unbiased": False,
                "efficient": False
            }
        }
    
    def test_academic_style_setup(self):
        """Test academic style configuration"""
        # Test that setup doesn't raise errors
        REHVisualizations.setup_academic_style()
        
        # Test some key matplotlib rcParams are set correctly
        import matplotlib as mpl
        
        # Check font settings
        assert 'serif' in mpl.rcParams['font.family']
        
        # Check figure settings
        assert mpl.rcParams['figure.dpi'] >= 100
        assert mpl.rcParams['savefig.dpi'] >= 300
        
        # Check grid settings
        assert mpl.rcParams['axes.grid'] is True
    
    def test_academic_colors_defined(self):
        """Test that academic color palette is properly defined"""
        colors = REHVisualizations.ACADEMIC_COLORS
        
        # Test required colors exist
        required_colors = ['primary', 'secondary', 'error', 'success', 'neutral']
        for color_name in required_colors:
            assert color_name in colors
            assert colors[color_name].startswith('#')  # Hex color format
            assert len(colors[color_name]) == 7  # #RRGGBB format
    
    def test_acf_pacf_analysis_plot(self):
        """Test ACF/PACF autocorrelation analysis plots"""
        # Add some autocorrelation to forecast errors for realistic testing
        autocorrelated_errors = self.forecast_errors.copy()
        for i in range(1, len(autocorrelated_errors)):
            autocorrelated_errors.iloc[i] += 0.3 * autocorrelated_errors.iloc[i-1]
        
        # Test plot generation
        fig = REHVisualizations.plot_acf_pacf_analysis(
            autocorrelated_errors,
            max_lags=12,
            title="Test ACF/PACF Analysis"
        )
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 2  # Should have ACF and PACF subplots
        
        # Check that plots have content
        for ax in axes:
            assert len(ax.get_children()) > 0  # Has plot elements
            assert ax.get_title()  # Has title
            assert ax.get_xlabel()  # Has x-label
            assert ax.get_ylabel()  # Has y-label
        
        plt.close(fig)
    
    def test_acf_pacf_with_no_autocorrelation(self):
        """Test ACF/PACF plots with white noise (no autocorrelation)"""
        white_noise = pd.Series(np.random.normal(0, 1, 100))
        
        fig = REHVisualizations.plot_acf_pacf_analysis(white_noise, max_lags=10)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.get_axes()) == 2
        
        plt.close(fig)
    
    def test_enhanced_qq_normality_plot(self):
        """Test enhanced Q-Q normality plots with confidence bands"""
        fig = REHVisualizations.plot_qq_normality(
            self.forecast_errors,
            title="Test Q-Q Normality Analysis"
        )
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        # Check for plot elements
        ax = axes[0]
        assert ax.get_title()
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        
        # Should have multiple line elements (Q-Q line, confidence bands, points)
        lines = ax.get_lines()
        assert len(lines) >= 2  # At least Q-Q line and confidence bands
        
        plt.close(fig)
    
    def test_qq_normality_with_non_normal_data(self):
        """Test Q-Q plot with clearly non-normal data"""
        # Create obviously non-normal data (bimodal)
        non_normal_data = pd.Series(np.concatenate([
            np.random.normal(-2, 0.5, 50),
            np.random.normal(2, 0.5, 50)
        ]))
        
        fig = REHVisualizations.plot_qq_normality(non_normal_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_comprehensive_diagnostics_enhanced(self):
        """Test enhanced comprehensive diagnostic plots"""
        fig = REHVisualizations.create_comprehensive_diagnostics(
            self.forecast,
            self.realized,
            self.mock_results,
            figsize=(16, 10)
        )
        
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 6  # Should have multiple diagnostic plots
        
        # Check figure size
        assert fig.get_size_inches()[0] == 16
        assert fig.get_size_inches()[1] == 10
        
        # Each subplot should have content
        for ax in axes:
            assert len(ax.get_children()) > 0
        
        plt.close(fig)
    
    def test_enhanced_forecast_vs_realization_plot(self):
        """Test enhanced forecast vs realization scatter plot"""
        fig = REHVisualizations.plot_forecast_vs_realization(
            self.forecast,
            self.realized,
            title="Test Enhanced Scatter Plot"
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        
        # Should have scatter points and regression line
        collections = ax.collections
        lines = ax.get_lines()
        assert len(collections) >= 1 or len(lines) >= 1  # Scatter or line plot
        
        # Check labels and title
        assert ax.get_title()
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        
        plt.close(fig)
    
    def test_enhanced_error_timeseries_plot(self):
        """Test enhanced forecast error time series plot"""
        fig = REHVisualizations.plot_forecast_errors_timeseries(
            self.forecast_errors,
            title="Test Enhanced Time Series"
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        
        # Should have line plot
        lines = ax.get_lines()
        assert len(lines) >= 1
        
        # Check for confidence bands or zero line
        assert len(ax.get_children()) > 1  # More than just the main line
        
        plt.close(fig)
    
    def test_enhanced_error_distribution_plot(self):
        """Test enhanced error distribution plot"""
        fig = REHVisualizations.plot_error_distribution(
            self.forecast_errors,
            title="Test Enhanced Distribution"
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        
        # Should have histogram or KDE plot
        patches = ax.patches  # Histogram bars
        lines = ax.get_lines()  # KDE line
        assert len(patches) > 0 or len(lines) > 0
        
        plt.close(fig)
    
    def test_plot_export_functionality(self):
        """Test plot export with high DPI settings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test individual plot export
            fig = REHVisualizations.plot_forecast_vs_realization(
                self.forecast,
                self.realized
            )
            
            # Save with high DPI
            output_file = os.path.join(temp_dir, "test_plot.png")
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            
            assert os.path.exists(output_file)
            
            # Check file size is reasonable (high DPI should create larger files)
            file_size = os.path.getsize(output_file)
            assert file_size > 10000  # Should be reasonably large for 300 DPI
            
            plt.close(fig)
    
    def test_plot_export_to_files(self):
        """Test comprehensive plot export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            REHVisualizations.export_plots_to_files(
                self.forecast,
                self.realized,
                self.mock_results,
                output_dir=temp_dir,
                dpi=150
            )
            
            # Check that multiple files were created
            plot_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
            assert len(plot_files) >= 4  # Should have multiple diagnostic plots
            
            # Check specific expected files
            file_names = [f.lower() for f in plot_files]
            assert any("forecast_vs_realization" in name for name in file_names)
            assert any("error_distribution" in name for name in file_names)
            assert any("errors_timeseries" in name for name in file_names)
    
    def test_colorblind_friendly_colors(self):
        """Test that academic colors are suitable for colorblind users"""
        colors = REHVisualizations.ACADEMIC_COLORS
        
        # Test that colors have sufficient contrast (basic check)
        # This is a simplified test - in practice you'd use more sophisticated
        # colorblind simulation tools
        
        def hex_to_rgb(hex_color):
            """Convert hex color to RGB values"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Test that primary colors are distinct enough
        primary_rgb = hex_to_rgb(colors['primary'])
        secondary_rgb = hex_to_rgb(colors['secondary'])
        
        # Basic color distance check
        color_distance = sum((a - b) ** 2 for a, b in zip(primary_rgb, secondary_rgb))
        assert color_distance > 10000  # Should be sufficiently different
    
    def test_academic_style_font_handling(self):
        """Test academic style font configuration"""
        # Test that font configuration doesn't break on systems without certain fonts
        try:
            REHVisualizations.setup_academic_style()
            
            # Create a simple plot to test font rendering
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title("Test Academic Font")
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            
            # This should not raise font-related errors
            plt.close(fig)
            
        except Exception as e:
            # If font issues occur, they should be handled gracefully
            assert "font" not in str(e).lower() or "glyph" not in str(e).lower()
    
    def test_plot_scaling_and_sizing(self):
        """Test plot scaling for different figure sizes"""
        sizes = [(8, 6), (12, 8), (16, 12)]
        
        for width, height in sizes:
            fig = REHVisualizations.create_comprehensive_diagnostics(
                self.forecast,
                self.realized,
                self.mock_results,
                figsize=(width, height)
            )
            
            actual_size = fig.get_size_inches()
            assert abs(actual_size[0] - width) < 0.1
            assert abs(actual_size[1] - height) < 0.1
            
            plt.close(fig)


class TestVisualizationEdgeCases:
    """Test visualization edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """Test visualization with empty or minimal data"""
        empty_series = pd.Series([], dtype=float)
        
        # Should handle empty data gracefully
        with pytest.raises(ValueError, match="insufficient data|empty"):
            REHVisualizations.plot_forecast_vs_realization(empty_series, empty_series)
    
    def test_single_value_data(self):
        """Test visualization with single data point"""
        single_series = pd.Series([1.0])
        
        # Should handle single value gracefully or raise appropriate error
        try:
            fig = REHVisualizations.plot_forecast_vs_realization(
                single_series, single_series
            )
            plt.close(fig)
        except ValueError:
            pass  # Expected for insufficient data
    
    def test_constant_data_handling(self):
        """Test visualization with constant values"""
        constant_series = pd.Series([4.0] * 100)
        
        # Should handle constant values without errors
        fig = REHVisualizations.plot_forecast_vs_realization(
            constant_series, constant_series
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_extreme_values_handling(self):
        """Test visualization with extreme outliers"""
        normal_data = pd.Series(np.random.normal(0, 1, 100))
        extreme_data = normal_data.copy()
        extreme_data.iloc[50] = 1000  # Extreme outlier
        
        # Should handle extreme values without breaking
        fig = REHVisualizations.plot_error_distribution(extreme_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])