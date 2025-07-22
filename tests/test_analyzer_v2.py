"""
Comprehensive tests for BrazilianREHAnalyzer v2.0.0 Enhanced Academic Framework
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from brazilian_reh_analyzer.analyzer import BrazilianREHAnalyzer


class TestBrazilianREHAnalyzerV2:
    """Test cases for v2.0.0 enhanced features"""
    
    def setup_method(self):
        """Set up test data for each test"""
        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize analyzer
        self.analyzer = BrazilianREHAnalyzer(
            start_date="2020-01-01",
            end_date="2023-12-31",
            cache_dir=os.path.join(self.temp_dir, "cache")
        )
        
        # Create comprehensive mock data
        self._setup_mock_data()
    
    def _setup_mock_data(self):
        """Set up realistic mock data"""
        # Mock IPCA data (monthly)
        ipca_dates = pd.date_range("2019-01-31", "2024-12-31", freq="ME")
        self.analyzer.ipca_data = pd.Series(
            np.random.normal(4.0, 1.5, len(ipca_dates)),
            index=ipca_dates,
            name="IPCA_12m"
        )
        
        # Mock Focus data (daily)
        focus_dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
        self.analyzer.focus_data = pd.DataFrame({
            "Mediana": np.random.normal(4.5, 1.2, len(focus_dates)),
            "Media": np.random.normal(4.6, 1.3, len(focus_dates)),
            "numeroRespondentes": np.random.randint(15, 50, len(focus_dates))
        }, index=focus_dates)
        
        # Create aligned data
        self.analyzer.aligned_data = self.analyzer.align_forecast_realization_data()
        self.analyzer.forecast_errors = self.analyzer.aligned_data['forecast_error']
    
    def test_enhanced_comprehensive_analysis(self):
        """Test v2.0.0 enhanced comprehensive analysis"""
        results = self.analyzer.comprehensive_analysis(fetch_data=False)
        
        # Test basic structure
        assert isinstance(results, dict)
        assert "descriptive_stats" in results
        assert "rationality_assessment" in results
        
        # Test v2.0.0 enhanced features
        assert "rich_descriptive_stats" in results
        assert "detailed_mincer_zarnowitz" in results
        assert "sub_period_analysis" in results
        assert "rolling_window_analysis" in results
        assert "economic_interpretation" in results
        
        # Test rich descriptive statistics
        rich_stats = results["rich_descriptive_stats"]
        assert "errors" in rich_stats
        assert "forecast" in rich_stats
        assert "realized" in rich_stats
        
        error_stats = rich_stats["errors"]
        assert "skewness" in error_stats
        assert "kurtosis" in error_stats
        assert "q25" in error_stats
        assert "q75" in error_stats
        assert "n_obs" in error_stats
        
        # Test detailed Mincer-Zarnowitz
        detailed_mz = results["detailed_mincer_zarnowitz"]
        assert "alpha_95_ci" in detailed_mz
        assert "beta_95_ci" in detailed_mz
        assert "joint_f_statistic" in detailed_mz
        assert "alpha_significant" in detailed_mz
        assert "beta_significantly_different_from_1" in detailed_mz
        
        # Test economic interpretation
        econ_interp = results["economic_interpretation"]
        assert "bias_analysis" in econ_interp
        assert "policy_implications" in econ_interp
        
        bias_analysis = econ_interp["bias_analysis"]
        assert "severity" in bias_analysis
        assert "direction" in bias_analysis
        assert "economic_significance" in bias_analysis
    
    def test_sub_period_analysis(self):
        """Test automatic structural break detection and sub-period analysis"""
        results = self.analyzer.comprehensive_analysis(fetch_data=False)
        sub_periods = results["sub_period_analysis"]
        
        # Should detect at least some sub-periods for multi-year data
        assert isinstance(sub_periods, dict)
        assert len(sub_periods) >= 1
        
        # Check structure of each sub-period
        for period_name, period_data in sub_periods.items():
            assert "start_date" in period_data
            assert "end_date" in period_data
            assert "n_observations" in period_data
            assert "mean_error" in period_data
            assert "bias_direction" in period_data
            assert "reh_tests" in period_data
    
    def test_rolling_window_analysis(self):
        """Test rolling window time-varying bias analysis"""
        results = self.analyzer.comprehensive_analysis(fetch_data=False)
        rolling_analysis = results["rolling_window_analysis"]
        
        assert "window_size" in rolling_analysis
        assert "max_abs_bias" in rolling_analysis
        assert "min_abs_bias" in rolling_analysis
        assert "bias_range" in rolling_analysis
        assert "significant_mean_changes" in rolling_analysis
        assert "significant_volatility_changes" in rolling_analysis
        
        # Window size should be reasonable
        assert rolling_analysis["window_size"] > 20
    
    def test_economic_interpretation_engine(self):
        """Test automated economic interpretation engine"""
        results = self.analyzer.comprehensive_analysis(fetch_data=False)
        econ_interp = results["economic_interpretation"]
        
        # Test bias analysis
        bias_analysis = econ_interp["bias_analysis"]
        assert bias_analysis["severity"] in ["minimal", "moderate", "substantial", "severe"]
        assert bias_analysis["direction"] in ["overestimation", "underestimation", "none"]
        assert bias_analysis["economic_significance"] in ["low", "moderate", "high"]
        
        # Test policy implications
        policy_impl = econ_interp["policy_implications"]
        assert "central_bank" in policy_impl
        assert "market_participants" in policy_impl
        assert "researchers" in policy_impl
        
        # Each should be a non-empty list or string
        for key, value in policy_impl.items():
            assert isinstance(value, (str, list))
            if isinstance(value, list):
                assert len(value) > 0
                assert all(isinstance(item, str) for item in value)
            else:
                assert len(value) > 0
    
    def test_latex_export(self):
        """Test LaTeX report generation"""
        # First run analysis
        results = self.analyzer.comprehensive_analysis(fetch_data=False)
        
        # Test LaTeX export
        latex_file = os.path.join(self.temp_dir, "test_report.tex")
        returned_path = self.analyzer.export_latex_report(
            latex_file,
            "Test Brazilian REH Analysis",
            "Test Author"
        )
        
        assert returned_path == latex_file
        assert os.path.exists(latex_file)
        
        # Read and verify LaTeX content
        with open(latex_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        # Check essential LaTeX elements
        assert "\\documentclass" in latex_content
        assert "\\begin{document}" in latex_content
        assert "\\end{document}" in latex_content
        assert "Test Brazilian REH Analysis" in latex_content
        assert "Test Author" in latex_content
        
        # Check academic features
        assert "booktabs" in latex_content  # Professional tables
        assert "\\toprule" in latex_content
        assert "\\definecolor{academicred}" in latex_content  # Academic colors
        assert "\\alpha" in latex_content  # Mathematical notation
        assert "\\beta" in latex_content
        
        # Check structured content
        assert "\\section{" in latex_content
        assert "Executive Summary" in latex_content
        assert "Comprehensive Descriptive Statistics" in latex_content
    
    def test_latex_export_without_results(self):
        """Test LaTeX export fails without analysis results"""
        latex_file = os.path.join(self.temp_dir, "empty_report.tex")
        
        with pytest.raises(ValueError, match="No results available"):
            self.analyzer.export_latex_report(
                latex_file,
                "Test Report",
                "Test Author"
            )
    
    def test_enhanced_plots_generation(self):
        """Test enhanced diagnostic plots with v2.0.0 features"""
        # Run analysis first
        self.analyzer.comprehensive_analysis(fetch_data=False)
        
        # Test enhanced plots generation
        fig = self.analyzer.plot_enhanced_diagnostics(show_plots=False)
        
        assert fig is not None
        assert len(fig.get_axes()) >= 6  # Should have multiple subplots
        
        # Test plots export
        plots_dir = os.path.join(self.temp_dir, "plots")
        self.analyzer.export_plots(plots_dir, dpi=150)
        
        assert os.path.exists(plots_dir)
        plot_files = os.listdir(plots_dir)
        
        # Should have multiple plot files
        assert len(plot_files) >= 4
        
        # Check for specific enhanced plots
        plot_names = [f.lower() for f in plot_files]
        assert any("comprehensive_diagnostics" in name for name in plot_names)  # Comprehensive diagnostics with ACF/PACF
        assert any("qq_plot" in name for name in plot_names)  # Enhanced Q-Q plots
    
    def test_data_alignment_robustness(self):
        """Test data alignment handles edge cases properly"""
        # Test with missing data
        self.analyzer.ipca_data = self.analyzer.ipca_data.dropna()
        self.analyzer.focus_data = self.analyzer.focus_data.dropna()
        
        aligned_data = self.analyzer.align_forecast_realization_data()
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert len(aligned_data) > 0
        assert not aligned_data.isnull().any().any()
        
        # Test required columns
        required_cols = ['forecast', 'realized', 'forecast_error', 'respondents', 'forecast_mean']
        for col in required_cols:
            assert col in aligned_data.columns
    
    def test_results_caching_and_persistence(self):
        """Test that results are properly cached and can be reused"""
        # Run analysis
        results1 = self.analyzer.comprehensive_analysis(fetch_data=False)
        
        # Results should be stored
        assert hasattr(self.analyzer, 'results')
        assert self.analyzer.results is not None
        
        # Second call should use cached results
        results2 = self.analyzer.comprehensive_analysis(fetch_data=False)
        
        # Results should be identical
        assert results1.keys() == results2.keys()
        assert results1['descriptive_stats'] == results2['descriptive_stats']
    
    def test_error_handling_invalid_dates(self):
        """Test error handling with invalid date ranges"""
        with pytest.raises(ValueError):
            BrazilianREHAnalyzer(
                start_date="2025-01-01",  # Future date
                end_date="2024-01-01"     # Before start date
            )
    
    def test_export_summary_enhanced_format(self):
        """Test enhanced results summary export"""
        # Run analysis
        self.analyzer.comprehensive_analysis(fetch_data=False)
        
        # Export summary
        summary_file = os.path.join(self.temp_dir, "enhanced_summary.txt")
        self.analyzer.export_results_summary(summary_file)
        
        assert os.path.exists(summary_file)
        
        # Read and verify content
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        # Check for v2.0.0 enhanced content
        assert "ENHANCED ACADEMIC FRAMEWORK" in summary_content
        assert "STATISTICAL INTERPRETATION:" in summary_content
        assert "ECONOMIC INTERPRETATION:" in summary_content
        assert "STRUCTURAL BREAK INTERPRETATION:" in summary_content
        assert "BIAS ANALYSIS:" in summary_content
        
        # Check for economic significance indicators
        assert "EFFICIENCY ANALYSIS:" in summary_content
        assert "OVERALL ASSESSMENT:" in summary_content
        
        # Check for policy implications
        assert "FOR CENTRAL BANK POLICYMAKERS:" in summary_content
        assert "FOR MARKET PARTICIPANTS:" in summary_content


class TestEnhancedDataHandling:
    """Test enhanced data handling and validation in v2.0.0"""
    
    def test_robust_data_validation(self):
        """Test robust data validation with various edge cases"""
        analyzer = BrazilianREHAnalyzer()
        
        # Test with insufficient data
        analyzer.ipca_data = pd.Series([1, 2], index=pd.date_range("2020-01-01", periods=2, freq="D"))
        analyzer.focus_data = pd.DataFrame({"Mediana": [1, 2]}, index=pd.date_range("2020-01-01", periods=2, freq="D"))
        
        with pytest.raises(ValueError, match="No aligned data available"):
            analyzer.comprehensive_analysis(fetch_data=False)
    
    def test_external_variables_integration(self):
        """Test integration of external variables for orthogonality testing"""
        analyzer = BrazilianREHAnalyzer()
        
        # Setup mock data
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        analyzer.focus_data = pd.DataFrame({
            "Mediana": np.random.normal(4.5, 1.0, len(dates)),
            "Media": np.random.normal(4.5, 1.0, len(dates)),
            "numeroRespondentes": np.random.randint(20, 50, len(dates))
        }, index=dates)
        
        ipca_dates = pd.date_range("2019-12-31", "2024-12-31", freq="ME")
        analyzer.ipca_data = pd.Series(
            np.random.normal(4.0, 1.5, len(ipca_dates)),
            index=ipca_dates
        )
        
        # Create external variables
        external_vars = pd.DataFrame({
            "selic_rate": np.random.normal(10.0, 2.0, len(dates)),
            "exchange_rate": np.random.normal(5.0, 0.5, len(dates))
        }, index=dates)
        
        # Align data before analysis
        analyzer.aligned_data = analyzer.align_forecast_realization_data()
        analyzer.forecast_errors = analyzer.aligned_data['forecast_error']
        
        # Run analysis with external variables
        results = analyzer.comprehensive_analysis(
            fetch_data=False,
            external_vars=external_vars
        )
        
        # Should include orthogonality test results
        assert "orthogonality" in results
        orthogonality = results["orthogonality"]
        assert "passes_orthogonality_test" in orthogonality
        assert "variable_results" in orthogonality


if __name__ == "__main__":
    pytest.main([__file__, "-v"])