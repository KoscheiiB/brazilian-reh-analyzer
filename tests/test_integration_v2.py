"""
Integration tests for v2.0.0 Enhanced Academic Framework
Tests complete workflows and component interactions
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from brazilian_reh_analyzer import BrazilianREHAnalyzer
from brazilian_reh_analyzer.data_fetcher import RespectfulBCBClient, DataCache
from brazilian_reh_analyzer.utils import ProgressTracker, format_results_summary


class TestEndToEndWorkflows:
    """Test complete end-to-end analysis workflows"""
    
    def setup_method(self):
        """Set up for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
    
    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_analysis_workflow_with_mock_data(self):
        """Test complete analysis workflow from initialization to export"""
        # Initialize analyzer
        analyzer = BrazilianREHAnalyzer(
            start_date="2020-01-01",
            end_date="2023-12-31",
            cache_dir=os.path.join(self.temp_dir, "cache")
        )
        
        # Mock the data fetching to avoid API calls
        self._mock_data_for_analyzer(analyzer)
        
        # Run complete analysis
        results = analyzer.comprehensive_analysis(fetch_data=False)
        
        # Verify complete results structure
        self._verify_complete_results_structure(results)
        
        # Test all export functionalities
        self._test_all_exports(analyzer)
        
        # Test plot generation
        analyzer.plot_enhanced_diagnostics(show_plots=False)
    
    def _mock_data_for_analyzer(self, analyzer):
        """Create comprehensive mock data for analyzer"""
        # Mock IPCA data (monthly, realistic Brazilian inflation)
        ipca_dates = pd.date_range("2019-01-31", "2024-06-30", freq="ME")
        ipca_values = []
        base_rate = 4.0
        
        for i, date in enumerate(ipca_dates):
            # Add seasonal and crisis effects
            seasonal = 0.5 * np.sin(2 * np.pi * i / 12)
            crisis_effect = 2.0 if date.year == 2020 else 0.0
            noise = np.random.normal(0, 1.0)
            ipca_values.append(base_rate + seasonal + crisis_effect + noise)
        
        analyzer.ipca_data = pd.Series(ipca_values, index=ipca_dates, name="IPCA_12m")
        
        # Mock Focus data (daily)
        focus_dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
        n_focus = len(focus_dates)
        
        # Create realistic Focus forecasts with systematic bias
        base_forecast = 4.5  # Slight overestimation
        trend = np.linspace(0, 0.5, n_focus)  # Small trend
        seasonal_focus = 0.3 * np.sin(2 * np.pi * np.arange(n_focus) / 365)
        noise_focus = np.random.normal(0, 0.8, n_focus)
        
        forecasts_median = base_forecast + trend + seasonal_focus + noise_focus
        forecasts_mean = forecasts_median + np.random.normal(0, 0.1, n_focus)
        respondents = np.random.randint(15, 45, n_focus)
        
        analyzer.focus_data = pd.DataFrame({
            "Mediana": forecasts_median,
            "Media": forecasts_mean,
            "numeroRespondentes": respondents
        }, index=focus_dates)
        
        # Align the data so analyzer can use it
        analyzer.aligned_data = analyzer.align_forecast_realization_data()
        analyzer.forecast_errors = analyzer.aligned_data['forecast_error']
    
    def _verify_complete_results_structure(self, results):
        """Verify the complete structure of v2.0.0 results"""
        # Basic components
        basic_keys = [
            "descriptive_stats", "mincer_zarnowitz", "autocorrelation", 
            "bias_test", "rationality_assessment"
        ]
        for key in basic_keys:
            assert key in results
        
        # Verify basic structure of each component
        assert isinstance(results["descriptive_stats"], dict)
        assert isinstance(results["mincer_zarnowitz"], dict)
        assert isinstance(results["autocorrelation"], dict)
        assert isinstance(results["bias_test"], dict)
        assert isinstance(results["rationality_assessment"], dict)
        
        # Verify key elements in mincer_zarnowitz
        mz = results["mincer_zarnowitz"]
        for key in ["alpha", "beta", "joint_test_pvalue", "passes_joint_test"]:
            assert key in mz
        
        # Verify rationality assessment
        ra = results["rationality_assessment"]
        for key in ["overall_rational", "n_observations"]:
            assert key in ra
        
        # Verify descriptive stats has basic information
        desc = results["descriptive_stats"]
        assert isinstance(desc, dict)
    
    def _test_all_exports(self, analyzer):
        """Test all export functionalities"""
        # Test text summary export
        summary_file = os.path.join(self.temp_dir, "test_summary.txt")
        analyzer.export_results_summary(summary_file)
        assert os.path.exists(summary_file)
        
        # Verify summary content
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        assert "BRAZILIAN REH ANALYZER" in summary_content
        assert "Analysis Date" in summary_content
        assert len(summary_content) > 100  # Basic check for meaningful content
        
        # Test data export
        data_file = os.path.join(self.temp_dir, "test_data.csv")
        analyzer.save_data(data_file)
        assert os.path.exists(data_file)
        
        # Test plots export
        plots_dir = os.path.join(self.temp_dir, "plots")
        analyzer.export_plots(plots_dir, dpi=150)
        assert os.path.exists(plots_dir)
        assert len(os.listdir(plots_dir)) >= 4  # Multiple plot files
        
        # Test LaTeX export
        latex_file = os.path.join(self.temp_dir, "test_report.tex")
        returned_path = analyzer.export_latex_report(
            latex_file,
            "Integration Test Report",
            "Test Framework"
        )
        assert returned_path == latex_file
        assert os.path.exists(latex_file)
        
        # Verify LaTeX content structure
        with open(latex_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        latex_requirements = [
            "\\documentclass", "\\begin{document}", "\\end{document}",
            "Integration Test Report", "Test Framework",
            "\\section{", "booktabs", "\\alpha", "\\beta"
        ]
        for requirement in latex_requirements:
            assert requirement in latex_content
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple periods"""
        periods = {
            "Period1": ("2020-01-01", "2021-12-31"),
            "Period2": ("2021-01-01", "2022-12-31"),
            "Period3": ("2022-01-01", "2023-12-31")
        }
        
        batch_results = {}
        
        for period_name, (start_date, end_date) in periods.items():
            analyzer = BrazilianREHAnalyzer(
                start_date=start_date,
                end_date=end_date,
                cache_dir=os.path.join(self.temp_dir, f"cache_{period_name}")
            )
            
            # Mock data for this period
            self._mock_data_for_analyzer(analyzer)
            
            # Run analysis
            results = analyzer.comprehensive_analysis(fetch_data=False)
            batch_results[period_name] = results
            
            # Export results for this period
            period_dir = os.path.join(self.temp_dir, period_name)
            os.makedirs(period_dir, exist_ok=True)
            
            analyzer.export_results_summary(
                os.path.join(period_dir, f"{period_name}_summary.txt")
            )
            analyzer.export_plots(
                os.path.join(period_dir, "plots"),
                dpi=150
            )
        
        # Verify all periods processed
        assert len(batch_results) == 3
        for period_name in periods.keys():
            assert period_name in batch_results
            self._verify_complete_results_structure(batch_results[period_name])
    
    def test_external_variables_integration_workflow(self):
        """Test integration with external macroeconomic variables"""
        analyzer = BrazilianREHAnalyzer(
            start_date="2020-01-01",
            end_date="2023-12-31",
            cache_dir=os.path.join(self.temp_dir, "cache_external")
        )
        
        # Mock main data
        self._mock_data_for_analyzer(analyzer)
        
        # Create external variables
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        external_vars = pd.DataFrame({
            "selic_rate": np.random.normal(10.5, 2.0, len(dates)),
            "exchange_rate": np.random.normal(5.2, 0.8, len(dates)),
            "gdp_growth": np.random.normal(2.1, 1.5, len(dates)),
            "unemployment": np.random.normal(12.0, 2.5, len(dates))
        }, index=dates)
        
        # Run analysis with external variables
        results = analyzer.comprehensive_analysis(
            fetch_data=False,
            external_vars=external_vars
        )
        
        # Verify orthogonality test was included
        assert "orthogonality" in results
        orthogonality = results["orthogonality"]
        assert "passes_orthogonality_test" in orthogonality
        assert "variable_results" in orthogonality
        
        # Verify all external variables were tested
        var_results = orthogonality["variable_results"]
        for var_name in external_vars.columns:
            assert var_name in var_results


class TestDataCachingIntegration:
    """Test data caching and persistence integration"""
    
    def setup_method(self):
        """Set up caching tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up caching tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_cache_integration(self):
        """Test data caching integration"""
        cache_dir = os.path.join(self.temp_dir, "test_cache")
        cache = DataCache(cache_dir)
        
        # Create test data
        test_data = pd.DataFrame({
            "test_col": [1, 2, 3, 4, 5],
            "date": pd.date_range("2020-01-01", periods=5)
        })
        
        # Test save and load
        cache.save_data(test_data, "test_data", "2020-01-01", "2020-12-31")
        loaded_data = cache.load_data("test_data", "2020-01-01", "2020-12-31", max_age_hours=24)
        
        assert loaded_data is not None
        pd.testing.assert_frame_equal(test_data, loaded_data)
        
        # Test cache directory structure
        assert os.path.exists(cache_dir)
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) >= 1
    
    def test_progress_tracking_integration(self):
        """Test progress tracking throughout analysis"""
        tracker = ProgressTracker(5, "Integration Test")
        
        steps = [
            "Initializing analyzer",
            "Fetching IPCA data", 
            "Fetching Focus data",
            "Running statistical tests",
            "Generating results"
        ]
        
        for step in steps:
            tracker.update(step)
        
        tracker.finish()
        
        # Should complete without errors
        assert tracker.current_step == len(steps)


class TestErrorHandlingIntegration:
    """Test error handling throughout integrated workflows"""
    
    def test_missing_data_error_handling(self):
        """Test graceful handling of missing data scenarios"""
        analyzer = BrazilianREHAnalyzer()
        
        # Test analysis without any data
        with pytest.raises(ValueError, match="Must fetch both IPCA and Focus data first"):
            analyzer.align_forecast_realization_data()
        
        # Test analysis without aligned data
        with pytest.raises(ValueError, match="No aligned data available"):
            analyzer.comprehensive_analysis(fetch_data=False)
    
    def test_export_without_results_error_handling(self):
        """Test export functions fail gracefully without results"""
        analyzer = BrazilianREHAnalyzer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test exports without results
            with pytest.raises(ValueError, match="No results available"):
                analyzer.export_results_summary(os.path.join(temp_dir, "test.txt"))
            
            with pytest.raises(ValueError, match="No results available"):
                analyzer.export_latex_report(
                    os.path.join(temp_dir, "test.tex"),
                    "Test", "Test"
                )
    
    def test_invalid_date_range_handling(self):
        """Test handling of invalid date ranges"""
        # End date before start date
        with pytest.raises(ValueError):
            BrazilianREHAnalyzer(
                start_date="2023-01-01",
                end_date="2022-01-01"
            )
        
        # Invalid date format
        with pytest.raises(ValueError):
            BrazilianREHAnalyzer(
                start_date="invalid-date",
                end_date="2023-01-01"
            )


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets (performance test)"""
        # Create large mock dataset
        n_large = 2000  # Large number of observations
        
        analyzer = BrazilianREHAnalyzer()
        
        # Create large IPCA dataset
        ipca_dates = pd.date_range("2010-01-31", periods=int(n_large/12), freq="ME")
        analyzer.ipca_data = pd.Series(
            np.random.normal(4.0, 2.0, len(ipca_dates)),
            index=ipca_dates
        )
        
        # Create large Focus dataset
        focus_dates = pd.date_range("2010-01-01", periods=n_large, freq="D")
        analyzer.focus_data = pd.DataFrame({
            "Mediana": np.random.normal(4.5, 1.5, n_large),
            "Media": np.random.normal(4.6, 1.6, n_large),
            "numeroRespondentes": np.random.randint(15, 50, n_large)
        }, index=focus_dates)
        
        # Should handle large dataset without excessive memory usage or time
        import time
        start_time = time.time()
        
        results = analyzer.comprehensive_analysis(fetch_data=False)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Analysis should complete in reasonable time (< 30 seconds)
        assert analysis_time < 30.0
        
        # Results should still be complete
        assert isinstance(results, dict)
        assert "rationality_assessment" in results
    
    def test_memory_usage_monitoring(self):
        """Test that memory usage remains reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple analyses
        for i in range(3):
            analyzer = BrazilianREHAnalyzer()
            
            # Mock medium-sized datasets
            n_obs = 500
            ipca_dates = pd.date_range("2020-01-31", periods=int(n_obs/12), freq="ME")
            analyzer.ipca_data = pd.Series(
                np.random.normal(4.0, 1.5, len(ipca_dates)),
                index=ipca_dates
            )
            
            focus_dates = pd.date_range("2020-01-01", periods=n_obs, freq="D")
            analyzer.focus_data = pd.DataFrame({
                "Mediana": np.random.normal(4.5, 1.0, n_obs),
                "Media": np.random.normal(4.5, 1.0, n_obs),
                "numeroRespondentes": np.random.randint(20, 40, n_obs)
            }, index=focus_dates)
            
            analyzer.comprehensive_analysis(fetch_data=False)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100 MB for 3 analyses)
        assert memory_increase < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])