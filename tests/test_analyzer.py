"""
Tests for the main BrazilianREHAnalyzer class
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from brazilian_reh_analyzer.analyzer import BrazilianREHAnalyzer


class TestBrazilianREHAnalyzer:
    """Test cases for BrazilianREHAnalyzer"""
    
    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = BrazilianREHAnalyzer(
            start_date="2020-01-01",
            end_date="2023-12-31",
            cache_dir="test_cache"
        )
        
        assert analyzer.start_date == pd.to_datetime("2020-01-01")
        assert analyzer.end_date == pd.to_datetime("2023-12-31")
        assert analyzer.ipca_data is None
        assert analyzer.focus_data is None
        assert analyzer.aligned_data is None
        assert analyzer.forecast_errors is None
    
    def test_alignment_with_mock_data(self):
        """Test data alignment with mock data"""
        analyzer = BrazilianREHAnalyzer()
        
        # Create mock IPCA data
        ipca_dates = pd.date_range("2017-01-31", "2023-12-31", freq="M")
        analyzer.ipca_data = pd.Series(
            np.random.normal(4.0, 2.0, len(ipca_dates)),
            index=ipca_dates,
            name="IPCA_12m"
        )
        
        # Create mock Focus data
        focus_dates = pd.date_range("2017-01-01", "2022-12-31", freq="D")
        analyzer.focus_data = pd.DataFrame({
            "Mediana": np.random.normal(4.5, 1.5, len(focus_dates)),
            "Media": np.random.normal(4.6, 1.6, len(focus_dates)),
            "numeroRespondentes": np.random.randint(15, 50, len(focus_dates))
        }, index=focus_dates)
        
        # Test alignment
        aligned_data = analyzer.align_forecast_realization_data()
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert "forecast" in aligned_data.columns
        assert "realized" in aligned_data.columns
        assert "forecast_error" in aligned_data.columns
        assert len(aligned_data) > 0
    
    def test_results_summary_without_results(self):
        """Test results summary when no analysis has been run"""
        analyzer = BrazilianREHAnalyzer()
        summary = analyzer.get_results_summary()
        assert "No results available" in summary
    
    @patch('brazilian_reh_analyzer.data_fetcher.BCB_AVAILABLE', False)
    def test_initialization_without_bcb(self):
        """Test that analyzer handles missing BCB library gracefully"""
        with pytest.raises(ImportError):
            analyzer = BrazilianREHAnalyzer()
            analyzer.fetch_ipca_data()


class TestDataValidation:
    """Test data validation and error handling"""
    
    def test_alignment_without_data(self):
        """Test alignment fails without data"""
        analyzer = BrazilianREHAnalyzer()
        
        with pytest.raises(ValueError, match="Must fetch both IPCA and Focus data first"):
            analyzer.align_forecast_realization_data()
    
    def test_analysis_without_aligned_data(self):
        """Test analysis fails without aligned data"""
        analyzer = BrazilianREHAnalyzer()
        
        with pytest.raises(ValueError, match="No aligned data available"):
            analyzer.comprehensive_analysis(fetch_data=False)
    
    def test_export_without_results(self):
        """Test export fails without results"""
        analyzer = BrazilianREHAnalyzer()
        
        with pytest.raises(ValueError, match="No results to export"):
            analyzer.export_results_summary()


if __name__ == "__main__":
    pytest.main([__file__])