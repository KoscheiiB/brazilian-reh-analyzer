"""
Tests for econometric statistical tests
"""

import pytest
import pandas as pd
import numpy as np
from brazilian_reh_analyzer.tests import REHTests


class TestREHTests:
    """Test cases for REH statistical tests"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        n_obs = 100
        
        # Create mock forecast and realized data
        self.forecast = pd.Series(
            np.random.normal(4.0, 1.0, n_obs),
            index=pd.date_range("2017-01-01", periods=n_obs, freq="D"),
            name="forecast"
        )
        
        # Realized values with some correlation to forecasts but with errors
        self.realized = pd.Series(
            self.forecast + np.random.normal(0, 0.5, n_obs),
            index=self.forecast.index,
            name="realized"
        )
        
        self.forecast_errors = self.realized - self.forecast
    
    def test_mincer_zarnowitz_test(self):
        """Test Mincer-Zarnowitz regression test"""
        result = REHTests.mincer_zarnowitz_test(self.forecast, self.realized)
        
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "beta" in result
        assert "joint_test_pvalue" in result
        assert "r_squared" in result
        assert "passes_joint_test" in result
        
        # Check that beta is reasonably close to 1 and alpha close to 0
        assert 0.5 < result["beta"] < 1.5
        assert -1.0 < result["alpha"] < 1.0
    
    def test_mincer_zarnowitz_insufficient_data(self):
        """Test MZ test with insufficient data"""
        short_forecast = self.forecast.head(5)
        short_realized = self.realized.head(5)
        
        result = REHTests.mincer_zarnowitz_test(short_forecast, short_realized)
        assert "error" in result
    
    def test_autocorrelation_test(self):
        """Test autocorrelation test"""
        result = REHTests.autocorrelation_test(self.forecast_errors, max_lags=5)
        
        assert isinstance(result, dict)
        assert "ljung_box_stat" in result
        assert "ljung_box_pvalue" in result
        assert "significant_autocorr" in result
        assert "passes_efficiency_test" in result
        assert "max_lags_tested" in result
    
    def test_autocorrelation_insufficient_data(self):
        """Test autocorrelation test with insufficient data"""
        short_errors = self.forecast_errors.head(3)
        
        result = REHTests.autocorrelation_test(short_errors, max_lags=10)
        assert "error" in result or result["max_lags_tested"] < 10
    
    def test_bias_test(self):
        """Test bias test"""
        result = REHTests.bias_test(self.forecast_errors)
        
        assert isinstance(result, dict)
        assert "mean_error" in result
        assert "t_statistic" in result
        assert "p_value" in result
        assert "is_biased" in result
        assert "bias_direction" in result
        assert "passes_unbiasedness_test" in result
        assert "confidence_interval_95" in result
    
    def test_bias_test_insufficient_data(self):
        """Test bias test with insufficient data"""
        short_errors = self.forecast_errors.head(3)
        
        result = REHTests.bias_test(short_errors)
        assert "error" in result
    
    def test_orthogonality_test_with_external_vars(self):
        """Test orthogonality test with external variables"""
        # Create mock external variables
        external_vars = pd.DataFrame({
            "var1": np.random.normal(0, 1, len(self.forecast_errors)),
            "var2": np.random.normal(0, 1, len(self.forecast_errors))
        }, index=self.forecast_errors.index)
        
        result = REHTests.orthogonality_test(self.forecast_errors, external_vars)
        
        assert isinstance(result, dict)
        assert "f_statistic" in result
        assert "f_pvalue" in result
        assert "passes_orthogonality_test" in result
        assert "variable_results" in result
    
    def test_orthogonality_test_no_external_vars(self):
        """Test orthogonality test without external variables"""
        result = REHTests.orthogonality_test(self.forecast_errors, None)
        assert "error" in result
    
    def test_comprehensive_reh_assessment(self):
        """Test comprehensive REH assessment"""
        result = REHTests.comprehensive_reh_assessment(
            self.forecast, 
            self.realized,
            max_autocorr_lags=5
        )
        
        assert isinstance(result, dict)
        assert "mincer_zarnowitz" in result
        assert "autocorrelation" in result
        assert "bias_test" in result
        assert "rationality_assessment" in result
        
        # Check rationality assessment structure
        ra = result["rationality_assessment"]
        assert "unbiased" in ra
        assert "mz_rational" in ra
        assert "efficient" in ra
        assert "overall_rational" in ra
        assert "n_observations" in ra


class TestStatisticalTestsEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_perfect_forecasts(self):
        """Test with perfect forecasts (no errors)"""
        forecast = pd.Series([1, 2, 3, 4, 5])
        realized = forecast.copy()  # Perfect forecasts
        
        result = REHTests.mincer_zarnowitz_test(forecast, realized)
        
        # With perfect forecasts, beta should be 1 and alpha should be 0
        assert abs(result["beta"] - 1.0) < 0.001
        assert abs(result["alpha"]) < 0.001
        assert result["r_squared"] == 1.0
    
    def test_constant_forecasts(self):
        """Test with constant forecasts"""
        forecast = pd.Series([4.0] * 50)
        realized = pd.Series(np.random.normal(4.0, 1.0, 50))
        
        # This should handle the case where forecast variance is zero
        result = REHTests.mincer_zarnowitz_test(forecast, realized)
        
        # The test should still work, though results may be unusual
        assert isinstance(result, dict)
    
    def test_misaligned_series(self):
        """Test with misaligned series"""
        forecast = pd.Series([1, 2, 3], index=[1, 2, 3])
        realized = pd.Series([1.1, 2.1, 3.1], index=[2, 3, 4])
        
        result = REHTests.mincer_zarnowitz_test(forecast, realized)
        
        # Should only use aligned data
        assert isinstance(result, dict)
        assert result["n_observations"] == 2  # Only indices 2 and 3 align


if __name__ == "__main__":
    pytest.main([__file__])