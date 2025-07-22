"""
Tests for utility functions
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from brazilian_reh_analyzer.utils import (
    ensure_scalar,
    split_date_range,
    validate_data_types,
    format_results_summary,
    ProgressTracker
)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_ensure_scalar_with_series(self):
        """Test ensure_scalar with pandas Series"""
        # Single value Series
        single_series = pd.Series([3.14])
        result = ensure_scalar(single_series)
        assert result == 3.14
        assert isinstance(result, float)
        
        # Multiple value Series should raise error
        multi_series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Expected single value, got Series"):
            ensure_scalar(multi_series)
    
    def test_ensure_scalar_with_array(self):
        """Test ensure_scalar with numpy array"""
        single_array = np.array([2.71])
        result = ensure_scalar(single_array)
        assert result == 2.71
        assert isinstance(result, float)
    
    def test_ensure_scalar_with_list(self):
        """Test ensure_scalar with list"""
        single_list = [1.618]
        result = ensure_scalar(single_list)
        assert result == 1.618
        assert isinstance(result, float)
    
    def test_ensure_scalar_with_scalar(self):
        """Test ensure_scalar with already scalar value"""
        scalar_int = 42
        result = ensure_scalar(scalar_int)
        assert result == 42.0
        assert isinstance(result, float)
    
    def test_split_date_range(self):
        """Test date range splitting"""
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2022-12-31")
        
        ranges = split_date_range(start_date, end_date, months=12)
        
        assert isinstance(ranges, list)
        assert len(ranges) == 3  # 2020, 2021, 2022
        
        # Check first range
        assert ranges[0][0] == start_date
        assert ranges[0][1] == pd.Timestamp("2021-01-01")
        
        # Check last range
        assert ranges[-1][1] == end_date
    
    def test_validate_data_types_success(self):
        """Test successful data type validation"""
        df = pd.DataFrame({
            "forecast": [1.0, 2.0, 3.0],
            "realized": [1.1, 2.1, 3.1],
            "forecast_error": [0.1, 0.1, 0.1],
            "respondents": [20, 25, 30]
        })
        
        result = validate_data_types(df, ["forecast", "realized"])
        assert result is True
    
    def test_validate_data_types_missing_columns(self):
        """Test validation with missing columns"""
        df = pd.DataFrame({
            "forecast": [1.0, 2.0, 3.0]
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data_types(df, ["forecast", "realized"])
    
    def test_validate_data_types_non_numeric(self):
        """Test validation with non-numeric columns"""
        df = pd.DataFrame({
            "forecast": ["a", "b", "c"],
            "realized": [1.1, 2.1, 3.1]
        })
        
        with pytest.raises(ValueError, match="contains non-numeric values"):
            validate_data_types(df, ["forecast", "realized"])
    
    def test_format_results_summary(self):
        """Test results summary formatting"""
        mock_results = {
            "descriptive_stats": {
                "date_range": "2020-01-01 to 2023-12-31",
                "n_observations": 100,
                "error_mean": 0.123
            },
            "rationality_assessment": {
                "unbiased": True,
                "mz_rational": False,
                "efficient": True,
                "overall_rational": False
            }
        }
        
        summary = format_results_summary(mock_results)
        
        assert isinstance(summary, str)
        assert "RATIONAL EXPECTATIONS HYPOTHESIS" in summary
        assert "2020-01-01 to 2023-12-31" in summary
        assert "100" in summary
        assert "0.123" in summary
        assert "PASS" in summary
        assert "FAIL" in summary


class TestProgressTracker:
    """Test ProgressTracker utility"""
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization"""
        tracker = ProgressTracker(10, "Test Process")
        
        assert tracker.total_steps == 10
        assert tracker.current_step == 0
        assert tracker.description == "Test Process"
        assert isinstance(tracker.start_time, datetime)
    
    def test_progress_tracker_update(self):
        """Test ProgressTracker update"""
        tracker = ProgressTracker(5, "Test Process")
        
        tracker.update("Step 1")
        assert tracker.current_step == 1
        
        tracker.update("Step 2")
        assert tracker.current_step == 2
    
    def test_progress_tracker_finish(self):
        """Test ProgressTracker finish"""
        tracker = ProgressTracker(3, "Test Process")
        
        # This should not raise any exceptions
        tracker.finish()
    
    def test_progress_tracker_no_total(self):
        """Test ProgressTracker with no total steps"""
        tracker = ProgressTracker(0, "Indefinite Process")
        
        tracker.update("Step 1")
        assert tracker.current_step == 1
        
        tracker.update("Step 2") 
        assert tracker.current_step == 2


if __name__ == "__main__":
    pytest.main([__file__])