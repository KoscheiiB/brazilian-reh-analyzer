"""
Utility functions and decorators for the Brazilian REH Analyzer.
"""

import pandas as pd
import numpy as np
import time
import random
import logging
from functools import wraps
from typing import Union, List
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


def rate_limit_decorator(
    min_delay=1.0,
    max_delay=3.0,
    requests_per_batch=10,
    batch_delay_min=10,
    batch_delay_max=20,
):
    """
    Decorator to add respectful rate limiting to API calls
    
    Parameters:
    - min_delay: Minimum delay between requests (seconds)
    - max_delay: Maximum delay between requests (seconds)
    - requests_per_batch: Number of requests before taking a longer break
    - batch_delay_min: Minimum batch delay (seconds)
    - batch_delay_max: Maximum batch delay (seconds)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Random delay between requests
            delay = random.uniform(min_delay, max_delay)
            logger.debug(f"Rate limiting: waiting {delay:.2f} seconds...")
            time.sleep(delay)

            # Check if we need a longer break (every N requests)
            if not hasattr(wrapper, "request_count"):
                wrapper.request_count = 0

            wrapper.request_count += 1

            if wrapper.request_count % requests_per_batch == 0:
                batch_delay = random.uniform(batch_delay_min, batch_delay_max)
                logger.info(
                    f"Batch break: waiting {batch_delay:.1f} seconds after {wrapper.request_count} requests..."
                )
                time.sleep(batch_delay)

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Add delay even on errors to be respectful
                error_delay = random.uniform(2.0, 5.0)
                logger.warning(
                    f"Error occurred, waiting {error_delay:.1f} seconds before continuing..."
                )
                time.sleep(error_delay)
                raise

        return wrapper

    return decorator


def ensure_scalar(value) -> float:
    """
    Ensure a value is a scalar float, not a Series or other pandas object
    
    Parameters:
    - value: Input value that might be a pandas Series, numpy array, or scalar
    
    Returns:
    - float: Scalar float value
    
    Raises:
    - ValueError: If Series contains multiple values or conversion fails
    """
    if isinstance(value, pd.Series):
        if len(value) == 1:
            return float(value.iloc[0])
        else:
            raise ValueError(
                f"Expected single value, got Series with {len(value)} values"
            )
    elif isinstance(value, (np.ndarray, list)) and len(value) == 1:
        return float(value[0])
    else:
        return float(value)


def split_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp, months: int = 12) -> List[tuple]:
    """
    Split date range into smaller chunks to be respectful to APIs
    
    Parameters:
    - start_date: Start date
    - end_date: End date  
    - months: Number of months per chunk
    
    Returns:
    - List of (start, end) date tuples
    """
    ranges = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + pd.DateOffset(months=months), end_date)
        ranges.append((current_start, current_end))
        current_start = current_end + pd.DateOffset(days=1)

    return ranges


def validate_data_types(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns with appropriate data types
    
    Parameters:
    - df: DataFrame to validate
    - required_columns: List of required column names
    
    Returns:
    - bool: True if validation passes
    
    Raises:
    - ValueError: If validation fails
    """
    # Check required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for numeric columns that should be numeric
    numeric_columns = ['forecast', 'realized', 'forecast_error', 'respondents']
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} contains non-numeric values")
    
    return True


def format_results_summary(results: dict) -> str:
    """
    Format analysis results into a human-readable summary string
    
    Parameters:
    - results: Dictionary containing analysis results
    
    Returns:
    - str: Formatted summary text
    """
    summary_lines = []
    summary_lines.append("RATIONAL EXPECTATIONS HYPOTHESIS ANALYSIS")
    summary_lines.append("=" * 50)
    
    # Basic statistics
    if "descriptive_stats" in results:
        stats = results["descriptive_stats"]
        summary_lines.append(f"Analysis Period: {stats.get('date_range', 'N/A')}")
        summary_lines.append(f"Observations: {stats.get('n_observations', 'N/A')}")
        summary_lines.append(f"Mean Forecast Error: {stats.get('error_mean', 0):.3f} p.p.")
        summary_lines.append("")
    
    # Rationality assessment
    if "rationality_assessment" in results:
        ra = results["rationality_assessment"]
        summary_lines.append("Rationality Tests:")
        summary_lines.append(f"✗ Unbiased: {'PASS' if ra.get('unbiased', False) else 'FAIL'}")
        summary_lines.append(f"✗ MZ Test: {'PASS' if ra.get('mz_rational', False) else 'FAIL'}")
        summary_lines.append(f"✗ Efficient: {'PASS' if ra.get('efficient', False) else 'FAIL'}")
        summary_lines.append(f"✗ Overall Rational: {'PASS' if ra.get('overall_rational', False) else 'FAIL'}")
    
    return "\n".join(summary_lines)


class ProgressTracker:
    """
    Simple progress tracker for long-running operations
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step_description: str = ""):
        """Update progress and log current status"""
        self.current_step += 1
        elapsed = datetime.now() - self.start_time
        
        if self.total_steps > 0:
            progress = (self.current_step / self.total_steps) * 100
            logger.info(f"{self.description}: {progress:.1f}% - {step_description}")
        else:
            logger.info(f"{self.description}: Step {self.current_step} - {step_description}")
    
    def finish(self):
        """Mark progress as complete"""
        elapsed = datetime.now() - self.start_time
        logger.info(f"{self.description} completed in {elapsed.total_seconds():.1f} seconds")