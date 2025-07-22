# Brazilian REH Analyzer - API Reference

## Table of Contents
1. [BrazilianREHAnalyzer](#brazilianrehanalyzer)
2. [Data Fetching Components](#data-fetching-components)
3. [Statistical Tests](#statistical-tests)
4. [Visualizations](#visualizations)
5. [Utility Functions](#utility-functions)

## BrazilianREHAnalyzer

### Class: `BrazilianREHAnalyzer`

The main analyzer class for comprehensive econometric analysis of Brazilian inflation forecasts.

#### Constructor

```python
BrazilianREHAnalyzer(
    start_date: str = "2017-01-01",
    end_date: str = "2024-12-31",
    cache_dir: str = "data_cache"
)
```

**Parameters:**
- `start_date` (str): Analysis start date in YYYY-MM-DD format
- `end_date` (str): Analysis end date in YYYY-MM-DD format
- `cache_dir` (str): Directory for data caching

**Example:**
```python
from brazilian_reh_analyzer import BrazilianREHAnalyzer

analyzer = BrazilianREHAnalyzer(
    start_date="2020-01-01",
    end_date="2023-12-31"
)
```

#### Methods

##### `fetch_ipca_data(force_refresh: bool = False) -> pd.Series`

Fetch IPCA 12-month accumulated data from BCB with caching.

**Parameters:**
- `force_refresh` (bool): If True, ignore cache and fetch fresh data

**Returns:**
- `pd.Series`: IPCA 12-month accumulated data

**Example:**
```python
ipca_data = analyzer.fetch_ipca_data(force_refresh=True)
print(f"Fetched {len(ipca_data)} IPCA observations")
```

##### `fetch_focus_data(force_refresh: bool = False) -> pd.DataFrame`

Fetch Focus Bulletin IPCA expectations data with caching.

**Parameters:**
- `force_refresh` (bool): If True, ignore cache and fetch fresh data

**Returns:**
- `pd.DataFrame`: Focus Bulletin expectations data with columns:
  - `Mediana`: Median forecast
  - `Media`: Mean forecast  
  - `numeroRespondentes`: Number of respondents

**Example:**
```python
focus_data = analyzer.fetch_focus_data()
print(f"Fetched {len(focus_data)} Focus observations")
```

##### `align_forecast_realization_data() -> pd.DataFrame`

Align forecasts with their corresponding realizations using proper 12-month matching.

**Returns:**
- `pd.DataFrame`: Aligned data with columns:
  - `forecast`: Forecast values
  - `realized`: Realized values
  - `forecast_error`: Calculated errors (realized - forecast)
  - `respondents`: Number of respondents
  - `forecast_mean`: Mean forecast values

**Example:**
```python
aligned_data = analyzer.align_forecast_realization_data()
print(f"Aligned {len(aligned_data)} forecast-realization pairs")
```

##### `comprehensive_analysis(fetch_data: bool = True, force_refresh: bool = False, external_vars: Optional[pd.DataFrame] = None) -> Dict`

Run complete enhanced analysis with real Brazilian data.

**Parameters:**
- `fetch_data` (bool): Whether to fetch data from APIs
- `force_refresh` (bool): Force refresh of cached data
- `external_vars` (pd.DataFrame, optional): External variables for orthogonality testing

**Returns:**
- `Dict`: Comprehensive analysis results containing:
  - `descriptive_stats`: Basic statistical measures
  - `mincer_zarnowitz`: MZ regression test results
  - `autocorrelation`: Ljung-Box test results
  - `bias_test`: Holden-Peel bias test results
  - `rationality_assessment`: Overall rationality verdict

**Example:**
```python
results = analyzer.comprehensive_analysis(
    fetch_data=True,
    force_refresh=False
)

# Access results
print(f"Overall Rational: {results['rationality_assessment']['overall_rational']}")
print(f"Mean Error: {results['descriptive_stats']['error_mean']:.3f} p.p.")
```

##### `plot_enhanced_diagnostics(show_plots: bool = True) -> matplotlib.figure.Figure`

Generate comprehensive diagnostic plots.

**Parameters:**
- `show_plots` (bool): Whether to display plots immediately

**Returns:**
- `matplotlib.figure.Figure`: Figure object with diagnostic plots

**Example:**
```python
fig = analyzer.plot_enhanced_diagnostics(show_plots=True)
fig.savefig("diagnostics.png", dpi=300, bbox_inches='tight')
```

##### `export_results_summary(filename: str = "reh_analysis_summary.txt") -> None`

Export human-readable summary of results to text file.

**Parameters:**
- `filename` (str): Output filename

**Example:**
```python
analyzer.export_results_summary("my_analysis_results.txt")
```

##### `export_plots(output_dir: str = "plots", dpi: int = 300) -> None`

Export all diagnostic plots to individual files.

**Parameters:**
- `output_dir` (str): Directory to save plots
- `dpi` (int): Resolution for saved plots

**Example:**
```python
analyzer.export_plots(output_dir="publication_plots", dpi=600)
```

##### `save_data(filename: str = "aligned_data.csv") -> None`

Save aligned forecast-realization data to CSV.

**Parameters:**
- `filename` (str): Output CSV filename

##### `load_data(filename: str) -> None`

Load previously saved aligned data from CSV.

**Parameters:**
- `filename` (str): Input CSV filename

## Data Fetching Components

### Class: `RespectfulBCBClient`

Wrapper for BCB API calls with built-in rate limiting and error handling.

#### Methods

##### `get_sgs_data(series_code, start_date, end_date)`

Rate-limited SGS data fetching.

##### `get_expectations_data(endpoint_name, query_params)`

Rate-limited Expectations data fetching.

### Class: `DataCache`

Handles caching of fetched data to avoid repeated API calls.

#### Constructor

```python
DataCache(cache_dir: str = "data_cache")
```

#### Methods

##### `load_data(data_type: str, start_date: str, end_date: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]`

Load data from cache if available and recent.

##### `save_data(data: pd.DataFrame, data_type: str, start_date: str, end_date: str)`

Save data to cache with automatic cleanup.

## Statistical Tests

### Class: `REHTests`

Collection of econometric tests for Rational Expectations Hypothesis analysis.

#### Static Methods

##### `mincer_zarnowitz_test(forecast: pd.Series, realized: pd.Series) -> Dict`

Perform Mincer-Zarnowitz test for forecast unbiasedness and efficiency.

**Mathematical Model:**
```
P_t = α + β · E_{t-h}[P_t] + u_t
H_0: (α, β) = (0, 1)
```

**Returns:**
- `Dict` with keys:
  - `alpha`, `beta`: Regression coefficients
  - `alpha_pvalue`, `beta_pvalue`: Individual p-values
  - `joint_test_pvalue`: Joint test p-value
  - `r_squared`: R-squared value
  - `passes_joint_test`: Boolean result

##### `autocorrelation_test(forecast_errors: pd.Series, max_lags: int = 10) -> Dict`

Test forecast errors for autocorrelation using Ljung-Box test.

**Returns:**
- `Dict` with keys:
  - `ljung_box_stat`: LB test statistic
  - `ljung_box_pvalue`: LB test p-value
  - `significant_autocorr`: Boolean indicator
  - `passes_efficiency_test`: Boolean result

##### `bias_test(forecast_errors: pd.Series) -> Dict`

Test forecast errors for systematic bias (Holden-Peel test).

**Returns:**
- `Dict` with keys:
  - `mean_error`: Mean forecast error
  - `t_statistic`: t-test statistic
  - `p_value`: t-test p-value
  - `is_biased`: Boolean indicator
  - `bias_direction`: "overestimation" or "underestimation"

##### `orthogonality_test(forecast_errors: pd.Series, external_vars: pd.DataFrame) -> Dict`

Test forecast errors for orthogonality with available information.

**Returns:**
- `Dict` with keys:
  - `f_statistic`: F-test statistic
  - `f_pvalue`: F-test p-value
  - `passes_orthogonality_test`: Boolean result
  - `variable_results`: Individual variable results

##### `comprehensive_reh_assessment(forecast: pd.Series, realized: pd.Series, external_vars: Optional[pd.DataFrame] = None, max_autocorr_lags: int = 10) -> Dict`

Run complete suite of REH tests and provide overall assessment.

**Returns:**
- `Dict` containing all individual test results plus:
  - `rationality_assessment`: Overall verdict with boolean indicators

## Visualizations

### Class: `REHVisualizations`

Visualization components for REH analysis results.

#### Static Methods

##### `plot_forecast_vs_realization(forecast: pd.Series, realized: pd.Series, ax: Optional[plt.Axes] = None, title: str = "Focus Forecasts vs Realized IPCA") -> plt.Figure`

Create scatter plot of forecasts vs realizations with regression line.

##### `plot_forecast_errors_timeseries(forecast_errors: pd.Series, ax: Optional[plt.Axes] = None, title: str = "Forecast Errors Over Time") -> plt.Figure`

Plot forecast errors as time series with confidence bands.

##### `plot_error_distribution(forecast_errors: pd.Series, ax: Optional[plt.Axes] = None, title: str = "Distribution of Forecast Errors") -> plt.Figure`

Plot distribution of forecast errors with normality tests.

##### `create_comprehensive_diagnostics(forecast: pd.Series, realized: pd.Series, results: Dict, figsize: Tuple[int, int] = (18, 12)) -> plt.Figure`

Create comprehensive diagnostic plot with all key visualizations.

##### `export_plots_to_files(forecast: pd.Series, realized: pd.Series, results: Dict, output_dir: str = "plots", dpi: int = 300)`

Export all plots to individual files for publication.

## Utility Functions

### Functions

##### `ensure_scalar(value) -> float`

Ensure a value is a scalar float, handling pandas Series and arrays.

##### `split_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp, months: int = 12) -> List[tuple]`

Split date range into smaller chunks for API rate limiting.

##### `validate_data_types(df: pd.DataFrame, required_columns: List[str]) -> bool`

Validate DataFrame contains required columns with appropriate data types.

##### `format_results_summary(results: dict) -> str`

Format analysis results into human-readable summary string.

### Decorators

##### `@rate_limit_decorator(min_delay=1.0, max_delay=3.0, requests_per_batch=10, batch_delay_min=10, batch_delay_max=20)`

Decorator to add respectful rate limiting to API calls.

**Parameters:**
- `min_delay`: Minimum delay between requests (seconds)
- `max_delay`: Maximum delay between requests (seconds)
- `requests_per_batch`: Number of requests before longer break
- `batch_delay_min`: Minimum batch delay (seconds)
- `batch_delay_max`: Maximum batch delay (seconds)

**Example:**
```python
@rate_limit_decorator(min_delay=0.5, max_delay=2.0)
def my_api_function():
    # Your API call here
    pass
```

### Classes

##### `ProgressTracker(total_steps: int, description: str = "Processing")`

Simple progress tracker for long-running operations.

**Methods:**
- `update(step_description: str = "")`: Update progress
- `finish()`: Mark progress as complete

**Example:**
```python
progress = ProgressTracker(100, "Fetching data")
for i in range(100):
    # Do work
    progress.update(f"Processing item {i+1}")
progress.finish()
```

## Error Handling

### Common Exceptions

- `ImportError`: Raised when python-bcb library is not installed
- `ValueError`: Raised for invalid data or parameters
- `ConnectionError`: Raised for API connection issues
- `KeyError`: Raised when required data fields are missing

### Example Error Handling

```python
try:
    analyzer = BrazilianREHAnalyzer()
    results = analyzer.comprehensive_analysis()
except ImportError:
    print("Please install python-bcb: pip install python-bcb")
except ValueError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Analysis failed: {e}")
```

## Configuration

### Environment Variables

- `BCB_CACHE_DIR`: Override default cache directory
- `BCB_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Default Settings

```python
DEFAULT_CONFIG = {
    'cache_ttl_hours': 24,
    'max_cache_files': 3,
    'rate_limit_min_delay': 0.8,
    'rate_limit_max_delay': 2.5,
    'significance_level': 0.05,
    'min_respondents': 10
}
```