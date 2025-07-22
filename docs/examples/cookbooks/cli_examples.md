# Command Line Interface Examples

The Brazilian REH Analyzer provides a comprehensive command-line interface for automated analysis. Here are common usage patterns:

## Basic Usage

### 1. Default Analysis

Run analysis with default date range (2017-01-01 to 2024-12-31):

```bash
python -m brazilian_reh_analyzer
```

Or using the installed command:

```bash
reh-analyzer
```

### 2. Custom Date Range

Analyze a specific period:

```bash
python -m brazilian_reh_analyzer --start-date 2020-01-01 --end-date 2023-12-31
```

### 3. Force Data Refresh

Ignore cached data and fetch fresh data from APIs:

```bash
python -m brazilian_reh_analyzer --force-refresh
```

## Export Options

### 4. Export Results Summary

Save analysis results to a text file:

```bash
python -m brazilian_reh_analyzer --export-summary --summary-file my_analysis.txt
```

### 5. Export Plots

Save diagnostic plots to files:

```bash
python -m brazilian_reh_analyzer --export-plots --output-dir publication_plots/
```

### 6. Save Raw Data

Export the aligned forecast-realization data:

```bash
python -m brazilian_reh_analyzer --save-data --data-file my_data.csv
```

## Display Options

### 7. Suppress Interactive Plots

Run analysis without showing plots (useful for batch processing):

```bash
python -m brazilian_reh_analyzer --no-plots
```

### 8. Verbose Output

Enable detailed logging:

```bash
python -m brazilian_reh_analyzer --verbose
```

### 9. Quiet Mode

Minimize console output:

```bash
python -m brazilian_reh_analyzer --quiet --export-summary
```

## Batch Processing Examples

### 10. Complete Analysis with All Exports

Run comprehensive analysis with all outputs:

```bash
python -m brazilian_reh_analyzer \
    --start-date 2019-01-01 \
    --end-date 2024-12-31 \
    --export-summary \
    --summary-file comprehensive_results.txt \
    --export-plots \
    --output-dir comprehensive_plots/ \
    --save-data \
    --data-file comprehensive_data.csv
```

### 11. Multiple Period Analysis Script

Create a bash script to analyze multiple periods:

```bash
#!/bin/bash
# analyze_periods.sh

echo "Analyzing multiple periods..."

# Pre-COVID period
python -m brazilian_reh_analyzer \
    --start-date 2017-01-01 \
    --end-date 2020-02-29 \
    --export-summary \
    --summary-file pre_covid_results.txt \
    --export-plots \
    --output-dir pre_covid_plots/ \
    --no-plots \
    --quiet

# COVID period
python -m brazilian_reh_analyzer \
    --start-date 2020-03-01 \
    --end-date 2022-12-31 \
    --export-summary \
    --summary-file covid_results.txt \
    --export-plots \
    --output-dir covid_plots/ \
    --no-plots \
    --quiet

# Post-COVID period
python -m brazilian_reh_analyzer \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --export-summary \
    --summary-file post_covid_results.txt \
    --export-plots \
    --output-dir post_covid_plots/ \
    --no-plots \
    --quiet

echo "Analysis completed! Check the generated files."
```

Make it executable and run:

```bash
chmod +x analyze_periods.sh
./analyze_periods.sh
```

### 12. Weekly Automated Analysis

Create a cron job for weekly analysis updates:

```bash
# Add to crontab (run every Monday at 9 AM)
# crontab -e
0 9 * * 1 /path/to/python -m brazilian_reh_analyzer --force-refresh --export-summary --summary-file /path/to/weekly_analysis.txt --quiet
```

## Advanced Usage

### 13. Custom Cache Directory

Use a specific directory for data caching:

```bash
python -m brazilian_reh_analyzer --cache-dir /path/to/my/cache/
```

### 14. Publication-Quality Output

Generate high-resolution plots for academic papers:

```bash
python -m brazilian_reh_analyzer \
    --start-date 2017-01-01 \
    --end-date 2024-12-31 \
    --export-plots \
    --output-dir paper_figures/ \
    --export-summary \
    --summary-file paper_results.txt \
    --no-plots
```

### 15. Integration with Other Tools

Pipe results to other analysis tools:

```bash
# Export data and analyze with R
python -m brazilian_reh_analyzer --save-data --data-file analysis_data.csv --no-plots --quiet
Rscript my_additional_analysis.R analysis_data.csv

# Convert results to JSON for web applications
python -c "
import json
from brazilian_reh_analyzer import BrazilianREHAnalyzer
analyzer = BrazilianREHAnalyzer('2020-01-01', '2024-12-31')
results = analyzer.comprehensive_analysis()
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
"
```

## Error Handling and Troubleshooting

### 16. Handle Missing Dependencies

```bash
# Check for required dependencies
python -c "import brazilian_reh_analyzer; print('✅ Package installed correctly')" || echo "❌ Package not found"

# Install if missing
pip install python-bcb pandas statsmodels matplotlib seaborn
```

### 17. Network Issues

If you encounter network issues:

```bash
# Try with longer timeouts and force refresh
python -m brazilian_reh_analyzer --force-refresh --verbose
```

### 18. Memory Issues

For very large datasets:

```bash
# Use smaller date ranges
python -m brazilian_reh_analyzer \
    --start-date 2022-01-01 \
    --end-date 2022-12-31 \
    --no-plots
```

## Environment Variables

You can set environment variables to configure default behavior:

```bash
# Set default cache directory
export BCB_CACHE_DIR=/path/to/cache

# Set log level
export BCB_LOG_LEVEL=DEBUG

# Run analysis
python -m brazilian_reh_analyzer
```

## Docker Usage

If using Docker:

```bash
# Build image
docker build -t brazilian-reh-analyzer .

# Run analysis in container
docker run -v $(pwd)/results:/app/results brazilian-reh-analyzer \
    python -m brazilian_reh_analyzer \
    --export-summary \
    --summary-file /app/results/analysis.txt \
    --export-plots \
    --output-dir /app/results/plots/
```

## Help and Documentation

### 19. Get Help

```bash
python -m brazilian_reh_analyzer --help
```

This shows all available options:

```
usage: __main__.py [-h] [--start-date START_DATE] [--end-date END_DATE]
                   [--force-refresh] [--cache-dir CACHE_DIR]
                   [--export-summary] [--summary-file SUMMARY_FILE]
                   [--export-plots] [--output-dir OUTPUT_DIR] [--save-data]
                   [--data-file DATA_FILE] [--no-plots] [--verbose] [--quiet]

Brazilian REH Analyzer - Econometric analysis of inflation forecast rationality

optional arguments:
  -h, --help            show this help message and exit
  --start-date START_DATE
                        Analysis start date (YYYY-MM-DD). Default: 2017-01-01
  --end-date END_DATE   Analysis end date (YYYY-MM-DD). Default: 2024-12-31
  --force-refresh       Force refresh data from APIs (ignore cache)
  --cache-dir CACHE_DIR
                        Directory for data caching. Default: data_cache
  --export-summary      Export text summary of results
  --summary-file SUMMARY_FILE
                        Filename for exported summary. Default: reh_analysis_results.txt
  --export-plots        Export diagnostic plots to files
  --output-dir OUTPUT_DIR
                        Directory for exported plots. Default: plots
  --save-data           Save aligned data to CSV file
  --data-file DATA_FILE
                        Filename for saved data. Default: aligned_data.csv
  --no-plots            Skip displaying interactive plots
  --verbose             Enable verbose logging
  --quiet               Minimize console output

Examples:
  # Basic analysis with default date range
  python -m brazilian_reh_analyzer
  
  # Analysis for specific period
  python -m brazilian_reh_analyzer --start-date 2017-01-01 --end-date 2023-12-31
  
  # Force refresh data (ignore cache)
  python -m brazilian_reh_analyzer --force-refresh
  
  # Save plots to specific directory
  python -m brazilian_reh_analyzer --export-plots --output-dir results/
```

## Output Files

The CLI generates several types of output files:

1. **Summary Text File** (`--export-summary`): Human-readable analysis results
2. **Diagnostic Plots** (`--export-plots`): PNG files with visualizations  
3. **Raw Data** (`--save-data`): CSV file with aligned forecast-realization pairs
4. **Cache Files** (automatic): Cached API data for faster subsequent runs

These files can be used for:
- Academic papers and presentations
- Further statistical analysis
- Integration with other tools
- Reproducible research workflows