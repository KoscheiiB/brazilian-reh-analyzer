# Analysis Scripts

This directory contains professional analysis scripts for the Brazilian REH Analyzer v2.0.0 Enhanced Academic Framework.

## Available Scripts

### 1. Pre-COVID Analysis (`run_pre_covid_analysis.py`)
- **Period**: 2017-01-01 to 2020-02-29
- **Focus**: Pre-pandemic inflation expectations rationality
- **Output**: `outputs/pre_covid_analysis_2017_2020/`

### 2. COVID Period Analysis (`run_covid_analysis.py`)
- **Period**: 2020-03-01 to 2022-12-31
- **Focus**: COVID-19 pandemic impact on expectations
- **Output**: `outputs/covid_analysis_2020_2022/`

### 3. Post-COVID Analysis (`run_post_covid_analysis.py`)
- **Period**: 2023-01-01 to 2025-07-01
- **Focus**: Post-pandemic recovery patterns
- **Output**: `outputs/post_covid_analysis_2023_2025/`

### 4. Comprehensive Analysis (`run_comprehensive_analysis.py`)
- **Period**: 2017-01-01 to 2025-07-01
- **Focus**: Complete historical analysis
- **Output**: `outputs/comprehensive_analysis_2017_2025/`

## Usage

Run any script from the project root directory:

```bash
python docs/examples/scripts/run_pre_covid_analysis.py
python docs/examples/scripts/run_covid_analysis.py
python docs/examples/scripts/run_post_covid_analysis.py
python docs/examples/scripts/run_comprehensive_analysis.py
```

## Output Structure

Each script generates:

- **Text Summary** (`.txt`): Comprehensive results summary
- **LaTeX Report** (`.tex`): Publication-ready academic report
- **Diagnostic Plots** (`diagnostic_plots/`): Professional visualizations
- **Aligned Data** (`.csv`): Processed dataset
- **Cache** (`cache/`): Data cache for faster subsequent runs

## Requirements

- Brazilian REH Analyzer v2.0.0 installed
- Internet connection for BCB API data fetching
- Sufficient disk space for outputs (approximately 50MB per analysis)

## Data Sources

All scripts automatically fetch data from:
- **BCB Focus Bulletin API**: Market inflation expectations
- **BCB SGS API**: Realized IPCA inflation data

Data is cached locally to improve performance on subsequent runs.