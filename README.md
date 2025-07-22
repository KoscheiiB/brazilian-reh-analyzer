# Brazilian REH Analyzer 🇧🇷📈

**v2.0.0 Enhanced Academic Framework for Brazilian Inflation Forecast Rationality**

A comprehensive, **publication-quality academic research framework** for assessing the rationality of Brazil's Focus Bulletin inflation forecasts according to the Rational Expectations Hypothesis (REH). Features advanced econometric analysis, professional visualizations, LaTeX report generation, and automated economic interpretation suitable for journal submission.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v2.0.0_academic-brightgreen.svg)]()
[![Analysis](https://img.shields.io/badge/analysis-REH-orange.svg)]()

## 🎯 Overview

This tool provides automated, reproducible analysis of Brazilian inflation forecast rationality using real-time data from the Central Bank of Brazil (BCB). It implements advanced econometric methodologies with comprehensive economic interpretation, making it suitable for academic research, policy analysis, and investment strategy development.

### 🌟 v2.0.0 Enhanced Key Features

- **🔗 Direct BCB Integration**: Seamless data fetching from SGS and Expectations APIs
- **🧠 Advanced REH Analysis**: Comprehensive rationality testing with rich economic interpretation
- **📊 Rich Descriptive Statistics**: Detailed statistical tables with skewness, kurtosis, and distribution analysis  
- **🔍 Automatic Structural Break Detection**: Dynamic sub-period identification adapting to any time period
- **📈 Enhanced Mincer-Zarnowitz Analysis**: Full regression output with 95% confidence intervals and residual analysis
- **🎯 Rolling Window Analysis**: Time-varying bias detection with professional color-corrected visualizations
- **📋 ACF/PACF Autocorrelation Analysis**: Professional econometric diagnostic plots with significance testing
- **📐 Q-Q Normality Testing**: Enhanced plots with confidence bands and multiple statistical tests
- **💡 Economic Interpretation Engine**: Automated generation of policy implications and economic significance assessment
- **📝 LaTeX Academic Export**: Professional publication-ready reports with mathematical equations and structured tables
- **🎨 Academic Color Scheme**: Colorblind-friendly professional palette for publication-quality figures
- **🚀 Smart Caching**: Persistent data storage with organized directory structures (results/, plots/, data/, cache/)
- **📈 Publication-Quality Visualizations**: Journal-ready plots with academic styling and high-DPI export
- **🇧🇷 Brazilian Context**: Handles institutional nuances, crisis periods, and monetary regime changes
- **⚡ Rate-Limited API Access**: Respectful data fetching with automatic retry logic
- **🔬 Batch Processing**: Multiple period analysis with comparative reporting capabilities

## 🏛️ Academic Context

This project implements the methodology described in:

> **"Assessment of the Rationality of Focus Bulletin Inflation Forecasts for the 12-Month Ahead IPCA (January 2017 – April 2025)"**
> *Analysis of Brazilian Central Bank Focus Survey Data*

The tool addresses critical questions in Brazilian monetary policy:
- Are market inflation expectations rational according to REH?
- Do Focus Bulletin forecasts exhibit systematic biases?
- How do structural breaks affect forecast efficiency?
- What institutional factors influence expectation formation?

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brazilian-reh-analyzer.git
cd brazilian-reh-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from brazilian_reh_analyzer import BrazilianREHAnalyzer

# Initialize analyzer with any date range
analyzer = BrazilianREHAnalyzer(
    start_date='2017-01-01',
    end_date='2024-12-31'
)

# Run comprehensive enhanced analysis (uses cached data when available)
results = analyzer.comprehensive_analysis()

# Display key findings from enhanced v2.0.0 output
print(f"Overall Rational: {'✓' if results['rationality_assessment']['overall_rational'] else '✗'}")
print(f"Mean Forecast Error: {results['descriptive_stats']['error_mean']:.3f} p.p.")
print(f"Bias Severity: {results['economic_interpretation']['bias_analysis']['severity']}")
print(f"Sub-periods Detected: {len(results['sub_period_analysis'])}")

# Generate enhanced diagnostic plots with ACF/PACF analysis
analyzer.plot_enhanced_diagnostics()

# Export comprehensive text analysis
analyzer.export_results_summary("path/to/your/results/enhanced_analysis.txt")

# 🆕 Export professional LaTeX report for academic publication
analyzer.export_latex_report(
    "path/to/your/results/academic_report.tex",
    "Brazilian Focus Bulletin Rationality Assessment (2017-2024)",
    "Your Research Team"
)
```

### Command Line Interface - Enhanced

#### 🚀 **Comprehensive Analysis with Organized Output**

```bash
# Create organized output structure and run full analysis
mkdir -p path/to/your/analysis_output/{results,plots,data,cache}

python -m brazilian_reh_analyzer \
    --start-date 2017-01-01 \
    --end-date 2025-07-01 \
    --export-summary \
    --summary-file path/to/your/analysis_output/results/comprehensive_analysis.txt \
    --export-plots \
    --output-dir path/to/your/analysis_output/plots/ \
    --save-data \
    --data-file path/to/your/analysis_output/data/aligned_forecast_data.csv \
    --cache-dir path/to/your/analysis_output/cache/ \
    --verbose
```

#### 📊 **Quick Analysis Examples**

```bash
# Default analysis (2017-2024)
python -m brazilian_reh_analyzer

# Custom period analysis  
python -m brazilian_reh_analyzer --start-date 2020-01-01 --end-date 2022-12-31

# Force refresh data (ignore cache)
python -m brazilian_reh_analyzer --force-refresh

# Silent mode with exports only
python -m brazilian_reh_analyzer --quiet --export-summary --export-plots --no-plots

# Event study analysis
python -m brazilian_reh_analyzer --start-date 2020-03-01 --end-date 2021-12-31 \
    --export-summary --summary-file covid_impact_analysis.txt
```

#### 🎯 **Batch Processing for Multiple Periods**

```bash
#!/bin/bash
# analyze_periods.sh - Batch analysis script

echo "Running comprehensive REH analysis for multiple periods..."

# Pre-COVID Analysis
python -m brazilian_reh_analyzer \
    --start-date 2017-01-01 --end-date 2020-02-29 \
    --export-summary --summary-file path/to/your/output/pre_covid_analysis.txt \
    --export-plots --output-dir path/to/your/output/pre_covid_plots/ \
    --no-plots --quiet

# COVID Period Analysis  
python -m brazilian_reh_analyzer \
    --start-date 2020-03-01 --end-date 2022-12-31 \
    --export-summary --summary-file path/to/your/output/covid_analysis.txt \
    --export-plots --output-dir path/to/your/output/covid_plots/ \
    --no-plots --quiet

# Post-COVID Analysis
python -m brazilian_reh_analyzer \
    --start-date 2023-01-01 --end-date 2025-07-01 \
    --export-summary --summary-file path/to/your/output/post_covid_analysis.txt \
    --export-plots --output-dir path/to/your/output/post_covid_plots/ \
    --no-plots --quiet

echo "Analysis completed! Check your output directory."
```

## 📊 Enhanced Analysis Results

The framework now generates **comprehensive, academic-quality output** with rich economic interpretation:

### 🎯 Executive Summary Format
```
╔════════════════════════════════════════════════════════════════════╗
║      BRAZILIAN REH ANALYZER - COMPREHENSIVE ECONOMIC ANALYSIS      ║
╚════════════════════════════════════════════════════════════════════╝

OVERALL ASSESSMENT: ❌ Forecasts VIOLATE Rational Expectations Hypothesis

• Analysis Period: 2017-01-02 to 2024-06-28 (7.5 years)
• Systematic OVERESTIMATION: -3.805 p.p. mean error
• Bias Severity: SEVERE (High Economic Significance)
• Learning Failure: YES (Extreme autocorrelation detected)
• Sub-periods Analyzed: 3 (with substantial time-variation)
```

### 📊 Rich Descriptive Statistics Table
```
                   ┌───────────────┬────────────────┬────────────────┐
                   │ Observed IPCA │ Focus Forecast │ Forecast Error │
                   │      (%)      │       (%)      │     (p.p.)     │
───────────────────┼───────────────┼────────────────┼────────────────┤
Mean               │      0.438    │       4.243    │      -3.805    │
Median             │      0.400    │       4.060    │      -3.700    │
Standard Deviation │      0.417    │       0.799    │       0.971    │
Minimum            │     -0.680    │       2.290    │      -6.500    │
Maximum            │      1.620    │       6.457    │      -1.460    │
Skewness           │      0.264    │       0.584    │      -0.179    │
Kurtosis           │      0.386    │      -0.115    │      -0.564    │
Observations       │       1878    │        1878    │        1878    │
───────────────────┴───────────────┴────────────────┴────────────────┘
```

### 📈 Detailed Mincer-Zarnowitz Regression Analysis
```
Regression: Realized = α + β × Forecast + ε
Null Hypothesis: H₀: α = 0, β = 1 (rational expectations)

Coefficient    │ Estimate │ Std Error │ t-stat │ p-value │ 95% Confidence Interval
───────────────┼──────────┼───────────┼────────┼─────────┼──────────────────────────
α (Intercept)  │   0.874  │    0.051  │  17.15 │  0.0000 │ [ 0.774,  0.974]
β (Slope)      │  -0.103  │    0.012  │  -8.70 │  0.0000 │ [-0.126, -0.080]

R² = 0.0388    │    F-statistic = 85,672.94    │    REJECT H₀

ECONOMIC INTERPRETATION:
• α = 0.874 ≠ 0: Systematic forecast bias present
• β = -0.103 ≠ 1: Forecasters under-respond to their own predictions  
• Joint test rejection indicates violations of both unbiasedness AND efficiency
```

### 🔍 Sub-Period Analysis (Automatic Structural Break Detection)
```
                   │  Period  │   Period   │ Mean Error │ REH Status │
                   │  Start   │    End     │    (p.p.)  │  Overall   │
───────────────────┼──────────┼────────────┼────────────┼────────────┤
Period 1          │  2017-01 │    2019-07 │     -3.799 │    FAIL    │
Period 2          │  2019-07 │    2021-12 │     -3.265 │    FAIL    │  
Period 3          │  2021-12 │    2024-06 │     -4.349 │    FAIL    │

STRUCTURAL BREAK INTERPRETATION:
• Bias ranges from -4.349 to -3.265 p.p. across sub-periods
• SUBSTANTIAL time-variation in forecast bias detected
• Worsening performance in most recent period
```

### 💡 Economic Interpretation & Policy Implications
```
BIAS ANALYSIS:
• Direction: OVERESTIMATION
• Magnitude: 3.805 percentage points  
• Severity: SEVERE
• Economic Significance: HIGH

FOR CENTRAL BANK POLICYMAKERS:
• Focus forecasts show severe systematic overestimation
• Market expectations exhibit extreme autocorrelation
• Consider enhanced communication strategies

FOR MARKET PARTICIPANTS:  
• Systematic biases present contrarian opportunities
• Forecast errors are predictable, violating efficiency
• Alternative forecasting models recommended

FOR RESEARCHERS:
• REH violations persistent over 7.5-year period
• Adaptive/sticky information models more appropriate
• Structural breaks warrant further investigation
```

## 📁 Data Sources

### Primary Data
- **IPCA (Realized Inflation)**: BCB SGS series 433 (12-month accumulated)
- **Focus Forecasts**: BCB Expectations API (`ExpectativasMercadoInflacao12Meses`)

### Data Coverage
- **Period**: January 2017 - Present
- **Frequency**: Daily Focus forecasts, Monthly IPCA realizations
- **Quality**: Minimum 10 respondents for Focus median calculations

### API Documentation
- [BCB SGS API](https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do)
- [BCB Expectations API](https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/documentacao)

## 🔬 Enhanced Methodology

### 📊 Comprehensive Statistical Analysis Framework

#### 1. **Rich Descriptive Statistics**
- **Comprehensive Distribution Analysis**: Mean, median, standard deviation, skewness, kurtosis
- **Quartile Analysis**: Q25, Q75 for robust central tendency measures
- **Outlier Detection**: Statistical identification of extreme forecast errors
- **Sample Quality Assessment**: Respondent count analysis and data coverage metrics

#### 2. **Advanced Econometric Tests**

**Enhanced Mincer-Zarnowitz Regression**
```
P_t = α + β · E_{t-12}[P_t] + ε_t
H₀: (α, β) = (0, 1)
```
- Full regression diagnostics with confidence intervals
- Individual coefficient significance tests
- Joint hypothesis testing with F-statistics  
- Economic interpretation of coefficient deviations

**Comprehensive Autocorrelation Analysis**
- Ljung-Box Q-test for serial correlation (up to 10 lags)
- Breusch-Godfrey LM test
- Rolling correlation analysis for time-varying patterns
- Partial autocorrelation function analysis

**Enhanced Bias Testing**
- Holden-Peel simple bias test with full statistics
- t-test for zero mean forecast errors
- Bias magnitude classification (minimal/moderate/substantial/severe)
- Economic significance assessment

**Advanced Orthogonality Tests**
- Regression on comprehensive information sets
- External variables: Selic rate, GDP, unemployment, exchange rate
- Information efficiency violation quantification

#### 3. **Structural Break Analysis**

**Automatic Break Detection**
- Statistical change point detection using rolling variance/mean analysis
- Adaptive segmentation based on data length
- Minimum segment size requirements for robust analysis
- Dynamic period identification (works with any date range)

**Sub-Period Analysis**  
- Individual REH testing for each detected period
- Cross-period bias comparison and evolution analysis
- Structural break economic interpretation
- Crisis period and regime change identification

#### 4. **Time-Varying Analysis**

**Rolling Window Framework**
- Configurable window sizes (auto-calculated based on sample size)
- Rolling bias detection and significance testing
- Change point identification in mean and volatility
- Dynamic forecast quality assessment

**Temporal Pattern Detection**
- Trend analysis in forecast errors
- Seasonal bias pattern identification
- Event study capabilities around major economic events

#### 5. **Economic Interpretation Engine**

**Automated Significance Assessment**
- Bias severity classification with economic thresholds
- Learning failure identification based on autocorrelation magnitude
- Forecast quality scoring (excellent/good/moderate/poor)
- Period-specific challenge identification

**Policy Implication Generation**
- Central bank policy guidance based on findings
- Market participant strategy implications
- Academic research direction suggestions
- Investment strategy insights for systematic bias exploitation

### 🇧🇷 Enhanced Brazilian Context Features

- **Institutional Integration**: Complete BCB API integration with rate limiting
- **Crisis Period Modeling**: Automatic detection of major economic disruptions
- **Focus Bulletin Composition Analysis**: Respondent quality and count analysis
- **Monetary Regime Awareness**: Policy regime change detection and analysis
- **Real-Time Data Handling**: Latest available data integration with proper alignment

## 📈 Visualization Gallery

The tool generates publication-quality plots including:

- **Forecast vs. Realization Scatter**: Assessment of forecast accuracy
- **Error Time Series**: Temporal patterns in forecast bias
- **Distribution Analysis**: Normality tests and outlier detection
- **Autocorrelation Functions**: Efficiency violation identification
- **Rolling Statistics**: Dynamic bias pattern analysis
- **Structural Break Charts**: Major economic event impacts

## 🛠️ Advanced Usage - Enhanced Framework

### 🎯 **Enhanced Analysis with Rich Output**

```python
from brazilian_reh_analyzer import BrazilianREHAnalyzer

# Initialize with custom cache directory and date range
analyzer = BrazilianREHAnalyzer(
    start_date='2020-01-01', 
    end_date='2023-12-31',
    cache_dir='path/to/your/custom_cache/'
)

# Run comprehensive enhanced analysis
results = analyzer.comprehensive_analysis()

# Access rich descriptive statistics
rich_stats = results['rich_descriptive_stats']
print(f"Error Skewness: {rich_stats['errors']['skewness']:.3f}")
print(f"Error Kurtosis: {rich_stats['errors']['kurtosis']:.3f}")

# Access sub-period analysis results
sub_periods = results['sub_period_analysis']
for period_name, period_data in sub_periods.items():
    print(f"{period_name}: Mean Error = {period_data['mean_error']:.3f} p.p.")

# Access economic interpretation
econ_interpretation = results['economic_interpretation']
bias_analysis = econ_interpretation['bias_analysis']
print(f"Bias Severity: {bias_analysis['severity']}")
print(f"Economic Significance: {bias_analysis['economic_significance']}")

# Export comprehensive academic-style summary
analyzer.export_results_summary('path/to/your/output/comprehensive_analysis.txt')
```

### 🔍 **Structural Break Analysis**

```python
# Analyze periods with automatic structural break detection
analyzer = BrazilianREHAnalyzer('2017-01-01', '2025-07-01')
results = analyzer.comprehensive_analysis(fetch_data=False)  # Use cached data

# Examine detected structural breaks
sub_periods = results['sub_period_analysis']
print(f"Detected {len(sub_periods)} structural periods:")

for period_name, period_data in sub_periods.items():
    print(f"""
    {period_name}:
    - Period: {period_data['start_date']} to {period_data['end_date']}
    - Mean Error: {period_data['mean_error']:.3f} p.p.
    - Bias Direction: {period_data['bias_direction']}
    - Observations: {period_data['n_observations']}
    """)

# Access rolling window analysis
rolling_analysis = results['rolling_window_analysis']
print(f"Rolling Window Size: {rolling_analysis['window_size']} observations")
print(f"Max Bias Detected: {rolling_analysis['max_abs_bias']:.3f} p.p.")
```

### 📊 **Detailed Statistical Analysis**

```python
# Access detailed Mincer-Zarnowitz regression results
detailed_mz = results['detailed_mincer_zarnowitz']

print("Detailed Regression Analysis:")
print(f"Alpha: {detailed_mz['alpha']:.6f} (95% CI: {detailed_mz['alpha_95_ci']})")
print(f"Beta: {detailed_mz['beta']:.6f} (95% CI: {detailed_mz['beta_95_ci']})")
print(f"Joint Test F-stat: {detailed_mz['joint_f_statistic']:.4f}")
print(f"P-value: {detailed_mz['joint_p_value']:.6f}")
print(f"R²: {detailed_mz['r_squared']:.4f}")

# Check economic significance
if detailed_mz['alpha_significant']:
    print("⚠️  Significant systematic bias detected (α ≠ 0)")
if detailed_mz['beta_significantly_different_from_1']:
    print("⚠️  Forecasters under/over-respond to their own predictions")
```

### 🎯 **Event Study Analysis**

```python
# Analyze specific economic events
event_studies = {
    'COVID_Impact': ('2020-03-01', '2021-06-30'),
    'Post_COVID_Recovery': ('2021-07-01', '2023-12-31'),
    'Election_Period': ('2018-06-01', '2019-01-31'),
    'Truckers_Strike': ('2018-05-01', '2018-08-31')
}

for event_name, (start, end) in event_studies.items():
    print(f"\n=== {event_name} Analysis ===")
    analyzer = BrazilianREHAnalyzer(start, end)
    
    try:
        results = analyzer.comprehensive_analysis()
        
        # Extract key metrics
        desc_stats = results['descriptive_stats']
        econ_interp = results['economic_interpretation']
        
        print(f"Mean Error: {desc_stats['error_mean']:.3f} p.p.")
        print(f"Bias Severity: {econ_interp['bias_analysis']['severity']}")
        print(f"REH Compatible: {results['rationality_assessment']['overall_rational']}")
        
        # Export event-specific results
        analyzer.export_results_summary(f'path/to/your/output/{event_name}_analysis.txt')
        
    except Exception as e:
        print(f"Analysis failed for {event_name}: {e}")
```

### 📁 **Batch Processing with Organized Output**

```python
import os
from pathlib import Path

# Define analysis periods
analysis_periods = {
    'Pre_COVID': ('2017-01-01', '2020-02-29'),
    'COVID_Era': ('2020-03-01', '2022-12-31'), 
    'Post_COVID': ('2023-01-01', '2025-07-01'),
    'Full_Period': ('2017-01-01', '2025-07-01')
}

# Create organized output structure
base_output_dir = Path('path/to/your/comprehensive_analysis/')
for period_name in analysis_periods.keys():
    (base_output_dir / period_name / 'results').mkdir(parents=True, exist_ok=True)
    (base_output_dir / period_name / 'plots').mkdir(parents=True, exist_ok=True)
    (base_output_dir / period_name / 'data').mkdir(parents=True, exist_ok=True)

# Run batch analysis
batch_results = {}

for period_name, (start_date, end_date) in analysis_periods.items():
    print(f"\n🔄 Processing {period_name}...")
    
    # Initialize analyzer with period-specific cache
    analyzer = BrazilianREHAnalyzer(
        start_date=start_date,
        end_date=end_date,
        cache_dir=str(base_output_dir / period_name / 'cache')
    )
    
    try:
        # Run comprehensive analysis
        results = analyzer.comprehensive_analysis()
        batch_results[period_name] = results
        
        # Export all outputs to organized directories
        period_output = base_output_dir / period_name
        
        # Export comprehensive summary
        analyzer.export_results_summary(
            str(period_output / 'results' / f'{period_name}_comprehensive_analysis.txt')
        )
        
        # Export plots
        analyzer.export_plots(
            output_dir=str(period_output / 'plots'),
            dpi=300
        )
        
        # Save raw data
        analyzer.save_data(
            str(period_output / 'data' / f'{period_name}_aligned_data.csv')
        )
        
        print(f"✅ {period_name} completed successfully")
        
    except Exception as e:
        print(f"❌ {period_name} failed: {e}")

# Generate comparative summary
print("\n📊 COMPARATIVE ANALYSIS SUMMARY")
print("=" * 70)

for period_name, results in batch_results.items():
    if 'economic_interpretation' in results:
        econ_interp = results['economic_interpretation']
        bias_analysis = econ_interp['bias_analysis']
        
        print(f"{period_name:15} | "
              f"Error: {results['descriptive_stats']['error_mean']:6.3f} p.p. | "
              f"Severity: {bias_analysis['severity']:12} | "
              f"REH: {'PASS' if results['rationality_assessment']['overall_rational'] else 'FAIL'}")
```

### 🔗 **External Variables Integration**

```python
# Add external macroeconomic variables for enhanced orthogonality testing
import pandas as pd

# Prepare external variables (example with synthetic data)
# In practice, fetch from BCB or other sources
external_data = pd.DataFrame({
    'selic_rate': selic_data,          # From BCB SGS series 432
    'exchange_rate': usd_brl_data,     # From BCB SGS series 1
    'gdp_growth': gdp_data,            # From BCB or IBGE
    'unemployment': unemployment_data   # From IBGE PNAD
}, index=date_range)

# Run analysis with external variables
results = analyzer.comprehensive_analysis(external_vars=external_data)

# The framework will automatically include these in orthogonality tests
# and provide enhanced interpretation of information efficiency violations
```

## 🏗️ Project Structure

```
brazilian-reh-analyzer/
├── brazilian_reh_analyzer/
│   ├── __init__.py
│   ├── analyzer.py          # Main analysis engine
│   ├── data_fetcher.py      # BCB API integration
│   ├── tests.py            # Econometric tests
│   ├── visualizations.py   # Plotting functions
│   └── utils.py            # Utility functions
├── tests/
│   ├── test_analyzer.py
│   ├── test_data_fetcher.py
│   └── test_statistical_tests.py
├── docs/
│   ├── methodology.md       # Technical methodology
│   ├── api_reference.md     # API documentation
│   └── examples/           # Usage examples
├── data_cache/             # Cached API data
├── requirements.txt
├── setup.py
└── README.md
```

## 🧪 Testing

```bash
# Run full test suite
pytest tests/

# Run with coverage
pytest --cov=brazilian_reh_analyzer tests/

# Run specific test categories
pytest tests/test_statistical_tests.py -v
```

## 📚 Dependencies

### Core Requirements
- `pandas >= 1.5.0` - Data manipulation and analysis
- `numpy >= 1.21.0` - Numerical computing
- `statsmodels >= 0.13.0` - Statistical modeling
- `scipy >= 1.7.0` - Scientific computing
- `python-bcb >= 0.3.0` - Brazilian Central Bank API client

### Visualization
- `matplotlib >= 3.5.0` - Plotting library
- `seaborn >= 0.11.0` - Statistical visualizations

### Optional
- `jupyter >= 1.0.0` - Interactive analysis notebooks
- `plotly >= 5.0.0` - Interactive visualizations

## 🤝 Contributing

We welcome contributions from the research community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/brazilian-reh-analyzer.git
cd brazilian-reh-analyzer
python -m venv venv
source venv/bin/activate
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Types of Contributions

- **Bug Reports**: Issues with data fetching, analysis, or visualizations
- **Feature Requests**: Additional econometric tests or analysis capabilities
- **Documentation**: Improvements to methodology explanations or examples
- **Code**: Implementation of new features or performance improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/KoscheiiB/brazilian-reh-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KoscheiiB/brazilian-reh-analyzer/discussions)
- **Email**: KoscheiiB@users.noreply.github.com

## 🎓 Citation

If you use this tool in your research, please cite:

```bibtex
@software{brazilian_reh_analyzer_2025,
  author = {KoscheiiB},
  title = {Brazilian REH Analyzer: Econometric Analysis of Focus Bulletin Forecasts},
  url = {https://github.com/KoscheiiB/brazilian-reh-analyzer},
  year = {2025}
}
```

## 🔄 Changelog

### v2.0.0 - Enhanced Academic Framework (2025-07-22)
**🎉 Major Release: Publication-Quality Academic Research Framework**

#### 🎨 **Visual Enhancements**
- **NEW:** Professional academic color scheme (colorblind-friendly palette)
- **FIXED:** Rolling statistics hard-to-see light blue color issue
- **NEW:** Enhanced Mincer-Zarnowitz plots with 95% confidence intervals
- **NEW:** Q-Q plots with confidence bands for normality testing
- **NEW:** ACF/PACF autocorrelation analysis plots (replaces basic summary table)
- **NEW:** Academic journal-quality styling with serif fonts
- **NEW:** High-DPI export ready (300 DPI default)

#### 📊 **Enhanced Statistical Analysis**
- **NEW:** Rich descriptive statistics with skewness, kurtosis, and quartile analysis
- **NEW:** Automatic structural break detection with dynamic sub-period identification
- **NEW:** Detailed Mincer-Zarnowitz regression with full diagnostics and confidence intervals
- **NEW:** Rolling window analysis with time-varying bias detection
- **NEW:** Comprehensive autocorrelation analysis (ACF/PACF) with significance testing
- **NEW:** Enhanced normality testing with multiple statistical tests

#### 🧠 **Economic Intelligence**
- **NEW:** Economic interpretation engine with automated significance assessment
- **NEW:** Policy implications generator for Central Bank, markets, and researchers
- **NEW:** Bias severity classification (minimal/moderate/substantial/severe)
- **NEW:** Learning failure identification based on autocorrelation patterns
- **NEW:** Crisis period and regime change detection

#### 📝 **Academic Output Formats**
- **NEW:** LaTeX report export for academic publication (`export_latex_report()`)
- **NEW:** Professional tables with booktabs styling
- **NEW:** Mathematical equations in proper LaTeX format
- **NEW:** Structured academic sections with colored status indicators
- **ENHANCED:** Comprehensive text reports with rich economic interpretation

#### 🔬 **Technical Improvements**
- **NEW:** Scikit-learn integration for advanced statistical methods
- **NEW:** Enhanced caching system for better performance
- **NEW:** Organized output directory structure (results/, plots/, data/, cache/)
- **NEW:** Batch processing capabilities with multiple period analysis
- **NEW:** Individual high-resolution plot exports for publication
- **ENHANCED:** Error handling and validation throughout framework

#### 🎯 **User Experience**  
- **NEW:** Enhanced CLI with comprehensive output organization
- **NEW:** Academic-style progress reporting and validation
- **NEW:** Detailed usage examples for event studies and batch processing
- **NEW:** Interactive analysis capabilities with rich return structures
- **ENHANCED:** Documentation with academic methodological explanations

### v1.0.0 - Core Framework (2025-01-15)
- ✅ Initial release with core REH testing functionality
- ✅ BCB API integration with intelligent caching
- ✅ Basic publication-quality visualizations  
- ✅ Comprehensive documentation and examples
- ✅ Command-line interface implementation
- ✅ Organized project structure with testing framework

### v0.1.0 - Foundation (2024-12-20) 
- ✅ Basic data fetching and alignment from BCB APIs
- ✅ Initial econometric test implementations (Mincer-Zarnowitz, bias tests)
- ✅ Prototype visualization system
- ✅ Project structure and development environment setup

## 🌟 Acknowledgments

- **Brazilian Central Bank** for providing open access to economic data APIs
- **Jose Luis Oreiro** for the foundational academic analysis
- **python-bcb** library maintainers for BCB API integration
- **Academic community** for methodological foundations and validation

---

**Made with ❤️ for Brazilian Economic Research**
