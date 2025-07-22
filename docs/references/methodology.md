# Brazilian REH Analyzer - Methodology & Knowledge Base

## Table of Contents
1. [Theoretical Framework](#theoretical-framework)
2. [Econometric Testing Methodology](#econometric-testing-methodology)
3. [Data Sources & Access](#data-sources--access)
4. [Implementation Challenges](#implementation-challenges)
5. [Brazilian Institutional Context](#brazilian-institutional-context)
6. [Academic References](#academic-references)
7. [Technical Specifications](#technical-specifications)

## Theoretical Framework

### Rational Expectations Hypothesis (REH)

The Rational Expectations Hypothesis, introduced by John Muth (1961) and developed by Lucas and Sargent, posits that economic agents form expectations optimally using all available information and their understanding of economic relationships.

#### Core Properties

**Unbiasedness Condition:**
```
E[P_t - E_{t-h}[P_t]] = 0
```
Where:
- `P_t` = realized inflation in period t
- `E_{t-h}[P_t]` = forecast made h periods earlier
- Forecast errors must have zero mean over time

**Efficiency Condition:**
```
e_t = P_t - E_{t-h}[P_t]
```
Forecast errors must be unpredictable using information available at time t-h.

#### Weak vs Strong Form Efficiency

**Weak Form (No Autocorrelation):**
```
Cov(e_t, e_{t-j}) = 0, ∀j > 0
```

**Strong Form (Orthogonality):**
```
Cov(e_t, X_{t-h}) = 0
```
Where X_{t-h} represents any information available at forecast time.

## Econometric Testing Methodology

### 1. Mincer-Zarnowitz (MZ) Regression

**Base Regression:**
```
P_t = α + β · E_{t-h}[P_t] + u_t
```

**Null Hypothesis for Rationality:**
```
H_0: (α, β) = (0, 1)
```

**Joint Test Statistic:**
```
F = [(RSS_r - RSS_u)/q] / [RSS_u/(n-k)]
```
Where:
- RSS_r = restricted sum of squares
- RSS_u = unrestricted sum of squares
- q = number of restrictions (2)
- n = sample size, k = parameters

**Implementation Notes:**
- Use robust standard errors for heteroskedasticity
- Consider small sample corrections for finite samples
- Check residual properties for model validity

### 2. Autocorrelation Testing

**Ljung-Box Test Statistic:**
```
LB = n(n+2) Σ(k=1 to h) [ρ_k²/(n-k)]
```
Where:
- n = sample size
- h = maximum lag
- ρ_k = sample autocorrelation at lag k
- Under H_0: LB ~ χ²(h)

**Breusch-Godfrey Test:**
Auxiliary regression:
```
ê_t = α + β₁X_t + β₂ê_{t-1} + ... + β_pê_{t-p} + v_t
```
Test statistic: `LM = (n-p)R² ~ χ²(p)`

### 3. Bias Testing (Holden-Peel)

**Simple Bias Test:**
```
e_t = λ + u_t
```
**Test:** `H_0: λ = 0`

**t-statistic:**
```
t = λ̂ / SE(λ̂)
```

### 4. Orthogonality Testing

**General Form:**
```
e_t = α + Σβᵢ Xᵢ,t-h + u_t
```

**F-test for joint significance:**
```
H_0: β₁ = β₂ = ... = βₖ = 0
```

**Relevant X variables for Brazil:**
- Lagged inflation rates
- Selic interest rate
- Unemployment rate
- GDP growth
- Exchange rate changes
- Political uncertainty indices

## Data Sources & Access

### Primary Data Sources

#### 1. IPCA Data (Realized Inflation)
- **Source**: IBGE (Brazilian Institute of Geography and Statistics)
- **Access**: BCB SGS API, series code 433
- **Description**: 12-month accumulated IPCA
- **Frequency**: Monthly
- **URL**: `https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados`

#### 2. Focus Bulletin Forecasts
- **Source**: BCB Market Expectations System
- **Access**: BCB Expectations API (OData)
- **Endpoint**: `ExpectativasMercadoInflacao12Meses`
- **Description**: 12-month ahead IPCA forecasts
- **Frequency**: Daily
- **URL**: `https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/`

### API Access Specifications

#### SGS API Format
```
GET https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados
Parameters:
- formato: json|csv|xml
- dataInicial: DD/MM/YYYY
- dataFinal: DD/MM/YYYY
```

#### Expectations API Format
```
GET https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoInflacao12Meses
OData Filters:
- $filter: Indicador eq 'IPCA' and Suavizada eq 'N'
- $select: Data,Mediana,Media,numeroRespondentes
- $orderby: Data desc
```

### Data Quality Requirements

#### Focus Data Filtering
- **Minimum Respondents**: ≥10 institutions
- **Non-smoothed Data**: `Suavizada = 'N'`
- **Median Usage**: More robust than mean to outliers
- **Quality Control**: Remove obvious data errors and outliers

#### Temporal Alignment
- **Critical Issue**: 12-month forecast alignment
- **Forecast Date**: Date when forecast was made
- **Target Date**: Forecast date + 12 months
- **Realization Date**: IPCA release date (typically 10th of month)
- **Matching Logic**: Find IPCA within 15 days of target month-end

## Implementation Challenges

### 1. Data Alignment Issues

**Problem**: Series objects vs scalar values
```python
# WRONG - Returns Series
realized_ipca = ipca_candidates.iloc[0]

# CORRECT - Extract scalar value
realized_ipca = float(ipca_candidates.iloc[0])
```

**Solution**: Implement `_ensure_scalar()` method:
```python
def _ensure_scalar(self, value) -> float:
    if isinstance(value, pd.Series):
        if len(value) == 1:
            return float(value.iloc[0])
        else:
            raise ValueError(f"Expected single value, got Series with {len(value)} values")
    return float(value)
```

### 2. API Rate Limiting

**Problem**: BCB APIs have undocumented rate limits
**Solution**: Implement respectful rate limiting:
- Random delays: 0.8-2.5 seconds between requests
- Batch breaks: 10-20 seconds every 8 requests
- Error handling: 2-5 second delays on failures
- Browser-like headers to avoid blocking

### 3. Date Handling Complexities

**Issues**:
- Brazilian holidays affect data release schedules
- Weekend adjustments for forecast matching
- IPCA release timing variations
- Focus survey timing irregularities

**Solutions**:
- Flexible date matching with tolerance windows
- Business day calendar integration
- Manual override capabilities for special cases

### 4. Pandas Deprecation Warnings

**Issue**: `resample('M')` deprecated in favor of `resample('ME')`
**Solution**: Use month-end frequency designation:
```python
# OLD (deprecated)
ipca_monthly = ipca_clean.resample('M').last()

# NEW (correct)
ipca_monthly = ipca_clean.resample('ME').last()
```

## Brazilian Institutional Context

### Focus Bulletin Background

**History**: Launched in 2001 as part of inflation targeting framework
**Participants**: ~130 financial institutions
**Frequency**: Daily collection, weekly publication
**Variables**: IPCA, GDP, Selic, Exchange rate, fiscal indicators

### Institutional Critiques

**Corecon-SP Criticism** (2025):
- Sample bias toward financial market institutions
- Potential strategic overestimation incentives
- Limited representation of real economy perspectives
- Calls for broader participant base

**Academic Findings**:
- Historical underestimation bias (pre-2016)
- Possible overestimation in recent periods
- Persistent inefficiencies in forecast errors
- Limited use of available macroeconomic information

### Policy Regime Context

**Key Periods for Analysis**:
- **2017-2018**: Pre-election uncertainty, truckers' strike
- **2018-2019**: Election period, new government transition
- **2020-2021**: COVID-19 pandemic, unprecedented shocks
- **2022-2024**: Post-pandemic normalization, inflation pressures

**Structural Breaks**: Major events affecting expectation formation:
- May 2018: Truckers' strike (supply disruption)
- October 2018: Presidential election
- March 2020: COVID-19 pandemic onset
- 2021-2022: Global supply chain disruptions
- 2023-2024: Monetary normalization period

## Academic References

### Primary Sources

1. **Oreiro, José Luis** (2025). "Assessment of the Rationality of Focus Bulletin Inflation Forecasts for the 12-Month Ahead IPCA (January 2017 – April 2025)". AI Gemini Deep Research Report.

2. **BCB Working Paper 464**: Focus professional forecasters analysis with historical bias findings.

3. **UFRGS Thesis (Baldusco)**: 2003-2008 period analysis concluding systematic underestimation.

4. **IPEA Discussion Paper 2814**: 2016-2019 analysis showing potential overestimation shift.

5. **BCB Working Paper TD227**: 2002-2010 analysis with adaptive expectations findings.

### Methodological References

6. **Mincer, J. & Zarnowitz, V.** (1969). "The Evaluation of Economic Forecasts". In Economic Forecasts and Expectations: Analysis of Forecasting Behavior and Performance, NBER.

7. **Holden, K. & Peel, D.A.** (1990). "On Testing for Unbiasedness and Efficiency of Forecasts". Manchester School, 58(2), 120-127.

8. **Muth, J.F.** (1961). "Rational Expectations and the Theory of Price Movements". Econometrica, 29(3), 315-335.

### Brazilian Context References

9. **BCB Inflation Reports** (2023-2024): Concerns about expectations de-anchoring.

10. **Corecon-SP Critique** (2025): "O Boletim Focus e o viés financeiro: quando as expectativas criam a realidade".

11. **IMF Working Paper** (February 2025): Monetary policy impact on Brazilian inflation expectations.

## Technical Specifications

### Required Python Libraries

```python
# Core data handling
pandas >= 1.5.0
numpy >= 1.21.0

# Econometric analysis
statsmodels >= 0.13.0
scipy >= 1.7.0

# Brazilian data access
python-bcb >= 0.3.0

# Visualization
matplotlib >= 3.5.0
seaborn >= 0.11.0

# Utilities
requests >= 2.28.0
pathlib  # built-in
logging  # built-in
```

### API Rate Limiting Configuration

```python
RATE_LIMIT_CONFIG = {
    'min_delay': 0.8,           # Minimum seconds between requests
    'max_delay': 2.5,           # Maximum seconds between requests
    'requests_per_batch': 8,    # Requests before longer break
    'batch_delay_min': 10,      # Minimum batch break seconds
    'batch_delay_max': 20,      # Maximum batch break seconds
    'error_delay_min': 2.0,     # Minimum delay after errors
    'error_delay_max': 5.0      # Maximum delay after errors
}
```

### Cache Configuration

```python
CACHE_CONFIG = {
    'ipca_ttl_hours': 168,      # 1 week for IPCA data
    'focus_ttl_hours': 24,      # 1 day for Focus data
    'max_cache_files': 3,       # Keep latest 3 cache files
    'cache_format': 'pickle',   # Serialization format
    'compression': True         # Compress cache files
}
```

### Statistical Test Parameters

```python
TEST_CONFIG = {
    'significance_level': 0.05,     # Standard 5% significance
    'autocorr_max_lags': 10,        # Maximum lags for autocorrelation
    'bootstrap_iterations': 1000,   # Bootstrap replications
    'rolling_window_months': 12,    # Rolling analysis window
    'min_observations': 30          # Minimum sample size
}
```

### Data Validation Rules

```python
VALIDATION_RULES = {
    'min_respondents': 10,          # Minimum Focus respondents
    'max_forecast_error': 20.0,     # Maximum reasonable error (p.p.)
    'ipca_reasonable_range': (-5, 25),  # IPCA bounds (%)
    'forecast_reasonable_range': (0, 15),  # Forecast bounds (%)
    'max_missing_months': 3         # Maximum consecutive missing data
}
```

This knowledge base provides the complete technical foundation for implementing the Brazilian REH Analyzer. All mathematical formulations, data sources, implementation challenges, and contextual information needed for successful development are documented here.
