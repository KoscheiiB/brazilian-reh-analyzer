"""
Interactive Analysis Example for Brazilian REH Analyzer

This example demonstrates interactive usage patterns suitable for
Jupyter notebooks and exploratory data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from brazilian_reh_analyzer import BrazilianREHAnalyzer
from brazilian_reh_analyzer.visualizations import REHVisualizations
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def interactive_setup():
    """Set up analyzer for interactive analysis"""
    print("Setting up Brazilian REH Analyzer...")
    print("This may take a few moments to fetch data from BCB...")
    
    # Initialize analyzer
    analyzer = BrazilianREHAnalyzer(
        start_date="2019-01-01",
        end_date="2024-12-31"
    )
    
    # Fetch and align data
    print("📊 Fetching IPCA data...")
    analyzer.fetch_ipca_data()
    
    print("📈 Fetching Focus data...")
    analyzer.fetch_focus_data()
    
    print("🔄 Aligning forecast and realization data...")
    aligned_data = analyzer.align_forecast_realization_data()
    
    print(f"✅ Setup complete! Aligned {len(aligned_data)} observations")
    print(f"   Period: {aligned_data.index.min().date()} to {aligned_data.index.max().date()}")
    
    return analyzer


def explore_data(analyzer):
    """Explore the aligned dataset interactively"""
    print("\n" + "="*50)
    print("📊 DATA EXPLORATION")
    print("="*50)
    
    data = analyzer.aligned_data
    
    # Basic statistics
    print("\n🔍 Basic Statistics:")
    print(f"   Total observations: {len(data)}")
    print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"   Average respondents: {data['respondents'].mean():.1f}")
    
    print(f"\n📈 Forecast Statistics:")
    print(f"   Mean forecast: {data['forecast'].mean():.3f}%")
    print(f"   Forecast std: {data['forecast'].std():.3f}%")
    print(f"   Forecast range: {data['forecast'].min():.2f}% to {data['forecast'].max():.2f}%")
    
    print(f"\n🎯 Realization Statistics:")
    print(f"   Mean realized: {data['realized'].mean():.3f}%")
    print(f"   Realized std: {data['realized'].std():.3f}%")
    print(f"   Realized range: {data['realized'].min():.2f}% to {data['realized'].max():.2f}%")
    
    print(f"\n❌ Forecast Error Statistics:")
    errors = data['forecast_error']
    print(f"   Mean error: {errors.mean():.3f} p.p.")
    print(f"   Error std: {errors.std():.3f} p.p.")
    print(f"   Error range: {errors.min():.2f} to {errors.max():.2f} p.p.")
    print(f"   Absolute mean error: {errors.abs().mean():.3f} p.p.")
    
    # Quick visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Forecast vs Realized scatter
    axes[0, 0].scatter(data['forecast'], data['realized'], alpha=0.6)
    min_val = min(data['forecast'].min(), data['realized'].min())
    max_val = max(data['forecast'].max(), data['realized'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Focus Forecast (%)')
    axes[0, 0].set_ylabel('Realized IPCA (%)')
    axes[0, 0].set_title('Forecast vs Realization')
    
    # Time series of both
    axes[0, 1].plot(data.index, data['forecast'], label='Forecast', alpha=0.8)
    axes[0, 1].plot(data.index, data['realized'], label='Realized', alpha=0.8)
    axes[0, 1].set_ylabel('IPCA 12-month (%)')
    axes[0, 1].set_title('Forecasts and Realizations Over Time')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Error time series
    axes[1, 0].plot(errors.index, errors.values, color='red', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=errors.mean(), color='green', linestyle=':', alpha=0.8)
    axes[1, 0].set_ylabel('Forecast Error (p.p.)')
    axes[1, 0].set_title('Forecast Errors Over Time')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Error distribution
    axes[1, 1].hist(errors, bins=20, alpha=0.7, density=True)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[1, 1].axvline(x=errors.mean(), color='green', linestyle=':', alpha=0.8)
    axes[1, 1].set_xlabel('Forecast Error (p.p.)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Forecast Errors')
    
    plt.tight_layout()
    plt.show()
    
    return data


def run_individual_tests(analyzer):
    """Run and examine individual econometric tests"""
    print("\n" + "="*50)
    print("🧪 INDIVIDUAL ECONOMETRIC TESTS")
    print("="*50)
    
    data = analyzer.aligned_data
    forecast = data['forecast']
    realized = data['realized']
    errors = data['forecast_error']
    
    from brazilian_reh_analyzer.tests import REHTests
    
    # 1. Mincer-Zarnowitz Test
    print("\n1️⃣ MINCER-ZARNOWITZ TEST")
    print("-" * 30)
    mz_results = REHTests.mincer_zarnowitz_test(forecast, realized)
    
    if 'error' not in mz_results:
        print(f"   Regression: Realized = {mz_results['alpha']:.4f} + {mz_results['beta']:.4f} × Forecast")
        print(f"   α (intercept): {mz_results['alpha']:.4f} (p-value: {mz_results['alpha_pvalue']:.4f})")
        print(f"   β (slope): {mz_results['beta']:.4f} (p-value: {mz_results['beta_pvalue']:.4f})")
        print(f"   Joint test p-value: {mz_results['joint_test_pvalue']:.4f}")
        print(f"   R-squared: {mz_results['r_squared']:.4f}")
        print(f"   🎯 Result: {'RATIONAL' if mz_results['passes_joint_test'] else 'NOT RATIONAL'}")
        
        # Interpretation
        if mz_results['joint_test_pvalue'] < 0.05:
            print("   📝 Interpretation: Forecasts violate unbiasedness/efficiency (reject H₀: α=0, β=1)")
        else:
            print("   📝 Interpretation: Forecasts are unbiased and efficient")
    else:
        print(f"   ❌ Test failed: {mz_results['error']}")
    
    # 2. Autocorrelation Test
    print("\n2️⃣ AUTOCORRELATION TEST (Ljung-Box)")
    print("-" * 40)
    autocorr_results = REHTests.autocorrelation_test(errors, max_lags=10)
    
    if 'error' not in autocorr_results:
        print(f"   Ljung-Box statistic: {autocorr_results['ljung_box_stat']:.4f}")
        print(f"   p-value: {autocorr_results['ljung_box_pvalue']:.4f}")
        print(f"   Lags tested: {autocorr_results['max_lags_tested']}")
        print(f"   🎯 Result: {'EFFICIENT' if autocorr_results['passes_efficiency_test'] else 'NOT EFFICIENT'}")
        
        # Interpretation
        if autocorr_results['significant_autocorr']:
            print("   📝 Interpretation: Forecast errors show significant autocorrelation")
        else:
            print("   📝 Interpretation: No significant autocorrelation in forecast errors")
    else:
        print(f"   ❌ Test failed: {autocorr_results['error']}")
    
    # 3. Bias Test
    print("\n3️⃣ BIAS TEST (Holden-Peel)")
    print("-" * 30)
    bias_results = REHTests.bias_test(errors)
    
    if 'error' not in bias_results:
        print(f"   Mean forecast error: {bias_results['mean_error']:.4f} p.p.")
        print(f"   t-statistic: {bias_results['t_statistic']:.4f}")
        print(f"   p-value: {bias_results['p_value']:.4f}")
        print(f"   95% CI: [{bias_results['confidence_interval_95'][0]:.4f}, {bias_results['confidence_interval_95'][1]:.4f}]")
        print(f"   🎯 Result: {'UNBIASED' if bias_results['passes_unbiasedness_test'] else 'BIASED'}")
        
        # Interpretation
        if bias_results['is_biased']:
            print(f"   📝 Interpretation: Systematic {bias_results['bias_direction']} detected")
        else:
            print("   📝 Interpretation: No systematic bias in forecasts")
    else:
        print(f"   ❌ Test failed: {bias_results['error']}")


def examine_time_patterns(analyzer):
    """Examine temporal patterns in forecast performance"""
    print("\n" + "="*50)
    print("📅 TEMPORAL PATTERNS ANALYSIS")
    print("="*50)
    
    data = analyzer.aligned_data
    
    # Rolling statistics
    window = 12  # 12 observations rolling window
    data['rolling_mean_error'] = data['forecast_error'].rolling(window=window).mean()
    data['rolling_std_error'] = data['forecast_error'].rolling(window=window).std()
    
    # Monthly patterns
    data['month'] = data.index.month
    data['year'] = data.index.year
    
    monthly_stats = data.groupby('month')['forecast_error'].agg(['mean', 'std', 'count']).round(4)
    print("\n📊 Monthly Pattern in Forecast Errors:")
    print(monthly_stats)
    
    # Yearly patterns
    yearly_stats = data.groupby('year')['forecast_error'].agg(['mean', 'std', 'count']).round(4)
    print("\n📊 Yearly Pattern in Forecast Errors:")
    print(yearly_stats)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Rolling statistics
    axes[0, 0].plot(data.index, data['rolling_mean_error'], label='Rolling Mean', linewidth=2)
    axes[0, 0].fill_between(data.index, 
                           data['rolling_mean_error'] - data['rolling_std_error'],
                           data['rolling_mean_error'] + data['rolling_std_error'],
                           alpha=0.3, label='±1 Std Dev')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title(f'Rolling Statistics ({window}-obs window)')
    axes[0, 0].set_ylabel('Forecast Error (p.p.)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Monthly boxplot
    monthly_data = [data[data['month'] == m]['forecast_error'].values for m in range(1, 13)]
    axes[0, 1].boxplot(monthly_data, labels=[f'{m:02d}' for m in range(1, 13)])
    axes[0, 1].set_title('Monthly Distribution of Forecast Errors')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Forecast Error (p.p.)')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Yearly trend
    yearly_stats['mean'].plot(kind='bar', ax=axes[1, 0], color='skyblue', alpha=0.8)
    axes[1, 0].set_title('Mean Forecast Error by Year')
    axes[1, 0].set_ylabel('Mean Forecast Error (p.p.)')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Forecast accuracy over time (absolute errors)
    data['abs_error'] = data['forecast_error'].abs()
    data['rolling_mae'] = data['abs_error'].rolling(window=window).mean()
    axes[1, 1].plot(data.index, data['rolling_mae'], color='orange', linewidth=2)
    axes[1, 1].set_title(f'Rolling Mean Absolute Error ({window}-obs window)')
    axes[1, 1].set_ylabel('Mean Absolute Error (p.p.)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def compare_forecast_vs_naive():
    """Compare Focus forecasts against naive benchmarks"""
    print("\n" + "="*50)
    print("🔄 FORECAST COMPARISON: FOCUS vs NAIVE MODELS")
    print("="*50)
    
    # Initialize analyzer
    analyzer = BrazilianREHAnalyzer("2020-01-01", "2024-12-31")
    analyzer.fetch_ipca_data()
    analyzer.fetch_focus_data()
    data = analyzer.align_forecast_realization_data()
    
    # Create naive forecasts
    # Naive 1: Previous realization (random walk)
    data = data.sort_index()
    data['naive_rw'] = data['realized'].shift(12)  # Previous year's realization
    
    # Naive 2: Simple average of last 3 observations
    data['naive_avg3'] = data['realized'].rolling(window=3).mean().shift(12)
    
    # Naive 3: Long-term average (expanding window)
    data['naive_lt_avg'] = data['realized'].expanding().mean().shift(12)
    
    # Remove missing values for fair comparison
    comparison_data = data.dropna()
    
    if len(comparison_data) > 0:
        # Calculate errors for each method
        comparison_data['focus_error'] = comparison_data['forecast_error']
        comparison_data['rw_error'] = comparison_data['realized'] - comparison_data['naive_rw']
        comparison_data['avg3_error'] = comparison_data['realized'] - comparison_data['naive_avg3']
        comparison_data['lt_avg_error'] = comparison_data['realized'] - comparison_data['naive_lt_avg']
        
        # Calculate performance metrics
        metrics = {}
        for method in ['focus', 'rw', 'avg3', 'lt_avg']:
            error_col = f'{method}_error'
            metrics[method] = {
                'MAE': comparison_data[error_col].abs().mean(),
                'MSE': (comparison_data[error_col] ** 2).mean(),
                'RMSE': np.sqrt((comparison_data[error_col] ** 2).mean()),
                'Mean_Error': comparison_data[error_col].mean(),
                'Std_Error': comparison_data[error_col].std()
            }
        
        # Display results
        metrics_df = pd.DataFrame(metrics).round(4)
        print("\n📊 Performance Comparison:")
        print(metrics_df)
        
        # Determine best performer
        best_mae = metrics_df.loc['MAE'].idxmin()
        best_rmse = metrics_df.loc['RMSE'].idxmin()
        
        print(f"\n🏆 Best MAE: {best_mae.upper()} ({metrics_df.loc['MAE', best_mae]:.4f})")
        print(f"🏆 Best RMSE: {best_rmse.upper()} ({metrics_df.loc['RMSE', best_rmse]:.4f})")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE comparison
        mae_values = metrics_df.loc['MAE']
        mae_values.plot(kind='bar', ax=axes[0, 0], color='lightblue')
        axes[0, 0].set_title('Mean Absolute Error Comparison')
        axes[0, 0].set_ylabel('MAE (p.p.)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison  
        rmse_values = metrics_df.loc['RMSE']
        rmse_values.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
        axes[0, 1].set_title('Root Mean Squared Error Comparison')
        axes[0, 1].set_ylabel('RMSE (p.p.)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Error distributions
        error_cols = ['focus_error', 'rw_error', 'avg3_error', 'lt_avg_error']
        for i, col in enumerate(error_cols):
            axes[1, 0].hist(comparison_data[col], alpha=0.5, 
                           label=col.replace('_error', '').upper(), bins=15)
        axes[1, 0].set_title('Error Distributions')
        axes[1, 0].set_xlabel('Forecast Error (p.p.)')
        axes[1, 0].legend()
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Time series of absolute errors
        for col in error_cols:
            axes[1, 1].plot(comparison_data.index, comparison_data[col].abs(), 
                           alpha=0.7, label=col.replace('_error', '').upper())
        axes[1, 1].set_title('Absolute Errors Over Time')
        axes[1, 1].set_ylabel('Absolute Error (p.p.)')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df
    else:
        print("❌ Insufficient data for naive model comparison")
        return None


def interactive_main():
    """Main function for interactive analysis"""
    print("🇧🇷 BRAZILIAN REH ANALYZER - INTERACTIVE ANALYSIS")
    print("="*60)
    print("This interactive example demonstrates step-by-step analysis")
    print("suitable for Jupyter notebooks and exploratory data analysis.")
    print("="*60)
    
    # Setup
    analyzer = interactive_setup()
    
    # Explore data
    data = explore_data(analyzer)
    
    # Run individual tests
    run_individual_tests(analyzer)
    
    # Examine patterns
    examine_time_patterns(analyzer)
    
    # Compare with naive models
    comparison_metrics = compare_forecast_vs_naive()
    
    # Final comprehensive analysis
    print("\n" + "="*50)
    print("🎯 FINAL COMPREHENSIVE ANALYSIS")
    print("="*50)
    
    results = analyzer.comprehensive_analysis(fetch_data=False)
    
    # Display final verdict
    ra = results['rationality_assessment']
    print(f"\n📋 FINAL RATIONALITY ASSESSMENT:")
    print(f"   Unbiased: {'✅ PASS' if ra['unbiased'] else '❌ FAIL'}")
    print(f"   MZ Efficient: {'✅ PASS' if ra['mz_rational'] else '❌ FAIL'}")
    print(f"   No Autocorr: {'✅ PASS' if ra['efficient'] else '❌ FAIL'}")
    print(f"   OVERALL: {'✅ RATIONAL' if ra['overall_rational'] else '❌ NOT RATIONAL'}")
    
    # Save results for further analysis
    analyzer.export_results_summary("interactive_analysis_results.txt")
    print(f"\n💾 Results saved to: interactive_analysis_results.txt")
    
    print(f"\n🎉 Interactive analysis completed!")
    return analyzer, results


if __name__ == "__main__":
    # Run the interactive analysis
    analyzer, results = interactive_main()
    
    # Display final message
    print("\n" + "="*60)
    print("✨ ANALYSIS COMPLETE!")
    print("="*60)
    print("You can now:")
    print("• Examine the 'analyzer' object for detailed data")
    print("• Explore the 'results' dictionary for test outcomes")
    print("• Create custom plots using the aligned data")
    print("• Run additional periods or configurations")
    print("="*60)