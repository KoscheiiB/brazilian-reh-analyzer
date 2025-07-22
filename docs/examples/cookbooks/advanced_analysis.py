"""
Advanced Analysis Example for Brazilian REH Analyzer

This example demonstrates advanced usage patterns including:
- Multiple period analysis
- External variables integration
- Custom data loading
- Publication-quality outputs
"""

import pandas as pd
import numpy as np
from brazilian_reh_analyzer import BrazilianREHAnalyzer
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def analyze_multiple_periods():
    """Analyze multiple economic periods separately"""
    print("\n" + "="*50)
    print("MULTIPLE PERIODS ANALYSIS")
    print("="*50)
    
    # Define periods of interest
    periods = [
        ("2017-01-01", "2018-12-31", "Pre-Election Period"),
        ("2019-01-01", "2020-03-31", "Early Bolsonaro Era"),
        ("2020-04-01", "2021-12-31", "COVID-19 Period"),
        ("2022-01-01", "2024-12-31", "Post-Pandemic Period")
    ]
    
    results_summary = []
    
    for start, end, label in periods:
        print(f"\nAnalyzing {label}: {start} to {end}")
        
        try:
            analyzer = BrazilianREHAnalyzer(start, end)
            results = analyzer.comprehensive_analysis()
            
            # Extract key metrics
            desc_stats = results['descriptive_stats']
            rationality = results['rationality_assessment']
            
            period_summary = {
                'period': label,
                'start_date': start,
                'end_date': end,
                'n_observations': desc_stats['n_observations'],
                'mean_error': desc_stats['error_mean'],
                'error_std': desc_stats['error_std'],
                'overall_rational': rationality['overall_rational'],
                'unbiased': rationality['unbiased'],
                'efficient': rationality['efficient']
            }
            
            results_summary.append(period_summary)
            
            # Save period-specific results
            safe_label = label.replace(" ", "_").replace("-", "_").lower()
            analyzer.export_results_summary(f"results_{safe_label}.txt")
            
        except Exception as e:
            print(f"Failed to analyze {label}: {e}")
            continue
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_summary)
    comparison_df.to_csv("periods_comparison.csv", index=False)
    
    print("\n" + "="*50)
    print("PERIODS COMPARISON SUMMARY")
    print("="*50)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def analyze_with_external_variables():
    """Demonstrate analysis with external macroeconomic variables"""
    print("\n" + "="*50)
    print("ANALYSIS WITH EXTERNAL VARIABLES")
    print("="*50)
    
    # Initialize analyzer
    analyzer = BrazilianREHAnalyzer("2020-01-01", "2023-12-31")
    
    # First, get the aligned data
    analyzer.fetch_ipca_data()
    analyzer.fetch_focus_data()
    aligned_data = analyzer.align_forecast_realization_data()
    
    # Create mock external variables (in practice, you'd fetch real data)
    print("Creating mock external variables...")
    external_vars = pd.DataFrame({
        'selic_rate': np.random.normal(10.0, 3.0, len(aligned_data)),
        'gdp_growth': np.random.normal(1.5, 2.0, len(aligned_data)),
        'unemployment': np.random.normal(12.0, 2.0, len(aligned_data)),
        'exchange_rate_change': np.random.normal(0.0, 5.0, len(aligned_data))
    }, index=aligned_data.index)
    
    # Add some correlation with forecast errors to make it realistic
    errors = aligned_data['forecast_error']
    external_vars['selic_rate'] += 0.3 * errors + np.random.normal(0, 0.5, len(errors))
    
    # Run analysis with external variables
    results = analyzer.comprehensive_analysis(
        fetch_data=False,  # Data already fetched
        external_vars=external_vars
    )
    
    # Display orthogonality test results
    if 'orthogonality' in results and 'error' not in results['orthogonality']:
        ortho = results['orthogonality']
        print(f"\nOrthogonality Test Results:")
        print(f"F-statistic: {ortho['f_statistic']:.4f}")
        print(f"F p-value: {ortho['f_pvalue']:.4f}")
        print(f"Passes orthogonality test: {'YES' if ortho['passes_orthogonality_test'] else 'NO'}")
        
        print(f"\nIndividual Variable Results:")
        for var, var_results in ortho['variable_results'].items():
            print(f"  {var}: coeff={var_results['coefficient']:.4f}, "
                  f"p-value={var_results['pvalue']:.4f}")
    
    analyzer.export_results_summary("analysis_with_external_vars.txt")
    
    return results


def create_publication_plots():
    """Create high-quality plots suitable for academic publication"""
    print("\n" + "="*50)
    print("CREATING PUBLICATION-QUALITY PLOTS")
    print("="*50)
    
    # Initialize analyzer with longer period for more data
    analyzer = BrazilianREHAnalyzer("2017-01-01", "2024-12-31")
    
    # Run analysis
    results = analyzer.comprehensive_analysis()
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'font.family': 'serif'
    })
    
    # Export high-quality plots
    analyzer.export_plots("publication_plots/", dpi=600)
    
    # Create custom summary plot for paper
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot forecast errors over time with key events marked
    forecast_errors = analyzer.forecast_errors
    ax.plot(forecast_errors.index, forecast_errors.values, 
           linewidth=1.5, alpha=0.8, color='steelblue')
    
    # Add horizontal lines
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=forecast_errors.mean(), color='green', linestyle=':', 
              linewidth=2, label=f'Mean Error: {forecast_errors.mean():.2f} p.p.')
    
    # Add shaded regions for key periods
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-31'), 
              alpha=0.1, color='red', label='COVID-19 Period')
    
    ax.set_xlabel('Forecast Date')
    ax.set_ylabel('Forecast Error (percentage points)')
    ax.set_title('Brazilian Focus Bulletin Forecast Errors (2017-2024)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('publication_plots/forecast_errors_with_events.pdf', 
                dpi=600, bbox_inches='tight', format='pdf')
    plt.savefig('publication_plots/forecast_errors_with_events.png', 
                dpi=600, bbox_inches='tight')
    plt.show()
    
    print("✓ Publication plots saved to publication_plots/")


def batch_processing_example():
    """Demonstrate batch processing of multiple configurations"""
    print("\n" + "="*50)
    print("BATCH PROCESSING EXAMPLE")
    print("="*50)
    
    # Define different analysis configurations
    configurations = [
        {
            'name': 'full_period',
            'start': '2017-01-01',
            'end': '2024-12-31',
            'description': 'Full available period'
        },
        {
            'name': 'pre_covid',
            'start': '2017-01-01', 
            'end': '2020-02-29',
            'description': 'Pre-COVID period'
        },
        {
            'name': 'covid_era',
            'start': '2020-03-01',
            'end': '2022-12-31',
            'description': 'COVID era'
        },
        {
            'name': 'recent',
            'start': '2022-01-01',
            'end': '2024-12-31', 
            'description': 'Recent period'
        }
    ]
    
    batch_results = []
    
    for config in configurations:
        print(f"\nProcessing configuration: {config['name']}")
        print(f"Description: {config['description']}")
        
        try:
            analyzer = BrazilianREHAnalyzer(
                start_date=config['start'],
                end_date=config['end']
            )
            
            results = analyzer.comprehensive_analysis()
            
            # Extract summary metrics
            desc_stats = results['descriptive_stats']
            rationality = results['rationality_assessment']
            mz = results.get('mincer_zarnowitz', {})
            bias = results.get('bias_test', {})
            
            batch_result = {
                'configuration': config['name'],
                'description': config['description'],
                'period': f"{config['start']} to {config['end']}",
                'n_observations': desc_stats.get('n_observations', 0),
                'mean_error': desc_stats.get('error_mean', np.nan),
                'error_std': desc_stats.get('error_std', np.nan),
                'overall_rational': rationality.get('overall_rational', False),
                'mz_alpha': mz.get('alpha', np.nan),
                'mz_beta': mz.get('beta', np.nan),
                'mz_joint_pvalue': mz.get('joint_test_pvalue', np.nan),
                'bias_pvalue': bias.get('p_value', np.nan)
            }
            
            batch_results.append(batch_result)
            
            # Save individual results
            analyzer.export_results_summary(f"batch_{config['name']}_results.txt")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Create comprehensive comparison
    batch_df = pd.DataFrame(batch_results)
    batch_df.to_csv("batch_analysis_comparison.csv", index=False)
    
    print("\n" + "="*60)
    print("BATCH ANALYSIS SUMMARY")
    print("="*60)
    
    # Display key columns
    display_cols = ['configuration', 'n_observations', 'mean_error', 
                   'overall_rational', 'mz_joint_pvalue']
    print(batch_df[display_cols].to_string(index=False))
    
    print(f"\n✓ Batch processing completed!")
    print(f"✓ Detailed results saved to batch_analysis_comparison.csv")
    
    return batch_df


def main():
    """Run all advanced analysis examples"""
    print("BRAZILIAN REH ANALYZER - ADVANCED EXAMPLES")
    print("="*60)
    
    try:
        # Run different types of advanced analysis
        periods_results = analyze_multiple_periods()
        
        external_vars_results = analyze_with_external_variables()
        
        create_publication_plots()
        
        batch_results = batch_processing_example()
        
        print("\n" + "="*60)
        print("ALL ADVANCED EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- periods_comparison.csv")
        print("- analysis_with_external_vars.txt")
        print("- publication_plots/ directory")
        print("- batch_analysis_comparison.csv")
        print("- Individual period and configuration results")
        
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        print("Make sure you have python-bcb installed and internet connection")


if __name__ == "__main__":
    main()