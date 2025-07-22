"""
Basic Usage Example for Brazilian REH Analyzer

This example demonstrates the most common usage patterns for analyzing
Brazilian inflation forecast rationality.
"""

from brazilian_reh_analyzer import BrazilianREHAnalyzer
import logging

# Configure logging to see progress
logging.basicConfig(level=logging.INFO)

def main():
    """Run basic REH analysis"""
    
    print("=" * 60)
    print("BRAZILIAN REH ANALYZER - BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize analyzer for recent period
    analyzer = BrazilianREHAnalyzer(
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Run comprehensive analysis (uses cached data when available)
    print("\n1. Running comprehensive analysis...")
    results = analyzer.comprehensive_analysis()
    
    # Display key results
    print("\n2. Key Results:")
    desc_stats = results['descriptive_stats']
    rationality = results['rationality_assessment']
    
    print(f"   Analysis Period: {desc_stats['date_range']}")
    print(f"   Observations: {desc_stats['n_observations']}")
    print(f"   Mean Forecast Error: {desc_stats['error_mean']:.3f} p.p.")
    
    print(f"\n   Rationality Tests:")
    print(f"   ✓ Unbiased: {'PASS' if rationality['unbiased'] else 'FAIL'}")
    print(f"   ✓ MZ Test: {'PASS' if rationality['mz_rational'] else 'FAIL'}")
    print(f"   ✓ Efficient: {'PASS' if rationality['efficient'] else 'FAIL'}")
    print(f"   ✓ Overall Rational: {'PASS' if rationality['overall_rational'] else 'FAIL'}")
    
    # Generate diagnostic plots
    print("\n3. Generating diagnostic plots...")
    analyzer.plot_enhanced_diagnostics()
    
    # Export results for academic use
    print("\n4. Exporting results...")
    analyzer.export_results_summary("basic_analysis_results.txt")
    analyzer.export_plots("basic_plots/", dpi=300)
    
    print(f"\n✓ Analysis completed successfully!")
    print(f"✓ Results saved to basic_analysis_results.txt")
    print(f"✓ Plots saved to basic_plots/ directory")


if __name__ == "__main__":
    main()