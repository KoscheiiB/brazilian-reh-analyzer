"""
Basic Usage Example for Brazilian REH Analyzer v2.0.0

This example demonstrates the enhanced v2.0.0 academic framework features 
for analyzing Brazilian inflation forecast rationality, including LaTeX export,
enhanced diagnostics, and professional academic output.
"""

from brazilian_reh_analyzer import BrazilianREHAnalyzer
import logging
import os

# Configure logging to see progress
logging.basicConfig(level=logging.INFO)

def main():
    """Run enhanced v2.0.0 REH analysis with academic features"""
    
    print("=" * 70)
    print("BRAZILIAN REH ANALYZER v2.0.0 - ENHANCED ACADEMIC FRAMEWORK")
    print("=" * 70)
    
    # Initialize analyzer for recent period
    analyzer = BrazilianREHAnalyzer(
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Run comprehensive analysis (uses cached data when available)
    print("\n1. Running comprehensive analysis...")
    results = analyzer.comprehensive_analysis()
    
    # Display enhanced v2.0.0 results
    print("\n2. Enhanced Analysis Results:")
    desc_stats = results['descriptive_stats']
    rationality = results['rationality_assessment']
    rich_stats = results.get('rich_descriptive_stats', {})
    econ_interp = results.get('economic_interpretation', {})
    
    print(f"   Analysis Period: {desc_stats['date_range']}")
    print(f"   Observations: {desc_stats['n_observations']}")
    print(f"   Mean Forecast Error: {desc_stats['error_mean']:.3f} p.p.")
    
    # Enhanced statistical information
    if rich_stats and 'errors' in rich_stats:
        error_stats = rich_stats['errors']
        print(f"   Error Skewness: {error_stats.get('skewness', 0):.3f}")
        print(f"   Error Kurtosis: {error_stats.get('kurtosis', 0):.3f}")
    
    print(f"\n   Rationality Tests:")
    print(f"   ✓ Unbiased: {'PASS' if rationality['unbiased'] else 'FAIL'}")
    print(f"   ✓ MZ Test: {'PASS' if rationality['mz_rational'] else 'FAIL'}")
    print(f"   ✓ Efficient: {'PASS' if rationality['efficient'] else 'FAIL'}")
    print(f"   ✓ Overall Rational: {'PASS' if rationality['overall_rational'] else 'FAIL'}")
    
    # Enhanced v2.0.0 features
    if econ_interp and 'bias_analysis' in econ_interp:
        bias_analysis = econ_interp['bias_analysis']
        print(f"\n   🆕 Economic Analysis:")
        print(f"   📊 Bias Severity: {bias_analysis.get('severity', 'unknown').title()}")
        print(f"   📈 Bias Direction: {bias_analysis.get('direction', 'unknown').title()}")
    
    sub_periods = results.get('sub_period_analysis', {})
    if sub_periods:
        print(f"   🔍 Sub-periods Detected: {len(sub_periods)}")
    
    # Generate enhanced diagnostic plots with ACF/PACF
    print("\n3. Generating enhanced diagnostic plots...")
    analyzer.plot_enhanced_diagnostics()
    
    # Create organized output structure
    print("\n4. Creating organized output structure...")
    os.makedirs("enhanced_analysis/results", exist_ok=True)
    os.makedirs("enhanced_analysis/plots", exist_ok=True) 
    os.makedirs("enhanced_analysis/data", exist_ok=True)
    os.makedirs("enhanced_analysis/latex", exist_ok=True)
    
    # Export comprehensive results
    print("\n5. Exporting enhanced v2.0.0 results...")
    
    # Text summary
    analyzer.export_results_summary("enhanced_analysis/results/comprehensive_analysis.txt")
    
    # High-resolution plots
    analyzer.export_plots("enhanced_analysis/plots/", dpi=300)
    
    # Raw data
    analyzer.save_data("enhanced_analysis/data/aligned_forecast_data.csv")
    
    # 🆕 NEW: LaTeX academic report
    latex_file = analyzer.export_latex_report(
        "enhanced_analysis/latex/academic_report.tex",
        "Brazilian Focus Bulletin Rationality Assessment (2020-2023)",
        "Enhanced REH Framework v2.0.0"
    )
    
    print(f"\n🎉 Enhanced v2.0.0 Analysis Completed Successfully!")
    print(f"✅ Comprehensive analysis: enhanced_analysis/results/")
    print(f"✅ Professional plots: enhanced_analysis/plots/")
    print(f"✅ Raw data export: enhanced_analysis/data/")
    print(f"✅ LaTeX academic report: {latex_file}")
    print(f"\n🔬 New v2.0.0 Features Used:")
    print(f"   📊 Rich descriptive statistics with skewness/kurtosis")
    print(f"   🎯 ACF/PACF autocorrelation analysis") 
    print(f"   📈 Enhanced Mincer-Zarnowitz diagnostics")
    print(f"   🧠 Automated economic interpretation")
    print(f"   📝 Academic LaTeX report generation")
    print(f"   📁 Organized output directory structure")


if __name__ == "__main__":
    main()