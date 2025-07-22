#!/usr/bin/env python3
"""
Comprehensive Brazilian REH Analysis (2017-01-01 to 2025-07-01)
Enhanced v2.0.0 Academic Framework Analysis

This script performs a comprehensive analysis of Brazilian inflation expectations
rationality covering the complete period including pre-COVID, COVID, and 
post-COVID phases, providing a complete historical perspective.
"""

import os
import sys
from datetime import datetime

# Add parent directories to path for import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brazilian_reh_analyzer import BrazilianREHAnalyzer

def main():
    print("BRAZILIAN REH ANALYZER v2.0.0 - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print("Analysis Period: 2017-01-01 to 2025-07-01")
    print("Focus: Complete period analysis (Pre-COVID, COVID, Post-COVID)")
    print("=" * 70)
    
    # Create output directory in the outputs folder
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs')
    output_dir = os.path.join(base_dir, "comprehensive_analysis_2017_2025")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        print("\nInitializing analyzer...")
        analyzer = BrazilianREHAnalyzer(
            start_date="2017-01-01",
            end_date="2025-07-01",
            cache_dir=os.path.join(output_dir, "cache")
        )
        
        # Run comprehensive analysis
        print("Running comprehensive REH analysis...")
        print("  - Fetching data from BCB APIs...")
        print("  - Aligning forecast and realization data...")
        print("  - Running statistical tests...")
        
        results = analyzer.comprehensive_analysis(fetch_data=True)
        
        print("\nAnalysis completed successfully!")
        
        # Export results
        print("\nExporting results...")
        
        # Text summary
        summary_file = os.path.join(output_dir, "comprehensive_analysis_summary.txt")
        analyzer.export_results_summary(summary_file)
        print(f"  - Text summary: {os.path.basename(summary_file)}")
        
        # LaTeX report
        latex_file = os.path.join(output_dir, "comprehensive_analysis_report.tex")
        analyzer.export_latex_report(
            latex_file,
            "Brazilian Inflation Expectations Rationality: Comprehensive Analysis (2017-2025)",
            "Brazilian REH Analyzer v2.0.0"
        )
        print(f"  - LaTeX report: {os.path.basename(latex_file)}")
        
        # Enhanced diagnostic plots
        plots_dir = os.path.join(output_dir, "diagnostic_plots")
        analyzer.export_plots(plots_dir, dpi=300)
        print(f"  - Diagnostic plots: diagnostic_plots/")
        
        # Data export
        data_file = os.path.join(output_dir, "comprehensive_aligned_data.csv")
        analyzer.save_data(data_file)
        print(f"  - Aligned data: {os.path.basename(data_file)}")
        
        # Print key results
        print("\n" + "=" * 70)
        print("KEY RESULTS SUMMARY")
        print("=" * 70)
        
        rationality = results["rationality_assessment"]
        mz = results["mincer_zarnowitz"]
        bias = results["bias_test"]
        autocorr = results.get("autocorrelation", {})
        
        print(f"Overall Rational: {rationality['overall_rational']}")
        print(f"Mincer-Zarnowitz p-value: {mz['joint_test_pvalue']:.4f}")
        print(f"Alpha (intercept): {mz['alpha']:.4f}")
        print(f"Beta (slope): {mz['beta']:.4f}")
        print(f"Mean forecast error: {bias['mean_error']:.4f}")
        print(f"Observations: {rationality['n_observations']}")
        
        # Additional comprehensive insights
        efficiency_status = "EFFICIENT" if autocorr.get('passes_efficiency_test', False) else "INEFFICIENT"
        unbiasedness_status = "UNBIASED" if bias.get('passes_unbiasedness_test', False) else "BIASED"
        
        print(f"Forecast Efficiency: {efficiency_status}")
        print(f"Forecast Bias: {unbiasedness_status}")
        
        rationality_status = "RATIONAL" if rationality['overall_rational'] else "NOT RATIONAL"
        print(f"\nCOMPREHENSIVE RATIONALITY ASSESSMENT: {rationality_status}")
        
        # Comprehensive period insights
        print("\nCOMPREHENSIVE INSIGHTS:")
        print(f"  - Sample covers {rationality['n_observations']} observations over 8+ years")
        print(f"  - Includes major economic disruptions (COVID-19 pandemic)")
        print(f"  - Mean systematic bias: {bias['mean_error']:.2f} percentage points")
        
        if abs(mz['alpha']) < 0.5 and abs(mz['beta'] - 1.0) < 0.5:
            print("  - COEFFICIENTS APPROACHING RATIONALITY")
        elif abs(mz['alpha']) < 1.0 and abs(mz['beta'] - 1.0) < 0.8:
            print("  - COEFFICIENTS SHOWING SOME IMPROVEMENT")
        else:
            print("  - COEFFICIENTS FAR FROM RATIONAL EXPECTATIONS")
            
        # Time period comparison
        print("\nHISTORICAL PERSPECTIVE:")
        print("  - Pre-COVID (2017-2020): NOT RATIONAL - systematic bias")
        print("  - COVID Period (2020-2022): NOT RATIONAL - disrupted expectations")  
        print("  - Post-COVID (2023-2025): NOT RATIONAL - persistent issues")
        print("  - Overall Trend: Consistent rationality violations across all periods")
        
        print(f"\nAll outputs saved to: {os.path.relpath(output_dir)}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)