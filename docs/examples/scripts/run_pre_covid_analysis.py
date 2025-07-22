#!/usr/bin/env python3
"""
Pre-COVID Brazilian REH Analysis (2017-01-01 to 2020-02-29)
Enhanced v2.0.0 Academic Framework Analysis

This script analyzes Brazilian inflation expectations rationality during 
the pre-pandemic period, providing comprehensive statistical testing and
publication-quality outputs.
"""

import os
import sys
from datetime import datetime

# Add parent directories to path for import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brazilian_reh_analyzer import BrazilianREHAnalyzer

def main():
    print("BRAZILIAN REH ANALYZER v2.0.0 - PRE-COVID ANALYSIS")
    print("=" * 70)
    print("Analysis Period: 2017-01-01 to 2020-02-29")
    print("Focus: Pre-pandemic inflation expectations rationality")
    print("=" * 70)
    
    # Create output directory in the outputs folder
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs')
    output_dir = os.path.join(base_dir, "pre_covid_analysis_2017_2020")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        print("\nInitializing analyzer...")
        analyzer = BrazilianREHAnalyzer(
            start_date="2017-01-01",
            end_date="2020-02-29",
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
        summary_file = os.path.join(output_dir, "pre_covid_analysis_summary.txt")
        analyzer.export_results_summary(summary_file)
        print(f"  - Text summary: {os.path.basename(summary_file)}")
        
        # LaTeX report
        latex_file = os.path.join(output_dir, "pre_covid_analysis_report.tex")
        analyzer.export_latex_report(
            latex_file,
            "Brazilian Inflation Expectations Rationality: Pre-COVID Analysis (2017-2020)",
            "Brazilian REH Analyzer v2.0.0"
        )
        print(f"  - LaTeX report: {os.path.basename(latex_file)}")
        
        # Enhanced diagnostic plots
        plots_dir = os.path.join(output_dir, "diagnostic_plots")
        analyzer.export_plots(plots_dir, dpi=300)
        print(f"  - Diagnostic plots: diagnostic_plots/")
        
        # Data export
        data_file = os.path.join(output_dir, "pre_covid_aligned_data.csv")
        analyzer.save_data(data_file)
        print(f"  - Aligned data: {os.path.basename(data_file)}")
        
        # Print key results
        print("\n" + "=" * 70)
        print("KEY RESULTS SUMMARY")
        print("=" * 70)
        
        rationality = results["rationality_assessment"]
        mz = results["mincer_zarnowitz"]
        bias = results["bias_test"]
        
        print(f"Overall Rational: {rationality['overall_rational']}")
        print(f"Mincer-Zarnowitz p-value: {mz['joint_test_pvalue']:.4f}")
        print(f"Alpha (intercept): {mz['alpha']:.4f}")
        print(f"Beta (slope): {mz['beta']:.4f}")
        print(f"Mean forecast error: {bias['mean_error']:.4f}")
        print(f"Observations: {rationality['n_observations']}")
        
        rationality_status = "RATIONAL" if rationality['overall_rational'] else "NOT RATIONAL"
        print(f"\nPRE-COVID RATIONALITY ASSESSMENT: {rationality_status}")
        
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