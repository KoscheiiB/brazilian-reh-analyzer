#!/usr/bin/env python3
"""
Post-COVID Brazilian REH Analysis (2023-01-01 to 2025-07-01)
Enhanced v2.0.0 Academic Framework Analysis

This script analyzes Brazilian inflation expectations rationality during 
the post-pandemic recovery period, examining whether expectations have
returned to rational behavior patterns.
"""

import os
import sys
from datetime import datetime

# Add parent directories to path for import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brazilian_reh_analyzer import BrazilianREHAnalyzer

def main():
    print("BRAZILIAN REH ANALYZER v2.0.0 - POST-COVID ANALYSIS")
    print("=" * 70)
    print("Analysis Period: 2023-01-01 to 2025-07-01")
    print("Focus: Post-pandemic inflation expectations recovery")
    print("=" * 70)
    
    # Create output directory in the outputs folder
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs')
    output_dir = os.path.join(base_dir, "post_covid_analysis_2023_2025")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        print("\nInitializing analyzer...")
        analyzer = BrazilianREHAnalyzer(
            start_date="2023-01-01",
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
        summary_file = os.path.join(output_dir, "post_covid_analysis_summary.txt")
        analyzer.export_results_summary(summary_file)
        print(f"  - Text summary: {os.path.basename(summary_file)}")
        
        # LaTeX report
        latex_file = os.path.join(output_dir, "post_covid_analysis_report.tex")
        analyzer.export_latex_report(
            latex_file,
            "Brazilian Inflation Expectations Rationality: Post-COVID Recovery Analysis (2023-2025)",
            "Brazilian REH Analyzer v2.0.0"
        )
        print(f"  - LaTeX report: {os.path.basename(latex_file)}")
        
        # Enhanced diagnostic plots
        plots_dir = os.path.join(output_dir, "diagnostic_plots")
        analyzer.export_plots(plots_dir, dpi=300)
        print(f"  - Diagnostic plots: diagnostic_plots/")
        
        # Data export
        data_file = os.path.join(output_dir, "post_covid_aligned_data.csv")
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
        print(f"\nPOST-COVID RATIONALITY ASSESSMENT: {rationality_status}")
        
        # Post-COVID specific insights
        if rationality['overall_rational']:
            print("   RECOVERY SUCCESS - Expectations returned to rationality")
        else:
            if abs(bias['mean_error']) < 1.0:
                print("   PARTIAL RECOVERY - Bias reduced but still present")
            elif abs(bias['mean_error']) < 2.0:
                print("   SLOW RECOVERY - Some improvement from COVID period")
            else:
                print("   PERSISTENT ISSUES - High bias remains post-COVID")
                
        # Compare beta coefficient improvement
        if abs(mz['beta'] - 1.0) < 0.3:
            print("   SLOPE COEFFICIENT IMPROVING - Moving toward rationality")
        elif abs(mz['beta'] - 1.0) < 0.6:
            print("   SLOPE COEFFICIENT MODERATE - Some improvement visible")
        else:
            print("   SLOPE COEFFICIENT POOR - No significant improvement")
            
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