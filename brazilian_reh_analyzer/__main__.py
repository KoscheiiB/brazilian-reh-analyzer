"""
Command-line interface for Brazilian REH Analyzer
"""

import argparse
import logging
import sys
from datetime import datetime
from .analyzer import BrazilianREHAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main command-line interface for the Brazilian REH Analyzer
    """
    parser = argparse.ArgumentParser(
        description="Brazilian REH Analyzer - Econometric analysis of inflation forecast rationality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default date range
  python -m brazilian_reh_analyzer
  
  # Analysis for specific period
  python -m brazilian_reh_analyzer --start-date 2017-01-01 --end-date 2023-12-31
  
  # Force refresh data (ignore cache)
  python -m brazilian_reh_analyzer --force-refresh
  
  # Save plots to specific directory
  python -m brazilian_reh_analyzer --export-plots --output-dir results/
        """
    )
    
    # Date range arguments
    parser.add_argument(
        "--start-date", 
        type=str, 
        default="2017-01-01",
        help="Analysis start date (YYYY-MM-DD). Default: 2017-01-01"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        default="2024-12-31",
        help="Analysis end date (YYYY-MM-DD). Default: 2024-12-31"
    )
    
    # Data fetching options
    parser.add_argument(
        "--force-refresh", 
        action="store_true",
        help="Force refresh data from APIs (ignore cache)"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="data_cache",
        help="Directory for data caching. Default: data_cache"
    )
    
    # Output options
    parser.add_argument(
        "--export-summary", 
        action="store_true",
        help="Export text summary of results"
    )
    parser.add_argument(
        "--summary-file", 
        type=str, 
        default="reh_analysis_results.txt",
        help="Filename for exported summary. Default: reh_analysis_results.txt"
    )
    parser.add_argument(
        "--export-plots", 
        action="store_true",
        help="Export diagnostic plots to files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="plots",
        help="Directory for exported plots. Default: plots"
    )
    parser.add_argument(
        "--save-data", 
        action="store_true",
        help="Save aligned data to CSV file"
    )
    parser.add_argument(
        "--data-file", 
        type=str, 
        default="aligned_data.csv",
        help="Filename for saved data. Default: aligned_data.csv"
    )
    
    # Display options
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Skip displaying interactive plots"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Minimize console output"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Initialize analyzer
        logger.info("=" * 70)
        logger.info("BRAZILIAN REH ANALYZER - COMMAND LINE INTERFACE")
        logger.info("=" * 70)
        
        analyzer = BrazilianREHAnalyzer(
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir
        )
        
        # Run comprehensive analysis
        logger.info(f"Analyzing period: {args.start_date} to {args.end_date}")
        if args.force_refresh:
            logger.info("Force refresh enabled - fetching fresh data...")
        
        results = analyzer.comprehensive_analysis(
            fetch_data=True,
            force_refresh=args.force_refresh
        )
        
        # Display key results
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)
        
        desc_stats = results.get("descriptive_stats", {})
        print(f"Analysis Period: {desc_stats.get('date_range', 'N/A')}")
        print(f"Number of Observations: {desc_stats.get('n_observations', 'N/A')}")
        print(f"Mean Forecast Error: {desc_stats.get('error_mean', 0):.3f} p.p.")
        print(f"Error Standard Deviation: {desc_stats.get('error_std', 0):.3f} p.p.")
        print(f"Average Respondents: {desc_stats.get('mean_respondents', 0):.1f}")
        
        if "rationality_assessment" in results:
            ra = results["rationality_assessment"]
            print(f"\nRationality Tests:")
            print(f"Unbiased: {'✓ PASS' if ra.get('unbiased', False) else '✗ FAIL'}")
            print(f"MZ Test Passed: {'✓ PASS' if ra.get('mz_rational', False) else '✗ FAIL'}")
            print(f"Efficient: {'✓ PASS' if ra.get('efficient', False) else '✗ FAIL'}")
            print(f"Overall Rational: {'✓ PASS' if ra.get('overall_rational', False) else '✗ FAIL'}")
        
        # Display specific test results if not quiet
        if not args.quiet:
            if "mincer_zarnowitz" in results and "error" not in results["mincer_zarnowitz"]:
                mz = results["mincer_zarnowitz"]
                print(f"\nMincer-Zarnowitz Test:")
                print(f"  α (intercept): {mz['alpha']:.4f} (p-value: {mz['alpha_pvalue']:.4f})")
                print(f"  β (slope): {mz['beta']:.4f} (p-value: {mz['beta_pvalue']:.4f})")
                print(f"  Joint test p-value: {mz['joint_test_pvalue']:.4f}")
                print(f"  R-squared: {mz['r_squared']:.4f}")
            
            if "bias_test" in results and "error" not in results["bias_test"]:
                bias = results["bias_test"]
                print(f"\nBias Test:")
                print(f"  Mean error: {bias['mean_error']:.4f} p.p.")
                print(f"  t-statistic: {bias['t_statistic']:.4f}")
                print(f"  p-value: {bias['p_value']:.4f}")
                print(f"  Bias direction: {bias['bias_direction']}")
        
        # Export summary if requested
        if args.export_summary:
            analyzer.export_results_summary(args.summary_file)
            logger.info(f"Results summary exported to {args.summary_file}")
        
        # Save data if requested
        if args.save_data:
            analyzer.save_data(args.data_file)
            logger.info(f"Aligned data saved to {args.data_file}")
        
        # Generate and export plots
        if not args.no_plots:
            logger.info("Generating diagnostic plots...")
            analyzer.plot_enhanced_diagnostics(show_plots=True)
        
        if args.export_plots:
            analyzer.export_plots(output_dir=args.output_dir)
            logger.info(f"Plots exported to {args.output_dir}/")
        
        print(f"\nAnalysis completed successfully!")
        if not args.force_refresh:
            print(f"Data cached in {args.cache_dir}/ directory for future runs")
        
    except ImportError as e:
        logger.error("python-bcb library required for real data access.")
        logger.error("Install with: pip install python-bcb")
        logger.error("Then run this script again.")
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error("Check your internet connection and try again.")
        logger.error("If the error persists, try running with --force-refresh")
        sys.exit(1)


if __name__ == "__main__":
    main()