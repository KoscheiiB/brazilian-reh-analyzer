"""
Advanced LaTeX Export Example for Brazilian REH Analyzer v2.0.0

This example demonstrates the professional LaTeX report generation capabilities
introduced in v2.0.0, suitable for academic publication and policy analysis.
"""

from brazilian_reh_analyzer import BrazilianREHAnalyzer
import logging
import os
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Demonstrate advanced LaTeX export capabilities"""
    
    print("=" * 75)
    print("BRAZILIAN REH ANALYZER v2.0.0 - ADVANCED LATEX EXPORT EXAMPLE")
    print("=" * 75)
    
    # Create output directory structure
    output_dir = "latex_publication_example"
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Analyze multiple periods for comprehensive academic study
    analysis_periods = {
        'Pre_COVID': ('2018-01-01', '2020-02-29'),
        'COVID_Era': ('2020-03-01', '2022-12-31'),
        'Full_Period': ('2018-01-01', '2022-12-31')
    }
    
    print("\n1. Running comprehensive multi-period analysis...")
    
    for period_name, (start_date, end_date) in analysis_periods.items():
        print(f"\n   🔄 Analyzing {period_name}: {start_date} to {end_date}")
        
        try:
            # Initialize analyzer for period
            analyzer = BrazilianREHAnalyzer(start_date, end_date)
            results = analyzer.comprehensive_analysis()
            
            # Export high-quality LaTeX report
            latex_title = f"Brazilian REH Analysis - {period_name.replace('_', ' ')} ({start_date[:4]}-{end_date[:4]})"
            latex_author = "Enhanced REH Framework v2.0.0\\\\Academic Research Team"
            
            latex_file = analyzer.export_latex_report(
                f"{output_dir}/reports/{period_name}_academic_report.tex",
                latex_title,
                latex_author
            )
            
            # Export corresponding plots
            analyzer.export_plots(f"{output_dir}/plots/{period_name}/", dpi=300)
            
            # Display key findings
            desc_stats = results['descriptive_stats']
            econ_interp = results.get('economic_interpretation', {})
            bias_analysis = econ_interp.get('bias_analysis', {}) if econ_interp else {}
            
            print(f"   ✅ {period_name}: {desc_stats.get('n_observations', 0)} obs, "
                  f"Error: {desc_stats.get('error_mean', 0):.3f} p.p., "
                  f"Severity: {bias_analysis.get('severity', 'unknown')}")
            
        except Exception as e:
            print(f"   ⚠️  {period_name} analysis failed: {e}")
    
    print("\n2. Generated LaTeX reports with academic features:")
    for filename in os.listdir(f"{output_dir}/reports"):
        if filename.endswith('.tex'):
            filepath = f"{output_dir}/reports/{filename}"
            file_size = os.path.getsize(filepath)
            print(f"   📄 {filename}: {file_size:,} bytes")
    
    print("\n3. LaTeX compilation example (requires LaTeX installation):")
    print(f"   # Navigate to reports directory")
    print(f"   cd {output_dir}/reports/")
    print(f"   ")
    print(f"   # Compile any report to PDF")
    print(f"   pdflatex Full_Period_academic_report.tex")
    print(f"   pdflatex Full_Period_academic_report.tex  # Run twice for references")
    
    print("\n4. Key LaTeX features included:")
    sample_tex_file = f"{output_dir}/reports"
    tex_files = [f for f in os.listdir(sample_tex_file) if f.endswith('.tex')]
    
    if tex_files:
        with open(f"{sample_tex_file}/{tex_files[0]}", 'r') as f:
            content = f.read()
        
        latex_features = [
            ('Professional document class', 'documentclass[11pt,a4paper]{article}' in content),
            ('Academic color definitions', 'definecolor{academicred}' in content),
            ('Professional tables', 'booktabs' in content and 'toprule' in content),
            ('Mathematical equations', 'equation' in content),
            ('Statistical symbols', 'alpha' in content and 'beta' in content),
            ('Structured sections', 'section{' in content),
            ('Professional bibliography', 'cite{' in content or 'bibliography' in content)
        ]
        
        for feature_name, is_present in latex_features:
            status = "✅" if is_present else "❌"
            print(f"   {status} {feature_name}")
    
    print("\n5. Academic usage recommendations:")
    print("   📚 Journal Submission: Use Full_Period report as main analysis")
    print("   🏛️  Policy Brief: Use individual period reports for targeted analysis") 
    print("   📊 Conference Presentation: Extract tables and figures from LaTeX")
    print("   🎓 Thesis Chapter: Combine multiple period analyses")
    
    print(f"\n🎉 Advanced LaTeX export demonstration completed!")
    print(f"📁 All files saved to: {output_dir}/")
    print(f"📝 LaTeX reports ready for academic publication")
    print(f"📈 High-resolution plots ready for journal submission")
    
    print(f"\n💡 Pro tip: Install LaTeX (TeX Live/MiKTeX) to compile reports to PDF")
    print(f"🔗 LaTeX installation: https://www.latex-project.org/get/")

if __name__ == "__main__":
    main()