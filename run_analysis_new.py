#!/usr/bin/env python3
"""
Command-line interface for comprehensive embedding analysis.
Run this script to analyze your behavioral embeddings with all available methods.
ADAPTED for new Zarr-based embedding format.
"""

import argparse
import sys
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Behavioral Embedding Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis without bodyweight
  python run_analysis.py /path/to/embeddings metadata.csv --output-dir results/

  # Full analysis with bodyweight data
  python run_analysis.py /path/to/embeddings metadata.csv \\
    --bodyweight-file BW.csv \\
    --tissue-file TissueCollection_PP.csv \\
    --summary-file summary_metadata.csv \\
    --output-dir results_with_bodyweight/

  # Specify aggregation and embedding level
  python run_analysis.py /path/to/embeddings metadata.csv \\
    --aggregation 1day \\
    --embedding-level combined \\
    --output-dir results/

  # Quick analysis (fewer samples for heatmaps, fewer workers)
  python run_analysis.py /path/to/embeddings metadata.csv \\
    --quick-mode \\
    --max-workers 4
        """
    )
    
    # Required arguments
    parser.add_argument("embeddings_path", 
                       help="Path to embeddings directory (contains aggregation subdirs)")
    parser.add_argument("metadata_path", 
                       help="Path to metadata CSV file")
    
    # Output options
    parser.add_argument("--output-dir", default="embedding_analysis_results",
                       help="Output directory for all results (default: embedding_analysis_results)")
    
    # NEW: Embedding format options
    parser.add_argument("--aggregation", default="1day",
                       help="Aggregation level to analyze (e.g., '1day', '1.5h', '1week')")
    parser.add_argument("--embedding-level", default="combined",
                       help="Embedding level to use ('level_1_pooled', 'combined', etc.)")
    
    # Bodyweight integration options
    parser.add_argument("--bodyweight-file", 
                       help="Path to BW.csv file (optional)")
    parser.add_argument("--tissue-file", 
                       help="Path to TissueCollection_PP.csv file (required if using bodyweight)")
    parser.add_argument("--summary-file", 
                       help="Path to summary_metadata.csv file (required if using bodyweight)")
    parser.add_argument("--bodyweight-strategy", 
                       choices=["closest", "interpolate", "most_recent", "gam_spline"],
                       default="gam_spline",
                       help="Strategy for matching bodyweight to timepoints (default: gam_spline)")
    
    # Analysis options
    parser.add_argument("--analysis-type", 
                       choices=["basic", "enhanced", "both"],
                       default="both",
                       help="Type of analysis to run (default: both)")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count()-4 if os.cpu_count() > 8 else 4,
                       help="Maximum number of worker threads (default: CPU count - 4)")
    parser.add_argument("--n-heatmap-samples", type=int, default=1000,
                       help="Number of samples for activity heatmaps (default: 1000)")
    
    # Performance options
    parser.add_argument("--quick-mode", action="store_true",
                       help="Quick mode: fewer samples for visualization, faster processing")
    parser.add_argument("--skip-interactive", action="store_true",
                       help="Skip interactive plot generation (faster)")
    parser.add_argument("--skip-temporal", action="store_true",
                       help="Skip temporal pattern analysis (faster)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.bodyweight_file:
        if not args.tissue_file or not args.summary_file:
            parser.error("--tissue-file and --summary-file are required when using --bodyweight-file")
    
    # Check file existence
    if not Path(args.embeddings_path).exists():
        print(f"Error: Embeddings directory not found: {args.embeddings_path}")
        sys.exit(1)
    
    # Check if aggregation subdirectory exists
    agg_dir = Path(args.embeddings_path) / args.aggregation
    if not agg_dir.exists():
        print(f"Error: Aggregation directory not found: {agg_dir}")
        print(f"Available aggregations in {args.embeddings_path}:")
        for item in Path(args.embeddings_path).iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        sys.exit(1)
    
    if not Path(args.metadata_path).exists():
        print(f"Error: Metadata file not found: {args.metadata_path}")
        sys.exit(1)
    
    # Adjust settings for quick mode
    if args.quick_mode:
        args.max_workers = min(args.max_workers, 4)
        args.n_heatmap_samples = min(args.n_heatmap_samples, 50)
        print("Running in quick mode: reduced samples and workers for faster processing")
    
    # Prepare bodyweight files
    bodyweight_files = None
    if args.bodyweight_file:
        bodyweight_files = {
            'bodyweight': args.bodyweight_file,
            'tissue': args.tissue_file,
            'summary': args.summary_file
        }
        print("Bodyweight integration enabled")
    
    print("=" * 80)
    print("COMPREHENSIVE BEHAVIORAL EMBEDDING ANALYSIS")
    print("=" * 80)
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Embedding level: {args.embedding_level}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis type: {args.analysis_type}")
    print(f"Workers: {args.max_workers}")
    if bodyweight_files:
        print(f"Bodyweight strategy: {args.bodyweight_strategy}")
    print("=" * 80)
    
    # Import analysis modules
    try:
        from downstream_tasks.embedding_analysis import EmbeddingAnalyzer
        print("✓ Loaded basic analysis module")
    except ImportError as e:
        print(f"Error importing basic analysis module: {e}")
        sys.exit(1)
    
    if args.analysis_type in ["enhanced", "both"] or bodyweight_files:
        try:
            from downstream_tasks.BW_integration_and_analysis import EnhancedEmbeddingAnalyzer
            print("✓ Loaded enhanced analysis module")
        except ImportError as e:
            print(f"Error importing enhanced analysis module: {e}")
            if args.analysis_type == "enhanced":
                sys.exit(1)
            else:
                print("Falling back to basic analysis only")
                args.analysis_type = "basic"
    
    # Run analyses
    try:
        if args.analysis_type in ["basic", "both"]:
            print("\n" + "="*50)
            print("RUNNING BASIC ANALYSIS")
            print("="*50)
            
            basic_analyzer = EmbeddingAnalyzer(
                embeddings_path=args.embeddings_path,
                metadata_path=args.metadata_path,
                output_dir=f"{args.output_dir}/basic_analysis",
                max_workers=args.max_workers,
                aggregation=args.aggregation,
                embedding_level=args.embedding_level
            )
            
            # Run basic analysis components
            if not args.skip_temporal:
                basic_analyzer.create_activity_heatmaps(n_samples=args.n_heatmap_samples)
            
            basic_analyzer.age_regression_analysis(include_bodyweight=False)
            if bodyweight_files:
                basic_analyzer.age_regression_analysis(include_bodyweight=True)
            
            basic_analyzer.strain_classification_analysis()
            basic_analyzer.perform_pca_analysis()
            basic_analyzer.perform_clustering_analysis()
            basic_analyzer.detect_anomalous_samples()
            
            print("✓ Basic analysis completed")
        
        if args.analysis_type in ["enhanced", "both"]:
            print("\n" + "="*50)
            print("RUNNING ENHANCED ANALYSIS")
            print("="*50)
            
            enhanced_analyzer = EnhancedEmbeddingAnalyzer(
                embeddings_path=args.embeddings_path,
                metadata_path=args.metadata_path,
                output_dir=f"{args.output_dir}/enhanced_analysis",
                bodyweight_files=bodyweight_files,
                aggregation=args.aggregation,
                embedding_level=args.embedding_level
            )
            
            # Run enhanced analysis components
            if not args.skip_temporal:
                enhanced_analyzer.perform_comprehensive_temporal_analysis()
            
            enhanced_analyzer.create_comprehensive_correlation_analysis()
            enhanced_analyzer.perform_enhanced_feature_importance_analysis()
            
            if not args.skip_interactive:
                enhanced_analyzer.create_interactive_exploration_dashboard()
            
            print("✓ Enhanced analysis completed")
    
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nOutput structure:")
    
    if args.analysis_type in ["basic", "both"]:
        print(f"  {args.output_dir}/basic_analysis/")
        print("    ├── heatmaps/           # Activity heatmaps for each dimension")
        print("    ├── regression/         # Age regression results and plots")
        print("    ├── classification/     # Strain classification results")
        print("    ├── feature_selection/  # Feature selection analysis")
        print("    ├── pca_analysis/       # PCA results and plots")
        print("    ├── clustering/         # Clustering analysis")
        print("    ├── sliding_window/     # Sliding window analysis")
        print("    └── anomaly_detection/  # Anomalous samples detection")
    
    if args.analysis_type in ["enhanced", "both"]:
        print(f"  {args.output_dir}/enhanced_analysis/")
        print("    ├── temporal_analysis/    # Comprehensive temporal patterns")
        print("    ├── correlation_analysis/ # Enhanced correlation analysis")
        print("    ├── feature_importance/   # Multi-method feature importance")
        print("    └── interactive_plots/    # Interactive exploration plots")
    
    print(f"\nKey files to check:")
    print(f"  - *_results.json files contain numerical results")
    print(f"  - *.pdf files contain high-quality plots for publications")
    print(f"  - *.html files contain interactive visualizations")
    print(f"  - anomalous_samples.txt contains detected outliers")
    
    if bodyweight_files:
        print(f"\nBodyweight covariate analysis:")
        print(f"  - Both with/without bodyweight regression performed")
        print(f"  - Bodyweight correlations analyzed")
        print(f"  - Strategy used: {args.bodyweight_strategy}")
    
    print(f"\nTo explore results:")
    print(f"  1. Open interactive HTML files in a web browser")
    print(f"  2. Review PDF plots for publication-quality figures")
    print(f"  3. Check JSON files for numerical results and statistics")
    print(f"  4. Examine anomalous_samples.txt for potential experimental issues")
    print(f"\nFor questions about specific results, refer to the analysis code or documentation.")


if __name__ == "__main__":
    main()