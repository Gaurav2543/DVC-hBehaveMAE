#!/usr/bin/env python3
"""
Comprehensive script to compare embeddings extracted from original vs optimized methods.
Verifies that embeddings are identical for the same cage and timestamp combinations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description="Compare embeddings from original vs optimized extraction methods")
    parser.add_argument("--original_dir", required=True, type=str, help="Directory containing original embeddings")
    parser.add_argument("--optimized_dir", required=True, type=str, help="Directory containing optimized embeddings")
    parser.add_argument("--aggregation_name", default="3days", type=str, help="Aggregation name to compare")
    parser.add_argument("--output_dir", type=str, help="Directory to save comparison results")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Numerical tolerance for comparison")
    parser.add_argument("--detailed", action="store_true", help="Perform detailed element-wise comparison")
    return parser.parse_args()

def load_embedding_data(file_path: Path) -> Dict:
    """Load embedding data from .npy file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True).item()
    print(f"Loaded {file_path.name}:")
    print(f"  - Embeddings shape: {data['embeddings'].shape}")
    print(f"  - Frame map entries: {len(data['frame_number_map'])}")
    return data

def compare_frame_maps(original_map: Dict, optimized_map: Dict) -> Tuple[bool, List[str]]:
    """Compare frame mapping between original and optimized versions"""
    issues = []
    
    # Check if both have same keys
    orig_keys = set(original_map.keys())
    opt_keys = set(optimized_map.keys())
    
    if orig_keys != opt_keys:
        missing_in_opt = orig_keys - opt_keys
        extra_in_opt = opt_keys - orig_keys
        if missing_in_opt:
            issues.append(f"Keys missing in optimized: {missing_in_opt}")
        if extra_in_opt:
            issues.append(f"Extra keys in optimized: {extra_in_opt}")
    
    # Check frame ranges for common keys
    common_keys = orig_keys.intersection(opt_keys)
    for key in common_keys:
        orig_range = original_map[key]
        opt_range = optimized_map[key]
        if orig_range != opt_range:
            issues.append(f"Frame range mismatch for {key}: original={orig_range}, optimized={opt_range}")
    
    return len(issues) == 0, issues

def compare_embeddings_shape_and_basic_stats(original_emb: np.ndarray, optimized_emb: np.ndarray) -> Tuple[bool, List[str]]:
    """Compare basic properties of embedding matrices"""
    issues = []
    
    # Shape comparison
    if original_emb.shape != optimized_emb.shape:
        issues.append(f"Shape mismatch: original={original_emb.shape}, optimized={optimized_emb.shape}")
        return False, issues
    
    # Basic statistics comparison
    orig_mean = np.mean(original_emb)
    opt_mean = np.mean(optimized_emb)
    orig_std = np.std(original_emb)
    opt_std = np.std(optimized_emb)
    orig_min = np.min(original_emb)
    opt_min = np.min(optimized_emb)
    orig_max = np.max(original_emb)
    opt_max = np.max(optimized_emb)
    
    print(f"  Original - Mean: {orig_mean:.6f}, Std: {orig_std:.6f}, Min: {orig_min:.6f}, Max: {orig_max:.6f}")
    print(f"  Optimized - Mean: {opt_mean:.6f}, Std: {opt_std:.6f}, Min: {opt_min:.6f}, Max: {opt_max:.6f}")
    
    return True, issues

def detailed_embedding_comparison(original_emb: np.ndarray, optimized_emb: np.ndarray, 
                                tolerance: float = 1e-6) -> Dict:
    """Perform detailed element-wise comparison of embeddings"""
    
    print("Performing detailed element-wise comparison...")
    
    # Element-wise absolute difference
    abs_diff = np.abs(original_emb - optimized_emb)
    
    # Element-wise relative difference (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / (np.abs(original_emb) + 1e-10)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
    
    # Statistics
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Count elements within tolerance
    within_tolerance = np.sum(abs_diff <= tolerance)
    total_elements = original_emb.size
    within_tolerance_pct = (within_tolerance / total_elements) * 100
    
    # Correlation
    flattened_orig = original_emb.flatten()
    flattened_opt = optimized_emb.flatten()
    correlation, p_value = pearsonr(flattened_orig, flattened_opt)
    
    # Cosine similarity
    cosine_sim = 1 - cosine(flattened_orig, flattened_opt)
    
    # Check for identical matrices
    are_identical = np.allclose(original_emb, optimized_emb, atol=tolerance, rtol=tolerance)
    are_exactly_equal = np.array_equal(original_emb, optimized_emb)
    
    results = {
        'max_absolute_difference': max_abs_diff,
        'mean_absolute_difference': mean_abs_diff,
        'max_relative_difference': max_rel_diff,
        'mean_relative_difference': mean_rel_diff,
        'within_tolerance_count': within_tolerance,
        'within_tolerance_percentage': within_tolerance_pct,
        'pearson_correlation': correlation,
        'correlation_p_value': p_value,
        'cosine_similarity': cosine_sim,
        'are_identical_within_tolerance': are_identical,
        'are_exactly_equal': are_exactly_equal,
        'total_elements': total_elements
    }
    
    return results

def compare_specific_sequences(original_data: Dict, optimized_data: Dict, tolerance: float) -> Dict:
    """Compare embeddings for specific sequences/cages"""
    
    original_map = original_data['frame_number_map']
    optimized_map = optimized_data['frame_number_map']
    original_emb = original_data['embeddings']
    optimized_emb = optimized_data['embeddings']
    
    sequence_results = {}
    
    # Find common sequences
    common_sequences = set(original_map.keys()).intersection(set(optimized_map.keys()))
    
    print(f"\nComparing {len(common_sequences)} common sequences...")
    
    for seq_key in sorted(common_sequences):
        orig_start, orig_end = original_map[seq_key]
        opt_start, opt_end = optimized_map[seq_key]
        
        # Extract sequence embeddings
        orig_seq_emb = original_emb[orig_start:orig_end]
        opt_seq_emb = optimized_emb[opt_start:opt_end]
        
        if orig_seq_emb.shape != opt_seq_emb.shape:
            sequence_results[seq_key] = {
                'error': f"Shape mismatch: {orig_seq_emb.shape} vs {opt_seq_emb.shape}"
            }
            continue
        
        # Detailed comparison for this sequence
        seq_comparison = detailed_embedding_comparison(orig_seq_emb, opt_seq_emb, tolerance)
        seq_comparison['sequence_length'] = orig_seq_emb.shape[0]
        seq_comparison['embedding_dim'] = orig_seq_emb.shape[1]
        
        sequence_results[seq_key] = seq_comparison
    
    return sequence_results

def visualize_comparison_results(comparison_results: Dict, output_dir: Path = None):
    """Create visualizations of comparison results"""
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for plotting
    sequences = list(comparison_results.keys())
    correlations = [comparison_results[seq]['pearson_correlation'] for seq in sequences 
                   if 'pearson_correlation' in comparison_results[seq]]
    cosine_sims = [comparison_results[seq]['cosine_similarity'] for seq in sequences 
                  if 'cosine_similarity' in comparison_results[seq]]
    max_abs_diffs = [comparison_results[seq]['max_absolute_difference'] for seq in sequences 
                    if 'max_absolute_difference' in comparison_results[seq]]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Correlation distribution
    axes[0,0].hist(correlations, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_title('Pearson Correlation Distribution')
    axes[0,0].set_xlabel('Correlation')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(np.mean(correlations), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(correlations):.6f}')
    axes[0,0].legend()
    
    # Cosine similarity distribution
    axes[0,1].hist(cosine_sims, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].set_title('Cosine Similarity Distribution')
    axes[0,1].set_xlabel('Cosine Similarity')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].axvline(np.mean(cosine_sims), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(cosine_sims):.6f}')
    axes[0,1].legend()
    
    # Max absolute difference distribution (log scale)
    axes[1,0].hist(max_abs_diffs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Max Absolute Difference Distribution')
    axes[1,0].set_xlabel('Max Absolute Difference')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_yscale('log')
    axes[1,0].axvline(np.mean(max_abs_diffs), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(max_abs_diffs):.2e}')
    axes[1,0].legend()
    
    # Correlation vs Max Absolute Difference scatter
    axes[1,1].scatter(correlations, max_abs_diffs, alpha=0.6, color='purple')
    axes[1,1].set_xlabel('Pearson Correlation')
    axes[1,1].set_ylabel('Max Absolute Difference')
    axes[1,1].set_title('Correlation vs Max Absolute Difference')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'embedding_comparison_plots.png', dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_dir / 'embedding_comparison_plots.png'}")
    
    plt.show()

def generate_comparison_report(overall_results: Dict, sequence_results: Dict, 
                             level_name: str, output_dir: Path = None) -> str:
    """Generate a comprehensive comparison report"""
    
    report_lines = []
    report_lines.append(f"=== EMBEDDING COMPARISON REPORT - {level_name.upper()} ===")
    report_lines.append(f"Generated on: {pd.Timestamp.now()}")
    report_lines.append("")
    
    # Overall results
    report_lines.append("OVERALL COMPARISON RESULTS:")
    report_lines.append(f"  - Total elements compared: {overall_results['total_elements']:,}")
    report_lines.append(f"  - Are exactly equal: {overall_results['are_exactly_equal']}")
    report_lines.append(f"  - Are identical within tolerance: {overall_results['are_identical_within_tolerance']}")
    report_lines.append(f"  - Elements within tolerance: {overall_results['within_tolerance_count']:,} ({overall_results['within_tolerance_percentage']:.2f}%)")
    report_lines.append("")
    
    report_lines.append("STATISTICAL MEASURES:")
    report_lines.append(f"  - Max absolute difference: {overall_results['max_absolute_difference']:.2e}")
    report_lines.append(f"  - Mean absolute difference: {overall_results['mean_absolute_difference']:.2e}")
    report_lines.append(f"  - Max relative difference: {overall_results['max_relative_difference']:.2e}")
    report_lines.append(f"  - Mean relative difference: {overall_results['mean_relative_difference']:.2e}")
    report_lines.append(f"  - Pearson correlation: {overall_results['pearson_correlation']:.8f}")
    report_lines.append(f"  - Cosine similarity: {overall_results['cosine_similarity']:.8f}")
    report_lines.append("")
    
    # Sequence-level results summary
    if sequence_results:
        correlations = [seq_res['pearson_correlation'] for seq_res in sequence_results.values() 
                       if 'pearson_correlation' in seq_res]
        max_abs_diffs = [seq_res['max_absolute_difference'] for seq_res in sequence_results.values() 
                        if 'max_absolute_difference' in seq_res]
        
        report_lines.append("SEQUENCE-LEVEL SUMMARY:")
        report_lines.append(f"  - Number of sequences compared: {len(sequence_results)}")
        report_lines.append(f"  - Mean correlation across sequences: {np.mean(correlations):.8f}")
        report_lines.append(f"  - Min correlation across sequences: {np.min(correlations):.8f}")
        report_lines.append(f"  - Mean max absolute difference: {np.mean(max_abs_diffs):.2e}")
        report_lines.append(f"  - Max absolute difference overall: {np.max(max_abs_diffs):.2e}")
        report_lines.append("")
        
        # Individual sequence results
        report_lines.append("INDIVIDUAL SEQUENCE RESULTS:")
        for seq_key, seq_res in sorted(sequence_results.items()):
            if 'error' in seq_res:
                report_lines.append(f"  - {seq_key}: ERROR - {seq_res['error']}")
            else:
                report_lines.append(f"  - {seq_key}:")
                report_lines.append(f"    * Correlation: {seq_res['pearson_correlation']:.8f}")
                report_lines.append(f"    * Max abs diff: {seq_res['max_absolute_difference']:.2e}")
                report_lines.append(f"    * Identical within tolerance: {seq_res['are_identical_within_tolerance']}")
    
    report_text = "\n".join(report_lines)
    
    # Save report if output directory is specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"comparison_report_{level_name}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"Saved report to {report_file}")
    
    return report_text

def main():
    args = get_args()
    
    original_dir = Path(args.original_dir)
    optimized_dir = Path(args.optimized_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Define levels to compare
    levels = ['level1', 'level2', 'level3', 'level4', 'level5', 'comb']
    
    print(f"Comparing embeddings between:")
    print(f"  Original: {original_dir}")
    print(f"  Optimized: {optimized_dir}")
    print(f"  Aggregation: {args.aggregation_name}")
    print(f"  Tolerance: {args.tolerance}")
    print("=" * 70)
    
    all_results = {}
    
    for level in levels:
        print(f"\n{'='*20} COMPARING {level.upper()} {'='*20}")
        
        # Load embedding files
        orig_file = original_dir / f"test_{args.aggregation_name}_{level}.npy"
        opt_file = optimized_dir / f"test_{args.aggregation_name}_{level}.npy"
        
        try:
            original_data = load_embedding_data(orig_file)
            optimized_data = load_embedding_data(opt_file)
        except FileNotFoundError as e:
            print(f"Skipping {level}: {e}")
            continue
        
        # Compare frame maps
        print(f"\nComparing frame maps for {level}...")
        maps_match, map_issues = compare_frame_maps(
            original_data['frame_number_map'], 
            optimized_data['frame_number_map']
        )
        
        if not maps_match:
            print("Frame map issues found:")
            for issue in map_issues:
                print(f"  - {issue}")
        else:
            print("✓ Frame maps are identical")
        
        # Compare embedding shapes and basic stats
        print(f"\nComparing embedding matrices for {level}...")
        shapes_match, shape_issues = compare_embeddings_shape_and_basic_stats(
            original_data['embeddings'], 
            optimized_data['embeddings']
        )
        
        if not shapes_match:
            print("Shape/basic stats issues found:")
            for issue in shape_issues:
                print(f"  - {issue}")
            continue
        
        # Detailed embedding comparison
        overall_results = detailed_embedding_comparison(
            original_data['embeddings'], 
            optimized_data['embeddings'], 
            args.tolerance
        )
        
        print(f"\nOverall comparison results for {level}:")
        print(f"  - Exactly equal: {overall_results['are_exactly_equal']}")
        print(f"  - Identical within tolerance: {overall_results['are_identical_within_tolerance']}")
        print(f"  - Max absolute difference: {overall_results['max_absolute_difference']:.2e}")
        print(f"  - Pearson correlation: {overall_results['pearson_correlation']:.8f}")
        print(f"  - Cosine similarity: {overall_results['cosine_similarity']:.8f}")
        
        # Sequence-level comparison if detailed flag is set
        sequence_results = {}
        if args.detailed:
            print(f"\nPerforming detailed sequence-level comparison for {level}...")
            sequence_results = compare_specific_sequences(
                original_data, optimized_data, args.tolerance
            )
        
        # Store results
        all_results[level] = {
            'overall': overall_results,
            'sequences': sequence_results,
            'maps_match': maps_match,
            'map_issues': map_issues
        }
        
        # Generate and save report
        if output_dir:
            report = generate_comparison_report(
                overall_results, sequence_results, level, output_dir
            )
        
        # Create visualizations for this level
        if args.detailed and sequence_results and output_dir:
            level_output_dir = output_dir / level
            visualize_comparison_results(sequence_results, level_output_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY:")
    print(f"{'='*70}")
    
    for level, results in all_results.items():
        overall = results['overall']
        print(f"{level:>8}: Exactly equal: {overall['are_exactly_equal']:>5}, "
              f"Within tolerance: {overall['are_identical_within_tolerance']:>5}, "
              f"Correlation: {overall['pearson_correlation']:.6f}")
    
    # Check if all levels are identical
    all_exactly_equal = all(results['overall']['are_exactly_equal'] for results in all_results.values())
    all_within_tolerance = all(results['overall']['are_identical_within_tolerance'] for results in all_results.values())
    
    print(f"\nAll levels exactly equal: {all_exactly_equal}")
    print(f"All levels within tolerance: {all_within_tolerance}")
    
    if all_exactly_equal:
        print("\n🎉 SUCCESS: All embeddings are exactly identical!")
    elif all_within_tolerance:
        print(f"\n✅ SUCCESS: All embeddings are identical within tolerance ({args.tolerance})")
    else:
        print(f"\n⚠️  WARNING: Some embeddings differ beyond tolerance ({args.tolerance})")
        print("Check the detailed reports for more information.")

if __name__ == "__main__":
    main()