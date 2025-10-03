#!/usr/bin/env python3
"""
Diagnostic script to compare data loading between original and optimized methods
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import zarr

# Try to import the original data loading function
try:
    from data_pipeline.load_dvc import load_dvc_data
    HAS_ORIGINAL_LOADER = True
except ImportError:
    HAS_ORIGINAL_LOADER = False
    print("WARNING: Cannot import load_dvc_data - will skip original method comparison")

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvc_root", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--output_dir", default="./data_comparison")
    return parser.parse_args()

def build_test_split_original_style(root, csv):
    """Replicate original's build_test_split exactly"""
    summary_df = pd.read_csv(csv)
    summary_df = summary_df.head(10)  # Match the limit
    
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])
    summary_df.sort_values(by=['cage_id', 'from_tpt'], inplace=True)
    
    time_delta_minutes = (summary_df['to_tpt'] - summary_df['from_tpt']).dt.total_seconds().iloc[0] / 60
    expected_chunk_size = int(round(time_delta_minutes)) + 1
    
    mapping, order = {}, []
    valid_chunks = 0
    
    for _, r in summary_df.iterrows():
        day = str(r["from_tpt"]).split()[0]
        cage_csv_path = Path(root) / f"{r['cage_id']}.csv"
        if not cage_csv_path.exists():
            continue
            
        cage_df = pd.read_csv(cage_csv_path, usecols=['timestamp'] + ELEC_COLS)
        cage_df['timestamp'] = pd.to_datetime(cage_df['timestamp'])
        
        chunk = cage_df[(cage_df['timestamp'] >= r['from_tpt']) & (cage_df['timestamp'] <= r['to_tpt'])]
        
        if len(chunk) != expected_chunk_size or chunk[ELEC_COLS].isnull().values.any():
            continue
            
        mapping.setdefault(r["cage_id"], []).append(day)
        order.append(f"{r['cage_id']}_{day}")
        valid_chunks += 1
        
    return mapping, order

def load_sequences_original_style(root, mapping, order):
    """Replicate original's load_sequences exactly"""
    if not HAS_ORIGINAL_LOADER:
        print("Cannot load using original method - load_dvc_data not available")
        return []
        
    raw = load_dvc_data(root, mapping)
    seqs = []
    for cage, df in raw.items():
        df["__day"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        for day, sub in df.groupby("__day"):
            key = f"{cage}_{day}"
            if key not in order:
                continue
            missing = [c for c in ELEC_COLS if c not in sub.columns]
            if missing:
                print(f"[WARN] {key} missing {missing}")
                continue
            seqs.append((key, sub[ELEC_COLS].values.astype(np.float32)))
    
    seqs.sort(key=lambda x: order.index(x[0]))
    return seqs

def load_sequences_optimized_style(zarr_path):
    """Load sequences from Zarr store"""
    zarr_store = zarr.open(zarr_path, mode='r')
    sequence_keys = [k for k in zarr_store.keys()]
    
    sequences = []
    for seq_key in sequence_keys:
        seq_data = zarr_store[seq_key][:]
        sequences.append((seq_key, seq_data))
    
    return sequences

def compare_sequences(original_seqs, optimized_seqs, output_dir):
    """Compare sequences loaded by both methods"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Original method loaded: {len(original_seqs)} sequences")
    print(f"Optimized method loaded: {len(optimized_seqs)} sequences")
    
    # Create dictionaries for easier lookup
    orig_dict = {key: data for key, data in original_seqs}
    opt_dict = {key: data for key, data in optimized_seqs}
    
    # Find common sequences
    orig_keys = set(orig_dict.keys())
    opt_keys = set(opt_dict.keys())
    common_keys = orig_keys.intersection(opt_keys)
    
    print(f"Common sequences: {len(common_keys)}")
    print(f"Original-only sequences: {orig_keys - opt_keys}")
    print(f"Optimized-only sequences: {opt_keys - orig_keys}")
    
    # Compare common sequences
    detailed_comparison = []
    
    for seq_key in sorted(common_keys):
        orig_data = orig_dict[seq_key]
        opt_data = opt_dict[seq_key]
        
        comparison = {
            'sequence_key': seq_key,
            'original_shape': orig_data.shape,
            'optimized_shape': opt_data.shape,
            'shapes_match': orig_data.shape == opt_data.shape
        }
        
        if comparison['shapes_match']:
            # Detailed numerical comparison
            abs_diff = np.abs(orig_data - opt_data)
            comparison.update({
                'max_abs_diff': np.max(abs_diff),
                'mean_abs_diff': np.mean(abs_diff),
                'exactly_equal': np.array_equal(orig_data, opt_data),
                'original_mean': np.mean(orig_data),
                'original_std': np.std(orig_data),
                'original_min': np.min(orig_data),
                'original_max': np.max(orig_data),
                'optimized_mean': np.mean(opt_data),
                'optimized_std': np.std(opt_data),
                'optimized_min': np.min(opt_data),
                'optimized_max': np.max(opt_data),
                'dtype_original': str(orig_data.dtype),
                'dtype_optimized': str(opt_data.dtype)
            })
            
            # Save sample data for manual inspection
            sample_size = min(10, orig_data.shape[0])
            orig_sample = orig_data[:sample_size]
            opt_sample = opt_data[:sample_size]
            
            np.save(Path(output_dir) / f"{seq_key}_original_sample.npy", orig_sample)
            np.save(Path(output_dir) / f"{seq_key}_optimized_sample.npy", opt_sample)
            
            print(f"\n{seq_key}:")
            print(f"  Shapes: orig={orig_data.shape}, opt={opt_data.shape}")
            print(f"  Exactly equal: {comparison['exactly_equal']}")
            print(f"  Max abs diff: {comparison['max_abs_diff']:.2e}")
            print(f"  Dtypes: orig={comparison['dtype_original']}, opt={comparison['dtype_optimized']}")
            print(f"  Stats orig: mean={comparison['original_mean']:.6f}, std={comparison['original_std']:.6f}")
            print(f"  Stats opt:  mean={comparison['optimized_mean']:.6f}, std={comparison['optimized_std']:.6f}")
            
        else:
            print(f"\n{seq_key}: SHAPE MISMATCH - orig={orig_data.shape}, opt={opt_data.shape}")
        
        detailed_comparison.append(comparison)
    
    # Save detailed comparison
    comparison_df = pd.DataFrame(detailed_comparison)
    comparison_df.to_csv(Path(output_dir) / "sequence_comparison.csv", index=False)
    
    return detailed_comparison

def main():
    args = get_args()
    
    print("=== Data Loading Diagnostic ===")
    print(f"DVC Root: {args.dvc_root}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Zarr Path: {args.zarr_path}")
    
    # Build mapping using original style
    print("\nBuilding test split...")
    mapping, order = build_test_split_original_style(args.dvc_root, args.summary_csv)
    print(f"Found mapping: {mapping}")
    print(f"Order: {order}")
    
    # Load sequences using original method
    print("\nLoading sequences using original method...")
    original_seqs = []
    if HAS_ORIGINAL_LOADER:
        original_seqs = load_sequences_original_style(args.dvc_root, mapping, order)
        print(f"Original method loaded: {len(original_seqs)} sequences")
    else:
        print("Skipping original method - load_dvc_data not available")
    
    # Load sequences using optimized method
    print("\nLoading sequences using optimized method...")
    if Path(args.zarr_path).exists():
        optimized_seqs = load_sequences_optimized_style(args.zarr_path)
        print(f"Optimized method loaded: {len(optimized_seqs)} sequences")
    else:
        print(f"Zarr store not found at {args.zarr_path}")
        return
    
    # Compare the results
    if original_seqs and optimized_seqs:
        print("\n=== COMPARISON RESULTS ===")
        compare_sequences(original_seqs, optimized_seqs, args.output_dir)
    elif optimized_seqs:
        print("\n=== OPTIMIZED SEQUENCES INFO ===")
        for seq_key, seq_data in optimized_seqs[:3]:  # Show first 3
            print(f"{seq_key}: shape={seq_data.shape}, dtype={seq_data.dtype}")
            print(f"  Mean: {np.mean(seq_data):.6f}, Std: {np.std(seq_data):.6f}")
            print(f"  Min: {np.min(seq_data):.6f}, Max: {np.max(seq_data):.6f}")

if __name__ == "__main__":
    main()