"""
Memory and time-optimized comprehensive hBehaveMAE feature extraction
Combines comprehensive extraction with Zarr+Dask optimizations
"""
import argparse
import os
import gc
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import zarr
import dask.array as da
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import original data loading functions
from data_pipeline.load_dvc import load_dvc_data
from models import models_defs
from util import misc

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    p = argparse.ArgumentParser(description="Memory-Optimized Comprehensive hBehaveMAE Feature Extraction")
    p.add_argument("--dvc_root", required=True, type=str)
    p.add_argument("--summary_csv", required=True, type=str)
    p.add_argument("--ckpt_dir", required=True, type=str)
    p.add_argument("--ckpt_name", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--zarr_path", required=True, type=str)
    p.add_argument("--time_aggregations", nargs='+', type=int, required=True)
    p.add_argument("--aggregation_names", nargs='+', type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)  # Smaller for comprehensive extraction
    p.add_argument("--force_zarr_rebuild", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument("--chunk_size", type=int, default=65536, help="Zarr chunk size for optimization")
    
    # Comprehensive extraction options
    p.add_argument("--extract_attention", action="store_true", help="Extract attention weights")
    p.add_argument("--extract_spatial_features", action="store_true", help="Preserve spatial structure")
    p.add_argument("--extract_temporal_dynamics", action="store_true", help="Preserve temporal dynamics")
    p.add_argument("--extract_raw_features", action="store_true", help="Extract raw encoder outputs")
    p.add_argument("--extract_all", action="store_true", help="Extract all possible features")
    
    return p.parse_args()

def create_optimized_zarr_store(sequences, zarr_path, chunk_size=65536):
    """Create memory-efficient Zarr store with optimal chunking"""
    print("Creating optimized Zarr store...")
    
    root = zarr.open(zarr_path, mode='w')
    
    # Process sequences in parallel with memory management
    def store_sequence(seq_item):
        seq_key, seq_data = seq_item
        try:
            # Optimal chunking for sequential access
            optimal_chunk_size = min(chunk_size, seq_data.shape[0])
            data_array = root.create_dataset(
                seq_key,
                data=seq_data,
                chunks=(optimal_chunk_size, None),
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
                dtype=np.float32  # Consistent dtype
            )
            data_array.attrs['description'] = f"Sequence data for {seq_key}"
            data_array.attrs['original_shape'] = seq_data.shape
            return True
        except Exception as e:
            print(f"Error storing {seq_key}: {e}")
            return False
    
    # Use ThreadPoolExecutor for I/O operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store_sequence, seq_item) for seq_item in sequences]
        success_count = sum(1 for future in tqdm(as_completed(futures), 
                                               total=len(futures), 
                                               desc="Storing sequences") if future.result())
    
    print(f"Zarr store created: {success_count}/{len(sequences)} sequences stored successfully")
    return root

def create_comprehensive_model_wrapper(model):
    """Memory-efficient comprehensive model wrapper"""
    class MemoryEfficientComprehensiveWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x, mask_ratio=0.0):
            """Extract comprehensive features with memory management"""
            all_features = {}
            
            # Use gradient checkpointing if available
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable()
            
            with torch.no_grad():
                # Get hierarchical outputs efficiently
                encoder_output, mask, intermediates = self.base_model.forward_encoder(
                    x, mask_ratio=mask_ratio, return_intermediates=True
                )
                
                # Store only essential features to save memory
                all_features['hierarchical_levels'] = [level.detach().cpu() for level in intermediates]
                all_features['final_encoder_output'] = encoder_output.detach().cpu()
                if mask is not None:
                    all_features['mask'] = mask.detach().cpu()
                
                # Clear GPU memory immediately
                del encoder_output, intermediates
                if mask is not None:
                    del mask
                torch.cuda.empty_cache()
            
            return all_features
    
    return MemoryEfficientComprehensiveWrapper(model)

"""
Enhanced embedding extraction with timestamp mapping and fixed feature extraction
"""

# Key fixes needed in your existing script:

def extract_comprehensive_features_optimized(batch, model_wrapper, extract_options):
    """
    FIXED: Memory-optimized comprehensive feature extraction
    """
    B, C, T, H, W = batch.shape
    
    # Get model outputs
    all_features = model_wrapper(batch)
    comprehensive_results = {}
    
    # 1. HIERARCHICAL FEATURES (working correctly)
    if 'hierarchical_levels' in all_features:
        hierarchical_levels = all_features['hierarchical_levels']
        comprehensive_results['hierarchical'] = {}
        
        for i, level_feat in enumerate(hierarchical_levels):
            level_name = f'level_{i+1}'
            
            if extract_options.get('temporal_dynamics', False):
                temporal_features = level_feat.flatten(2, -2).mean(-1)
                comprehensive_results['hierarchical'][f'{level_name}_temporal'] = temporal_features.numpy()
            
            if extract_options.get('spatial_features', False):
                spatial_features = level_feat.mean(1)
                comprehensive_results['hierarchical'][f'{level_name}_spatial'] = spatial_features.numpy()
            
            if extract_options.get('spatio_temporal', True):
                spatiotemporal_features = level_feat.flatten(1, -2)
                comprehensive_results['hierarchical'][f'{level_name}_spatiotemporal'] = spatiotemporal_features.numpy()
            
            # Standard pooled features
            pooled_features = level_feat.flatten(1, -2).mean(1)
            comprehensive_results['hierarchical'][f'{level_name}_pooled'] = pooled_features.numpy()
            
            del level_feat
    
    # 2. ELECTRODE-SPECIFIC FEATURES (FIX: This was not extracting properly)
    if extract_options.get('electrode_specific', True) and 'hierarchical_levels' in all_features:
        comprehensive_results['electrode_features'] = {}
        
        for i, level_feat in enumerate(all_features['hierarchical_levels']):
            if len(level_feat.shape) == 5:  # (B, T, H, W, C)
                H_dim, W_dim = level_feat.shape[2], level_feat.shape[3]
                
                # Extract ALL electrode positions (not just subset)
                for h in range(H_dim):
                    for w in range(W_dim):
                        electrode_key = f'level_{i+1}_electrode_{h}_{w}'
                        # Shape: (B, T, C) -> pool temporal to (B, C)
                        electrode_features = level_feat[:, :, h, w, :].mean(1)
                        comprehensive_results['electrode_features'][electrode_key] = electrode_features.numpy()
    
    # 3. TEMPORAL ANALYSIS (FIX: This was not working)
    if extract_options.get('temporal_analysis', True) and 'hierarchical_levels' in all_features:
        comprehensive_results['temporal_analysis'] = {}
        
        for i, level_feat in enumerate(all_features['hierarchical_levels']):
            if len(level_feat.shape) == 5 and level_feat.shape[1] > 1:  # (B, T, H, W, C)
                # Pool spatial dimensions to get temporal features: (B, T, C)
                level_temporal = level_feat.flatten(2, -2).mean(-1)
                
                # Temporal derivatives (changes over time)
                if level_temporal.shape[1] > 1:
                    temporal_diff = (level_temporal[:, 1:] - level_temporal[:, :-1])
                    comprehensive_results['temporal_analysis'][f'level_{i+1}_temporal_diff'] = temporal_diff.numpy()
                
                # Temporal variance across the time dimension
                temporal_var = level_temporal.var(dim=1)
                comprehensive_results['temporal_analysis'][f'level_{i+1}_temporal_var'] = temporal_var.numpy()
    
    # Clear all intermediate results
    del all_features
    gc.collect()
    
    return comprehensive_results

def create_timestamp_mapping(sequence_key, start_frame, end_frame, summary_df):
    """
    Create detailed timestamp mapping for a sequence
    """
    # Parse sequence key (format: EHDP-00083_2019-11-11)
    parts = sequence_key.split('_')
    cage_id = parts[0]
    date_str = '_'.join(parts[1:])  # 2019-11-11
    
    # Find matching summary entry
    matching_summary = summary_df[
        (summary_df['cage_id'] == cage_id) & 
        (summary_df['from_tpt'].dt.strftime('%Y-%m-%d') == date_str)
    ]
    
    if matching_summary.empty:
        print(f"Warning: No summary found for {sequence_key}")
        return None
    
    summary_row = matching_summary.iloc[0]
    start_time = summary_row['from_tpt']  # e.g., 2019-11-11 06:00:00
    end_time = summary_row['to_tpt']      # e.g., 2019-11-14 06:00:00
    
    # Create minute-by-minute timestamps
    total_frames = end_frame - start_frame
    timestamps = pd.date_range(start=start_time, end=end_time, periods=total_frames)
    
    # Create mapping: frame_index -> timestamp_string
    timestamp_mapping = {}
    for i, timestamp in enumerate(timestamps):
        # Format: YYYY-MM-DD_HH:MM
        timestamp_str = timestamp.strftime('%Y-%m-%d_%H:%M')
        timestamp_mapping[i] = timestamp_str
    
    return {
        'sequence_key': sequence_key,
        'cage_id': cage_id,
        'date_range': date_str,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_frames': total_frames,
        'frame_to_timestamp': timestamp_mapping,
        'timestamps_array': [ts.strftime('%Y-%m-%d_%H:%M') for ts in timestamps]
    }

def save_comprehensive_results_with_timestamps(all_results, frame_map, output_dir, agg_name, summary_csv):
    """
    Save results with timestamp information
    """
    print(f"Saving comprehensive results with timestamps for {agg_name}...")
    
    agg_dir = output_dir / agg_name
    agg_dir.mkdir(exist_ok=True)
    
    # Load summary for timestamp creation
    summary_df = pd.read_csv(summary_csv)
    # take only first 10 rows to match original
    summary_df = summary_df.head(10)
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])
    
    # Create enhanced frame map with timestamps
    enhanced_frame_map = {}
    
    for seq_key, (start_frame, end_frame) in frame_map.items():
        timestamp_info = create_timestamp_mapping(seq_key, start_frame, end_frame, summary_df)
        if timestamp_info:
            enhanced_frame_map[seq_key] = {
                'frame_range': (start_frame, end_frame),
                'timestamp_info': timestamp_info
            }
    
    # Save enhanced frame mapping
    frame_map_path = agg_dir / 'frame_map_with_timestamps.npy'
    np.save(frame_map_path, enhanced_frame_map)
    
    # Save each sequence with timestamp keys
    for sequence_key, sequence_results in all_results.items():
        if sequence_key in enhanced_frame_map:
            timestamp_info = enhanced_frame_map[sequence_key]['timestamp_info']
            timestamps_array = timestamp_info['timestamps_array']
            
            for category, category_features in sequence_results.items():
                category_dir = agg_dir / category
                category_dir.mkdir(exist_ok=True)
                
                # Save as Zarr with timestamp information
                category_zarr_path = category_dir / f'{sequence_key}.zarr'
                category_zarr = zarr.open(str(category_zarr_path), mode='w')
                
                # Save features
                for feature_name, feature_array in category_features.items():
                    category_zarr[feature_name] = feature_array
                
                # Add timestamp array for each sequence
                # This creates a mapping: index -> timestamp
                category_zarr['timestamps'] = np.array(timestamps_array, dtype='U16')  # Unicode string
                category_zarr['frame_indices'] = np.arange(len(timestamps_array))
                
                # Add metadata
                category_zarr.attrs['sequence_key'] = sequence_key
                category_zarr.attrs['cage_id'] = timestamp_info['cage_id']
                category_zarr.attrs['date_range'] = timestamp_info['date_range']
                category_zarr.attrs['start_time'] = timestamp_info['start_time']
                category_zarr.attrs['end_time'] = timestamp_info['end_time']
    
    print(f"Enhanced results with timestamps saved to {agg_dir}")

# Usage example for accessing timestamped embeddings:
def example_access_timestamped_embeddings():
    """
    Example of how to access embeddings with timestamps
    """
    # Load a sequence
    zarr_path = "path/to/hierarchical/EHDP-00083_2019-11-11.zarr"
    zarr_store = zarr.open(zarr_path, mode='r')
    
    # Get embeddings and timestamps
    embeddings = zarr_store['level_1_pooled'][:]  # Shape: (4320, 128)
    timestamps = zarr_store['timestamps'][:]      # Shape: (4320,) - strings like '2019-11-11_06:00'
    
    # Access specific time
    # Example: Get embedding for 6:30 AM on first day
    target_time = '2019-11-11_06:30'
    time_index = np.where(timestamps == target_time)[0]
    if len(time_index) > 0:
        embedding_at_630am = embeddings[time_index[0]]  # Shape: (128,)
        print(f"Embedding at 6:30 AM: {embedding_at_630am[:5]}")
    
    # Get embeddings for specific hour ranges
    morning_indices = [i for i, ts in enumerate(timestamps) if ts.endswith('_06:00') or ts.endswith('_07:00')]
    morning_embeddings = embeddings[morning_indices]
    
    return embeddings, timestamps

# Quick diagnostic script to check why electrode_features and temporal_analysis are empty
def diagnose_missing_features():
    """
    Diagnostic script to understand why some features weren't extracted
    """
    print("DIAGNOSTIC: Checking feature extraction logic...")
    
    # Check the extract_options that were passed
    extract_options_used = {
        'attention': True,  # from --extract_all
        'spatial_features': True,
        'temporal_dynamics': True, 
        'raw_features': True,
        'electrode_specific': True,  # This should be True
        'temporal_analysis': True,   # This should be True
        'spatio_temporal': True
    }
    
    print("Extract options used:", extract_options_used)
    
    # The issue is likely in the conditions:
    # 1. electrode_specific extraction depends on 5D tensors (B, T, H, W, C)
    # 2. temporal_analysis depends on temporal dimension > 1
    
    # Check if hierarchical features have the right shape
    # From your output: level_1_spatial: (4320, 144, 1, 1, 128) - this is 5D, should work
    # The issue might be in the indexing: level_feat[:, :, h, w, :] 
    # But your spatial features have shape (4320, 144, 1, 1, 128)
    # So H=1, W=1 which means only 1 electrode position?
    
    print("\nExpected tensor shapes for electrode extraction:")
    print("level_feat should be (B, T, H, W, C) where H, W > 1")
    print("Your actual shapes suggest H=1, W=1 for most levels")
    print("This might be why electrode_features is empty - no spatial structure to extract")

# def extract_comprehensive_features_optimized(batch, model_wrapper, extract_options):
#     """Memory-optimized comprehensive feature extraction"""
#     B, C, T, H, W = batch.shape
    
#     # Get model outputs
#     all_features = model_wrapper(batch)
#     comprehensive_results = {}
    
#     # 1. HIERARCHICAL FEATURES with memory-efficient processing
#     if 'hierarchical_levels' in all_features:
#         hierarchical_levels = all_features['hierarchical_levels']
#         comprehensive_results['hierarchical'] = {}
        
#         for i, level_feat in enumerate(hierarchical_levels):
#             level_name = f'level_{i+1}'
            
#             # Process features based on options, using float16 to save memory
#             if extract_options.get('temporal_dynamics', False):
#                 temporal_features = level_feat.flatten(2, -2).mean(-1)  # Use half precision
#                 comprehensive_results['hierarchical'][f'{level_name}_temporal'] = temporal_features.numpy()
            
#             if extract_options.get('spatial_features', False):
#                 spatial_features = level_feat.mean(1)
#                 comprehensive_results['hierarchical'][f'{level_name}_spatial'] = spatial_features.numpy()
            
#             if extract_options.get('spatio_temporal', True):
#                 # Limit spatio-temporal to prevent memory explosion
#                 spatiotemporal_features = level_feat.flatten(1, -2)
#                 comprehensive_results['hierarchical'][f'{level_name}_spatiotemporal'] = spatiotemporal_features.numpy()
            
#             # Standard pooled features (always include, but use float16)
#             pooled_features = level_feat.flatten(1, -2).mean(1)
#             comprehensive_results['hierarchical'][f'{level_name}_pooled'] = pooled_features.numpy()
            
#             # Clear memory immediately after processing each level
#             del level_feat
    
#     # 2. ELECTRODE-SPECIFIC FEATURES (optimized)
#     if extract_options.get('electrode_specific', True):
#         comprehensive_results['electrode_features'] = {}
        
#         for i, level_feat in enumerate(all_features.get('hierarchical_levels', [])):
#             if len(level_feat.shape) == 5:
#                 # Process only a subset of electrodes to save memory
#                 for h in range(min(2, level_feat.shape[2])):  # Limit to reduce memory
#                     for w in range(min(2, level_feat.shape[3])):
#                         electrode_key = f'level_{i+1}_electrode_{h}_{w}'
#                         electrode_features = level_feat[:, :, h, w, :].mean(1)
#                         comprehensive_results['electrode_features'][electrode_key] = electrode_features.numpy()
    
#     # 3. TEMPORAL ANALYSIS (memory-efficient)
#     if extract_options.get('temporal_analysis', True):
#         comprehensive_results['temporal_analysis'] = {}
        
#         for i, level_feat in enumerate(all_features.get('hierarchical_levels', [])):
#             if len(level_feat.shape) == 5:
#                 level_temporal = level_feat.flatten(2, -2).mean(-1)
#                 if level_temporal.shape[1] > 1:
#                     # Only compute first derivative to save memory
#                     temporal_diff = (level_temporal[:, 1:] - level_temporal[:, :-1])
#                     comprehensive_results['temporal_analysis'][f'level_{i+1}_temporal_diff'] = temporal_diff.numpy()
                
#                 # Temporal variance
#                 temporal_var = level_temporal.var(dim=1)
#                 comprehensive_results['temporal_analysis'][f'level_{i+1}_temporal_var'] = temporal_var.numpy()
    
#     # Clear all intermediate results
#     del all_features
#     gc.collect()
    
#     return comprehensive_results

def process_sequence_batch_comprehensive_optimized(sequences, zarr_store, model, num_frames, 
                                                 batch_size, device_id, extract_options, num_levels):
    """Memory-optimized comprehensive processing using Zarr and Dask"""
    device = torch.device(f"cuda:{device_id}")
    
    # Create comprehensive model wrapper
    model_wrapper = create_comprehensive_model_wrapper(model)
    
    # Results storage
    all_comprehensive_results = {}
    frame_map = {}
    ptr = 0
    
    # Process sequences with memory management
    for seq_key, _ in tqdm(sequences, desc=f"GPU {device_id} comprehensive processing", leave=False):
        try:
            # Load from Zarr using Dask for memory efficiency
            zarr_data = zarr_store[seq_key]
            dask_array = da.from_zarr(zarr_data)
            
            # Convert to numpy with controlled memory usage
            mat = dask_array.compute()
            
            # Normalize and aggregate
            mat = mat.astype(np.float32) / 100.0
            original_frames = mat.shape[0]
            
            if original_frames != num_frames:
                mat = aggregate_sequence_exact(mat, num_frames, aggregation_method='mean')
            
            n_frames = mat.shape[0]
            
            # Process with sliding windows
            dataset = SlidingWindowDataset(mat, num_frames)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, pin_memory=False)
            
            # Initialize storage for this sequence
            sequence_comprehensive_results = {}
            
            # Process batches with memory management
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dl, desc=f"GPU {device_id} batches", leave=False)):
                    batch = batch.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                    
                    # Extract comprehensive features
                    batch_comprehensive = extract_comprehensive_features_optimized(
                        batch, model_wrapper, extract_options
                    )
                    
                    # Accumulate features
                    for category, category_features in batch_comprehensive.items():
                        if category not in sequence_comprehensive_results:
                            sequence_comprehensive_results[category] = {}
                        
                        for feature_name, feature_array in category_features.items():
                            if feature_name not in sequence_comprehensive_results[category]:
                                sequence_comprehensive_results[category][feature_name] = []
                            sequence_comprehensive_results[category][feature_name].append(feature_array)
                    
                    # Clear batch from GPU immediately
                    del batch, batch_comprehensive
                    torch.cuda.empty_cache()
            
            # Concatenate results for this sequence using numpy for efficiency
            for category in sequence_comprehensive_results:
                for feature_name in sequence_comprehensive_results[category]:
                    feature_list = sequence_comprehensive_results[category][feature_name]
                    if feature_list:
                        sequence_comprehensive_results[category][feature_name] = np.concatenate(
                            feature_list, axis=0
                        ).astype(np.float32)  
            
            # Store results
            all_comprehensive_results[seq_key] = sequence_comprehensive_results
            frame_map[seq_key] = (ptr, ptr + n_frames)
            ptr += n_frames
            
            # Clear sequence data from memory
            del mat, sequence_comprehensive_results
            
        except Exception as e:
            print(f"Error processing sequence {seq_key}: {e}")
            continue
        finally:
            # Always clean up
            gc.collect()
            torch.cuda.empty_cache()
    
    return all_comprehensive_results, frame_map

def save_comprehensive_results_optimized(all_results, frame_map, output_dir, agg_name):
    """Save results with Zarr for efficient storage"""
    print(f"Saving optimized comprehensive results for {agg_name}...")
    
    agg_dir = output_dir / agg_name
    agg_dir.mkdir(exist_ok=True)
    
    # Save frame mapping as numpy file (can't serialize dict to Zarr easily)
    frame_map_path = agg_dir / 'frame_map.npy'
    np.save(frame_map_path, frame_map)
    
    # Use ThreadPoolExecutor for parallel saving
    def save_feature_category(args):
        sequence_key, sequence_results = args
        try:
            for category, category_features in sequence_results.items():
                category_dir = agg_dir / category
                category_dir.mkdir(exist_ok=True)
                
                # Save as Zarr for efficient access
                category_zarr_path = category_dir / f'{sequence_key}.zarr'
                category_zarr = zarr.open(str(category_zarr_path), mode='w')
                
                for feature_name, feature_array in category_features.items():
                    category_zarr[feature_name] = feature_array
            return True
        except Exception as e:
            print(f"Error saving {sequence_key}: {e}")
            return False
    
    # Parallel saving with controlled memory usage
    with ThreadPoolExecutor(max_workers=2) as executor:  # Limit workers to control memory
        futures = [executor.submit(save_feature_category, item) for item in all_results.items()]
        success_count = sum(1 for future in tqdm(as_completed(futures), 
                                               total=len(futures), 
                                               desc="Saving results") if future.result())
    
    print(f"Saved {success_count}/{len(all_results)} sequences successfully")

# Include the exact same functions from your second script for consistency
def build_test_split_original_exact(root, csv):
    """Exact same logic as original script"""
    summary_df = pd.read_csv(csv)
    summary_df = summary_df.head(10)  # Match original limit
    
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])
    summary_df.sort_values(by=['cage_id', 'from_tpt'], inplace=True)
    
    time_delta_minutes = (summary_df['to_tpt'] - summary_df['from_tpt']).dt.total_seconds().iloc[0] / 60
    expected_chunk_size = int(round(time_delta_minutes)) + 1
    print(f"Expected sequence chunk size from summary: {expected_chunk_size} minutes.")
    
    mapping, order = {}, []
    valid_chunks = 0
    
    for _, r in summary_df.iterrows():
        day = str(r["from_tpt"]).split()[0]
        cage_csv_path = Path(root) / f"{r['cage_id']}.csv"
        if not cage_csv_path.exists():
            print(f"Warning: CSV file not found for cage {r['cage_id']}. Skipping.")
            continue
            
        cage_df = pd.read_csv(cage_csv_path, usecols=['timestamp'] + ELEC_COLS)
        cage_df['timestamp'] = pd.to_datetime(cage_df['timestamp'])
        
        chunk = cage_df[(cage_df['timestamp'] >= r['from_tpt']) & (cage_df['timestamp'] <= r['to_tpt'])]
        
        if len(chunk) != expected_chunk_size:
            print(f"Warning: Cage {r['cage_id']} chunk starting {r['from_tpt']} has {len(chunk)} rows, expected {expected_chunk_size}. Skipping.")
            continue
        if chunk[ELEC_COLS].isnull().values.any():
            print(f"Warning: Cage {r['cage_id']} chunk starting {r['from_tpt']} contains NaN values. Skipping.")
            continue
            
        mapping.setdefault(r["cage_id"], []).append(day)
        order.append(f"{r['cage_id']}_{day}")
        valid_chunks += 1
        
    print(f"Built mapping for {len(mapping)} cages, {valid_chunks} valid chunks out of {len(summary_df)} total")
    return mapping, order

def load_sequences_original_exact(root, mapping, order, skip_missing):
    """Use original data loading method"""
    raw = load_dvc_data(root, mapping)  # Use original function
    seqs = []
    for cage, df in raw.items():
        df["__day"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        for day, sub in df.groupby("__day"):
            key = f"{cage}_{day}"
            if key not in order:
                continue
            missing = [c for c in ELEC_COLS if c not in sub.columns]
            if missing:
                if skip_missing:
                    print(f"[WARN] {key} missing {missing}")
                    continue
                raise KeyError(f"{key} missing {missing}")
            seqs.append((key, sub[ELEC_COLS].values.astype(np.float32)))
    
    seqs.sort(key=lambda x: order.index(x[0]))
    print(f"Loaded {len(seqs)} sequences")
    return seqs

def parse_q_strides_exact(q_strides_str):
    """Exact same parsing as original"""
    if isinstance(q_strides_str, str):
        stages = q_strides_str.split(';')
        parsed_strides = [[int(x) for x in stage.split(',')] for stage in stages]
        num_levels = len(parsed_strides) + 1
        return parsed_strides, num_levels
    return q_strides_str, len(q_strides_str) + 1

def load_model_for_gpu_exact(ckpt_path, model_args, device_id):
    """Load model exactly like original script"""
    device = torch.device(f"cuda:{device_id}")
    
    print(f"Loading model on GPU {device_id}")
    from iopath.common.file_io import g_pathmgr as pathmgr
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu")
    
    model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
    
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in ckpt["model"].items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                print(f"GPU {device_id}: Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
        else:
            print(f"GPU {device_id}: Skipping {k}: not found in model")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"GPU {device_id}: Loaded {len(pretrained_dict)}/{len(ckpt['model'])} layers")
    
    return model.to(device).eval().requires_grad_(False)

class SlidingWindowDataset(torch.utils.data.Dataset):
    """Exact same as original"""
    def __init__(self, mat, num_frames):
        self.mat = mat
        self.num_frames = num_frames
        self.pad = (num_frames - 1) // 2
        self.padded = np.pad(mat, ((self.pad, self.pad), (0, 0)), "edge")
        self.length = mat.shape[0]
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.padded[idx:idx + self.num_frames]
        
        if window.shape[0] < self.num_frames:
            pad_needed = self.num_frames - window.shape[0]
            window = np.pad(window, ((0, pad_needed), (0, 0)), 'edge')
        
        return torch.from_numpy(window.copy()).unsqueeze(0).unsqueeze(2)

def aggregate_sequence_exact(sequence, target_frames, aggregation_method='mean'):
    """Exact same as original"""
    current_frames = sequence.shape[0]
    if current_frames == target_frames:
        return sequence
    elif current_frames < target_frames:
        pad_needed = target_frames - current_frames
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        return np.pad(sequence, ((pad_before, pad_after), (0, 0)), 'edge')
    else:
        if aggregation_method == 'mean':
            excess = current_frames % target_frames
            if excess != 0:
                sequence = sequence[:current_frames - excess]
            group_size = sequence.shape[0] // target_frames
            reshaped = sequence.reshape(target_frames, group_size, -1)
            return reshaped.mean(axis=1)
        elif aggregation_method == 'subsample':
            indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
            return sequence[indices]
    return sequence

def main():
    mp.set_start_method('spawn', force=True)
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse extraction options
    extract_options = {
        'attention': args.extract_attention or args.extract_all,
        'spatial_features': args.extract_spatial_features or args.extract_all,
        'temporal_dynamics': args.extract_temporal_dynamics or args.extract_all,
        'raw_features': args.extract_raw_features or args.extract_all,
        'electrode_specific': args.extract_all,
        'temporal_analysis': args.extract_all,
        'spatio_temporal': True
    }
    
    print("MEMORY-OPTIMIZED COMPREHENSIVE EXTRACTION:")
    for option, enabled in extract_options.items():
        print(f"  - {option}: {'✓' if enabled else '✗'}")

    # Load data using original method
    print("Loading data using original approach...")
    mapping, order = build_test_split_original_exact(args.dvc_root, args.summary_csv)
    sequences = load_sequences_original_exact(args.dvc_root, mapping, order, args.skip_missing)

    # Create or use optimized Zarr store
    if args.force_zarr_rebuild or not Path(args.zarr_path).exists():
        zarr_store = create_optimized_zarr_store(sequences, args.zarr_path, args.chunk_size)
    else:
        print(f"Using existing Zarr store at {args.zarr_path}")
        zarr_store = zarr.open(args.zarr_path, mode='r')

    # Setup GPU
    gpu_ids = [0]  # Single GPU for comprehensive extraction
    print(f"Using GPU: {gpu_ids[0]}")

    # Load checkpoint and model args
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        if os.path.exists(os.path.join(args.ckpt_dir, "checkpoint-best.pth")):
            ckpt_path = os.path.join(args.ckpt_dir, "checkpoint-best.pth")
        else:
            checkpoint_files = [f for f in os.listdir(args.ckpt_dir) if f.startswith('checkpoint-') and f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.ckpt_dir, x)), reverse=True)
                ckpt_path = os.path.join(args.ckpt_dir, checkpoint_files[0])
            else:
                raise FileNotFoundError(f"No checkpoint found in {args.ckpt_dir}")
    
    print("Loading checkpoint", ckpt_path)
    from iopath.common.file_io import g_pathmgr as pathmgr
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu")
    base_model_args = ckpt.get("args", {})
    if isinstance(base_model_args, argparse.Namespace):
        base_model_args = vars(base_model_args)

    if 'q_strides' in base_model_args:
        base_model_args['q_strides'], num_levels = parse_q_strides_exact(base_model_args['q_strides'])
    else:
        base_model_args['q_strides'], num_levels = parse_q_strides_exact("4,1,1;3,1,1;2,1,1;3,1,1")
    
    print(f"Model architecture: {num_levels} hierarchical levels")

    # Process each aggregation with optimized comprehensive extraction
    for num_frames, agg_name in tqdm(zip(args.time_aggregations, args.aggregation_names), 
                                     total=len(args.time_aggregations), desc="Processing aggregations"):
        print(f"\n=== OPTIMIZED COMPREHENSIVE EXTRACTION: {agg_name} ({num_frames} frames) ===")
        
        model_args = {**base_model_args}
        model_args['input_size'] = [num_frames, 1, 12]
        model_args['num_frames'] = num_frames
        
        model = load_model_for_gpu_exact(ckpt_path, model_args, gpu_ids[0])
        
        # Optimized comprehensive feature extraction
        all_results, frame_map = process_sequence_batch_comprehensive_optimized(
            sequences, zarr_store, model, num_frames, args.batch_size, 
            gpu_ids[0], extract_options, num_levels
        )
        
        # Save optimized results
        save_comprehensive_results_optimized(all_results, frame_map, output_dir, agg_name)
        
        # Aggressive cleanup
        del model, all_results, frame_map
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nMemory-optimized comprehensive extraction completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    diagnose_missing_features()  # Run diagnostic to check feature extraction logic 
    main()