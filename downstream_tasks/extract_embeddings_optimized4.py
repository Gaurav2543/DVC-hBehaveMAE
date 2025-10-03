"""
Simplified, memory-efficient embedding extraction with proper timestamp mapping
Focus on what actually matters: core embeddings + timestamps
"""
import argparse
import os
import gc
import datetime
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
import warnings
warnings.filterwarnings('ignore')

# Import original data loading functions
from data_pipeline.load_dvc import load_dvc_data
from models import models_defs
from util import misc

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    p = argparse.ArgumentParser(description="Simplified Memory-Efficient Embedding Extraction")
    p.add_argument("--dvc_root", required=True, type=str)
    p.add_argument("--summary_csv", required=True, type=str)
    p.add_argument("--ckpt_dir", required=True, type=str)
    p.add_argument("--ckpt_name", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--zarr_path", required=True, type=str)
    p.add_argument("--time_aggregations", nargs='+', type=int, required=True)
    p.add_argument("--aggregation_names", nargs='+', type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--force_zarr_rebuild", action="store_true")
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument("--chunk_size", type=int, default=32768)
    p.add_argument("--include_spatial", action="store_true", help="Also save spatial embeddings if they make sense")
    
    return p.parse_args()

def build_test_split_original_exact(root, csv):
    """Exact same logic as original script"""
    summary_df = pd.read_csv(csv)
    # summary_df = summary_df.head(500)
        # take entries 100 to 500
    summary_df = summary_df.iloc[1500:] 
    
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
                if skip_missing:
                    print(f"[WARN] {key} missing {missing}")
                    continue
                raise KeyError(f"{key} missing {missing}")
            seqs.append((key, sub[ELEC_COLS].values.astype(np.float32)))
    
    seqs.sort(key=lambda x: order.index(x[0]))
    print(f"Loaded {len(seqs)} sequences")
    return seqs

def create_optimized_zarr_store(sequences, zarr_path, chunk_size=32768):
    """Create memory-efficient Zarr store"""
    print("Creating optimized Zarr store...")
    
    root = zarr.open(zarr_path, mode='w')
    
    def store_sequence(seq_item):
        seq_key, seq_data = seq_item
        try:
            optimal_chunk_size = min(chunk_size, seq_data.shape[0])
            data_array = root.create_dataset(
                seq_key,
                data=seq_data,
                chunks=(optimal_chunk_size, None),
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE),
                dtype=np.float32
            )
            data_array.attrs['description'] = f"Sequence data for {seq_key}"
            data_array.attrs['original_shape'] = seq_data.shape
            return True
        except Exception as e:
            print(f"Error storing {seq_key}: {e}")
            return False
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store_sequence, seq_item) for seq_item in sequences]
        success_count = sum(1 for future in tqdm(as_completed(futures), 
                                               total=len(futures), 
                                               desc="Storing sequences") if future.result())
    
    print(f"Zarr store created: {success_count}/{len(sequences)} sequences stored successfully")
    return root

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

def extract_core_embeddings(batch, model):
    """Extract only the core embeddings we actually need"""
    with torch.no_grad():
        # Get hierarchical levels
        encoder_output, mask, hierarchical_levels = model.forward_encoder(
            batch, mask_ratio=0.0, return_intermediates=True
        )
        
        results = {}
        spatial_results = {}
        
        # Extract core pooled features (main embeddings)
        for i, level_feat in enumerate(hierarchical_levels):
            level_name = f'level_{i+1}'
            
            # Core pooled embedding (this is what you primarily want)
            pooled = level_feat.flatten(1, -2).mean(1).float().cpu()
            results[f'{level_name}_pooled'] = pooled.numpy()
            
            # Optional spatial variants (only if they make sense)
            if len(level_feat.shape) >= 4:  # Has spatial structure
                # Temporal pooling but keep some spatial info
                if len(level_feat.shape) == 5:  # (B, T, H, W, C)
                    spatial_pooled = level_feat.mean(1).flatten(1, -2).mean(1).float().cpu()
                    spatial_results[f'{level_name}_spatial_pooled'] = spatial_pooled.numpy()
                elif len(level_feat.shape) == 3 and level_feat.shape[1] > 1:  # (B, T, Features)
                    # Keep some temporal structure
                    temporal_reduced = level_feat.mean(1).float().cpu()  # (B, Features)
                    spatial_results[f'{level_name}_temporal_mean'] = temporal_reduced.numpy()
            
            # Clear GPU memory immediately
            del level_feat
            
        # Combined embedding across all levels
        if len(results) > 1:
            combined_features = []
            for key in sorted(results.keys()):
                combined_features.append(torch.from_numpy(results[key].astype(np.float32)))
            combined = torch.cat(combined_features, dim=1)
            results['combined'] = combined.numpy()
            del combined_features, combined
        
        # Clear remaining GPU memory
        del encoder_output, hierarchical_levels
        if mask is not None:
            del mask
        torch.cuda.empty_cache()
        
        return results, spatial_results

def create_timestamp_mapping(sequence_key, num_frames, summary_df):
    """Create timestamp array for a sequence"""
    # Parse sequence key: EHDP-00083_2019-11-11
    parts = sequence_key.split('_')
    cage_id = parts[0]
    date_str = '_'.join(parts[1:])
    
    # Find matching summary entry
    matching_summary = summary_df[
        (summary_df['cage_id'] == cage_id) & 
        (summary_df['from_tpt'].dt.strftime('%Y-%m-%d') == date_str)
    ]
    
    if matching_summary.empty:
        print(f"Warning: No summary found for {sequence_key}")
        # Fallback: create generic timestamps
        base_date = pd.to_datetime(date_str + ' 06:00:00')
        timestamps = pd.date_range(start=base_date, periods=num_frames, freq='1T')
    else:
        summary_row = matching_summary.iloc[0]
        start_time = summary_row['from_tpt']
        end_time = summary_row['to_tpt']
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_frames)
    
    # Format as strings: YYYY-MM-DD_HH:MM
    timestamp_strings = [ts.strftime('%Y-%m-%d_%H:%M') for ts in timestamps]
    return timestamp_strings, timestamps[0], timestamps[-1]

def process_and_save_sequence(seq_key, zarr_store, model, num_frames, batch_size, 
                            device_id, output_dir, summary_df, include_spatial):
    """Process single sequence with memory optimization"""
    device = torch.device(f"cuda:{device_id}")
    
    try:
        # Load sequence data using Dask
        zarr_data = zarr_store[seq_key]
        dask_array = da.from_zarr(zarr_data)
        mat = dask_array.compute().astype(np.float32)
        
        # Normalize and aggregate
        mat = mat / 100.0
        original_frames = mat.shape[0]
        
        if original_frames != num_frames:
            mat = aggregate_sequence_exact(mat, num_frames, aggregation_method='mean')
        
        n_frames = mat.shape[0]
        
        # Create sliding window dataset
        dataset = SlidingWindowDataset(mat, num_frames)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                      num_workers=0, pin_memory=False)
        
        # Process batches and accumulate results
        core_embeddings = []
        spatial_embeddings = [] if include_spatial else None
        
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"Processing {seq_key}", leave=False):
                batch = batch.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                
                # Extract embeddings
                core_results, spatial_results = extract_core_embeddings(batch, model)
                core_embeddings.append(core_results)
                
                if include_spatial and spatial_results:
                    spatial_embeddings.append(spatial_results)
                
                # Clear batch immediately
                del batch, core_results, spatial_results
                torch.cuda.empty_cache()
        
        # Concatenate results
        final_embeddings = {}
        for key in core_embeddings[0].keys():
            arrays = [batch_result[key] for batch_result in core_embeddings]
            final_embeddings[key] = np.concatenate(arrays, axis=0)
        
        final_spatial = {}
        if include_spatial and spatial_embeddings:
            for key in spatial_embeddings[0].keys():
                arrays = [batch_result[key] for batch_result in spatial_embeddings]
                final_spatial[key] = np.concatenate(arrays, axis=0)
        
        # Create timestamps
        timestamps, start_time, end_time = create_timestamp_mapping(seq_key, n_frames, summary_df)
        
        # Save core embeddings
        core_zarr_path = output_dir / 'core_embeddings' / f'{seq_key}.zarr'
        core_zarr_path.parent.mkdir(parents=True, exist_ok=True)
        core_store = zarr.open(str(core_zarr_path), mode='w')
        
        # Save embeddings with efficient chunking
        for name, data in final_embeddings.items():
            chunk_size = min(1024, data.shape[0])
            core_store[name] = zarr.array(data, chunks=(chunk_size, None) if len(data.shape) > 1 else (chunk_size,))
        
        # Save timestamps and metadata
        core_store['timestamps'] = zarr.array(timestamps, chunks=(min(1024, len(timestamps)),))
        core_store['frame_indices'] = zarr.array(np.arange(len(timestamps)), chunks=(min(1024, len(timestamps)),))
        
        # Metadata
        core_store.attrs['sequence_key'] = seq_key
        core_store.attrs['cage_id'] = seq_key.split('_')[0]
        core_store.attrs['date'] = '_'.join(seq_key.split('_')[1:])
        core_store.attrs['start_time'] = start_time.isoformat()
        core_store.attrs['end_time'] = end_time.isoformat()
        core_store.attrs['total_frames'] = n_frames
        
        # Save spatial embeddings if available
        if include_spatial and final_spatial:
            spatial_zarr_path = output_dir / 'spatial_embeddings' / f'{seq_key}.zarr'
            spatial_zarr_path.parent.mkdir(parents=True, exist_ok=True)
            spatial_store = zarr.open(str(spatial_zarr_path), mode='w')
            
            for name, data in final_spatial.items():
                chunk_size = min(1024, data.shape[0])
                spatial_store[name] = zarr.array(data, chunks=(chunk_size, None) if len(data.shape) > 1 else (chunk_size,))
            
            # Same timestamps and metadata
            spatial_store['timestamps'] = zarr.array(timestamps, chunks=(min(1024, len(timestamps)),))
            spatial_store.attrs.update(core_store.attrs)
        
        # Memory cleanup
        del mat, final_embeddings, final_spatial, timestamps, core_embeddings, spatial_embeddings
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error processing {seq_key}: {e}")
        return False

def main():
    mp.set_start_method('spawn', force=True)
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SIMPLIFIED EMBEDDING EXTRACTION:")
    print(f"  - Core embeddings: Always extracted")
    print(f"  - Spatial embeddings: {'✓' if args.include_spatial else '✗'}")
    print(f"  - Timestamps: Always included")

    # Load data using original method
    print("Loading data using original approach...")
    mapping, order = build_test_split_original_exact(args.dvc_root, args.summary_csv)
    sequences = load_sequences_original_exact(args.dvc_root, mapping, order, args.skip_missing)

    # Load summary for timestamps
    summary_df = pd.read_csv(args.summary_csv)
    # take onyl the first 10 rows for testing
    # summary_df = summary_df.head(10)
    # summary_df = summary_df.head(500)
    summary_df = summary_df.iloc[1500:]
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])

    # Create or use Zarr store
    if args.force_zarr_rebuild or not Path(args.zarr_path).exists():
        zarr_store = create_optimized_zarr_store(sequences, args.zarr_path, args.chunk_size)
    else:
        print(f"Using existing Zarr store at {args.zarr_path}")
        zarr_store = zarr.open(args.zarr_path, mode='r')

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

    # Process each aggregation
    for num_frames, agg_name in tqdm(zip(args.time_aggregations, args.aggregation_names), 
                                     total=len(args.time_aggregations), desc="Processing aggregations"):
        print(f"\n=== SIMPLIFIED EXTRACTION: {agg_name} ({num_frames} frames) ===")
        
        model_args = {**base_model_args}
        model_args['input_size'] = [num_frames, 1, 12]
        model_args['num_frames'] = num_frames
        
        model = load_model_for_gpu_exact(ckpt_path, model_args, 0)
        
        # Create aggregation output directory
        agg_output_dir = output_dir / agg_name
        
        # Process all sequences
        success_count = 0
        for seq_key, _ in tqdm(sequences, desc=f"Processing sequences"):
            success = process_and_save_sequence(
                seq_key, zarr_store, model, num_frames, args.batch_size, 
                0, agg_output_dir, summary_df, args.include_spatial
            )
            if success:
                success_count += 1
        
        print(f"Successfully processed {success_count}/{len(sequences)} sequences")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nSimplified embedding extraction completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
