"""
FINAL CORRECTED VERSION: Uses original script's exact data loading approach
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

# Import the original data loading function
from data_pipeline.load_dvc import load_dvc_data
from models import models_defs
from util import misc

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    p = argparse.ArgumentParser(description="CORRECTED Optimized Script - Original Data Loading")
    p.add_argument("--dvc_root", required=True, type=str)
    p.add_argument("--summary_csv", required=True, type=str)
    p.add_argument("--ckpt_dir", required=True, type=str)
    p.add_argument("--ckpt_name", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--zarr_path", required=True, type=str)
    p.add_argument("--time_aggregations", nargs='+', type=int, required=True)
    p.add_argument("--aggregation_names", nargs='+', type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--force_zarr_rebuild", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    p.add_argument("--skip_missing", action="store_true")
    
    return p.parse_args()

def build_test_split_original_exact(root, csv):
    """Use EXACT same logic as original script"""
    summary_df = pd.read_csv(csv)
    # summary_df = summary_df.head(10)  # Match original limit
    
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
    """Use EXACT same logic as original script including load_dvc_data"""
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

def create_zarr_from_original_sequences(sequences, zarr_path):
    """Create Zarr store from sequences loaded by original method"""
    print("Creating Zarr store from original data loading method...")
    
    root = zarr.open(zarr_path, mode='w')
    
    for seq_key, seq_data in tqdm(sequences, desc="Storing sequences in Zarr"):
        # Store sequence
        data_array = root.create_dataset(
            seq_key,
            data=seq_data,
            chunks=(min(65536, seq_data.shape[0]), None),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        )
        data_array.attrs['description'] = f"Sequence data for {seq_key}"
        data_array.attrs['original_shape'] = seq_data.shape
    
    print(f"Zarr store created with {len(sequences)} sequences")

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

def process_sequence_batch_exact(key_mat_batch, model, num_frames, batch_size, device_id, num_levels):
    """Process sequences exactly like original"""
    device = torch.device(f"cuda:{device_id}")
    
    results = {}
    level_embeddings = [[] for _ in range(num_levels)]
    frame_map = {}
    ptr = 0
    
    for key, mat in tqdm(key_mat_batch, desc=f"GPU {device_id} processing batch", leave=False):
        # Normalize exactly like original
        mat = mat / 100.0
        original_frames = mat.shape[0]
        
        if original_frames != num_frames:
            mat = aggregate_sequence_exact(mat, num_frames, aggregation_method='mean')
        
        n_frames = mat.shape[0]

        dataset = SlidingWindowDataset(mat, num_frames)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=0, pin_memory=False)
        
        level_seqs = [[] for _ in range(num_levels)]
        
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"GPU {device_id} batches", leave=False):
                batch = batch.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                
                outputs = model.forward_encoder(batch, mask_ratio=0.0, return_intermediates=True)
                levels = outputs[-1]
                
                def pool(feat):
                    return feat.flatten(1, -2).mean(1).float().cpu()
                
                for i, level_feat in enumerate(levels):
                    level_seqs[i].append(pool(level_feat))
        
        for i in range(num_levels):
            level_seq = torch.cat(level_seqs[i]).numpy().astype(np.float16)
            level_embeddings[i].append(level_seq)
        
        frame_map[key] = (ptr, ptr + n_frames)
        ptr += n_frames
    
    results['frame_map'] = frame_map
    
    for i in range(num_levels):
        level_name = f"level{i+1}"
        if level_embeddings[i]:
            results[level_name] = np.concatenate(level_embeddings[i], axis=0)
        else:
            results[level_name] = np.array([]).astype(np.float16)
    
    if all(results[f"level{i+1}"].size > 0 for i in range(num_levels)):
        combined_embedding = np.concatenate([results[f"level{i+1}"] for i in range(num_levels)], axis=1)
        results['comb'] = combined_embedding
    else:
        results['comb'] = np.array([]).astype(np.float16)
    
    return results

def main():
    mp.set_start_method('spawn', force=True)
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data using ORIGINAL method
    print("Using ORIGINAL data loading approach...")
    mapping, order = build_test_split_original_exact(args.dvc_root, args.summary_csv)
    print(f"ORIGINAL: Found {len(mapping)} cages: {list(mapping.keys())}")
    print(f"ORIGINAL: Total sequences: {len(order)}")
    sequences = load_sequences_original_exact(args.dvc_root, mapping, order, args.skip_missing)
    
    # Create or use Zarr store
    if args.force_zarr_rebuild or not Path(args.zarr_path).exists():
        create_zarr_from_original_sequences(sequences, args.zarr_path)
    else:
        print(f"Using existing Zarr store at {args.zarr_path}")
        # Load from Zarr instead
        zarr_store = zarr.open(args.zarr_path, mode='r')
        sequence_keys = [k for k in zarr_store.keys()]
        sequences = []
        for seq_key in sequence_keys:
            seq_data = zarr_store[seq_key][:]
            sequences.append((seq_key, seq_data))

    # Setup GPU
    if args.multi_gpu and torch.cuda.is_available():
        gpu_ids = args.gpu_ids if args.gpu_ids else list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [0]
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    # Load checkpoint to get base model args EXACTLY like original
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
    
    print(f"Model will have {num_levels} hierarchical levels based on q_strides: {base_model_args['q_strides']}")

    # Validation
    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("time_aggregations and aggregation_names must have same length")

    # Process each aggregation
    all_results = {}
    
    for num_frames, agg_name in tqdm(zip(args.time_aggregations, args.aggregation_names), 
                                     total=len(args.time_aggregations), desc="Processing aggregations"):
        print(f"\n=== Processing {agg_name} aggregation ({num_frames} frames) ===")
        
        model_args = {**base_model_args}
        model_args['input_size'] = [num_frames, 1, 12]
        model_args['num_frames'] = num_frames
        
        print(f"Creating model with input_size: {model_args['input_size']}")
        
        model = load_model_for_gpu_exact(ckpt_path, model_args, gpu_ids[0])
        results = process_sequence_batch_exact(sequences, model, num_frames, 
                                             args.batch_size, gpu_ids[0], num_levels)
        del model
        torch.cuda.empty_cache()
        
        level_names = [f"level{i+1}" for i in range(num_levels)] + ['comb']
        
        for level_name in level_names:
            filename = f"test_{agg_name}_{level_name}.npy"
            filepath = os.path.join(args.output_dir, filename)
            save_data = {
                'frame_number_map': results['frame_map'],
                'embeddings': results[level_name],
                'aggregation_info': {
                    'num_frames': num_frames,
                    'aggregation_name': agg_name,
                    'level': level_name,
                    'shape': results[level_name].shape,
                    'total_levels': num_levels
                }
            }
            np.save(filepath, save_data)
            print(f"Saved {filename} – shape {results[level_name].shape}")
        
        all_results[agg_name] = results

    # Save metadata
    metadata = {
        'extraction_args': vars(args),
        'base_model_args': base_model_args,
        'num_hierarchical_levels': num_levels,
        'gpu_ids_used': gpu_ids,
        'multiprocessing_used': len(gpu_ids) > 1,
        'data_loading_method': 'original_load_dvc_data',
        'aggregation_levels': {
            name: {
                'num_frames': frames,
                'embedding_shapes': {
                    **{f'level{i+1}': all_results[name][f'level{i+1}'].shape for i in range(num_levels)},
                    'comb': all_results[name]['comb'].shape
                }
            }
            for name, frames in zip(args.aggregation_names, args.time_aggregations)
        },
        'total_sequences': len(sequences),
        'extraction_timestamp': datetime.datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(args.output_dir, 'extraction_metadata.npy')
    np.save(metadata_path, metadata)
    
    print(f"\n✓ FINAL CORRECTED extraction completed using original data loading!")

if __name__ == "__main__":
    main()