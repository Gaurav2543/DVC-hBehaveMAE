"""
Comprehensive hBehaveMAE feature extraction script
Extracts ALL available information from the trained model for downstream analysis
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
import h5py

# Import original data loading functions
from data_pipeline.load_dvc import load_dvc_data
from models import models_defs
from util import misc

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    p = argparse.ArgumentParser(description="Comprehensive hBehaveMAE Feature Extraction")
    p.add_argument("--dvc_root", required=True, type=str)
    p.add_argument("--summary_csv", required=True, type=str)
    p.add_argument("--ckpt_dir", required=True, type=str)
    p.add_argument("--ckpt_name", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--time_aggregations", nargs='+', type=int, required=True)
    p.add_argument("--aggregation_names", nargs='+', type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)  # Smaller for memory
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    p.add_argument("--skip_missing", action="store_true")
    
    # NEW: Comprehensive extraction options
    p.add_argument("--extract_attention", action="store_true", help="Extract attention weights")
    p.add_argument("--extract_spatial_features", action="store_true", help="Preserve spatial structure")
    p.add_argument("--extract_temporal_dynamics", action="store_true", help="Preserve temporal dynamics")
    p.add_argument("--extract_raw_features", action="store_true", help="Extract raw encoder outputs")
    p.add_argument("--extract_all", action="store_true", help="Extract all possible features")
    
    return p.parse_args()

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

def create_comprehensive_model_wrapper(model):
    """
    Wrap the model to capture ALL intermediate outputs
    """
    class ComprehensiveModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x, mask_ratio=0.0):
            """Extract comprehensive features"""
            # Store all intermediate outputs
            all_features = {}
            
            # Hook function to capture intermediate outputs
            def save_activation(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        all_features[name] = output[0].detach().cpu()
                    else:
                        all_features[name] = output.detach().cpu()
                return hook
            
            # Register hooks for all encoder blocks
            hooks = []
            for i, block in enumerate(self.base_model.blocks):
                hook = block.register_forward_hook(save_activation(f'encoder_block_{i}'))
                hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                # Get standard encoder output
                encoder_output, mask, intermediates = self.base_model.forward_encoder(
                    x, mask_ratio=mask_ratio, return_intermediates=True
                )
                
                # Store hierarchical outputs
                all_features['hierarchical_levels'] = intermediates
                all_features['final_encoder_output'] = encoder_output.detach().cpu()
                all_features['mask'] = mask.detach().cpu() if mask is not None else None
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return all_features
    
    return ComprehensiveModelWrapper(model)

def extract_comprehensive_features(batch, model_wrapper, extract_options):
    """
    Extract all possible features from a batch
    """
    device = batch.device
    B, C, T, H, W = batch.shape  # Batch, Channels, Time, Height, Width
    
    # Get comprehensive model outputs
    all_features = model_wrapper(batch, mask_ratio=0.0)
    
    comprehensive_results = {}
    
    # 1. HIERARCHICAL FEATURES (your current extraction)
    if 'hierarchical_levels' in all_features:
        hierarchical_levels = all_features['hierarchical_levels']
        comprehensive_results['hierarchical'] = {}
        
        for i, level_feat in enumerate(hierarchical_levels):
            level_name = f'level_{i+1}'
            
            if extract_options.get('temporal_dynamics', False):
                # Preserve temporal structure: pool only spatial dimensions
                # Shape: (B, T, H, W, C) -> (B, T, C)
                temporal_features = level_feat.flatten(2, -2).mean(-1)  # Pool spatial, keep temporal
                comprehensive_results['hierarchical'][f'{level_name}_temporal'] = temporal_features.numpy()
            
            if extract_options.get('spatial_features', False):
                # Preserve spatial structure: pool only temporal dimension
                # Shape: (B, T, H, W, C) -> (B, H, W, C)
                spatial_features = level_feat.mean(1)  # Pool temporal, keep spatial
                comprehensive_results['hierarchical'][f'{level_name}_spatial'] = spatial_features.numpy()
            
            if extract_options.get('spatio_temporal', True):
                # Full spatio-temporal features: minimal pooling
                # Shape: (B, T, H, W, C) -> (B, T*H*W, C) or (B, C, T*H*W)
                spatiotemporal_features = level_feat.flatten(1, -2)  # Flatten all except batch and feature dims
                comprehensive_results['hierarchical'][f'{level_name}_spatiotemporal'] = spatiotemporal_features.numpy()
            
            # Standard pooled features (your current approach)
            pooled_features = level_feat.flatten(1, -2).mean(1)  # Pool everything except batch and features
            comprehensive_results['hierarchical'][f'{level_name}_pooled'] = pooled_features.numpy()
    
    # 2. RAW ENCODER FEATURES
    if extract_options.get('raw_features', False):
        comprehensive_results['raw_encoder'] = {}
        
        for feature_name, feature_tensor in all_features.items():
            if feature_name.startswith('encoder_block_'):
                block_idx = feature_name.split('_')[-1]
                
                # Different pooling strategies for raw features
                if len(feature_tensor.shape) == 5:  # (B, T, H, W, C)
                    # Temporal preservation
                    temporal_raw = feature_tensor.flatten(2, -2).mean(-1)
                    comprehensive_results['raw_encoder'][f'block_{block_idx}_temporal'] = temporal_raw.numpy()
                    
                    # Spatial preservation  
                    spatial_raw = feature_tensor.mean(1)
                    comprehensive_results['raw_encoder'][f'block_{block_idx}_spatial'] = spatial_raw.numpy()
                    
                    # Full pooling
                    pooled_raw = feature_tensor.flatten(1, -2).mean(1)
                    comprehensive_results['raw_encoder'][f'block_{block_idx}_pooled'] = pooled_raw.numpy()
    
    # 3. ATTENTION PATTERNS (if model supports it)
    if extract_options.get('attention', False):
        comprehensive_results['attention'] = {}
        # This would require modifying the model to return attention weights
        # For now, we'll extract what we can from the available outputs
        
    # 4. ELECTRODE-SPECIFIC FEATURES
    if extract_options.get('electrode_specific', True):
        comprehensive_results['electrode_features'] = {}
        
        # Extract features for each electrode position in the 4x3 grid
        for i, level_feat in enumerate(all_features.get('hierarchical_levels', [])):
            if len(level_feat.shape) == 5:  # (B, T, H, W, C)
                # For each spatial position (electrode)
                for h in range(level_feat.shape[2]):  # Height (4 positions)
                    for w in range(level_feat.shape[3]):  # Width (3 positions)
                        electrode_key = f'level_{i+1}_electrode_{h}_{w}'
                        # Shape: (B, T, C)
                        electrode_features = level_feat[:, :, h, w, :].mean(1)  # Pool temporal
                        comprehensive_results['electrode_features'][electrode_key] = electrode_features.numpy()
    
    # 5. TEMPORAL DYNAMICS ANALYSIS
    if extract_options.get('temporal_analysis', True):
        comprehensive_results['temporal_analysis'] = {}
        
        for i, level_feat in enumerate(all_features.get('hierarchical_levels', [])):
            if len(level_feat.shape) == 5:  # (B, T, H, W, C)
                # Temporal derivatives (changes over time)
                level_temporal = level_feat.flatten(2, -2).mean(-1)  # (B, T, C)
                if level_temporal.shape[1] > 1:
                    temporal_diff = level_temporal[:, 1:] - level_temporal[:, :-1]  # First derivative
                    comprehensive_results['temporal_analysis'][f'level_{i+1}_temporal_diff'] = temporal_diff.numpy()
                
                # Temporal variance
                temporal_var = level_temporal.var(dim=1)  # Variance across time
                comprehensive_results['temporal_analysis'][f'level_{i+1}_temporal_var'] = temporal_var.numpy()
    
    return comprehensive_results

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

def process_sequence_batch_comprehensive(key_mat_batch, model, num_frames, batch_size, device_id, extract_options):
    """Process sequences with comprehensive feature extraction"""
    device = torch.device(f"cuda:{device_id}")
    
    # Create comprehensive model wrapper
    model_wrapper = create_comprehensive_model_wrapper(model)
    
    all_comprehensive_results = {}
    frame_map = {}
    ptr = 0
    
    for key, mat in tqdm(key_mat_batch, desc=f"GPU {device_id} comprehensive processing", leave=False):
        # Normalize exactly like original
        mat = mat / 100.0
        original_frames = mat.shape[0]
        
        if original_frames != num_frames:
            mat = aggregate_sequence_exact(mat, num_frames, aggregation_method='mean')
        
        n_frames = mat.shape[0]

        dataset = SlidingWindowDataset(mat, num_frames)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=0, pin_memory=False)
        
        # Initialize comprehensive storage for this sequence
        sequence_comprehensive_results = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dl, desc=f"GPU {device_id} batches", leave=False)):
                batch = batch.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                
                # Extract comprehensive features
                batch_comprehensive = extract_comprehensive_features(batch, model_wrapper, extract_options)
                
                # Accumulate features for this sequence
                for category, category_features in batch_comprehensive.items():
                    if category not in sequence_comprehensive_results:
                        sequence_comprehensive_results[category] = {}
                    
                    for feature_name, feature_array in category_features.items():
                        if feature_name not in sequence_comprehensive_results[category]:
                            sequence_comprehensive_results[category][feature_name] = []
                        sequence_comprehensive_results[category][feature_name].append(feature_array)
        
        # Concatenate all batches for this sequence
        for category in sequence_comprehensive_results:
            for feature_name in sequence_comprehensive_results[category]:
                sequence_comprehensive_results[category][feature_name] = np.concatenate(
                    sequence_comprehensive_results[category][feature_name], axis=0
                )
        
        # Store in overall results
        all_comprehensive_results[key] = sequence_comprehensive_results
        frame_map[key] = (ptr, ptr + n_frames)
        ptr += n_frames
        
        # Memory cleanup
        del sequence_comprehensive_results
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_comprehensive_results, frame_map

def save_comprehensive_results(all_results, frame_map, output_dir, agg_name):
    """Save comprehensive results in organized structure"""
    print(f"Saving comprehensive results for {agg_name}...")
    
    # Create directory structure
    agg_dir = output_dir / agg_name
    agg_dir.mkdir(exist_ok=True)
    
    # Save frame mapping
    frame_map_path = agg_dir / 'frame_map.npy'
    np.save(frame_map_path, frame_map)
    
    # Process and save each category of features
    for sequence_key, sequence_results in all_results.items():
        for category, category_features in sequence_results.items():
            category_dir = agg_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for feature_name, feature_array in category_features.items():
                feature_path = category_dir / f'{feature_name}.npy'
                np.save(feature_path, feature_array)
    
    # Create comprehensive index
    create_comprehensive_index(all_results, agg_dir)
    
    print(f"Comprehensive results saved to {agg_dir}")

def create_comprehensive_index(all_results, output_dir):
    """Create an index of all extracted features"""
    index_data = {
        'extraction_timestamp': datetime.datetime.now().isoformat(),
        'sequences': list(all_results.keys()),
        'feature_categories': {},
        'total_features_extracted': 0
    }
    
    # Analyze structure
    for seq_key, seq_results in all_results.items():
        for category, category_features in seq_results.items():
            if category not in index_data['feature_categories']:
                index_data['feature_categories'][category] = {}
            
            for feature_name, feature_array in category_features.items():
                if feature_name not in index_data['feature_categories'][category]:
                    index_data['feature_categories'][category][feature_name] = {
                        'shape': feature_array.shape,
                        'dtype': str(feature_array.dtype),
                        'description': get_feature_description(category, feature_name)
                    }
                index_data['total_features_extracted'] += 1
    
    # Save index
    index_path = output_dir / 'comprehensive_feature_index.npy'
    np.save(index_path, index_data)
    
    # Also save human-readable version
    readable_path = output_dir / 'feature_summary.txt'
    with open(readable_path, 'w') as f:
        f.write("COMPREHENSIVE FEATURE EXTRACTION SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Extraction timestamp: {index_data['extraction_timestamp']}\n")
        f.write(f"Number of sequences: {len(index_data['sequences'])}\n")
        f.write(f"Total features extracted: {index_data['total_features_extracted']}\n\n")
        
        for category, features in index_data['feature_categories'].items():
            f.write(f"\nCATEGORY: {category.upper()}\n")
            f.write("-" * 30 + "\n")
            for feature_name, feature_info in features.items():
                f.write(f"  {feature_name}:\n")
                f.write(f"    Shape: {feature_info['shape']}\n")
                f.write(f"    Type: {feature_info['dtype']}\n")
                f.write(f"    Description: {feature_info['description']}\n\n")

def get_feature_description(category, feature_name):
    """Generate human-readable descriptions of features"""
    descriptions = {
        'hierarchical': {
            'temporal': 'Hierarchical features preserving temporal dynamics',
            'spatial': 'Hierarchical features preserving spatial electrode arrangement',
            'spatiotemporal': 'Full spatio-temporal hierarchical features',
            'pooled': 'Standard pooled hierarchical features (current approach)'
        },
        'raw_encoder': {
            'temporal': 'Raw encoder outputs with temporal structure preserved',
            'spatial': 'Raw encoder outputs with spatial structure preserved', 
            'pooled': 'Standard pooled raw encoder outputs'
        },
        'electrode_features': 'Features specific to individual electrode positions',
        'temporal_analysis': {
            'temporal_diff': 'Temporal derivatives showing changes over time',
            'temporal_var': 'Temporal variance showing activity patterns'
        }
    }
    
    for desc_category, desc_features in descriptions.items():
        if category == desc_category:
            if isinstance(desc_features, dict):
                for desc_key, desc_text in desc_features.items():
                    if desc_key in feature_name:
                        return desc_text
            else:
                return desc_features
    
    return f"Feature from {category} category"

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
        'electrode_specific': args.extract_all,  # Always extract for DVC data
        'temporal_analysis': args.extract_all,
        'spatio_temporal': True  # Always extract comprehensive features
    }
    
    print("COMPREHENSIVE FEATURE EXTRACTION ENABLED:")
    for option, enabled in extract_options.items():
        print(f"  - {option}: {'✓' if enabled else '✗'}")

    # Load data using original method
    print("Using ORIGINAL data loading approach...")
    mapping, order = build_test_split_original_exact(args.dvc_root, args.summary_csv)
    sequences = load_sequences_original_exact(args.dvc_root, mapping, order, args.skip_missing)

    # Setup GPU
    gpu_ids = [0]  # Single GPU for now
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

    # Process each aggregation with comprehensive extraction
    for num_frames, agg_name in tqdm(zip(args.time_aggregations, args.aggregation_names), 
                                     total=len(args.time_aggregations), desc="Processing aggregations"):
        print(f"\n=== COMPREHENSIVE EXTRACTION: {agg_name} aggregation ({num_frames} frames) ===")
        
        model_args = {**base_model_args}
        model_args['input_size'] = [num_frames, 1, 12]
        model_args['num_frames'] = num_frames
        
        model = load_model_for_gpu_exact(ckpt_path, model_args, gpu_ids[0])
        
        # Comprehensive feature extraction
        all_results, frame_map = process_sequence_batch_comprehensive(
            sequences, model, num_frames, args.batch_size, gpu_ids[0], extract_options
        )
        
        # Save comprehensive results
        save_comprehensive_results(all_results, frame_map, output_dir, agg_name)
        
        del model, all_results
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n🎉 COMPREHENSIVE EXTRACTION COMPLETED!")
    print(f"All features saved to: {output_dir}")
    print(f"Check 'feature_summary.txt' in each aggregation folder for details")

if __name__ == "__main__":
    main()