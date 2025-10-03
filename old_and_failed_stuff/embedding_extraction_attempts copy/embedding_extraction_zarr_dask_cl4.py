"""
Memory-efficient hBehaveMAE embedding extractor using Zarr streaming
Matches exact training behavior with positional embedding interpolation
"""
import argparse, os, time, datetime, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from iopath.common.file_io import g_pathmgr as pathmgr
from models import models_defs
from util.pos_embed import interpolate_pos_embed
import zarr
import json

# ----------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    # Data arguments
    p.add_argument("--dvc_root", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", required=True)
    p.add_argument("--output_dir", required=True)
    
    # Model architecture arguments (must match training)
    p.add_argument("--model", default="hbehavemae")
    p.add_argument("--input_size", nargs=3, type=int, default=[1440, 1, 12])
    p.add_argument("--stages", nargs='+', type=int, default=[2, 3, 4])
    p.add_argument("--q_strides", default="4,1,1;3,1,1;2,1,1;3,1,1")
    p.add_argument("--mask_unit_attn", nargs='+', type=lambda x: x.lower()=='true',
                   default=[True, False, False])
    p.add_argument("--patch_kernel", nargs=3, type=int, default=[2, 1, 12])
    p.add_argument("--init_embed_dim", type=int, default=96)
    p.add_argument("--init_num_heads", type=int, default=2)
    p.add_argument("--out_embed_dims", nargs='+', type=int, default=[64, 96, 128])
    p.add_argument("--decoding_strategy", default="single")
    p.add_argument("--decoder_embed_dim", type=int, default=128)
    p.add_argument("--decoder_depth", type=int, default=1)
    p.add_argument("--decoder_num_heads", type=int, default=1)
    
    # Extraction arguments
    p.add_argument("--trained_num_frames", type=int, required=True,
                   help="Number of frames model was trained on (e.g., 4320 or 40320)")
    p.add_argument("--aggregation_name", type=str, required=True,
                   help="Name for this aggregation (e.g., '3days' or '4weeks')")
    
    # Processing arguments
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--skip_missing", action="store_true")
    
    return p.parse_args()

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

# ----------------------------------------------------------------------------------
class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

class StreamingValidator:
    """Validates extraction process"""
    
    def __init__(self, expected_frames: int, aggregation_name: str):
        self.expected_frames = expected_frames
        self.aggregation_name = aggregation_name
        self.errors = []
        self.warnings = []
        
    def validate_sequence_length(self, key: str, actual_frames: int, allow_off_by_one: bool = True):
        """Check if sequence has expected length (with off-by-one tolerance)"""
        if actual_frames == self.expected_frames:
            return True
        
        if allow_off_by_one and actual_frames == self.expected_frames - 1:
            self.warnings.append(f"{key}: {actual_frames} frames (off by 1, will pad)")
            return True
        
        self.errors.append(f"{key}: {actual_frames} frames != {self.expected_frames}")
        return False
    
    def validate_embeddings(self, key: str, embeddings: np.ndarray):
        """Check embedding validity"""
        if np.any(np.isnan(embeddings)):
            self.errors.append(f"{key}: contains NaN")
            return False
        if np.any(np.isinf(embeddings)):
            self.errors.append(f"{key}: contains Inf")
            return False
        if embeddings.shape[0] != self.expected_frames:
            self.errors.append(f"{key}: shape {embeddings.shape[0]} != {self.expected_frames}")
            return False
        return True
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY: {self.aggregation_name}")
        print(f"{'='*60}")
        if self.warnings:
            print(f"Warnings: {len(self.warnings)}")
            for w in self.warnings[:3]:
                print(f"  {w}")
        if self.errors:
            print(f"Errors: {len(self.errors)}")
            for e in self.errors[:3]:
                print(f"  {e}")
        else:
            print("All validations passed!")
        print(f"{'='*60}\n")

# ----------------------------------------------------------------------------------
def load_dvc_data_by_ranges(folder_path, cage_ranges):
    """Load DVC data using date ranges"""
    data_concat = {}
    
    for cage, date_ranges in tqdm(cage_ranges.items(), desc="Loading cage data"):
        path = os.path.join(folder_path, f"{cage}.csv")
        if not os.path.exists(path):
            continue
            
        df = pd.read_csv(path, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        cage_data_list = []
        for start_date, end_date in date_ranges:
            mask = (df["timestamp"] >= start_date) & (df["timestamp"] < end_date)
            range_df = df[mask].copy()
            if len(range_df) > 0:
                cage_data_list.append(range_df)
        
        if cage_data_list:
            data_concat[cage] = pd.concat(cage_data_list, ignore_index=True)
    
    return data_concat

def load_sequences_matching_training(root, csv_path, trained_num_frames, skip_missing=True):
    """Load sequences matching training with proper padding"""
    summary = pd.read_csv(csv_path, low_memory=False)
    # take only 100 rows for testing
    summary = summary.head(100) 
    summary.columns = summary.columns.str.strip()
    summary["from_tpt"] = pd.to_datetime(summary["from_tpt"])
    summary["to_tpt"] = pd.to_datetime(summary["to_tpt"])
    
    cage_ranges = {}
    for _, row in summary.iterrows():
        cage_id = row["cage_id"]
        if cage_id not in cage_ranges:
            cage_ranges[cage_id] = []
        cage_ranges[cage_id].append((row["from_tpt"], row["to_tpt"]))
    
    print(f"Loading data for {len(cage_ranges)} cages...")
    raw_data = load_dvc_data_by_ranges(root, cage_ranges)
    
    seqs = []
    stats = {
        "total": 0,
        "loaded": 0,
        "padded": 0,
        "incomplete": 0,
        "missing_cols": 0,
        "cage_missing": 0
    }
    
    for _, row in summary.iterrows():
        stats["total"] += 1
        cage_id = row["cage_id"]
        from_tpt = row["from_tpt"]
        to_tpt = row["to_tpt"]
        
        if cage_id not in raw_data:
            stats["cage_missing"] += 1
            continue
        
        df = raw_data[cage_id]
        mask = (df["timestamp"] >= from_tpt) & (df["timestamp"] < to_tpt)
        chunk_df = df[mask]
        
        missing = [c for c in ELEC_COLS if c not in chunk_df.columns]
        if missing:
            stats["missing_cols"] += 1
            continue
        
        data = chunk_df[ELEC_COLS].values.astype(np.float32)
        actual_frames = data.shape[0]
        
        # Match training: exact or pad if off-by-one
        if actual_frames == trained_num_frames:
            pass  # Perfect match
        elif actual_frames == trained_num_frames - 1:
            # Pad to match training
            data = np.pad(data, ((0, 1), (0, 0)), mode='edge')
            stats["padded"] += 1
            # CRITICAL: Update actual_frames after padding
            actual_frames = trained_num_frames
        else:
            # Incomplete
            stats["incomplete"] += 1
            if not skip_missing:
                print(f"[SKIP] {cage_id}_{from_tpt.strftime('%Y-%m-%d')}: "
                      f"{actual_frames} != {trained_num_frames}")
            continue
        
        # Final validation
        if data.shape[0] != trained_num_frames:
            print(f"[ERROR] {cage_id}: data shape {data.shape[0]} != {trained_num_frames} after processing")
            stats["incomplete"] += 1
            continue
        
        key = f"{cage_id}_{from_tpt.strftime('%Y-%m-%d')}"
        seqs.append((key, data))
        stats["loaded"] += 1
        
        if stats["loaded"] % 100 == 0:
            print(f"[INFO] Loaded {stats['loaded']} sequences...")
    
    print(f"\n{'='*60}")
    print("LOADING SUMMARY")
    print(f"{'='*60}")
    print(f"Total sequences: {stats['total']}")
    print(f"Successfully loaded: {stats['loaded']}")
    print(f"  - Padded (off-by-1): {stats['padded']}")
    print(f"Skipped incomplete: {stats['incomplete']}")
    print(f"Success rate: {stats['loaded']/stats['total']*100:.1f}%")
    print(f"{'='*60}\n")
    
    # Verify all sequences have correct length
    print("Verifying sequence lengths...")
    for key, data in seqs[:5]:
        print(f"  {key}: {data.shape[0]} frames")
    
    return seqs

def parse_q_strides(q_strides_str):
    """Parse q_strides string"""
    if isinstance(q_strides_str, str):
        stages = q_strides_str.split(';')
        parsed = [[int(x) for x in stage.split(',')] for stage in stages]
        return parsed, len(parsed) + 1
    return q_strides_str, len(q_strides_str) + 1

def load_model_with_interpolation(ckpt_path, model_args, device_id):
    """Load model with positional embedding interpolation"""
    device = torch.device(f"cuda:{device_id}")
    
    print(f"Loading model on GPU {device_id}")
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu", weights_only=False)
    
    model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
    
    # Load weights with interpolation for positional embeddings
    model_dict = model.state_dict()
    pretrained_dict = {}
    skipped = []
    
    for k, v in ckpt["model"].items():
        if k not in model_dict:
            skipped.append(f"{k} (not in model)")
            continue
        
        # Handle positional embeddings with interpolation
        if 'pos_embed' in k and v.shape != model_dict[k].shape:
            print(f"Interpolating {k}: {v.shape} → {model_dict[k].shape}")
            try:
                v = interpolate_pos_embed(v, model_dict[k])
            except Exception as e:
                print(f"  Interpolation failed: {e}, skipping")
                skipped.append(f"{k} (interpolation failed)")
                continue
        
        if v.shape == model_dict[k].shape:
            pretrained_dict[k] = v
        else:
            skipped.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"Loaded {len(pretrained_dict)}/{len(ckpt['model'])} layers")
    if skipped:
        print(f"Skipped {len(skipped)} layers:")
        for s in skipped[:5]:
            print(f"  - {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")
    
    return model.to(device).eval().requires_grad_(False)

class StreamingEmbeddingDataset:
    """Memory-efficient dataset for sliding windows"""
    
    def __init__(self, mat, num_frames):
        self.mat = mat
        self.num_frames = num_frames
        self.pad = (num_frames - 1) // 2
        self.padded = np.pad(mat, ((self.pad, self.pad), (0, 0)), "edge")
        self.length = mat.shape[0]  # THIS IS THE BUG
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        window = self.padded[idx:idx + self.num_frames]
        if window.shape[0] != self.num_frames:
            raise ValidationError(f"Window size mismatch: {window.shape[0]} != {self.num_frames}")
        return torch.from_numpy(window.copy()).unsqueeze(0).unsqueeze(2)

def process_sequence_streaming(key, mat, model, num_frames, batch_size, device_id, 
                               num_levels, validator, chunk_size):
    """Process sequence with streaming"""
    device = torch.device(f"cuda:{device_id}")
    
    # Normalize
    mat = mat / 100.0
    
    # CRITICAL: Verify exact length BEFORE creating dataset
    if mat.shape[0] != num_frames:
        raise ValidationError(f"{key}: Input must be exactly {num_frames} frames, got {mat.shape[0]}")
    
    # Create dataset (now guaranteed to have correct length)
    dataset = StreamingEmbeddingDataset(mat, num_frames)
    level_seqs = [[] for _ in range(num_levels)]
    
    # Process in chunks
    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        
        chunk_data = []
        for idx in range(chunk_start, chunk_end):
            try:
                chunk_data.append(dataset[idx])
            except ValidationError as e:
                validator.errors.append(f"{key}: {e}")
                continue
        
        if not chunk_data:
            continue
        
        batch = torch.stack(chunk_data).to(device, dtype=torch.float32, non_blocking=True)
        
        with torch.no_grad():
            try:
                outputs = model.forward_encoder(batch, mask_ratio=0.0, return_intermediates=True)
                levels = outputs[-1]
                
                for i, level_feat in enumerate(levels):
                    pooled = level_feat.flatten(1, -2).mean(1).float().cpu()
                    level_seqs[i].append(pooled)
                    
            except Exception as e:
                validator.errors.append(f"{key} forward error: {e}")
                return None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate results
    results = {}
    for i in range(num_levels):
        level_name = f"level{i+1}"
        if level_seqs[i]:
            emb = torch.cat(level_seqs[i]).numpy().astype(np.float16)
            if not validator.validate_embeddings(key, emb):
                return None
            results[level_name] = emb
        else:
            return None
    
    # Combined embedding
    if all(results[f"level{i+1}"].size > 0 for i in range(num_levels)):
        results['comb'] = np.concatenate([results[f"level{i+1}"] for i in range(num_levels)], axis=1)
    else:
        return None
    
    return results

def extract_embeddings_to_zarr(sequences, model, num_frames, batch_size, device_id,
                               num_levels, aggregation_name, output_dir, chunk_size):
    """Extract embeddings and stream to zarr"""
    
    validator = StreamingValidator(num_frames, aggregation_name)
    
    print("Determining embedding dimensions...")
    test_result = process_sequence_streaming(
        sequences[0][0], sequences[0][1], model, num_frames, batch_size,
        device_id, num_levels, validator, chunk_size
    )
    
    if test_result is None:
        raise ValidationError(f"Failed to process test sequence. Errors: {validator.errors}")
    
    embed_dims = {level: emb.shape[1] for level, emb in test_result.items()}
    print(f"Embedding dimensions: {embed_dims}")
    
    # Initialize zarr stores
    zarr_stores = {}
    frame_maps = {}
    
    for level_name, dim in embed_dims.items():
        zarr_path = os.path.join(output_dir, f"{aggregation_name}_{level_name}.zarr")
        zarr_stores[level_name] = zarr.open(
            zarr_path, mode='w',
            shape=(len(sequences) * num_frames, dim),
            chunks=(num_frames, dim),
            dtype='float16'
        )
        frame_maps[level_name] = {}
    
    # Process all sequences
    current_ptr = 0
    successful = 0
    
    for key, mat in tqdm(sequences, desc=f"Extracting {aggregation_name}"):
        result = process_sequence_streaming(
            key, mat, model, num_frames, batch_size,
            device_id, num_levels, validator, chunk_size
        )
        
        if result is None:
            continue
        
        # Write to zarr
        for level_name, emb in result.items():
            zarr_stores[level_name][current_ptr:current_ptr + num_frames] = emb
            frame_maps[level_name][key] = (current_ptr, current_ptr + num_frames)
        
        current_ptr += num_frames
        successful += 1
    
    # Trim arrays
    for level_name in zarr_stores:
        zarr_stores[level_name].resize(current_ptr, embed_dims[level_name])
    
    validator.print_summary()
    print(f"Successfully processed {successful}/{len(sequences)} sequences")
    
    return zarr_stores, frame_maps, embed_dims

def main():
    mp.set_start_method('spawn', force=True)
    
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"EMBEDDING EXTRACTION: {args.aggregation_name}")
    print(f"Trained on: {args.trained_num_frames} frames")
    print(f"{'='*80}\n")
    
    # Load sequences matching training
    sequences = load_sequences_matching_training(
        args.dvc_root, args.summary_csv, args.trained_num_frames, args.skip_missing
    )
    
    if not sequences:
        raise ValueError("No valid sequences found!")
    
    # Load checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu", weights_only=False)
    base_model_args = ckpt.get("args", {})
    if isinstance(base_model_args, argparse.Namespace):
        base_model_args = vars(base_model_args)
    
    # Parse q_strides
    if 'q_strides' in base_model_args:
        base_model_args['q_strides'], num_levels = parse_q_strides(base_model_args['q_strides'])
    else:
        base_model_args['q_strides'], num_levels = parse_q_strides(args.q_strides)
    
    print(f"Model has {num_levels} hierarchical levels")
    
    # Create model args
    model_args = {**base_model_args}
    model_args['input_size'] = [args.trained_num_frames, 1, 12]
    model_args['num_frames'] = args.trained_num_frames
    
    # Load model
    model = load_model_with_interpolation(ckpt_path, model_args, args.gpu_id)
    
    # Extract embeddings
    zarr_stores, frame_maps, embed_dims = extract_embeddings_to_zarr(
        sequences, model, args.trained_num_frames, args.batch_size, args.gpu_id,
        num_levels, args.aggregation_name, args.output_dir, args.chunk_size
    )
    
    # Save frame maps
    for level_name, frame_map in frame_maps.items():
        map_path = os.path.join(args.output_dir, f"{args.aggregation_name}_{level_name}_framemap.npy")
        np.save(map_path, {'frame_number_map': frame_map})
    
    # Save metadata
    metadata = {
        'extraction_args': vars(args),
        'model_args': base_model_args,
        'num_levels': num_levels,
        'num_frames': args.trained_num_frames,
        'aggregation_name': args.aggregation_name,
        'embedding_dims': embed_dims,
        'successful_sequences': len(next(iter(frame_maps.values()))),
        'total_sequences': len(sequences),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(args.output_dir, f'{args.aggregation_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Metadata: {metadata_path}")
    for level, path in {level: os.path.join(args.output_dir, f"{args.aggregation_name}_{level}.zarr") 
                        for level in embed_dims.keys()}.items():
        print(f"{level}: {path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()