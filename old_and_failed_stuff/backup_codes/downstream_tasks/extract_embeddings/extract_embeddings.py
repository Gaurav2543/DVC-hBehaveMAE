"""
hBehaveMAE embedding extractor with multi-level time aggregation
Extracts embeddings at different time scales to capture aging patterns
"""
import argparse, os, time, datetime, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from iopath.common.file_io import g_pathmgr as pathmgr

from data_pipeline.load_dvc import load_dvc_data
from models import models_defs
from util import misc
from util.pos_embed import interpolate_pos_embed


# ----------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    # Data arguments
    # p.add_argument("--dvc_root",      default="/scratch/bhole/dvc_data/smoothed/cage_activations_wo_nan")
    # p.add_argument("--summary_csv",
    #                default="/scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv")
    # p.add_argument("--ckpt_name",     default="checkpoint-00050.pth")
    p.add_argument("--dvc_root",      required=True)
    p.add_argument("--summary_csv",   required=True)
    p.add_argument("--ckpt_dir",      required=True)
    p.add_argument("--ckpt_name",     required=True)
    p.add_argument("--output_dir",    required=True)

    # Model architecture arguments (should match training)
    p.add_argument("--model",         default="hbehavemae")
    p.add_argument("--input_size",    nargs=3, type=int, default=[1440, 1, 12])
    p.add_argument("--stages",        nargs='+', type=int, default=[2, 3, 4])
    p.add_argument("--q_strides",     default="15,1,1;6,1,1")
    p.add_argument("--mask_unit_attn", nargs='+', type=lambda x: x.lower()=='true', 
                   default=[True, False, False])
    p.add_argument("--patch_kernel",  nargs=3, type=int, default=[2, 1, 12])
    p.add_argument("--init_embed_dim", type=int, default=96)
    p.add_argument("--init_num_heads", type=int, default=2)
    p.add_argument("--out_embed_dims", nargs='+', type=int, default=[64, 96, 128])
    p.add_argument("--decoding_strategy", default="single")
    p.add_argument("--decoder_embed_dim", type=int, default=128)
    p.add_argument("--decoder_depth", type=int, default=1)
    p.add_argument("--decoder_num_heads", type=int, default=1)

    # Multi-level extraction arguments
    p.add_argument("--time_aggregations", nargs='+', type=int, 
                   default=[90, 360, 720, 1440, 2160, 2880, 4320, 10080],  # 1.5h, 6h, 1day, 3days, 1week
                   help="Different time aggregation levels (in minutes)")
    p.add_argument("--aggregation_names", nargs='+', type=str,
                   default=["1.5h", "6h", "12h", "1day", "1.5days", "2days", "3days", "1week"],
                   help="Names for each aggregation level")

    # Processing arguments
    p.add_argument("--batch_size",    type=int, default=4096)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--num_workers",   type=int, default=8)
    p.add_argument("--skip_missing",  action="store_true")
    
    return p.parse_args()

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

# ----------------------------------------------------------------------------------
def build_test_split(root, csv):
    df = pd.read_csv(csv, low_memory=False)
    df = df.head(100)
    df.columns = df.columns.str.strip()
    # print(f"CSV columns: {list(df.columns)}")
    # print(f"Total rows in CSV: {len(df)}")
    
    mapping, order = {}, []
    for _, r in df.iterrows():
        day = str(r["from_tpt"]).split()[0]
        mapping.setdefault(r["cage_id"], []).append(day)
        order.append(f"{r['cage_id']}_{day}")
    
    print(f"Built mapping for {len(mapping)} cages, {len(order)} total sequences")
    return mapping, order


def load_sequences(root, mapping, order, skip_missing):
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


def parse_q_strides(q_strides_str):
    """Parse q_strides string into proper format"""
    if isinstance(q_strides_str, str):
        # Parse "15,1,1;16,1,1" into [[15,1,1], [16,1,1]]
        stages = q_strides_str.split(';')
        return [[int(x) for x in stage.split(',')] for stage in stages]
    return q_strides_str

# def load_modecl
def load_model(ckpt_dir, ckpt_name, device, model_args):
    path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(path):
        if os.path.exists(os.path.join(ckpt_dir, "checkpoint-best.pth")):
            path = os.path.join(ckpt_dir, "checkpoint-best.pth")
        else:
            # Find the latest checkpoint manually
            checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint-') and f.endswith('.pth')]
            if checkpoint_files:
                # Sort by modification time and get the latest
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
                path = os.path.join(ckpt_dir, checkpoint_files[0])
            else:
                path = None
    if not path:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    print("Loading checkpoint", path)
    ckpt = torch.load(pathmgr.open(path, "rb"), map_location="cpu")
    tr_args = ckpt.get("args", {})
    if isinstance(tr_args, argparse.Namespace):
        tr_args = vars(tr_args)

    # Use checkpoint args as base, override with provided model args
    final_args = {**tr_args}
    
    # Only override with model_args if they are explicitly provided (not default values)
    # This prevents overriding trained model parameters with default extraction parameters
    for key, value in model_args.items():
        if key in final_args and value is not None:
            # Only override if it's a non-default value or specifically set
            final_args[key] = value
    
    # Parse q_strides if it's a string
    if 'q_strides' in final_args:
        final_args['q_strides'] = parse_q_strides(final_args['q_strides'])
    
    print(f"Final model arguments: {final_args}")
    
    model = models_defs.__dict__[final_args.get("model", "hbehavemae")](**final_args)
    interpolate_pos_embed(model, ckpt["model"])
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval().requires_grad_(False)  # ONLY RETURN THE MODEL

def aggregate_sequence(sequence, target_frames, aggregation_method='mean'):
    """Aggregate sequence to target number of frames"""
    current_frames = sequence.shape[0]
    
    if current_frames == target_frames:
        return sequence
    elif current_frames < target_frames:
        # Pad with edge values if we need more frames
        pad_needed = target_frames - current_frames
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        return np.pad(sequence, ((pad_before, pad_after), (0, 0)), 'edge')
    else:
        # Downsample if we have more frames than needed
        if aggregation_method == 'mean':
            # Reshape and average
            excess = current_frames % target_frames
            if excess != 0:
                # Trim to make it divisible
                sequence = sequence[:current_frames - excess]
            group_size = sequence.shape[0] // target_frames
            reshaped = sequence.reshape(target_frames, group_size, -1)
            return reshaped.mean(axis=1)
        elif aggregation_method == 'subsample':
            # Simple subsampling
            indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
            return sequence[indices]
    
    return sequence


def extract_embeddings_for_aggregation(model, sequences, num_frames, batch_size, device, num_workers):
    """Extract embeddings for a specific time aggregation"""
    print(f"Extracting embeddings for {num_frames} frames...")
    
    frame_map, ptr = {}, 0
    LOW, MID, HIGH, COMB = [], [], [], []
    pad = (num_frames - 1) // 2

    for key, mat in tqdm(sequences, desc=f"Processing {num_frames}f", unit="seq"):
        # Normalize to [0,1] range
        mat = mat / 100.0
        original_frames = mat.shape[0]
        
        # Aggregate sequence to target frames
        if original_frames != num_frames:
            mat = aggregate_sequence(mat, num_frames, aggregation_method='mean')
        
        n_frames = mat.shape[0]
        
        # Create sliding windows
        padded = np.pad(mat, ((pad, pad), (0, 0)), "edge")
        windows = sliding_window_view(padded, (num_frames, 12), axis=(0, 1))[:, 0]
        windows = torch.from_numpy(windows.copy()).unsqueeze(1).unsqueeze(3)  # (N,1,T,1,12)

        dl = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        low_seq, mid_seq, high_seq = [], [], []

        with torch.no_grad():
            for batch in dl:
                batch = batch.to(device, dtype=next(model.parameters()).dtype)
                
                # # Create a dummy mask of None to avoid masking issues
                # # Or create a proper mask with all False (no masking)
                # B, C, T, H, W = batch.shape
                # mask = torch.zeros((B, T, H, W), dtype=torch.bool, device=device)
                
                # try:
                _, _mask, levels = model.forward_encoder(batch, mask_ratio=0.0,
                                                            return_intermediates=True)
                # except RuntimeError as e:
                #     if "Input and output sizes should be greater than 0" in str(e):
                #         # Try with explicit mask=None
                #         print(f"Trying with mask=None due to error: {e}")
                #         # Modify the forward call to bypass masking
                #         levels = model.forward_encoder_no_mask(batch)
                #     else:
                #         raise e
                
                l, m, h = levels[0], levels[1], levels[2]

                def pool(feat):  # (B, T, H, W, C) → (B,C)
                    return feat.flatten(1, -2).mean(1).float().cpu()

                low_seq.append(pool(l))
                mid_seq.append(pool(m))
                high_seq.append(pool(h))

        low_seq = torch.cat(low_seq).numpy().astype(np.float16)
        mid_seq = torch.cat(mid_seq).numpy().astype(np.float16)
        high_seq = torch.cat(high_seq).numpy().astype(np.float16)
        comb_seq = np.concatenate([low_seq, mid_seq, high_seq], axis=1)

        LOW.append(low_seq)
        MID.append(mid_seq)
        HIGH.append(high_seq)
        COMB.append(comb_seq)

        frame_map[key] = (ptr, ptr + n_frames)
        ptr += n_frames

    return {
        'low': np.concatenate(LOW, axis=0),
        'mid': np.concatenate(MID, axis=0),
        'high': np.concatenate(HIGH, axis=0),
        'comb': np.concatenate(COMB, axis=0),
        'frame_map': frame_map
    }
     

# def extract_embeddings_for_aggregation(model, sequences, num_frames, batch_size, device, num_workers):
#     """Extract embeddings for a specific frame aggregation"""
    
#     print(f"Extracting embeddings for {num_frames} frames...")
    
#     all_embeddings = []
#     all_metadata = []
    
#     # Process sequences in batches
#     batch_size = 32  # Adjust based on your memory
    
#     for i in tqdm(range(0, len(sequences), batch_size), desc=f"Processing {num_frames}f"):
#         batch_sequences = sequences[i:i+batch_size]
#         batch_inputs = []
        
#         for seq in batch_sequences:
#             if len(seq) >= num_frames:
#                 # Take the first num_frames
#                 truncated_seq = seq[:num_frames]
                
#                 # Reshape to match model input: (num_frames, 1, 12)
#                 if truncated_seq.shape[1] == 12:  # (T, 12)
#                     truncated_seq = truncated_seq.reshape(num_frames, 1, 12)
#                 elif truncated_seq.ndim == 4:  # (T, 1, 12, 1)
#                     truncated_seq = truncated_seq.squeeze(-1)  # (T, 1, 12)
                
#                 batch_inputs.append(truncated_seq)
        
#         if not batch_inputs:
#             continue
            
#         # Stack and move to device
#         inputs = torch.stack([torch.tensor(inp, dtype=torch.float32) for inp in batch_inputs])
#         inputs = inputs.to(device)
        
#         try:
#             # First try without any mask
#             with torch.no_grad():
#                 latent, _, _, _, _ = model.forward_encoder_no_mask(inputs)
            
#         except Exception as e:
#             print(f"Error with forward_encoder_no_mask: {e}")
#             try:
#                 # Fallback: try with mask_ratio=0 and mask=None
#                 with torch.no_grad():
#                     latent, _, _, _, _ = model.forward_encoder(inputs, mask_ratio=0.0, mask=None)
#             except Exception as e2:
#                 print(f"Error even with fallback: {e2}")
#                 continue
        
#         # Store embeddings
#         all_embeddings.append(latent.cpu().numpy())
        
#         # Store metadata (sequence indices)
#         for j, seq_idx in enumerate(range(i, min(i + len(batch_inputs), len(sequences)))):
#             all_metadata.append({
#                 'sequence_idx': seq_idx,
#                 'num_frames': num_frames
#             })
    
#     if all_embeddings:
#         embeddings = np.concatenate(all_embeddings, axis=0)
#         return embeddings, all_metadata
#     else:
#         return np.array([]), []

# def load_data_for_embeddings(csv_path, data_dir):
#     """Load data compatible with new format - minimal changes"""
#     print(f"Reading CSV from: {csv_path}")
#     df = pd.read_csv(csv_path, low_memory=False)
#     # df = df.head(1000)  # Limit to first 1000 rows for testing
#     print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")

#     training_list = {}
#     for _, row in df.iterrows():
#         cage = row['cage_id']
#         day = row['from_tpt'].split(' ')[0]
#         if cage not in training_list:
#             training_list[cage] = []
#         training_list[cage].append(day)

#     # Use your existing load_dvc_data function
#     loaded_data_dict = load_dvc_data(data_dir, training_list)
    
#     # Convert to the format expected by embedding extraction
#     sequences = []
#     for cage, data in loaded_data_dict.items():
#         # Extract v_ columns (electrodes) 
#         v_cols = [col for col in data.columns if col.startswith('v_')]
#         if v_cols:
#             seq_data = data[v_cols].values.astype(np.float32)
#             sequences.append(seq_data)
    
#     print(f"Loaded {len(sequences)} sequences")
#     return sequences


     
def main():
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Validate aggregation arguments
    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("time_aggregations and aggregation_names must have same length")

    # # Load data
    mapping, order = build_test_split(args.dvc_root, args.summary_csv)
    sequences = load_sequences(args.dvc_root, mapping, order, args.skip_missing)
    
    # # Replace the data loading section with:
    # csv_path = "/scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv"
    # data_dir = "/scratch/bhole/dvc_data/smoothed/cage_activations_wo_nan"
    
    # sequences = load_data_for_embeddings(csv_path, data_dir)
    
    # Load checkpoint to get base model args
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    print("Loading checkpoint", ckpt_path)
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu")
    base_model_args = ckpt.get("args", {})
    if isinstance(base_model_args, argparse.Namespace):
        base_model_args = vars(base_model_args)
    
    # Parse q_strides if it's a string
    if 'q_strides' in base_model_args:
        base_model_args['q_strides'] = parse_q_strides(base_model_args['q_strides'])
    
    # Extract embeddings for each time aggregation
    all_results = {}
    
    for num_frames, agg_name in zip(args.time_aggregations, args.aggregation_names):
        print(f"\n=== Processing {agg_name} aggregation ({num_frames} frames) ===")
        
        # Create model args with correct input size for this aggregation
        model_args = {**base_model_args}
        model_args['input_size'] = [num_frames, 1, 12]  # Update input size
        model_args['num_frames'] = num_frames
        
        print(f"Creating model with input_size: {model_args['input_size']}")
        
        # Create model with correct input size
        model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
        
        # Load weights (skip positional embeddings that don't match)
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        for k, v in ckpt["model"].items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"Skipping {k}: not found in model")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(ckpt['model'])} layers")
        
        model = model.to(device).eval().requires_grad_(False)
        
        # Extract embeddings for this aggregation level
        results = extract_embeddings_for_aggregation(
            model, sequences, num_frames, args.batch_size, device, args.num_workers
        )
        
        # Save results and cleanup
        for level in ['low', 'mid', 'high', 'comb']:
            filename = f"test_{agg_name}_{level}.npy"
            filepath = os.path.join(args.output_dir, filename)
            
            save_data = {
                'frame_number_map': results['frame_map'],
                'embeddings': results[level],
                'aggregation_info': {
                    'num_frames': num_frames,
                    'aggregation_name': agg_name,
                    'level': level,
                    'shape': results[level].shape
                }
            }
            
            np.save(filepath, save_data)
            print(f"Saved {filename} – shape {results[level].shape}")
        
        all_results[agg_name] = results
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save summary metadata
    metadata = {
        'extraction_args': vars(args),
        'base_model_args': base_model_args,
        'aggregation_levels': {
            name: {
                'num_frames': frames,
                'embedding_shapes': {
                    level: all_results[name][level].shape 
                    for level in ['low', 'mid', 'high', 'comb']
                }
            }
            for name, frames in zip(args.aggregation_names, args.time_aggregations)
        },
        'total_sequences': len(sequences),
        'extraction_timestamp': datetime.datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(args.output_dir, 'extraction_metadata.npy')
    np.save(metadata_path, metadata)
    print(f"\nSaved extraction metadata to {metadata_path}")
    
    print(f"\n✓ Multi-level extraction completed!")
    print(f"Extracted embeddings at {len(args.time_aggregations)} time scales:")
    for name, frames in zip(args.aggregation_names, args.time_aggregations):
        print(f"  - {name}: {frames} frames")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
