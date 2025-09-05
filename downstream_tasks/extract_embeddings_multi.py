"""
hBehaveMAE embedding extractor with multi-level time aggregation
Extracts embeddings at different time scales to capture aging patterns
Modified to use multiprocessing for true multi-GPU parallelization
"""
import argparse, os, time, datetime, warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from iopath.common.file_io import g_pathmgr as pathmgr
from data_pipeline.load_dvc import load_dvc_data
from models import models_defs
from util import misc
from util.pos_embed import interpolate_pos_embed
import queue
import threading
from multiprocessing import Queue, Process
import pickle

# ----------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    # Data arguments
    p.add_argument("--dvc_root", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", required=True)
    p.add_argument("--output_dir", required=True)
    
    # Model architecture arguments (should match training)
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
    
    # Multi-level extraction arguments
    p.add_argument("--time_aggregations", nargs='+', type=int,
                   default=[90, 360, 720, 1440, 2160, 2880, 4320, 10080], # 1.5h, 6h, 1day, 3days, 1week
                   help="Different time aggregation levels (in minutes)")
    p.add_argument("--aggregation_names", nargs='+', type=str,
                   default=["1.5h", "6h", "12h", "1day", "1.5days", "2days", "3days", "1week"],
                   help="Names for each aggregation level")
    
    # Processing arguments
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs with multiprocessing")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None, 
                   help="Specific GPU IDs to use (e.g., --gpu_ids 0 1 2)")
    p.add_argument("--max_processes", type=int, default=None,
                   help="Maximum number of processes (defaults to number of GPUs)")
    
    return p.parse_args()

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

# ----------------------------------------------------------------------------------
def build_test_split(root, csv):
    df = pd.read_csv(csv, low_memory=False)
    # df = df.head(100)
    df.columns = df.columns.str.strip()
    
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
    """Parse q_strides string into proper format and count levels"""
    if isinstance(q_strides_str, str):
        # Parse "15,1,1;16,1,1" into [[15,1,1], [16,1,1]]
        stages = q_strides_str.split(';')
        parsed_strides = [[int(x) for x in stage.split(',')] for stage in stages]
        num_levels = len(parsed_strides) + 1  # +1 for the initial level
        return parsed_strides, num_levels
    return q_strides_str, len(q_strides_str) + 1

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

def load_model_for_gpu(ckpt_path, model_args, device_id):
    """Load model on specific GPU"""
    device = torch.device(f"cuda:{device_id}")
    
    print(f"Loading model on GPU {device_id}")
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu")
    
    # Create model
    model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
    
    # Load weights (skip positional embeddings that don't match)
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

def process_sequence_batch(key_mat_batch, model, num_frames, batch_size, device_id, num_levels):
    """Process a batch of sequences on a specific GPU"""
    device = torch.device(f"cuda:{device_id}")
    
    results = {}
    level_embeddings = [[] for _ in range(num_levels)]
    frame_map = {}
    ptr = 0
    
    pad = (num_frames - 1) // 2
    
    for key, mat in tqdm(key_mat_batch, desc=f"GPU {device_id} processing batch", leave=False):
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
        
        dl = DataLoader(windows, batch_size=batch_size, shuffle=False, 
                       num_workers=0, pin_memory=False)  # No multiprocessing in DataLoader
        
        # Initialize level sequences for this sequence
        level_seqs = [[] for _ in range(num_levels)]
        
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"GPU {device_id} batches", leave=False):
                batch = batch.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                # # Create a dummy mask
                # B, C, T, H, W = batch.shape
                # # mask = torch.zeros((B, T, H, W), dtype=torch.bool, device=device)
                # mask = None 
                
                # try:
                outputs = model.forward_encoder(batch, mask_ratio=0.0,
                                                return_intermediates=True)
                levels = outputs[-1]  # Last element should be the intermediate levels
                # except RuntimeError as e:
                #     if "Input and output sizes should be greater than 0" in str(e):
                #         print(f"GPU {device_id}: Trying with mask=None due to error: {e}")
                #         levels = model.forward_encoder_no_mask(batch)
                #     else:
                #         raise e
                
                # Process all levels dynamically
                def pool(feat):  # (B, T, H, W, C) → (B,C)
                    return feat.flatten(1, -2).mean(1).float().cpu()
                
                for i, level_feat in enumerate(levels):
                    level_seqs[i].append(pool(level_feat))
        
        # Concatenate sequences for this sample and convert to numpy
        for i in range(num_levels):
            level_seq = torch.cat(level_seqs[i]).numpy().astype(np.float16)
            level_embeddings[i].append(level_seq)
        
        frame_map[key] = (ptr, ptr + n_frames)
        ptr += n_frames
    
    # Prepare results dictionary
    results['frame_map'] = frame_map
    
    # Add each level to results
    for i in range(num_levels):
        level_name = f"level{i+1}"
        if level_embeddings[i]:  # Check if not empty
            results[level_name] = np.concatenate(level_embeddings[i], axis=0)
        else:
            results[level_name] = np.array([]).astype(np.float16)
    
    # Create combined embedding by concatenating all levels
    if all(results[f"level{i+1}"].size > 0 for i in range(num_levels)):
        combined_embedding = np.concatenate([results[f"level{i+1}"] for i in range(num_levels)], axis=1)
        results['comb'] = combined_embedding
    else:
        results['comb'] = np.array([]).astype(np.float16)
    
    return results

def gpu_worker(gpu_id, work_queue, result_queue, ckpt_path, model_args, num_frames, batch_size, num_levels):
    """Worker process for a specific GPU"""
    try:
        # Load model on this GPU
        model = load_model_for_gpu(ckpt_path, model_args, gpu_id)
        
        processed_count = 0
        while True:
            try:
                # Get work from queue (with timeout to avoid hanging)
                work_item = work_queue.get(timeout=30)
                if work_item is None:  # Poison pill to stop worker
                    break
                
                batch_id, key_mat_batch = work_item
                
                # Process this batch of sequences
                results = process_sequence_batch(key_mat_batch, model, num_frames, 
                                               batch_size, gpu_id, num_levels)
                
                # Put results back
                result_queue.put((batch_id, results))
                processed_count += 1
                
            except queue.Empty:
                print(f"GPU {gpu_id}: Queue timeout, stopping worker")
                break
            except Exception as e:
                print(f"GPU {gpu_id}: Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"GPU {gpu_id}: Processed {processed_count} batches")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Worker initialization failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_embeddings_multiprocess(sequences, ckpt_path, model_args, num_frames, 
                                   batch_size, gpu_ids, num_levels, sequences_per_batch=10):
    """Extract embeddings using multiple GPUs with multiprocessing"""
    
    print(f"Starting multi-GPU extraction with {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"Processing {len(sequences)} sequences in batches of {sequences_per_batch}")
    
    # Create work and result queues
    work_queue = Queue(maxsize=len(gpu_ids) * 2)  # Buffer for smooth processing
    result_queue = Queue()
    
    # Start worker processes
    processes = []
    for gpu_id in gpu_ids:
        p = Process(target=gpu_worker, 
                   args=(gpu_id, work_queue, result_queue, ckpt_path, model_args, 
                         num_frames, batch_size, num_levels))
        p.start()
        processes.append(p)
    
    # Split sequences into batches and add to work queue
    sequence_batches = []
    for i in range(0, len(sequences), sequences_per_batch):
        batch = sequences[i:i+sequences_per_batch]
        sequence_batches.append((i // sequences_per_batch, batch))
    
    # Add work to queue
    for batch_id, batch in sequence_batches:
        work_queue.put((batch_id, batch))
    
    # Collect results
    results_dict = {}
    collected_results = 0
    total_batches = len(sequence_batches)
    
    print(f"Waiting for {total_batches} batches to complete...")
    
    with tqdm(total=total_batches, desc="Collecting results") as pbar:
        while collected_results < total_batches:
            try:
                batch_id, batch_results = result_queue.get(timeout=60)
                results_dict[batch_id] = batch_results
                collected_results += 1
                pbar.update(1)
            except queue.Empty:
                print("Timeout waiting for results, checking process status...")
                alive_processes = [p.is_alive() for p in processes]
                print(f"Process status: {alive_processes}")
                if not any(alive_processes):
                    print("All processes died, breaking...")
                    break
    
    # Send stop signals to workers
    for _ in processes:
        work_queue.put(None)
    
    # Wait for processes to finish
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    # Combine results from all batches
    print("Combining results from all batches...")
    combined_results = {'frame_map': {}}
    
    # Initialize level lists
    for i in range(num_levels):
        combined_results[f"level{i+1}"] = []
    combined_results['comb'] = []
    
    # Sort and combine results by batch_id
    for batch_id in sorted(results_dict.keys()):
        batch_results = results_dict[batch_id]
        combined_results['frame_map'].update(batch_results['frame_map'])
        
        for i in range(num_levels):
            level_name = f"level{i+1}"
            if batch_results[level_name].size > 0:
                combined_results[level_name].append(batch_results[level_name])
        
        if batch_results['comb'].size > 0:
            combined_results['comb'].append(batch_results['comb'])
    
    # Concatenate all embeddings
    for i in range(num_levels):
        level_name = f"level{i+1}"
        if combined_results[level_name]:
            combined_results[level_name] = np.concatenate(combined_results[level_name], axis=0)
        else:
            combined_results[level_name] = np.array([]).astype(np.float16)
    
    if combined_results['comb']:
        combined_results['comb'] = np.concatenate(combined_results['comb'], axis=0)
    else:
        combined_results['comb'] = np.array([]).astype(np.float16)
    
    return combined_results

def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Validate aggregation arguments
    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("time_aggregations and aggregation_names must have same length")
    
    # Setup GPUs
    if args.multi_gpu and torch.cuda.is_available():
        if args.gpu_ids:
            gpu_ids = args.gpu_ids
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if args.max_processes:
            gpu_ids = gpu_ids[:args.max_processes]
        
        print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
        use_multiprocessing = len(gpu_ids) > 1
    else:
        gpu_ids = [0]  # Single GPU
        use_multiprocessing = False
        print("Using single GPU")
    
    # Load data
    mapping, order = build_test_split(args.dvc_root, args.summary_csv)
    sequences = load_sequences(args.dvc_root, mapping, order, args.skip_missing)
    
    # Load checkpoint to get base model args
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        if os.path.exists(os.path.join(args.ckpt_dir, "checkpoint-best.pth")):
            ckpt_path = os.path.join(args.ckpt_dir, "checkpoint-best.pth")
        else:
            # Find the latest checkpoint manually
            checkpoint_files = [f for f in os.listdir(args.ckpt_dir) if f.startswith('checkpoint-') and f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.ckpt_dir, x)), reverse=True)
                ckpt_path = os.path.join(args.ckpt_dir, checkpoint_files[0])
            else:
                raise FileNotFoundError(f"No checkpoint found in {args.ckpt_dir}")
    
    print("Loading checkpoint", ckpt_path)
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu")
    base_model_args = ckpt.get("args", {})
    if isinstance(base_model_args, argparse.Namespace):
        base_model_args = vars(base_model_args)
    
    # Parse q_strides and determine number of levels
    if 'q_strides' in base_model_args:
        base_model_args['q_strides'], num_levels = parse_q_strides(base_model_args['q_strides'])
    else:
        # Fallback to args.q_strides if not in checkpoint
        base_model_args['q_strides'], num_levels = parse_q_strides(args.q_strides)
    
    print(f"Model will have {num_levels} hierarchical levels based on q_strides: {base_model_args['q_strides']}")
    
    # Extract embeddings for each time aggregation
    all_results = {}
    
    for num_frames, agg_name in tqdm(zip(args.time_aggregations, args.aggregation_names), 
                                     total=len(args.time_aggregations), desc="Processing aggregations"):
        print(f"\n=== Processing {agg_name} aggregation ({num_frames} frames) ===")
        
        # Create model args with correct input size for this aggregation
        model_args = {**base_model_args}
        model_args['input_size'] = [num_frames, 1, 12]  # Update input size
        model_args['num_frames'] = num_frames
        
        print(f"Creating model with input_size: {model_args['input_size']}")
        
        # Extract embeddings using multiprocessing or single GPU
        if use_multiprocessing:
            results = extract_embeddings_multiprocess(
                sequences, ckpt_path, model_args, num_frames, 
                args.batch_size, gpu_ids, num_levels, 
                sequences_per_batch=max(1, len(sequences) // (len(gpu_ids) * 4))
            )
        else:
            # Single GPU fallback
            model = load_model_for_gpu(ckpt_path, model_args, gpu_ids[0])
            results = process_sequence_batch(sequences, model, num_frames, 
                                           args.batch_size, gpu_ids[0], num_levels)
            del model
            torch.cuda.empty_cache()
        
        # Save results
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
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            for gpu_id in gpu_ids:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
    
    # Save summary metadata
    metadata = {
        'extraction_args': vars(args),
        'base_model_args': base_model_args,
        'num_hierarchical_levels': num_levels,
        'gpu_ids_used': gpu_ids,
        'multiprocessing_used': use_multiprocessing,
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
    print(f"\nSaved extraction metadata to {metadata_path}")
    
    print(f"\n✓ Multi-level extraction completed!")
    print(f"Extracted embeddings at {len(args.time_aggregations)} time scales with {num_levels} hierarchical levels:")
    for name, frames in zip(args.aggregation_names, args.time_aggregations):
        print(f" - {name}: {frames} frames")
        level_info = ", ".join([f"level{i+1}" for i in range(num_levels)]) + ", comb"
        print(f"   Levels: {level_info}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
