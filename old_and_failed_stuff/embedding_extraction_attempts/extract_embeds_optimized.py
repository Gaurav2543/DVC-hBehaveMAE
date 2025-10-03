"""
Memory-Optimized hBehaveMAE embedding extractor with aggressive memory management.
Designed for maximum memory and time efficiency with timestamp preservation.

Key Features:
- Aggressive Memory Management: Explicit deletion and garbage collection throughout
- Timestamp Preservation: Saves timestamps alongside embeddings for downstream analysis
- Streaming Processing: Minimal memory footprint during inference
- No Normalization: Skips unnecessary preprocessing
- Native Model Precision: Uses model's natural precision without conversions
- Zarr Optimization: Efficient storage and retrieval
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import zarr
import dask
import dask.array as da

# Assuming these imports are in your project structure
from models import models_defs
from util import misc

# Suppress Dask performance warnings
from dask.array.core import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

# ----------------------------------------------------------------------------------
# --- Configuration & Argument Parsing ---
# ----------------------------------------------------------------------------------

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    p = argparse.ArgumentParser(description="Memory-Optimized hBehaveMAE Embedding Extractor")
    # --- Critical Paths ---
    p.add_argument("--dvc_root", required=True, type=str, help="Path to the directory containing raw cage activation CSV files.")
    p.add_argument("--summary_csv", required=True, type=str, help="Path to the summary metadata CSV file defining the chunks.")
    p.add_argument("--ckpt_path", required=True, type=str, help="Direct path to the model .pth checkpoint file.")
    p.add_argument("--output_dir", required=True, type=str, help="Directory to save embeddings and metadata.")
    p.add_argument("--zarr_path", required=True, type=str, help="Path to save or load the intermediate Zarr data store.")

    # --- Model Architecture (must match training) ---
    p.add_argument("--model", default="hbehavemae", type=str)
    p.add_argument("--stages", nargs='+', type=int, default=[2, 3, 4, 4, 5, 5, 6])
    p.add_argument("--q_strides", default="6,1,1;4,1,1;3,1,1;4,1,1;2,1,1;2,1,1")
    p.add_argument("--mask_unit_attn", nargs='+', type=lambda x: x.lower()=='true',
                   default=[True, False, False, False, False, False, False])
    p.add_argument("--patch_kernel", nargs=3, type=int, default=[5, 1, 12])
    p.add_argument("--init_embed_dim", type=int, default=96)
    p.add_argument("--init_num_heads", type=int, default=2)
    p.add_argument("--out_embed_dims", nargs='+', type=int, default=[128, 160, 192, 192, 224, 224, 256])
    p.add_argument("--decoding_strategy", default="single")
    p.add_argument("--decoder_embed_dim", type=int, default=128)
    p.add_argument("--decoder_depth", type=int, default=1)
    p.add_argument("--decoder_num_heads", type=int, default=1)

    # --- Aggregation & Processing ---
    p.add_argument("--time_aggregations", nargs='+', type=int, required=True,
                   help="List of time aggregation window sizes in minutes (e.g., 4320 10080).")
    p.add_argument("--aggregation_names", nargs='+', type=str, required=True,
                   help="Corresponding names for each aggregation level (e.g., 3day 1week).")
    p.add_argument("--batch_size", type=int, default=128, help="GPU batch size for model inference (reduced default for memory)")
    p.add_argument("--force_zarr_rebuild", action="store_true", help="Force rebuild of Zarr store even if it exists.")

    # --- System & GPU ---
    p.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs with multiprocessing.")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None, help="Specific GPU IDs to use (e.g., 0 1 2).")
    
    # --- Memory Management ---
    p.add_argument("--max_cages_in_memory", type=int, default=10, help="Maximum number of cage results to keep in memory before flushing to disk")
    
    return p.parse_args()

# ----------------------------------------------------------------------------------
# --- Part 1: Data Conversion to Zarr Store (with timestamps) ---
# ----------------------------------------------------------------------------------

def create_zarr_store_from_csvs(dvc_root, summary_csv_path, zarr_path):
    """
    Memory-optimized Zarr store creation with timestamps.
    """
    print("--- Starting Memory-Optimized Zarr Store Creation ---")
    summary_df = pd.read_csv(summary_csv_path)
    
    summary_df = summary_df.head(10)  # For testing, limit to first 10 entries
    
    # Convert timestamps
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])
    summary_df.sort_values(by=['cage_id', 'from_tpt'], inplace=True)
    
    # Calculate expected chunk size
    time_delta_minutes = (summary_df['to_tpt'] - summary_df['from_tpt']).dt.total_seconds().iloc[0] / 60
    expected_chunk_size = int(round(time_delta_minutes)) + 1
    print(f"Expected sequence chunk size: {expected_chunk_size} minutes.")
    
    print(f"OPTIMIZED: Processing cage groups: {list(summary_df.groupby('cage_id').groups.keys())}")

    root = zarr.open(zarr_path, mode='w')
    
    for cage_id, group in tqdm(summary_df.groupby('cage_id'), desc="Processing cages into Zarr"):
        all_chunks_data = []
        all_chunks_ts = []
        
        # Load CSV once per cage
        cage_csv_path = Path(dvc_root) / f"{group.iloc[0]['cage_id']}.csv"
        if not cage_csv_path.exists():
            print(f"Warning: CSV file not found for cage {group.iloc[0]['cage_id']}. Skipping.")
            continue
        
        cage_df = pd.read_csv(cage_csv_path, usecols=['timestamp'] + ELEC_COLS)
        cage_df['timestamp'] = pd.to_datetime(cage_df['timestamp'])

        # Process chunks for this cage
        for _, row in group.iterrows():
            chunk = cage_df[(cage_df['timestamp'] >= row['from_tpt']) & (cage_df['timestamp'] <= row['to_tpt'])]
            
            # Validation
            if len(chunk) != expected_chunk_size:
                print(f"Warning: Cage {cage_id} chunk starting {row['from_tpt']} has {len(chunk)} rows, expected {expected_chunk_size}. Skipping.")
                continue
            if chunk[ELEC_COLS].isnull().values.any():
                print(f"Warning: Cage {cage_id} chunk starting {row['from_tpt']} contains NaN values. Skipping.")
                continue

            all_chunks_data.append(chunk[ELEC_COLS].values.astype(np.float32))
            all_chunks_ts.append(chunk['timestamp'].values)
            
            # Free chunk memory immediately
            del chunk
            gc.collect()

        # Free cage_df memory
        del cage_df
        gc.collect()

        if not all_chunks_data:
            print(f"No valid chunks found for cage {cage_id}.")
            continue
            
        # Concatenate all valid chunks for the cage
        full_sequence_data = np.concatenate(all_chunks_data, axis=0)
        full_sequence_ts = np.concatenate(all_chunks_ts, axis=0)
        
        # Free intermediate lists
        del all_chunks_data, all_chunks_ts
        gc.collect()
        
        # Save data to Zarr store
        data_array = root.create_dataset(
            cage_id,
            data=full_sequence_data,
            chunks=(65536, None),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        )
        data_array.attrs['description'] = f"Full time-series data for cage {cage_id}"
        
        # Save timestamps to Zarr store
        ts_array = root.create_dataset(
            f"{cage_id}_timestamps",
            data=full_sequence_ts,
            chunks=(65536,),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        )
        ts_array.attrs['description'] = f"Timestamps for cage {cage_id}"
        
        # Free memory immediately after saving
        del full_sequence_data, full_sequence_ts
        gc.collect()
    
    # Free summary_df
    del summary_df
    gc.collect()
    
    print(f"--- Zarr store successfully created at: {zarr_path} ---")

# ----------------------------------------------------------------------------------
# --- Part 2: Memory-Optimized Model Loading and Inference ---
# ----------------------------------------------------------------------------------
   
def load_model_for_gpu(ckpt_path, model_args, device_id):
    """Load model on specific GPU with memory optimization"""
    device = torch.device(f"cuda:{device_id}")
    
    # Clear GPU memory first
    torch.cuda.empty_cache()
    with torch.cuda.device(device_id):
        torch.cuda.empty_cache()
    
    print(f"Loading model on GPU {device_id}")
    
    from iopath.common.file_io import g_pathmgr as pathmgr
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu")
    
    # Use checkpoint's original model args (KEY FIX)
    base_model_args = ckpt.get("args", {})
    if isinstance(base_model_args, argparse.Namespace):
        base_model_args = vars(base_model_args)
    
    # Create model with checkpoint's original architecture
    model = models_defs.__dict__[base_model_args.get("model", "hbehavemae")](**base_model_args)
    
    # Load weights (same approach as working code)
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
    
    # Clear checkpoint from memory
    del ckpt, model_dict, pretrained_dict, base_model_args
    torch.cuda.empty_cache()
    
    print(f"GPU {device_id}: Loaded model, moving to GPU...")
    model = model.to(device).eval().requires_grad_(False)
    
    return model 


def process_cage_streaming(cage_id, dask_array_data, dask_array_ts, model, num_frames, batch_size, device_id, num_levels):
    """
    Memory-optimized cage processing with streaming and aggressive cleanup.
    """
    device = torch.device(f"cuda:{device_id}")

    # Validation
    if dask_array_data.shape[0] < num_frames:
        print(f"Skipping cage {cage_id}: Sequence length {dask_array_data.shape[0]} < window size {num_frames}.")
        return None

    num_windows = dask_array_data.shape[0] - num_frames + 1
    level_embeddings = [[] for _ in range(num_levels)]
    timestamps_list = []
    
    # Process in smaller chunks to avoid memory buildup
    chunk_size = min(batch_size * 10, num_windows)  # Process in larger chunks but not too large
    
    with torch.no_grad():
        for chunk_start in tqdm(range(0, num_windows, chunk_size), 
                               desc=f"GPU {device_id} processing {cage_id}", leave=False):
            
            chunk_end = min(chunk_start + chunk_size, num_windows)
            
            # Create sliding windows for this chunk using da.lib.stride_tricks
            chunk_windows = da.lib.stride_tricks.sliding_window_view(
                dask_array_data[chunk_start:chunk_start + chunk_end - chunk_start + num_frames - 1],
                window_shape=(num_frames, dask_array_data.shape[1])
            )[:, 0, :, :]
            
            # Get corresponding timestamps
            chunk_timestamps = dask_array_ts[chunk_start:chunk_end]
            
            # Process this chunk in batches
            for batch_start in range(0, chunk_windows.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, chunk_windows.shape[0])
                
                # Compute the batch (triggers Dask execution)
                batch_windows = chunk_windows[batch_start:batch_end].compute()
                batch_ts = chunk_timestamps[batch_start:batch_end].compute()
                
                # Fix: Copy array to make it writable and avoid undefined behavior
                batch_windows = np.array(batch_windows, copy=True)
                
                # Prepare tensor (use model's natural dtype)
                batch_tensor = torch.from_numpy(batch_windows).unsqueeze(1).unsqueeze(3)
                batch_tensor = batch_tensor.to(device, dtype=next(model.parameters()).dtype)

                # Model inference
                outputs = model.forward_encoder(batch_tensor, mask_ratio=0.0, return_intermediates=True)
                intermediate_levels = outputs[-1]

                # Pool and collect embeddings
                for level_idx, level_feat in enumerate(intermediate_levels):
                    pooled = level_feat.flatten(1, -2).mean(1).cpu().numpy().astype(np.float16)
                    level_embeddings[level_idx].append(pooled)
                    del pooled  # Free immediately
                
                timestamps_list.append(batch_ts)
                
                # Free batch memory immediately
                del batch_tensor, outputs, intermediate_levels, batch_windows, batch_ts
                torch.cuda.empty_cache()
                gc.collect()
            
            # Free chunk memory
            del chunk_windows, chunk_timestamps
            gc.collect()

    # Consolidate results for this cage
    cage_results = {}
    for i in range(num_levels):
        if level_embeddings[i]:
            cage_results[f'level{i+1}'] = np.concatenate(level_embeddings[i], axis=0)
        else:
            cage_results[f'level{i+1}'] = np.array([], dtype=np.float16)
    
    # Create combined embedding
    if all(cage_results[f"level{i+1}"].size > 0 for i in range(num_levels)):
        cage_results['comb'] = np.concatenate([cage_results[f"level{i+1}"] for i in range(num_levels)], axis=1)
    else:
        cage_results['comb'] = np.array([], dtype=np.float16)
    
    # Consolidate timestamps
    if timestamps_list:
        consolidated_timestamps = np.concatenate(timestamps_list, axis=0)
    else:
        consolidated_timestamps = np.array([], dtype='datetime64[ns]')
    
    # Free intermediate lists
    del level_embeddings, timestamps_list
    gc.collect()

    return cage_id, cage_results, consolidated_timestamps

def save_embeddings_incrementally(results_buffer, output_dir, agg_name, num_levels):
    """
    Save embeddings incrementally to avoid memory buildup.
    """
    if not results_buffer:
        return {}
    
    print(f"Saving incremental batch of {len(results_buffer)} cages...")
    
    # Generate consistent batch number
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(f"embeddings_{agg_name}_") and f.endswith('.npz')]
    batch_num = len([f for f in existing_files if 'level1' in f])  # Count level1 files for batch number
    
    # Combine current batch
    level_names = [f"level{i+1}" for i in range(num_levels)] + ['comb']
    combined_batch = {'frame_map': {}, 'timestamps': []}
    
    for name in level_names:
        combined_batch[name] = []
    
    ptr = 0
    for cage_id, (cage_results, cage_timestamps) in results_buffer.items():
        if cage_results and 'comb' in cage_results:
            num_windows = cage_results['comb'].shape[0]
            if num_windows > 0:
                combined_batch['frame_map'][cage_id] = (ptr, ptr + num_windows)
                ptr += num_windows
                for name in level_names:
                    combined_batch[name].append(cage_results[name])
                combined_batch['timestamps'].append(cage_timestamps)
    
    # Concatenate batch results
    for name in level_names:
        if combined_batch[name]:
            combined_batch[name] = np.concatenate(combined_batch[name], axis=0)
        else:
            combined_batch[name] = np.array([], dtype=np.float16)
    
    if combined_batch['timestamps']:
        combined_batch['timestamps'] = np.concatenate(combined_batch['timestamps'], axis=0)
    else:
        combined_batch['timestamps'] = np.array([], dtype='datetime64[ns]')
    
    # Save each level separately with consistent batch numbering
    for level_name in level_names:
        filename = f"embeddings_{agg_name}_{level_name}_batch_{batch_num}.npz"
        filepath = output_dir / filename
        np.savez_compressed(
            filepath,
            frame_map=combined_batch['frame_map'],
            embeddings=combined_batch[level_name],
            timestamps=combined_batch['timestamps']
        )
        print(f"Saved batch {filename} – shape {combined_batch[level_name].shape}")
    
    # Free memory
    del combined_batch
    gc.collect()
    
    return {}

def gpu_worker_optimized(gpu_id, work_queue, result_queue, ckpt_path, model_args, num_frames, batch_size, num_levels, zarr_path):
    """Memory-optimized GPU worker."""
    try:
        model = load_model_for_gpu(ckpt_path, model_args, gpu_id)
        zarr_store = zarr.open(zarr_path, mode='r')

        while True:
            cage_id = work_queue.get()
            if cage_id is None:  # Sentinel value to stop
                break

            # Load data arrays
            dask_array_data = da.from_zarr(zarr_store[cage_id])
            dask_array_ts = da.from_zarr(zarr_store[f"{cage_id}_timestamps"])
            
            # Process cage
            results = process_cage_streaming(cage_id, dask_array_data, dask_array_ts, model, 
                                           num_frames, batch_size, gpu_id, num_levels)
            result_queue.put(results)
            
            # Free references
            del dask_array_data, dask_array_ts
            gc.collect()

    except Exception as e:
        print(f"Error in GPU worker {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        result_queue.put(None)  # Signal completion

def extract_embeddings_with_memory_management(zarr_path, ckpt_path, model_args, num_frames, batch_size, gpu_ids, num_levels, output_dir, agg_name, max_cages_in_memory=10):
    """
    Main extraction function with aggressive memory management.
    """
    zarr_store = zarr.open(zarr_path, mode='r')
    cage_ids = [k for k in zarr_store.keys() if not k.endswith('_timestamps')]
    
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add work items
    for cage_id in cage_ids:
        work_queue.put(cage_id)
    for _ in gpu_ids:
        work_queue.put(None)  # Sentinel values

    # Start worker processes
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker_optimized, 
                      args=(gpu_id, work_queue, result_queue, ckpt_path, model_args, 
                           num_frames, batch_size, num_levels, zarr_path))
        p.start()
        processes.append(p)
    
    # Collect results with incremental saving
    results_buffer = {}
    completed_processes = 0
    processed_cages = 0
    
    with tqdm(total=len(cage_ids), desc="Processing and saving cages") as pbar:
        while completed_processes < len(gpu_ids):
            res = result_queue.get()
            if res is None:
                completed_processes += 1
                continue
            
            cage_id, cage_data, timestamps = res
            if cage_id is not None:
                results_buffer[cage_id] = (cage_data, timestamps)
                processed_cages += 1
                pbar.update(1)
                
                # Save incrementally when buffer is full
                if len(results_buffer) >= max_cages_in_memory:
                    results_buffer = save_embeddings_incrementally(results_buffer, output_dir, agg_name, num_levels)
                    gc.collect()
    
    # Save any remaining results
    if results_buffer:
        save_embeddings_incrementally(results_buffer, output_dir, agg_name, num_levels)
    
    # Clean up processes
    for p in processes:
        p.join()
    
    # Fix semaphore leak warning
    work_queue.close()
    work_queue.join_thread()
    result_queue.close() 
    result_queue.join_thread()
    
    print(f"Completed processing {processed_cages} cages with incremental saving.")
    return True

def parse_q_strides(q_strides_str):
    """Parses q_strides string into a list of tuples and counts levels."""
    if isinstance(q_strides_str, str):
        stages = q_strides_str.split(';')
        parsed = [tuple(int(x) for x in stage.split(',')) for stage in stages]
        return parsed, len(parsed) + 1
    return q_strides_str, len(q_strides_str) + 1

# ----------------------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------------------

def main():
    mp.set_start_method('spawn', force=True)
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare Zarr Data Store
    if args.force_zarr_rebuild or not Path(args.zarr_path).exists():
        create_zarr_store_from_csvs(args.dvc_root, args.summary_csv, args.zarr_path)
    else:
        print(f"Using existing Zarr store at {args.zarr_path}.")

    # Step 2: Setup GPUs
    if args.multi_gpu and torch.cuda.is_available():
        gpu_ids = args.gpu_ids if args.gpu_ids else list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [0]
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    # Step 3: Parse Model Hyperparameters
    base_model_args = vars(args)
    base_model_args['q_strides'], num_levels = parse_q_strides(args.q_strides)
    print(f"Model will have {num_levels} hierarchical levels.")

    # Step 4: Process Each Aggregation
    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("Mismatch between --time_aggregations and --aggregation_names counts.")

    for num_frames, agg_name in zip(args.time_aggregations, args.aggregation_names):
        print(f"\n{'='*20} Processing Aggregation: {agg_name} ({num_frames} minutes) {'='*20}")

        # Validation
        summary_df_check = pd.read_csv(args.summary_csv)
        summary_df_check = summary_df_check.head(10)
        time_delta_minutes = (pd.to_datetime(summary_df_check['to_tpt']) - pd.to_datetime(summary_df_check['from_tpt'])).dt.total_seconds().iloc[0] / 60
        expected_frames_from_summary = int(round(time_delta_minutes)) + 1
        
        if num_frames % expected_frames_from_summary != 0:
             print(f"FATAL MISMATCH: Aggregation window ({num_frames}) not a multiple of base chunk size ({expected_frames_from_summary}).")
             continue

        model_args = base_model_args.copy()
        model_args['input_size'] = [num_frames, 1, 12]

        # Run extraction with memory management
        success = extract_embeddings_with_memory_management(
            args.zarr_path, args.ckpt_path, model_args,
            num_frames, args.batch_size, gpu_ids, num_levels,
            output_dir, agg_name, args.max_cages_in_memory
        )
        
        # Free model args
        del model_args
        gc.collect()
        
        if success:
            print(f"Successfully completed extraction for {agg_name}")
        else:
            print(f"Issues encountered during extraction for {agg_name}")
            
    print("\nAll extractions complete with memory optimization!")

if __name__ == "__main__":
    main()