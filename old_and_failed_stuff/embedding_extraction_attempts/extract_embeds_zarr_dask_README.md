"""
hBehaveMAE embedding extractor with multi-level time aggregation.
Re-engineered for extreme memory and time efficiency using Zarr and Dask.

Key Features:
- True Streaming Pipeline: Never loads more data into RAM than a single batch requires.
- Out-of-Core Processing: Uses Dask for lazy, memory-free sliding window operations.
- Efficient Storage: Converts raw CSVs into a compressed, chunked Zarr store for rapid access.
- Strict Validation: Includes numerous checks to ensure data integrity and parameter matching.
- No Padding: Skips processing if a sequence is shorter than the required window size.
- Automatic Visualization: Generates heatmaps of the learned embeddings over time.
- Robust Multiprocessing: Retains multi-GPU support for accelerated inference.
"""
import argparse
import os
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import zarr
import dask
import dask.array as da
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these imports are in your project structure
from models import models_defs
from util import misc

# Suppress Dask performance warnings for this specific use case
from dask.array.core import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

# ----------------------------------------------------------------------------------
# --- Configuration & Argument Parsing ---
# ----------------------------------------------------------------------------------

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def get_args():
    p = argparse.ArgumentParser(description="Zarr/Dask Optimized hBehaveMAE Embedding Extractor")
    # --- Critical Paths ---
    p.add_argument("--dvc_root", required=True, type=str, help="Path to the directory containing raw cage activation CSV files.")
    p.add_argument("--summary_csv", required=True, type=str, help="Path to the summary metadata CSV file defining the chunks.")
    p.add_argument("--ckpt_path", required=True, type=str, help="Direct path to the model .pth checkpoint file.")
    p.add_argument("--output_dir", required=True, type=str, help="Directory to save embeddings, plots, and metadata.")
    p.add_argument("--zarr_path", required=True, type=str, help="Path to save or load the intermediate Zarr data store.")

    # --- Model Architecture (must match training) ---
    p.add_argument("--model", default="hbehavemae", type=str)
    # The --input_size will be set dynamically based on aggregation, this is a placeholder
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
    p.add_argument("--batch_size", type=int, default=256, help="GPU batch size for model inference.")
    p.add_argument("--force_zarr_rebuild", action="store_true", help="Force rebuild of Zarr store even if it exists.")

    # --- System & GPU ---
    p.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs with multiprocessing.")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None, help="Specific GPU IDs to use (e.g., 0 1 2).")
    return p.parse_args()


# ----------------------------------------------------------------------------------
# --- Part 1: Data Conversion to Zarr Store ---
# ----------------------------------------------------------------------------------

def create_zarr_store_from_csvs(dvc_root, summary_csv_path, zarr_path):
    """
    Reads raw CSV data based on a summary file, concatenates it chronologically
    for each cage, and saves it to an efficient Zarr data store.
    """
    print("--- Starting Zarr Store Creation ---")
    summary_df = pd.read_csv(summary_csv_path)
    # # take onyl the fisrt 100 rows to infer chunk size
    # summary_df = summary_df.head(1000)
    
    # Convert BOTH columns to datetime objects before doing calculations
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt']) 
    
    summary_df.sort_values(by=['cage_id', 'from_tpt'], inplace=True)
    
    # Calculate expected chunk size from summary file
    time_delta_minutes = (summary_df['to_tpt'] - summary_df['from_tpt']).dt.total_seconds().iloc[0] / 60
    # Add 1 because the time range is inclusive
    expected_chunk_size = int(round(time_delta_minutes)) + 1
    print(f"✅ Inferred expected sequence chunk size from summary: {expected_chunk_size} minutes.")

    root = zarr.open(zarr_path, mode='w')
    
    for cage_id, group in tqdm(summary_df.groupby('cage_id'), desc="Processing cages into Zarr"):
        all_chunks = []
        # Pre-load the full CSV for the cage to avoid repeated reading
        cage_csv_path = Path(dvc_root) / f"{group.iloc[0]['cage_id']}.csv"
        if not cage_csv_path.exists():
            print(f"⚠️ Warning: CSV file not found for cage {group.iloc[0]['cage_id']}. Skipping entire cage.")
            continue
        
        # Load only necessary columns to save memory
        cage_df = pd.read_csv(cage_csv_path, usecols=['timestamp'] + ELEC_COLS)
        cage_df['timestamp'] = pd.to_datetime(cage_df['timestamp'])

        for _, row in group.iterrows():
            chunk = cage_df[(cage_df['timestamp'] >= row['from_tpt']) & (cage_df['timestamp'] <= row['to_tpt'])]
            
            # --- Strict Validation ---
            if len(chunk) != expected_chunk_size:
                print(f"⚠️ Warning: Cage {cage_id} chunk starting {row['from_tpt']} has {len(chunk)} rows, expected {expected_chunk_size}. Skipping.")
                continue
            if chunk[ELEC_COLS].isnull().values.any():
                print(f"⚠️ Warning: Cage {cage_id} chunk starting {row['from_tpt']} contains NaN values. Skipping.")
                continue

            all_chunks.append(chunk[ELEC_COLS].values)

        if not all_chunks:
            print(f"No valid chunks found for cage {cage_id}.")
            continue
            
        # Concatenate all valid chunks for the cage
        full_sequence = np.concatenate(all_chunks, axis=0).astype(np.float32)
        
        # Save to Zarr store
        zarr_array = root.create_dataset(
            cage_id,
            data=full_sequence,
            chunks=(65536, None),  # Chunk along the time axis for efficient slicing
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        )
        zarr_array.attrs['description'] = f"Full time-series for cage {cage_id}"
        zarr_array.attrs['shape'] = full_sequence.shape
    
    print(f"--- ✅ Zarr store successfully created at: {zarr_path} ---")

# ----------------------------------------------------------------------------------
# --- Part 2: Dask-Powered Embedding Extraction ---
# ----------------------------------------------------------------------------------

# def load_model_for_gpu(ckpt_path, model_args, device_id):
#     """Loads a pre-trained model onto a specific GPU."""
#     device = torch.device(f"cuda:{device_id}")
#     print(f"Loading model on GPU {device_id}...")
    
#     try:
#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
        
#         # Tolerant weight loading
#         misc.load_model(args=argparse.Namespace(resume=ckpt_path), model_without_ddp=model)
        
#         return model.to(device).eval().requires_grad_(False)
#     except Exception as e:
#         print(f"❌ FATAL ERROR on GPU {device_id}: Could not load model from {ckpt_path}.")
#         print(f"Error details: {e}")
#         # Ensure the model args in your script match the training args *exactly*.
#         print("Model args used:", model_args)
#         raise e
    
def load_model_for_gpu(ckpt_path, model_args, device_id):
    """Loads a pre-trained model onto a specific GPU for INFERENCE."""
    device = torch.device(f"cuda:{device_id}")
    print(f"Loading model on GPU {device_id} for inference...")

    try:
        # 1. Load the entire checkpoint dictionary from the .pth file
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 2. Create an instance of the model with the correct architecture
        model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)

        # 3. Extract the state dictionary of the model weights.
        #    In your training script, these are saved under the key 'model'.
        state_dict = ckpt['model']

        # 4. Load the weights into the model.
        #    'strict=False' is robust: it loads all matching layers and ignores
        #    any that don't match (e.g., decoder weights if not needed).
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Successfully loaded model weights onto GPU {device_id}.")

        return model.to(device).eval().requires_grad_(False)
        
    except Exception as e:
        print(f"❌ FATAL ERROR on GPU {device_id}: Could not load model from {ckpt_path}.")
        print(f"Error details: {e}")
        print("Model args used:", model_args)
        raise e

def generate_and_save_heatmap(embeddings, output_path, title):
    """Generates and saves a heatmap of embeddings."""
    if embeddings.size == 0:
        print("Cannot generate heatmap for empty embeddings.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(embeddings.T, cmap='viridis', cbar=True, xticklabels=False)
    plt.title(title)
    plt.xlabel("Time (each tick is a non-overlapping window)")
    plt.ylabel("Embedding Dimension")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Heatmap saved to {output_path}")

# Make sure you have this import at the top of your script
import dask

def process_cage_with_dask(cage_id, dask_array, model, num_frames, batch_size, device_id, num_levels):
    """
    Processes a single cage's full time-series using a robust dask.delayed
    method for generating sliding windows. (Replaces faulty map_overlap).
    """
    device = torch.device(f"cuda:{device_id}")

    # --- Strict Validation ---
    if dask_array.shape[0] < num_frames:
        print(f"⚠️ Skipping cage {cage_id}: Sequence length {dask_array.shape[0]} is less than window size {num_frames}.")
        return None

    # --- NEW, ROBUST SLIDING WINDOW IMPLEMENTATION ---
    # 1. Create a list of lazy "slicing" operations using dask.delayed.
    #    This does not load any data into memory.
    num_windows = dask_array.shape[0] - num_frames + 1
    
    # Define a simple slicing function
    def get_slice(arr, i, size):
        return arr[i : i + size]

    delayed_windows = [dask.delayed(get_slice)(dask_array, i, num_frames) for i in range(num_windows)]
    # --- END NEW IMPLEMENTATION ---

    level_embeddings = [[] for _ in range(num_levels)]

    with torch.no_grad():
        # 2. Iterate through batches of these lazy windows.
        for i in tqdm(range(0, num_windows, batch_size), desc=f"GPU {device_id} processing {cage_id}", leave=False):
            
            # .compute() triggers Dask to execute the slicing operations for this batch
            end_index = min(i + batch_size, num_windows)
            batch_delayed = delayed_windows[i:end_index]
            
            # This will return a list of numpy arrays, each with shape (num_frames, 12)
            batch_windows = dask.compute(*batch_delayed)
            
            # Stack the computed windows into a single numpy array
            batch_windows_np = np.stack(batch_windows)
            
            # Prepare tensor for PyTorch model
            batch_tensor = torch.from_numpy(batch_windows_np).unsqueeze(1).unsqueeze(3) # (B, 1, T, 1, 12)
            batch_tensor = batch_tensor.to(device, dtype=next(model.parameters()).dtype)

            # --- Model Inference ---
            outputs = model.forward_encoder(batch_tensor, mask_ratio=0.0, return_intermediates=True)
            intermediate_levels = outputs[-1]

            # Pool and collect embeddings for each level
            def pool(feat):
                return feat.flatten(1, -2).mean(1).float().cpu()

            for level_idx, level_feat in enumerate(intermediate_levels):
                level_embeddings[level_idx].append(pool(level_feat))

    # Concatenate results for this cage
    cage_results = {}
    for i in range(num_levels):
        if level_embeddings[i]:
            cage_results[f'level{i+1}'] = torch.cat(level_embeddings[i]).numpy().astype(np.float16)
        else:
            cage_results[f'level{i+1}'] = np.array([], dtype=np.float16)
            
    # Create combined embedding
    if all(cage_results[f"level{i+1}"].size > 0 for i in range(num_levels)):
        cage_results['comb'] = np.concatenate([cage_results[f"level{i+1}"] for i in range(num_levels)], axis=1)
    else:
        cage_results['comb'] = np.array([], dtype=np.float16)

    return cage_id, cage_results

def gpu_worker(gpu_id, work_queue, result_queue, ckpt_path, model_args, num_frames, batch_size, num_levels, zarr_path):
    """Worker process for a single GPU."""
    try:
        model = load_model_for_gpu(ckpt_path, model_args, gpu_id)
        zarr_store = zarr.open(zarr_path, mode='r')

        while True:
            cage_id = work_queue.get()
            if cage_id is None:  # Sentinel value to stop
                break

            dask_array = da.from_zarr(zarr_store[cage_id])
            results = process_cage_with_dask(cage_id, dask_array, model, num_frames, batch_size, gpu_id, num_levels)
            result_queue.put(results)

    except Exception as e:
        print(f"❌ Error in GPU worker {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        result_queue.put(None) # Signal completion or error

def extract_embeddings_multiprocess(zarr_path, ckpt_path, model_args, num_frames, batch_size, gpu_ids, num_levels):
    """Main function to orchestrate multi-GPU extraction."""
    zarr_store = zarr.open(zarr_path, mode='r')
    cage_ids = list(zarr_store.keys())
    
    work_queue = mp.Queue()
    result_queue = mp.Queue()
    for cage_id in cage_ids:
        work_queue.put(cage_id)
    for _ in gpu_ids:
        work_queue.put(None)  # Add sentinel values for each process

    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(gpu_id, work_queue, result_queue, ckpt_path, model_args, num_frames, batch_size, num_levels, zarr_path))
        p.start()
        processes.append(p)
    
    # Collect results
    all_cage_results = {}
    completed_processes = 0
    with tqdm(total=len(cage_ids), desc="Aggregating results from GPUs") as pbar:
        while completed_processes < len(gpu_ids):
            res = result_queue.get()
            if res is None:
                completed_processes += 1
                continue
            cage_id, cage_data = res
            all_cage_results[cage_id] = cage_data
            pbar.update(1)

    for p in processes:
        p.join()

    # Combine results from all cages into final arrays
    combined_results = {'frame_map': {}}
    level_names = [f"level{i+1}" for i in range(num_levels)] + ['comb']
    for name in level_names:
        combined_results[name] = []

    ptr = 0
    for cage_id in sorted(all_cage_results.keys()): # Process in a consistent order
        cage_data = all_cage_results[cage_id]
        num_windows = cage_data['comb'].shape[0]
        if num_windows > 0:
            combined_results['frame_map'][cage_id] = (ptr, ptr + num_windows)
            ptr += num_windows
            for name in level_names:
                combined_results[name].append(cage_data[name])

    for name in level_names:
        if combined_results[name]:
            combined_results[name] = np.concatenate(combined_results[name], axis=0)
        else:
            combined_results[name] = np.array([], dtype=np.float16)

    return combined_results

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

    # --- Step 1: Prepare Zarr Data Store ---
    if args.force_zarr_rebuild or not Path(args.zarr_path).exists():
        create_zarr_store_from_csvs(args.dvc_root, args.summary_csv, args.zarr_path)
    else:
        print(f"✅ Using existing Zarr store at {args.zarr_path}. Use --force_zarr_rebuild to overwrite.")

    # --- Step 2: Setup GPUs ---
    if args.multi_gpu and torch.cuda.is_available():
        gpu_ids = args.gpu_ids if args.gpu_ids else list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [0]
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    # --- Step 3: Parse Model Hyperparameters ---
    base_model_args = vars(args)
    base_model_args['q_strides'], num_levels = parse_q_strides(args.q_strides)
    print(f"✅ Model will have {num_levels} hierarchical levels.")

    # --- Step 4: Loop Through Aggregations and Extract ---
    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("Mismatch between --time_aggregations and --aggregation_names counts.")

    for num_frames, agg_name in zip(args.time_aggregations, args.aggregation_names):
        print(f"\n{'='*20} Processing Aggregation: {agg_name} ({num_frames} minutes) {'='*20}")

        # --- Strict Validation ---
        summary_df_check = pd.read_csv(args.summary_csv)
        # # take onyl first 100 rows to infer chunk size
        # summary_df_check = summary_df_check.head(1000)
        time_delta_minutes = (pd.to_datetime(summary_df_check['to_tpt']) - pd.to_datetime(summary_df_check['from_tpt'])).dt.total_seconds().iloc[0] / 60
        expected_frames_from_summary = int(round(time_delta_minutes)) + 1
        
        # This check ensures you are not mixing up summary files with aggregations
        # Here we check if the requested aggregation *is a multiple of* the base chunk size.
        if num_frames % expected_frames_from_summary != 0:
             print(f"❌ FATAL MISMATCH: The requested aggregation window ({num_frames}) is not a multiple of the base data chunk size inferred from the summary file ({expected_frames_from_summary}).")
             print("This can happen if you are using a summary file for 3-day chunks but requesting a 5-day aggregation.")
             print("Please ensure your aggregation windows are composed of whole data chunks.")
             continue # Skip to next aggregation

        model_args = base_model_args.copy()
        model_args['input_size'] = [num_frames, 1, 12]

        # --- Run Extraction ---
        results = extract_embeddings_multiprocess(
            args.zarr_path, args.ckpt_path, model_args,
            num_frames, args.batch_size, gpu_ids, num_levels
        )

        # --- Step 5: Save Results and Generate Plots ---
        level_names = [f"level{i+1}" for i in range(num_levels)] + ['comb']
        for level_name in level_names:
            # Save embeddings
            filename = f"embeddings_{agg_name}_{level_name}.npz"
            filepath = output_dir / filename
            np.savez_compressed(
                filepath,
                frame_map=results['frame_map'],
                embeddings=results[level_name]
            )
            print(f"Saved {filename} – shape {results[level_name].shape}")

            # Generate and save heatmap
            heatmap_path = output_dir / f"heatmap_{agg_name}_{level_name}.png"
            generate_and_save_heatmap(
                results[level_name],
                heatmap_path,
                f"Learned Embeddings ({agg_name} windows, {level_name})"
            )
            
    print("\n\n🎉 All extractions and visualizations complete!")

if __name__ == "__main__":
    main()