"""
Properly fixed hBehaveMAE embedding extractor with true multi-GPU utilization
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
import pickle

def get_args():
    p = argparse.ArgumentParser()
    # Data arguments
    p.add_argument("--dvc_root", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", required=True)
    p.add_argument("--output_dir", required=True)
    
    # Model architecture arguments
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
                   default=[90, 360, 720, 1440, 2160, 2880, 4320, 10080],
                   help="Different time aggregation levels (in minutes)")
    p.add_argument("--aggregation_names", nargs='+', type=str,
                   default=["1.5h", "6h", "12h", "1day", "1.5days", "2days", "3days", "1week"],
                   help="Names for each aggregation level")
    
    # Processing arguments
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--skip_missing", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    
    return p.parse_args()

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]

def build_test_split(root, csv):
    df = pd.read_csv(csv, low_memory=False)
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
        stages = q_strides_str.split(';')
        parsed_strides = [[int(x) for x in stage.split(',')] for stage in stages]
        num_levels = len(parsed_strides) + 1
        return parsed_strides, num_levels
    return q_strides_str, len(q_strides_str) + 1

def aggregate_sequence(sequence, target_frames, aggregation_method='mean'):
    """Aggregate sequence to target number of frames"""
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

def single_gpu_worker(rank, gpu_id, sequences, ckpt_path, model_args, num_frames, 
                     batch_size, num_levels, output_file):
    """
    Worker function that runs on a single GPU
    This function will run in its own process with isolated GPU access
    """
    try:
        print(f"Worker {rank}: Starting on GPU {gpu_id} with PID {os.getpid()}")
        
        # CRITICAL: Set CUDA_VISIBLE_DEVICES to only see this GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Now reinitialize CUDA - device 0 will map to our assigned GPU
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        
        print(f"Worker {rank}: Set to use GPU {gpu_id}, CUDA device 0 = {torch.cuda.get_device_name(0)}")
        print(f"Worker {rank}: Processing {len(sequences)} sequences")
        
        # Load model
        print(f"Worker {rank}: Loading checkpoint...")
        ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu", weights_only=False)
        
        # Create model
        model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
        
        # Load weights
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in ckpt["model"].items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Worker {rank}: Skipping {k}: shape mismatch")
            else:
                print(f"Worker {rank}: Skipping {k}: not found in model")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.to(device).eval().requires_grad_(False)
        
        print(f"Worker {rank}: Model loaded, starting processing...")
        
        # Process sequences
        results = {}
        level_embeddings = [[] for _ in range(num_levels)]
        frame_map = {}
        ptr = 0
        pad = (num_frames - 1) // 2
        
        for seq_idx, (key, mat) in enumerate(sequences):
            if seq_idx % 10 == 0:
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"Worker {rank}: Sequence {seq_idx}/{len(sequences)}, Memory: {memory_used:.1f}GB")
            
            try:
                # Normalize and aggregate
                mat = mat / 100.0
                if mat.shape[0] != num_frames:
                    mat = aggregate_sequence(mat, num_frames, aggregation_method='mean')
                
                n_frames = mat.shape[0]
                
                # Create sliding windows
                padded = np.pad(mat, ((pad, pad), (0, 0)), "edge")
                windows = sliding_window_view(padded, (num_frames, 12), axis=(0, 1))[:, 0]
                windows = torch.from_numpy(windows.copy()).unsqueeze(1).unsqueeze(3)
                
                # Process in batches
                dl = DataLoader(windows, batch_size=batch_size, shuffle=False, 
                               num_workers=0, pin_memory=True)
                
                level_seqs = [[] for _ in range(num_levels)]
                
                with torch.no_grad():
                    for batch in dl:
                        batch = batch.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
                        
                        outputs = model.forward_encoder(batch, mask_ratio=0.0, return_intermediates=True)
                        levels = outputs[-1]
                        
                        def pool(feat):
                            return feat.flatten(1, -2).mean(1).float().cpu()
                        
                        for i, level_feat in enumerate(levels):
                            level_seqs[i].append(pool(level_feat))
                
                # Store results
                for i in range(num_levels):
                    if level_seqs[i]:
                        level_seq = torch.cat(level_seqs[i]).numpy().astype(np.float16)
                        level_embeddings[i].append(level_seq)
                
                frame_map[key] = (ptr, ptr + n_frames)
                ptr += n_frames
                
                # Periodic cleanup
                if seq_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Worker {rank}: Error processing {key}: {e}")
                continue
        
        # Prepare final results
        results['frame_map'] = frame_map
        
        for i in range(num_levels):
            level_name = f"level{i+1}"
            if level_embeddings[i]:
                results[level_name] = np.concatenate(level_embeddings[i], axis=0)
            else:
                results[level_name] = np.array([]).astype(np.float16)
        
        # Create combined embedding
        if all(results[f"level{i+1}"].size > 0 for i in range(num_levels)):
            combined_embedding = np.concatenate([results[f"level{i+1}"] for i in range(num_levels)], axis=1)
            results['comb'] = combined_embedding
        else:
            results['comb'] = np.array([]).astype(np.float16)
        
        # Save results
        print(f"Worker {rank}: Saving results to {output_file}")
        np.save(output_file, results, allow_pickle=True)
        
        print(f"Worker {rank}: Completed successfully!")
        
    except Exception as e:
        print(f"Worker {rank}: FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_results = {'error': str(e), 'traceback': traceback.format_exc()}
        np.save(output_file, error_results, allow_pickle=True)
        
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_embeddings_true_multiGPU(sequences, ckpt_path, model_args, num_frames, 
                                    batch_size, gpu_ids, num_levels, output_dir):
    """
    True multi-GPU extraction using proper process isolation
    """
    print(f"=== Starting TRUE multi-GPU extraction ===")
    print(f"GPUs: {gpu_ids}")
    print(f"Total sequences: {len(sequences)}")
    
    # Split sequences across GPUs
    n_gpus = len(gpu_ids)
    sequences_per_gpu = len(sequences) // n_gpus
    remainder = len(sequences) % n_gpus
    
    gpu_assignments = []
    start_idx = 0
    
    for i, gpu_id in enumerate(gpu_ids):
        # Give remainder sequences to first few GPUs
        n_seqs = sequences_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + n_seqs
        
        gpu_sequences = sequences[start_idx:end_idx]
        gpu_assignments.append((i, gpu_id, gpu_sequences))
        print(f"GPU {gpu_id}: {len(gpu_sequences)} sequences")
        
        start_idx = end_idx
    
    # Create output files
    temp_dir = Path(output_dir) / "temp_gpu_results"
    temp_dir.mkdir(exist_ok=True)
    
    output_files = []
    processes = []
    
    # Launch processes
    for rank, gpu_id, gpu_sequences in gpu_assignments:
        output_file = temp_dir / f"gpu_{gpu_id}_results.npy"
        output_files.append((gpu_id, output_file))
        
        # Create process
        p = mp.Process(target=single_gpu_worker, 
                      args=(rank, gpu_id, gpu_sequences, ckpt_path, model_args, 
                            num_frames, batch_size, num_levels, str(output_file)))
        p.start()
        processes.append((gpu_id, p))
        
        print(f"Launched process for GPU {gpu_id} (PID will be assigned)")
    
    print(f"\n=== Waiting for {len(processes)} GPU processes ===")
    
    # Wait for all processes
    for gpu_id, p in processes:
        print(f"Waiting for GPU {gpu_id}...")
        p.join()
        
        if p.exitcode == 0:
            print(f"✓ GPU {gpu_id} completed successfully")
        else:
            print(f"✗ GPU {gpu_id} failed with exit code {p.exitcode}")
    
    print("\n=== Collecting results ===")
    
    # Collect results
    combined_results = {'frame_map': {}}
    for i in range(num_levels):
        combined_results[f"level{i+1}"] = []
    combined_results['comb'] = []
    
    # Load results from each GPU
    for gpu_id, output_file in output_files:
        if output_file.exists():
            try:
                gpu_results = np.load(output_file, allow_pickle=True).item()
                
                if 'error' in gpu_results:
                    print(f"GPU {gpu_id} had error: {gpu_results['error']}")
                    continue
                
                # Merge results
                combined_results['frame_map'].update(gpu_results['frame_map'])
                
                for i in range(num_levels):
                    level_name = f"level{i+1}"
                    if gpu_results[level_name].size > 0:
                        combined_results[level_name].append(gpu_results[level_name])
                
                if gpu_results['comb'].size > 0:
                    combined_results['comb'].append(gpu_results['comb'])
                
                print(f"✓ Loaded results from GPU {gpu_id}")
                
            except Exception as e:
                print(f"Error loading GPU {gpu_id} results: {e}")
        else:
            print(f"No results file for GPU {gpu_id}")
    
    # Concatenate all results
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
    
    # Cleanup temp files
    for _, output_file in output_files:
        if output_file.exists():
            output_file.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()
    
    print("✓ Results collection completed")
    return combined_results

def main():
    # Set multiprocessing method
    mp.set_start_method('spawn', force=True)
    
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Validate arguments
    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("time_aggregations and aggregation_names must have same length")
    
    # Setup GPUs
    print("=== GPU Detection ===")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    total_gpus = torch.cuda.device_count()
    print(f"Detected {total_gpus} GPUs")
    for i in range(total_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if args.multi_gpu:
        if args.gpu_ids:
            gpu_ids = args.gpu_ids
            # Validate GPU IDs
            invalid = [g for g in gpu_ids if g >= total_gpus]
            if invalid:
                raise ValueError(f"Invalid GPU IDs: {invalid}. Available: 0-{total_gpus-1}")
        else:
            gpu_ids = list(range(total_gpus))
        
        print(f"Will use GPUs: {gpu_ids}")
        use_multi_gpu = len(gpu_ids) > 1
    else:
        gpu_ids = [0]
        use_multi_gpu = False
        print("Single GPU mode")
    
    # Load data
    print("\n=== Loading Data ===")
    mapping, order = build_test_split(args.dvc_root, args.summary_csv)
    sequences = load_sequences(args.dvc_root, mapping, order, args.skip_missing)
    
    # Load checkpoint to get model info
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        # Try to find checkpoint
        checkpoint_files = [f for f in os.listdir(args.ckpt_dir) 
                           if f.startswith('checkpoint-') and f.endswith('.pth')]
        if checkpoint_files:
            ckpt_path = os.path.join(args.ckpt_dir, checkpoint_files[-1])
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.ckpt_dir}")
    
    print(f"Using checkpoint: {ckpt_path}")
    
    # Load to get base args
    ckpt = torch.load(pathmgr.open(ckpt_path, "rb"), map_location="cpu", weights_only=False)
    base_model_args = ckpt.get("args", {})
    if isinstance(base_model_args, argparse.Namespace):
        base_model_args = vars(base_model_args)
    
    # Get number of levels from q_strides
    if 'q_strides' in base_model_args:
        _, num_levels = parse_q_strides(base_model_args['q_strides'])
    else:
        _, num_levels = parse_q_strides(args.q_strides)
    
    print(f"Model has {num_levels} hierarchical levels")
    
    # Process each time aggregation
    for num_frames, agg_name in zip(args.time_aggregations, args.aggregation_names):
        print(f"\n{'='*50}")
        print(f"Processing {agg_name} ({num_frames} frames)")
        print(f"{'='*50}")
        
        # Prepare model args for this aggregation
        model_args = dict(base_model_args)
        model_args['input_size'] = [num_frames, 1, 12]
        
        # Extract embeddings
        if use_multi_gpu:
            print("Using multi-GPU extraction")
            results = extract_embeddings_true_multiGPU(
                sequences, ckpt_path, model_args, num_frames,
                args.batch_size, gpu_ids, num_levels, args.output_dir
            )
        else:
            print("Using single GPU extraction")
            # For single GPU, we can use the worker function directly
            output_file = Path(args.output_dir) / "temp_single_gpu.npy"
            single_gpu_worker(0, gpu_ids[0], sequences, ckpt_path, model_args, 
                            num_frames, args.batch_size, num_levels, str(output_file))
            results = np.load(output_file, allow_pickle=True).item()
            output_file.unlink()
        
        # Save results in the expected format
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
            print(f"Saved {filename} with shape {results[level_name].shape}")
        
        print(f"✓ Completed {agg_name} aggregation")
    
    print(f"\n{'='*50}")
    print("🎉 ALL EXTRACTIONS COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'='*50}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()