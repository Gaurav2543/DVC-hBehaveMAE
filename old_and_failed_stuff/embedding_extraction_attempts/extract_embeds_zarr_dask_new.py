import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import zarr
import dask.array as da
import matplotlib.pyplot as plt
import seaborn as sns

from models import models_defs
from util import misc

from dask.array.core import PerformanceWarning
import warnings

warnings.filterwarnings("ignore", category=PerformanceWarning)

ELEC_COLS = [f"v_{i}" for i in range(1, 13)]


# --- ARGUMENT PARSING ---
def get_args():
    p = argparse.ArgumentParser(description="Zarr/Dask Optimized hBehaveMAE Embedding Extractor")
    p.add_argument("--dvc_root", required=True, type=str)
    p.add_argument("--summary_csv", required=True, type=str)
    p.add_argument("--ckpt_path", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--zarr_path", required=True, type=str)
    p.add_argument("--stats_path", type=str, default=None, help="(Recommended) Path to normalization stats .npy file.")
    p.add_argument("--no_normalization", action="store_true", help="Disable Z-score normalization. (Not recommended).")
    p.add_argument("--model", default="hbehavemae", type=str)
    p.add_argument("--stages", nargs='+', type=int, default=[2, 3, 4, 4, 5, 5, 6])
    p.add_argument("--q_strides", default="6,1,1;4,1,1;3,1,1;4,1,1;2,1,1;2,1,1")
    p.add_argument("--mask_unit_attn", nargs='+', type=lambda x: x.lower() == 'true',
                   default=[True, False, False, False, False, False, False])
    p.add_argument("--patch_kernel", nargs=3, type=int, default=[5, 1, 12])
    p.add_argument("--init_embed_dim", type=int, default=96)
    p.add_argument("--init_num_heads", type=int, default=2)
    p.add_argument("--out_embed_dims", nargs='+', type=int, default=[128, 160, 192, 192, 224, 224, 256])
    p.add_argument("--decoding_strategy", default="single")
    p.add_argument("--decoder_embed_dim", type=int, default=128)
    p.add_argument("--decoder_depth", type=int, default=1)
    p.add_argument("--decoder_num_heads", type=int, default=1)
    p.add_argument("--time_aggregations", nargs='+', type=int, required=True)
    p.add_argument("--aggregation_names", nargs='+', type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--force_zarr_rebuild", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    return p.parse_args()


# --- ZARR CREATION ---
def create_zarr_store_from_csvs(dvc_root, summary_csv_path, zarr_path):
    print("--- Starting Zarr Store Creation (with Timestamps) ---")
    summary_df = pd.read_csv(summary_csv_path)
    # take only 100
    summary_df = summary_df.head(58)
    summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
    summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])
    summary_df.sort_values(by=['cage_id', 'from_tpt'], inplace=True)

    time_delta_minutes = (summary_df['to_tpt'] - summary_df['from_tpt']).dt.total_seconds().iloc[0] / 60
    expected_chunk_size = int(round(time_delta_minutes)) + 1

    root = zarr.open(zarr_path, mode='w')

    for cage_id, group in tqdm(summary_df.groupby('cage_id'), desc="Processing cages into Zarr"):
        all_chunks_data, all_chunks_ts = [], []
        cage_csv_path = Path(dvc_root) / f"{group.iloc[0]['cage_id']}.csv"

        if not cage_csv_path.exists():
            continue

        cage_df = pd.read_csv(cage_csv_path, usecols=['timestamp'] + ELEC_COLS)
        cage_df['timestamp'] = pd.to_datetime(cage_df['timestamp'])

        for _, row in group.iterrows():
            chunk = cage_df[(cage_df['timestamp'] >= row['from_tpt']) & (cage_df['timestamp'] <= row['to_tpt'])]
            if len(chunk) != expected_chunk_size or chunk[ELEC_COLS].isnull().values.any():
                continue
            all_chunks_data.append(chunk[ELEC_COLS].values)
            all_chunks_ts.append(chunk['timestamp'].values)

        if not all_chunks_data:
            continue

        full_sequence_data = np.concatenate(all_chunks_data, axis=0).astype(np.float32)
        full_sequence_ts = np.concatenate(all_chunks_ts, axis=0)

        root.create_dataset(
            cage_id,
            data=full_sequence_data,
            chunks=(65536, None),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
        )
        root.create_dataset(
            f"{cage_id}_timestamps",
            data=full_sequence_ts,
            chunks=(65536,),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        )

    print(f"--- ✅ Zarr store successfully created at: {zarr_path} ---")


# --- CORE PROCESSING LOGIC ---
# class ModelInference:
#     def __init__(self, ckpt_path, model_args, device_id):
#         self.device = torch.device(f"cuda:{device_id}")
#         self.model = self._load_model(ckpt_path, model_args)

#     def _load_model(self, ckpt_path, model_args):
#         print(f"Loading model on GPU {self.device} for inference...")
#         try:
#             ckpt = torch.load(ckpt_path, map_location="cpu")
#             model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
#             model.load_state_dict(ckpt['model'], strict=False)
#             print(f"✅ Successfully loaded model weights onto GPU {self.device}.")
#             return model.to(self.device).eval().requires_grad_(False)
#         except Exception as e:
#             print(f"❌ FATAL ERROR on GPU {self.device}: Could not load model.")
#             print(f"Error details: {e}\nModel args used: {model_args}")
#             raise e

#     @torch.no_grad()
#     def __call__(self, window_batch):
#         batch_tensor = torch.from_numpy(window_batch).unsqueeze(1).unsqueeze(3).to(self.device, dtype=torch.float32)

#         outputs = self.model.forward_encoder(batch_tensor, mask_ratio=0.0, return_intermediates=True)
#         intermediate_levels = outputs[-1]
#         pooled_levels = [feat.flatten(1, -2).mean(1).cpu().numpy().astype(np.float16) for feat in intermediate_levels]
#         return pooled_levels
    
class ModelInference:
    def __init__(self, ckpt_path, model_args, device_id):
        self.device = torch.device(f"cuda:{device_id}")
        self.model = self._load_model(ckpt_path, model_args)

    def _load_model(self, ckpt_path, model_args):
        print(f"Loading model on GPU {self.device} for inference...")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = models_defs.__dict__[model_args.get("model", "hbehavemae")](**model_args)
            model.load_state_dict(ckpt['model'], strict=False)
            print(f"✅ Successfully loaded model weights onto GPU {self.device}.")
            
            # --- FIX: Convert the entire model to half-precision (float16) ---
            model = model.half()
            
            return model.to(self.device).eval().requires_grad_(False)
        except Exception as e:
            print(f"❌ FATAL ERROR on GPU {self.device}: Could not load model.")
            print(f"Error details: {e}\nModel args used: {model_args}")
            raise e

    @torch.no_grad()
    def __call__(self, window_batch):
        # The input tensor is correctly converted to float16 here
        batch_tensor = torch.from_numpy(window_batch).unsqueeze(1).unsqueeze(3).to(self.device, dtype=torch.float16)
        outputs = self.model.forward_encoder(batch_tensor, mask_ratio=0.0, return_intermediates=True)
        intermediate_levels = outputs[-1]
        # The output is converted back to float32 for CPU-based operations like pooling
        pooled_levels = [feat.flatten(1, -2).mean(1).float().cpu().numpy() for feat in intermediate_levels]
        return pooled_levels


def process_cage(cage_id, dask_array_data, dask_array_ts, model_inference, num_frames, batch_size, num_levels, norm_stats):
    if dask_array_data.shape[0] < num_frames:
        return None, None, None

    if norm_stats:
        dask_array_data = (dask_array_data - norm_stats['mean']) / norm_stats['std']

    windows = da.lib.stride_tricks.sliding_window_view(
        dask_array_data,
        window_shape=(num_frames, dask_array_data.shape[1])
    )[:, 0, :, :]
    timestamps = dask_array_ts[:windows.shape[0]]

    all_level_embeds = [[] for _ in range(num_levels)]

    with tqdm(range(0, windows.shape[0], batch_size), desc=f"Processing {cage_id}", leave=False) as pbar:
        for i in pbar:
            batch = windows[i:i + batch_size].compute()
            pooled_results = model_inference(batch)
            for level_idx in range(num_levels):
                all_level_embeds[level_idx].append(pooled_results[level_idx])

    cage_results = {}
    for i in range(num_levels):
        cage_results[f'level{i + 1}'] = np.concatenate(all_level_embeds[i], axis=0) if all_level_embeds[i] else np.array([], dtype=np.float16)

    cage_results['comb'] = np.concatenate(
        [cage_results[f"level{i + 1}"] for i in range(num_levels)],
        axis=1
    ) if all(cage_results[f"level{i + 1}"].size > 0 for i in range(num_levels)) else np.array([], dtype=np.float16)

    computed_timestamps = timestamps.compute()
    return cage_id, cage_results, computed_timestamps


def gpu_worker(gpu_id, work_queue, results_dict, ckpt_path, model_args, num_frames, batch_size, num_levels, zarr_path, args):
    try:
        model_inference = ModelInference(ckpt_path, model_args, gpu_id)
        zarr_store = zarr.open(zarr_path, mode='r')

        norm_stats = None
        if not args.no_normalization:
            if args.stats_path and os.path.exists(args.stats_path):
                norm_stats = np.load(args.stats_path, allow_pickle=True).item()
                print(f"GPU {gpu_id}: Loaded norm stats (mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f})")
            else:
                print(f"GPU {gpu_id}: WARNING - Normalization enabled, but --stats_path not provided or file not found.")

        while True:
            cage_id = work_queue.get()
            if cage_id is None:
                break
            dask_array_data = da.from_zarr(zarr_store[cage_id])
            dask_array_ts = da.from_zarr(zarr_store[f"{cage_id}_timestamps"])
            cage_id_out, results, timestamps = process_cage(
                cage_id, dask_array_data, dask_array_ts,
                model_inference, num_frames, batch_size, num_levels, norm_stats
            )
            if cage_id_out is not None:
                results_dict[cage_id_out] = (results, timestamps)

    except Exception as e:
        print(f"❌ Error in GPU worker {gpu_id}: {e}")
        import traceback
        traceback.print_exc()


def extract_embeddings_multiprocess(zarr_path, ckpt_path, model_args, num_frames, batch_size, gpu_ids, num_levels, args):
    zarr_store = zarr.open(zarr_path, mode='r')
    cage_ids = [k for k in zarr_store.keys() if not k.endswith('_timestamps')]

    manager = mp.Manager()
    work_queue, results_dict = manager.Queue(), manager.dict()

    for cage_id in cage_ids:
        work_queue.put(cage_id)
    for _ in gpu_ids:
        work_queue.put(None)

    processes = [
        mp.Process(target=gpu_worker, args=(gid, work_queue, results_dict, ckpt_path, model_args, num_frames, batch_size, num_levels, zarr_path, args))
        for gid in gpu_ids
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    all_cage_results = dict(results_dict)
    combined_results = {'frame_map': {}, 'timestamps': []}
    level_names = [f"level{i + 1}" for i in range(num_levels)] + ['comb']

    for name in level_names:
        combined_results[name] = []

    ptr = 0
    for cage_id in sorted(all_cage_results.keys()):
        cage_data, cage_timestamps = all_cage_results[cage_id]
        if cage_data and 'comb' in cage_data:
            num_windows = cage_data['comb'].shape[0]
            if num_windows > 0:
                combined_results['frame_map'][cage_id] = (ptr, ptr + num_windows)
                ptr += num_windows
                for name in level_names:
                    combined_results[name].append(cage_data[name])
                combined_results['timestamps'].append(cage_timestamps)

    for name in level_names:
        combined_results[name] = np.concatenate(combined_results[name]) if combined_results[name] else np.array([], dtype=np.float16)

    combined_results['timestamps'] = np.concatenate(combined_results['timestamps']) if combined_results['timestamps'] else np.array([], dtype='datetime64[ns]')
    return combined_results


# --- PLOTTING & SUMMARY ---
def _plot_heatmap(embeddings, timestamps, title, output_path):
    if embeddings.size == 0:
        print(f"Skipping empty plot for: {title}")
        return
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(embeddings.T, cmap='viridis', cbar=True, ax=ax, xticklabels=False)
    tick_indices = np.linspace(0, embeddings.shape[0] - 1, 10, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(pd.to_datetime(timestamps[tick_indices]).strftime('%Y-%m-%d %H:%M'), rotation=30, ha='right')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Embedding Dimension", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Heatmap saved to {output_path}")


def generate_day_night_heatmaps(embeddings, timestamps, frame_map, output_dir, agg_name, level_name):
    print(f"--- Generating Day/Night Heatmaps for {level_name} ---")
    if timestamps.size == 0:
        return
    ts_series = pd.to_datetime(pd.Series(timestamps))
    for cage_id, (start_idx, end_idx) in tqdm(frame_map.items(), desc=f"Plotting cages for {level_name}"):
        cage_embeds = embeddings[start_idx:end_idx]
        cage_ts = timestamps[start_idx:end_idx]
        cage_ts_series = ts_series.iloc[start_idx:end_idx]
        day_mask = (cage_ts_series.dt.hour >= 7) & (cage_ts_series.dt.hour < 19)
        night_mask = ~day_mask
        _plot_heatmap(
            cage_embeds[day_mask.values],
            cage_ts[day_mask.values],
            f"Embeddings: {cage_id} ({agg_name}, {level_name}) - DAY (7am-7pm)",
            output_dir / f"heatmap{agg_name}{level_name}{cage_id}DAY.png"
        )
        _plot_heatmap(
            cage_embeds[night_mask.values],
            cage_ts[night_mask.values],
            f"Embeddings: {cage_id} ({agg_name}, {level_name}) - NIGHT (7pm-7am)",
            output_dir / f"heatmap{agg_name}{level_name}_{cage_id}_NIGHT.png"
        )


def print_embedding_summary(results, num_levels, agg_name):
    print(f"\n--- 🔬 Embedding Summary for Aggregation: {agg_name} ---")
    level_names = [f"level{i + 1}" for i in range(num_levels)] + ['comb']
    for level_name in level_names:
        embeds = results.get(level_name)
        if embeds is not None and embeds.size > 0:
            print(f"\n[+] Level: {level_name}")
            print(f"  - Shape: {embeds.shape} (Windows, Dimensions)")
            print(f"  - Dtype: {embeds.dtype}")
            print(f"  - Example Head (first 2 rows, first 8 dims):\n{embeds[:2, :8]}")

    timestamps = results.get('timestamps')
    if timestamps is not None and timestamps.size > 0:
        print("\n[+] Timestamps")
        print(f"  - Shape: {timestamps.shape}")
        print(f"  - Dtype: {timestamps.dtype}")
        print(f"  - Time Range: {pd.to_datetime(timestamps[0])} -> {pd.to_datetime(timestamps[-1])}")

    frame_map = results.get('frame_map')
    if frame_map:
        print("\n[+] Frame Map")
        print(f"  - Contains mappings for {len(frame_map)} cages.")
        for i, (cage, indices) in enumerate(frame_map.items()):
            if i >= 3:
                break
            print(f"  - Example: Cage '{cage}' -> Indices {indices}")
    print("-" * (35 + len(agg_name)))


def parse_q_strides(q_strides_str):
    if isinstance(q_strides_str, str):
        stages = q_strides_str.split(';')
        parsed = [tuple(int(x) for x in stage.split(',')) for stage in stages]
        return parsed, len(parsed) + 1
    return q_strides_str, len(q_strides_str) + 1


def main():
    mp.set_start_method('spawn', force=True)
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_zarr_rebuild or not Path(args.zarr_path).exists():
        create_zarr_store_from_csvs(args.dvc_root, args.summary_csv, args.zarr_path)
    else:
        print(f"✅ Using existing Zarr store at {args.zarr_path}.")

    gpu_ids = args.gpu_ids if args.multi_gpu and torch.cuda.is_available() else [0]
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    base_model_args = vars(args)
    base_model_args['q_strides'], num_levels = parse_q_strides(args.q_strides)
    print(f"✅ Model will have {num_levels} hierarchical levels.")

    if len(args.time_aggregations) != len(args.aggregation_names):
        raise ValueError("Mismatch between --time_aggregations and --aggregation_names counts.")

    for num_frames, agg_name in zip(args.time_aggregations, args.aggregation_names):
        print(f"\n{'=' * 20} Processing Aggregation: {agg_name} ({num_frames} minutes) {'=' * 20}")
        summary_df_check = pd.read_csv(args.summary_csv)
        #  take only 100
        summary_df_check = summary_df_check.head(58)
        time_delta_minutes = (pd.to_datetime(summary_df_check['to_tpt']) - pd.to_datetime(summary_df_check['from_tpt'])).dt.total_seconds().iloc[0] / 60
        expected_frames_from_summary = int(round(time_delta_minutes)) + 1

        if num_frames % expected_frames_from_summary != 0:
            print(f"❌ FATAL MISMATCH: Aggregation window ({num_frames}) is not a multiple of the base chunk size ({expected_frames_from_summary}).")
            continue

        model_args = base_model_args.copy()
        model_args['input_size'] = [num_frames, 1, 12]

        results = extract_embeddings_multiprocess(
            args.zarr_path, args.ckpt_path, model_args,
            num_frames, args.batch_size, gpu_ids, num_levels, args
        )

        print_embedding_summary(results, num_levels, agg_name)

        level_names = [f"level{i + 1}" for i in range(num_levels)] + ['comb']
        for level_name in level_names:
            filename = f"embeddings_{agg_name}_{level_name}.npz"
            filepath = output_dir / filename
            np.savez_compressed(
                filepath,
                frame_map=results['frame_map'],
                embeddings=results[level_name],
                timestamps=results['timestamps']
            )
            print(f"Saved {filename} – shape {results[level_name].shape}")

            generate_day_night_heatmaps(
                results[level_name],
                results['timestamps'],
                results['frame_map'],
                output_dir,
                agg_name,
                level_name
            )

    print("\n\n🎉 All extractions and visualizations complete!")


if __name__ == "__main__":
    main()
