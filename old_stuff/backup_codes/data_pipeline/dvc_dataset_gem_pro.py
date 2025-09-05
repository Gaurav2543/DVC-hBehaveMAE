from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Assuming load_dvc_data is in this path and can handle loading data for a specific cage
from data_pipeline.load_dvc import load_dvc_data
from data_pipeline.pose_traj_dataset import BasePoseTrajDataset

class DVCDataset(BasePoseTrajDataset):
    """
    Corrected DVC dataset for hBehaveMAE.
    This version correctly handles discrete time-series chunks as defined in the summary CSV.
    """
    DEFAULT_FRAME_RATE = 1/60  # 1 sample per minute
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 1
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, 4, 3, KPTS_DIMENSIONS) # 4x3 grid

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        summary_csv: str,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 400, # This is the "page" size for the model
        sliding_window: int = 1,
        normalization_method: str = 'percentage',
        precomputed_stats_path: str = None,
        **kwargs
    ):
        # Determine the length of each continuous data chunk from the filename
        match = re.search(r'_(\d+)\.csv', summary_csv)
        if match:
            self.SAMPLE_LEN = int(match.group(1))
            print(f"Data chunks are expected to be {self.SAMPLE_LEN} minutes long (the 'Book' size).")
        else:
            raise ValueError(f"Could not determine sample length from summary file: {summary_csv}")
        
        # num_frames is the subsequence length the model sees (the 'Page' size)
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE
        self.mode = mode
        self.normalization_method = normalization_method
        self.precomputed_stats_path = precomputed_stats_path
        self.mean_val, self.std_val = None, None

        self.load_data(summary_csv)

        if self.normalization_method == 'z_score':
            self._prepare_zscore_stats()

        self.preprocess()

    def _prepare_zscore_stats(self):
        # This function remains correct
        if self.mode == 'pretrain':
            if self.precomputed_stats_path and os.path.exists(self.precomputed_stats_path):
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val, self.std_val = stats['mean'], stats['std']
            elif hasattr(self, 'raw_data') and self.raw_data:
                all_train_data = np.concatenate([seq.flatten() for seq in self.raw_data])
                self.mean_val = np.mean(all_train_data)
                self.std_val = np.std(all_train_data)
                if self.std_val == 0: self.std_val = 1e-6
                print(f"Calculated Z-score stats: Mean={self.mean_val}, Std={self.std_val}")
                if self.precomputed_stats_path:
                    np.save(self.precomputed_stats_path, {'mean': self.mean_val, 'std': self.std_val})

    def load_data(self, summary_csv):
        """
        CORRECTED: Loads data by treating each row in the summary CSV as a distinct sample,
        respecting the specified time chunks.
        """
        print(f"Reading summary file from: {summary_csv}")
        summary_df = pd.read_csv(summary_csv, low_memory=False)
        # # tae only 1000 entries for testing
        # summary_df = summary_df.head(1000)
        
        summary_df['from_tpt'] = pd.to_datetime(summary_df['from_tpt'])
        summary_df['to_tpt'] = pd.to_datetime(summary_df['to_tpt'])

        self.raw_data = []
        
        # Group by cage to load per-minute data efficiently
        for cage_id, group in tqdm(summary_df.groupby('cage_id'), desc="Loading cage data"):
            # It's more efficient to load the full per-minute file for a cage once
            # Note: This assumes load_dvc_data can be adapted or you have a similar utility
            # to load the raw, per-minute CSV for a given cage_id.
            try:
                # Load the raw per-minute data for the entire cage
                # This is a placeholder for your actual raw data loading function
                raw_cage_df = pd.read_csv(os.path.join(self.path, f"{cage_id}.csv"), parse_dates=['timestamp'])
                raw_cage_df = raw_cage_df.set_index('timestamp')
            except FileNotFoundError:
                print(f"Warning: Raw data file for cage {cage_id} not found. Skipping.")
                continue

            # Iterate through each defined chunk in the summary file for this cage
            for _, row in group.iterrows():
                start_time, end_time = row['from_tpt'], row['to_tpt']
                
                # Slice the raw data to get the exact time chunk
                chunk_df = raw_cage_df[start_time:end_time]
                
                # Validate the length of the chunk
                if len(chunk_df) != self.SAMPLE_LEN:
                    print(f"Warning: Chunk for {cage_id} from {start_time} has length {len(chunk_df)}, expected {self.SAMPLE_LEN}. Skipping.")
                    continue
                
                electrode_cols = [f"v_{i}" for i in range(1, 13)]
                chunk_data = chunk_df[electrode_cols].values.astype(np.float32)
                self.raw_data.append(chunk_data)

        print(f"Successfully loaded {len(self.raw_data)} continuous data chunks.")

    def preprocess(self):
        """
        This function is now correct because load_data provides a clean list
        of continuous data chunks with the correct SAMPLE_LEN.
        """
        if not hasattr(self, 'raw_data') or not self.raw_data:
            print("No raw data to preprocess.")
            self.seq_keypoints, self.keypoints_ids, self.items = [], [], []
            self.n_frames = 0
            return
        
        sequences_gridded = []
        for data_chunk in self.raw_data:
            # Reshape to (T, I, 4, 3, D)
            gridded_chunk = data_chunk.reshape(self.SAMPLE_LEN, self.NUM_INDIVIDUALS, 4, 3, self.KPTS_DIMENSIONS)
            sequences_gridded.append(gridded_chunk)

        seq_keypoints_list = []
        keypoints_ids_list = []
        sub_seq_length = self.max_keypoints_len # This comes from num_frames
        sliding_w = self.sliding_window

        for seq_ix, vec_seq_chunk in enumerate(sequences_gridded):
            # Padding is applied to handle subsequences at the edges of the chunk
            pad_length = sub_seq_length // 2
            pad_width = ((pad_length, pad_length), (0, 0), (0, 0), (0, 0), (0,0))
            pad_vec = np.pad(vec_seq_chunk, pad_width, mode="edge")
            seq_keypoints_list.append(pad_vec)

            # Generate IDs for the sliding windows
            num_possible_starts = len(vec_seq_chunk) # Iterate over the original length
            for i in range(0, num_possible_starts, sliding_w):
                keypoints_ids_list.append((seq_ix, i))
            
        self.seq_keypoints = seq_keypoints_list
        self.keypoints_ids = keypoints_ids_list
        self.n_frames = len(self.keypoints_ids)
        self.items = list(range(self.n_frames))
        
        del self.raw_data # Clean up memory

    # featurise_keypoints, normalize, unnormalize, prepare_subsequence_sample, and __getitem__
    # methods remain the same as they operate on the correctly preprocessed data.
    def featurise_keypoints(self, keypoints):
        keypoints_normalized = self.normalize(keypoints.astype(np.float32))
        keypoints_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)
        return keypoints_tensor

    def normalize(self, data: np.ndarray) -> np.ndarray:
        method = self.normalization_method
        if method in {"none", "raw", None}: return data.astype(np.float32)
        if method == "percentage": return data.astype(np.float32) / 100.0
        if method in {"z_score", "global_z_score"}:
            if self.mean_val is None or self.std_val is None: raise ValueError("Global μ/σ not initialised.")
            return (data - self.mean_val) / self.std_val
        if method == "local_z_score":
            mean, std = float(data.mean()), float(data.std())
            if std == 0: std = 1e-6
            return (data - mean) / std
        raise ValueError(f"Unknown normalisation method: {method}")

    def unnormalize(self, data: np.ndarray) -> np.ndarray:
        method = self.normalization_method

        if method in {"none", "raw", None}:
            return data

        if method == "percentage":
            return data * 100.0

        if method in {"z_score", "global_z_score"}:
            if self.mean_val is None or self.std_val is None:
                raise ValueError("Global μ/σ not initialised.")
            return (data * self.std_val) + self.mean_val

        if method == "local_z_score":
            mean = float(data.mean())
            std = float(data.std())
            if std == 0:
                std = 1e-6
            return (data * std) + mean

        raise ValueError(f"Unknown normalisation method: {method}")

    def prepare_subsequence_sample(self, sequence: np.ndarray) -> torch.Tensor:
        feats_tensor = self.featurise_keypoints(sequence)
        feats_reshaped = feats_tensor.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)
        return feats_reshaped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list]:
        if not self.keypoints_ids: raise IndexError("keypoints_ids is empty.")
        seq_idx, frame_start_in_original = self.keypoints_ids[idx]
        padded_sequence = self.seq_keypoints[seq_idx]
        
        # The start index needs to be adjusted for the padding
        pad_length_start = self.max_keypoints_len // 2
        frame_start_in_padded = frame_start_in_original
        
        subsequence = padded_sequence[frame_start_in_padded : frame_start_in_padded + self.max_keypoints_len]
        
        if subsequence.shape[0] != self.max_keypoints_len:
            raise ValueError(f"Subsequence shape error at index {idx}")

        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []