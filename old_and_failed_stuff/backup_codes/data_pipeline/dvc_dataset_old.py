from pathlib import Path
import os # Make sure os is imported

import numpy as np
import pandas as pd
import torch

from data_pipeline.load_dvc import load_dvc_data
from data_pipeline.pose_traj_dataset import BasePoseTrajDataset

class DVCDataset(BasePoseTrajDataset):
    DEFAULT_FRAME_RATE = 4 # DVC data is sampled at 1/60Hz
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 1 # Each electrode has 1 activation value
    NUM_INDIVIDUALS = 1 # Data per cage, treated as a single entity for input
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    SAMPLE_LEN = 1440  # Timesteps per day

    STR_BODY_PARTS = [
        "V1", "V2", "V3", "V4", "V5", "V6",
        "V7", "V8", "V9", "V10", "V11", "V12",
    ]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True, # This 'scale' arg might become redundant if normalization_method is used
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        summary_csv: str = None, # Path to the summary CSV file
        normalization_method: str = 'percentage', # NEW: 'percentage' or 'z_score'
        precomputed_stats_path: str = None, # NEW: Path to load/save Z-score stats
        **kwargs
    ):
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE
        self.mode = mode
        self.normalization_method = normalization_method
        self.precomputed_stats_path = precomputed_stats_path
        self.mean_val = None
        self.std_val = None

        self.load_data(summary_csv)

        if self.normalization_method == 'z_score':
            self._prepare_zscore_stats()

        self.preprocess() # Processes raw_data into seq_keypoints

    def _prepare_zscore_stats(self):
        if self.mode == 'pretrain':
            if self.precomputed_stats_path and os.path.exists(self.precomputed_stats_path):
                print(f"Loading Z-score stats from {self.precomputed_stats_path}")
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val = stats['mean']
                self.std_val = stats['std']
            elif hasattr(self, 'raw_data') and self.raw_data:
                print("Calculating Z-score stats from training data...")
                # Temporarily concatenate all training sequences for stat calculation
                # This assumes self.raw_data is a list of numpy arrays for DVC
                all_train_data = np.concatenate([seq.flatten() for seq in self.raw_data])
                self.mean_val = np.mean(all_train_data)
                self.std_val = np.std(all_train_data)
                if self.std_val == 0: # Avoid division by zero
                    self.std_val = 1e-6
                print(f"Calculated Mean: {self.mean_val}, Std: {self.std_val}")
                if self.precomputed_stats_path:
                    print(f"Saving Z-score stats to {self.precomputed_stats_path}")
                    # Ensure directory exists if precomputed_stats_path includes a directory
                    dirname = os.path.dirname(self.precomputed_stats_path)
                    if dirname and not os.path.exists(dirname): # Check if dirname is not empty
                        os.makedirs(dirname, exist_ok=True)
                    # --- MODIFICATION FOR ROBUST SAVE ---
                    try:
                        with open(self.precomputed_stats_path, 'wb') as f: # Open in binary write mode
                            np.save(f, {'mean': self.mean_val, 'std': self.std_val})
                        # Optionally, try to force a sync to disk if on a system that supports it
                        # This is OS-dependent and might not always be available or effective
                        if hasattr(os, 'sync'):
                            os.sync() 
                        print(f"Successfully saved stats to {self.precomputed_stats_path}")
                    except Exception as e:
                        print(f"ERROR saving stats to {self.precomputed_stats_path}: {e}")
                        raise
                # --- END MODIFICATION ---
                    # np.save(self.precomputed_stats_path, {'mean': self.mean_val, 'std': self.std_val})
            else:
                raise ValueError("Z-score stats cannot be computed: raw_data not loaded or empty for pretrain mode.")
        
        if self.mean_val is None or self.std_val is None:
             raise ValueError("Mean and Std for Z-score normalization are not set.")


    def load_data(self, summary_csv):
        # csv_path="/scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv"
        csv_path= summary_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print("Reading CSV from: ", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.head(1000)  # For testing, load only first 1000 rows
        print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")

        training_list = {}
        for _, row in df.iterrows():
            cage = row['cage_id']
            day = row['from_tpt'].split(' ')[0]
            if cage not in training_list:
                training_list[cage] = []
            training_list[cage].append(day)
        del df

        if self.mode == "pretrain":
            loaded_data_dict = load_dvc_data(self.path, training_list)
            self.raw_data = [df[[col for col in df.columns if col.startswith('v_')]].values.astype(np.float32) for df in loaded_data_dict.values()]
            print(f"Loaded {len(self.raw_data)} training sequences for pretrain mode.")
            if not self.raw_data:
                print("Warning: No training data loaded. Check CSV and file paths.")
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def featurise_keypoints(self, keypoints):
        # Normalization is now handled in the normalize method
        keypoints_normalized = self.normalize(keypoints.astype(np.float32))
        keypoints_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)
        return keypoints_tensor

    def preprocess(self):

        if not hasattr(self, 'raw_data') or not self.raw_data:
            print(f"No raw data to preprocess for mode: {self.mode}. Skipping preprocessing.")
            self.seq_keypoints = []
            self.keypoints_ids = []
            self.n_frames = 0
            self.items = []
            return
        
        # Reshape each sequence to have NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS
        # For DVC, this might mean ensuring it's (num_timesteps, 1, 12, 1) before flattening for padding
        sequences_reshaped = []
        for data_day in self.raw_data: # data_day is (timesteps_in_day, 12_electrodes)
            # We need to add NUM_INDIVIDUALS and KPTS_DIMENSIONS
            # Assuming data_day is (T, K), reshape to (T, I, K, D)
            # For DVC: (T, 12) -> (T, 1, 12, 1)
            if data_day.ndim == 2: # Expected (T, K)
                 sequences_reshaped.append(data_day.reshape(data_day.shape[0], self.NUM_INDIVIDUALS, self.NUM_KEYPOINTS, self.KPTS_DIMENSIONS))
            elif data_day.ndim == 4 and data_day.shape[1:] == (self.NUM_INDIVIDUALS, self.NUM_KEYPOINTS, self.KPTS_DIMENSIONS):
                 sequences_reshaped.append(data_day)
            else:
                raise ValueError(f"Unexpected data shape in raw_data: {data_day.shape}")


        seq_keypoints_list = []
        keypoints_ids_list = []
        sub_seq_length = self.max_keypoints_len # e.g., num_frames from args
        sliding_w = self.sliding_window

        for seq_ix, vec_seq_day in enumerate(sequences_reshaped):
            # vec_seq_day is now (timesteps_in_day, NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
            # Flatten the last 3 dims for padding: (timesteps_in_day, I*K*D)
            original_shape_spatial = vec_seq_day.shape[1:]
            vec_seq_flat_spatial = vec_seq_day.reshape(vec_seq_day.shape[0], -1)

            if sub_seq_length < 40: # This padding logic might need review based on sequence nature
                pad_length = sub_seq_length
            else:
                pad_length = 40

            pad_width_time = (pad_length // 2, pad_length - 1 - (pad_length // 2))
            
            # Pad only along the time axis (axis 0)
            pad_vec_flat_spatial = np.pad(
                vec_seq_flat_spatial,
                (pad_width_time, (0,0)), # Pad only time axis
                mode="edge",
            )
            
            # Reshape back after padding: (padded_timesteps, I, K, D)
            pad_vec_reshaped = pad_vec_flat_spatial.reshape(pad_vec_flat_spatial.shape[0], *original_shape_spatial)
            seq_keypoints_list.append(pad_vec_reshaped.astype(np.float32))

            # Generate IDs for sliding windows
            num_possible_starts = len(pad_vec_reshaped) - sub_seq_length + 1
            for i in range(0, num_possible_starts, sliding_w):
                keypoints_ids_list.append((seq_ix, i))
            
            # print(f"Sequence {seq_ix}: original_len={vec_seq_day.shape[0]}, padded_len={pad_vec_reshaped.shape[0]}, num_sub_seqs={num_possible_starts // sliding_w}")


        self.seq_keypoints = seq_keypoints_list # List of (Padded_T, I, K, D) arrays
        self.keypoints_ids = keypoints_ids_list # List of (seq_idx, start_frame_in_padded_seq)
        
        if not self.keypoints_ids:
            print(f"Warning: No keypoint IDs generated for mode {self.mode}. This might indicate an issue with data loading or preprocessing parameters (num_frames vs sequence length).")
            self.n_frames = 0
        else:
            self.n_frames = len(self.keypoints_ids)

        self.items = list(np.arange(self.n_frames))
        
        # print(f"Preprocessing done. Mode: {self.mode}. Number of sequences: {len(self.seq_keypoints)}. Total sub-sequences (items): {self.n_frames}")

        # It's good practice to delete raw_data if it's large and no longer needed in this form
        if hasattr(self, 'raw_data'):
            del self.raw_data


    @staticmethod
    def fill_holes(): # DVC data is imputed, so this might not be strictly necessary
        print("DVCData: fill_holes called, but data is expected to be imputed.")
        pass

    def normalize(self, data: np.ndarray) -> np.ndarray:
        method = self.normalization_method

        if method in {"none", "raw", None}:
            return data.astype(np.float32)

        if method == "percentage":
            return data.astype(np.float32) / 100.0

        if method in {"z_score", "global_z_score"}:
            if (self.mean_val is None or self.std_val is None) \
                and self.precomputed_stats_path \
                and os.path.exists(self.precomputed_stats_path):
                stats = np.load(self.precomputed_stats_path, allow_pickle=True).item()
                self.mean_val = stats["mean"]
                self.std_val  = stats["std"]
            if self.mean_val is None or self.std_val is None:
                raise ValueError("Global μ/σ not initialised.")
            return (data - self.mean_val) / self.std_val

        if method == "local_z_score":
            mean = float(data.mean())
            std = float(data.std())
            if std == 0:
                std = 1e-6
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
        # keypoints are already float32 from preprocessing
        feats_tensor = self.featurise_keypoints(sequence) # Normalization happens here

        # Reshape for the model: (num_frames, NUM_INDIVIDUALS, NUM_KEYPOINTS * KPTS_DIMENSIONS)
        # For DVC: (num_frames, 1, 12*1) -> (num_frames, 1, 12)
        feats_reshaped = feats_tensor.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)
        return feats_reshaped

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list]:
        if not self.keypoints_ids:
             raise IndexError(f"Attempting to get item {idx} but keypoints_ids is empty. Dataset might not have been loaded or preprocessed correctly.")

        seq_idx, frame_start_in_padded_seq = self.keypoints_ids[idx]
        
        # Retrieve the correct padded day sequence
        padded_day_sequence = self.seq_keypoints[seq_idx] 
        
        # Extract the subsequence
        # Padded_day_sequence shape: (Padded_T_day, I, K, D)
        # We need: (num_frames, I, K, D)
        subsequence = padded_day_sequence[
            frame_start_in_padded_seq : frame_start_in_padded_seq + self.max_keypoints_len
        ]
        
        # Ensure subsequence has the correct number of frames
        if subsequence.shape[0] != self.max_keypoints_len:
            raise ValueError(f"Subsequence at index {idx} has incorrect length: {subsequence.shape[0]}, expected {self.max_keypoints_len}. seq_idx={seq_idx}, frame_start={frame_start_in_padded_seq}")

        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, [] # Second element is usually for labels, empty for self-supervised MAE