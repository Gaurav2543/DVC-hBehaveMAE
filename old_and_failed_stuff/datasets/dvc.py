# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# For more details on our work, please refer to:
# Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders
# Lucas Stoffl, Andy Bonnetto, Stéphane d'Ascoli, Alexander Mathis
# https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1
# --------------------------------------------------------

import __future__

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import os
import random

from .pose_traj_dataset import BasePoseTrajDataset
from .DVC.load_dvc import load_data as load_dvc_data


class DVCDataset(BasePoseTrajDataset):
    """
    DVC dataset
    """

    DEFAULT_FRAME_RATE = 4
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 1
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    SAMPLE_LEN = 1440  # per day

    STR_BODY_PARTS = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",   
    ]

    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        **kwargs
    ):
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )

        self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed

        self.mode = mode

        self.load_data()

        self.preprocess()

    def load_data(self) -> None:
        """Loads dataset"""
        """
        Here I want to read the csv  file "summary_table_imputed_with_sets.csv" . In this file, I have the following columns: cage, day and sets.
        I want to create 2 lists : one for the training and one for the testing according to sets (0=training, 1= testing, NA= not used).
        I want the name of the element in the list to be the cage name and the values to be the days. I will then use these lists to load the data.
        """

       # Read the CSV file
        csv_path = os.path.join(self.path, "summary_table_imputed_with_sets_sub.csv")
        df = pd.read_csv(csv_path)

        # Create training and testing lists
        training_list = {}
        testing_list = {}

        for _, row in df.iterrows():
            cage = row['cage']
            day = row['day']
            set_type = row['sets']

            if pd.isna(set_type):
                continue
            elif set_type == 0:
                if cage not in training_list:
                    training_list[cage] = []
                training_list[cage].append(day)
            elif set_type == 1:
                if cage not in testing_list:
                    testing_list[cage] = []
                testing_list[cage].append(day)
        del df 
        if self.mode == "pretrain":
            self.raw_data = [data.values[:,0:-2] for data in load_dvc_data(self.path, training_list).values()]
        elif self.mode == "test":
            self.raw_data = [data.values[:,0:-2] for data in load_dvc_data(self.path, testing_list).values()]
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))


    def featurise_keypoints(self, keypoints):
        keypoints = self.normalize(keypoints) #TODO: normalize across samples
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints

    def preprocess(self):
        """
        Does initial preprocessing on entire dataset.
        """
        # add 1 dimension as number of samples to self.raw_data
        sequences = [
            data.reshape(1, -1, self.NUM_INDIVIDUALS, self.NUM_KEYPOINTS, self.KPTS_DIMENSIONS)
            for data in self.raw_data
        ]        
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window
        for seq_ix, vec_seq in enumerate(sequences):
            # Preprocess sequences
            vec_seq = vec_seq.squeeze(0)  # Remove the extra dimension

            # Pads the beginning and end of the sequence with duplicate frames
            if sub_seq_length < 40:
                pad_length = sub_seq_length
            else:
                pad_length = 40

            pad_width = [(pad_length // 2, pad_length - 1 - pad_length // 2)] + [(0, 0)] * (vec_seq.ndim - 1)
            pad_vec = np.pad(
                vec_seq,
                pad_width,
                mode="edge",
            )

            seq_keypoints.append(pad_vec)

            keypoints_ids.extend(
                [
                    (seq_ix, i)
                    for i in np.arange(
                        0, len(pad_vec) - sub_seq_length + 1, sliding_window
                    )
                ]
            )
            print(f"Sequence {seq_ix}: len(vec_seq)={len(vec_seq)}, sub_seq_length={sub_seq_length}, sliding_window={sliding_window}")

        seq_keypoints = [np.array(seq, dtype=np.float32) for seq in seq_keypoints]
        print("Number of sequences: ", len(seq_keypoints))
        print("Number of keypoints: ", len(keypoints_ids))
        self.items = list(np.arange(len(keypoints_ids)))

        self.seq_keypoints = seq_keypoints
        self.keypoints_ids = keypoints_ids
        self.n_frames = len(self.keypoints_ids)

        del self.raw_data

    @staticmethod
    def fill_holes():
        pass

    def normalize(self, data):
        # features are already percentages
        return data / 100
    
    def unnormalize(self):
        pass

    def transform_to_centered_data(self):
        pass

    def transform_to_centeralign_components(self):
        pass

    def prepare_subsequence_sample(self, sequence: np.ndarray):
        """
        Returns a training sample
        """

        feats = self.featurise_keypoints(sequence)

        feats = feats.reshape(self.max_keypoints_len, self.NUM_INDIVIDUALS, -1)

        return feats

    def __getitem__(self, idx: int):

        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0]][
            subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len
        ]
        inputs = self.prepare_subsequence_sample(subsequence)

        return inputs, []
