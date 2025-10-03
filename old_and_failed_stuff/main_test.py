# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# For more details on our work, please refer to:
# Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders
# Lucas Stoffl, Andy Bonnetto, Stéphane d'Ascoli, Alexander Mathis
# https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1
# --------------------------------------------------------

import argparse
import math
import os
import tempfile
from functools import reduce
from operator import mul

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr as pathmgr
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

import util.misc as misc
from datasets import hbabel as hbabel
from datasets import mabe22_mice as mice
from datasets import shot7m2 as shot7m2
from datasets import dvc as dvc
from datasets.DVC.load_dvc import load_data
from models import models_defs
from models.hbehave_mae import apply_fusion_head
from models.hiera_utils import conv_nd
from util.logging import master_print as print
from util.misc import parse_tuples, str2bool
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser("hBehaveMAE embeddings extraction", add_help=False)
    parser.add_argument(
        "--dataset",
        default="shot7m2",
        type=str,
        help="Type of dataset [mabe_mice, shot7m2, hbabel, dvc]",
    )
    parser.add_argument("--joints3d_procrustes", default=True, type=str2bool)

    parser.add_argument(
        "--embedsum",
        default=False,
        type=str2bool,
        help="single embeddings will be summed up instead of concatenated",
    )

    parser.add_argument(
        "--fast_inference",
        default=False,
        type=str2bool,
        help="if set, we do not perform any embedding averaging, but only take the middle embedding",
    )

    parser.add_argument(
        "--combine_embeddings",
        default=False,
        type=str2bool,
        help="combine embeddings from different hierarchical levels and save them",
    )
    parser.add_argument(
        "--fusion_head",
        default=False,
        type=str2bool,
        help="combined embeddings are created by (trained) fusion head",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    # Model parameters
    parser.add_argument(
        "--model",
        default="gen_hiera",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    # test a non-hierarchical model ("BehaveMAE")
    parser.add_argument("--non_hierarchical", default=False, type=str2bool)

    parser.add_argument(
        "--path_to_data_dir",
        default="",
        help="path where to load data from",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--num_frames", default=400, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--distributed", action="store_true")

    # hBehaveMAE specific parameters
    parser.add_argument("--input_size", default=(600, 3, 24), nargs="+", type=int)
    parser.add_argument("--stages", default=(2, 3, 4), nargs="+", type=int)
    parser.add_argument(
        "--q_strides", default=[(1, 1, 3), (1, 1, 4), (1, 3, 1)], type=parse_tuples
    )
    parser.add_argument(
        "--mask_unit_attn", default=(True, False, False), nargs="+", type=str2bool
    )
    parser.add_argument("--patch_kernel", default=(4, 1, 2), nargs="+", type=int)
    parser.add_argument("--init_embed_dim", default=48, type=int)
    parser.add_argument("--init_num_heads", default=2, type=int)
    parser.add_argument("--out_embed_dims", default=(32, 64, 96), nargs="+", type=int)

    parser.add_argument("--fill_holes", default=False, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")

    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.set_defaults(fp16=True)
    

    return parser


def load_model(args):

    # Device configurations
    device = torch.device(args.device)

    model = models_defs.__dict__[args.model](
        **vars(args),
    )
    # load last model checkpoint
    chkpt = misc.get_last_checkpoint(args)

    with pathmgr.open(chkpt, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    print("Load pre-trained checkpoint from: %s" % args.output_dir)
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    checkpoint_model = misc.convert_checkpoint(checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    model = model.eval()

    return model, device


def load_fusion_head(args, model):

    device = torch.device(args.device)

    multi_scale_fusion_heads = nn.ModuleList()
    curr_mu_size = model.mask_unit_size
    for ix, i in enumerate(
        model.stage_ends[: model.q_pool]
    ):  # resolution constant after q_pool
        overall_q_strides = list(
            map(lambda elements: reduce(mul, elements), zip(*model.q_strides))
        )
        mask_unit_spatial_shape = [
            i // s for i, s in zip(model.mask_unit_size, overall_q_strides)
        ]
        kernel = [i // s for i, s in zip(curr_mu_size, mask_unit_spatial_shape)]
        curr_mu_size = [i // s for i, s in zip(curr_mu_size, model.q_strides[ix])]
        multi_scale_fusion_heads.append(
            conv_nd(len(model.q_strides[0]))(
                model.projections[ix].out_features,
                model.projections[-1].out_features,
                kernel_size=kernel,
                stride=kernel,
            )
        )
    multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

    # load pre-trained weights
    chkpt = misc.get_last_checkpoint(args)
    with pathmgr.open(chkpt, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]

    checkpoint_model = {
        k[25:]: v
        for k, v in checkpoint_model.items()
        if k.startswith("multi_scale_fusion_heads")
    }

    # load pre-trained model
    msg = multi_scale_fusion_heads.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    print("Fusion Head = %s" % str(multi_scale_fusion_heads))

    # add also normalization layer
    multi_scale_fusion_heads.append(
        nn.LayerNorm(model.projections[-1].out_features, eps=1e-6)
    )

    return multi_scale_fusion_heads.to(device).eval()

def extract_hierarchical_embeddings(args):

    model, device = load_model(args)
    if args.fusion_head:
        fusion_head = load_fusion_head(args, model)
    
    if args.non_hierarchical:
        args.q_strides = [(1, 1, 1)] * len(args.stages)
        args.out_embed_dims = [args.out_embed_dims[0]] * len(args.stages)

    if args.dataset == "shot7m2":
        submission_clips = np.load(args.path_to_data_dir, allow_pickle=True).item()
        submission_clips["sequences"] = submission_clips["sequences"]["keypoints"]
        num_animals = 1
        max_frame_emb_size = 64
        nr_test_frames = 2720 * 1800
    elif args.dataset == "hbabel":
        submission_clips = {"sequences": dict()}
        val = joblib.load(
            os.path.join(args.path_to_data_dir, "babel-smplh-30fps-male/val.pth.tar")
        )
        if args.joints3d_procrustes:
            val = joblib.load(
                os.path.join(
                    args.path_to_data_dir,
                    "babel-smplh-30fps-male/val_proc_realigned_procrustes.pth.tar",
                )
            )
            submission_clips["sequences"].update(
                {
                    sample["babel_id"]: sample["joint_positions_processed"]
                    for sample in val
                }
            )
            nr_test_frames = sum(
                map(lambda lst: len(lst), submission_clips["sequences"].values())
            )
        else:
            submission_clips["sequences"].update(
                {sample["babel_id"]: sample["joint_positions"] for sample in val}
            )
            nr_test_frames = sum(
                [len(sample) for sample in submission_clips["sequences"].values()]
            )
        num_animals = 1
        max_frame_emb_size = 64
    elif args.dataset == "mabe_mice":
        submission_clips = np.load(
            os.path.join(args.path_to_data_dir, "mouse_triplet_test.npy"),
            allow_pickle=True,
        ).item()
        normalize = mice.MABeMouseDataset._normalize
        grid_size = 850
        fill_holes = mice.MABeMouseDataset.fill_holes
        num_animals = 3
        max_frame_emb_size = 128
        nr_test_frames = (
            mice.MABeMouseDataset.DEFAULT_NUM_TESTING_POINTS
            * mice.MABeMouseDataset.SAMPLE_LEN
        )
    elif args.dataset == "dvc":
        '''
        # e.g.
        Here I want to read the csv  file "summary_table_imputed_with_sets.csv" . In this file, I have the following columns: cage, day and sets.
        I want to create 1 lists : for the testing according to sets (0=training, 1= testing, NA= not used).
        I want the name of the element in the list to be the cage name and the values to be the days. I will then use these lists to load the data.
        '''
        # selected_cages_file = os.path.join(args.path_to_data_dir, "selected_cages_test.txt")
        # selected_cages = load_selected_cages(selected_cages_file)

        # updated to be the same as dvc.py 
        # Read the CSV file 
        csv_path = os.path.join(args.path_to_data_dir, "summary_table_imputed_with_sets.csv")
        df = pd.read_csv(csv_path)
        torch.cuda.empty_cache() 
        # Create training and testing lists
        testing_list = {}
        day_names    = []
        for _, row in df.iterrows():
            cage = row['cage']
            day = row['day']
            set_type = row['sets']

            if pd.isna(set_type):
                continue
            elif set_type == 1:
                if cage not in testing_list:
                    testing_list[cage] = []
                testing_list[cage].append(day)
                day_names.append(day)
        del df 
        torch.cuda.empty_cache()
        submission_clips = [data.values[:,0:-2] for data in load_data(args.path_to_data_dir, testing_list).values()]
        torch.cuda.empty_cache()

        num_animals = 1  # should stay 1 
        max_frame_emb_size = 64 
        nr_test_frames = sum(data.shape[0] for data in submission_clips) 
        print(nr_test_frames)  # Adjust this calculation as needed
    else:
        raise NotImplementedError(
            f"Your specified dataset -- {args.dataset} -- is not supported..."
        )

    frame_number_map = {}
    # create a temporary file to hold the dictionary
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    dummy_input = torch.ones(1, 1, *args.input_size).to(device)
    with torch.no_grad():
        _, interm = model(dummy_input, return_intermediates=True)
        if args.embedsum:
            shapes = list(map(lambda x: x.shape[-1], interm))
        else:
            shapes = list(map(lambda x: math.prod(x.shape[2:]), interm))

    submissions = {}
    with open(temp_file.name, "wb") as f:
        for lv in range(len(args.stages)):
            map_path = os.path.join(args.output_dir, f"test_submission_TEMP_{lv}.dat")
            submissions[lv] = np.memmap(
                map_path, dtype="float16", mode="w+", shape=(nr_test_frames, shapes[lv])
            )
        map_path = os.path.join(args.output_dir, f"test_submission_TEMP_combined.dat")
        if args.combine_embeddings:
            if args.fusion_head:
                submissions["combined"] = np.memmap(
                    map_path,
                    dtype="float16",
                    mode="w+",
                    shape=(nr_test_frames, shapes[-1]),
                )
            else:
                submissions["combined"] = np.memmap(
                    map_path,
                    dtype="float16",
                    mode="w+",
                    shape=(nr_test_frames, sum(shapes)),
                )

    sub_seq_length = args.num_frames

    if args.fast_inference and args.num_frames % 2 == 0:
        sliding_window = 2
    else:
        sliding_window = 1
    start_idx = 0

    if args.dataset == "dvc":
        loop = ((day_name, sequence) for day_name,sequence in zip(day_names, submission_clips))
    else:
        loop = (
            (name, sequence) for name, sequence in submission_clips["sequences"].items()
        )

    for name, sequence in tqdm(loop):
        # Preprocess sequences
        if args.dataset == "shot7m2":
            vec_seq = sequence[:, :, shot7m2.SHOT7M2Dataset.SPLIT_INDS]
        elif args.dataset == "hbabel":
            if args.joints3d_procrustes:
                features = torch.from_numpy(sequence).float()
            else:
                features = sequence[:, hbabel.hBABELDataset.NTU_KPTS, :]
                features = features.transpose(2, 0, 1)[:, :, :, np.newaxis]
                # Normalize (pre-process) in NTU RGBD-style
                features = hbabel.hBABELDataset.ntu_pre_normalization(features)
                features = features.transpose(1, 2, 3, 0).squeeze()
            vec_seq = features
        elif args.dataset == "dvc":
            # sequence is already normalized 
            vec_seq = sequence
        else:
            vec_seq = sequence["keypoints"]

        if args.fill_holes:
            vec_seq = fill_holes(vec_seq)

        vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)

        if not (args.dataset == "shot7m2"):
            if args.dataset == "hbabel":
                vec_seq = vec_seq.reshape(-1, 25, 3)
                vec_seq = vec_seq[:, hbabel.hBABELDataset.NTU_KPT_GROUPING, :].reshape(
                    len(vec_seq), -1
                )
            elif args.dataset == "dvc":
                vec_seq = vec_seq.astype(np.float16)  # Convert to float16
                vec_seq /= 100 
            else:
                vec_seq = normalize(vec_seq, grid_size)

        if args.centeralign:
            vec_seq = vec_seq.reshape(vec_seq.shape[0], mice.NUM_MICE, 12, 2)
            vec_seq = mice.transform_to_centeralign_components(vec_seq)

        full_seq_len = vec_seq.shape[0]

        # Pads the beginning and end of the sequence with duplicate frames
        vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
        if args.fast_inference:
            pad = int((args.num_frames - sliding_window) / 2)
        else:
            pad = args.num_frames - sliding_window
        pad_vec = np.pad(vec_seq, ((pad, pad), (0, 0)), mode="edge")

        # Converts sequence into [number of sub-sequences, frames in sub-sequence, x/y alternating keypoints]
        data_test = sliding_window_view(pad_vec, window_shape=sub_seq_length, axis=0)[
            ::sliding_window
        ].transpose(0, 2, 1)

        if args.fast_inference:
            if data_test.shape[0] * sliding_window != len(vec_seq):
                len_diff = len(vec_seq) - (data_test.shape[0] * sliding_window)
                pad_vec = np.pad(
                    vec_seq, ((pad, pad + sliding_window), (0, 0)), mode="edge"
                )
                data_test = sliding_window_view(
                    pad_vec, window_shape=sub_seq_length, axis=0
                )[::sliding_window].transpose(0, 2, 1)
            else:
                len_diff = 0

        data_test = data_test.reshape(
            data_test.shape[0], data_test.shape[1], num_animals, -1
        )
        data_test = torch.tensor(data_test, dtype=torch.float16)

        data_loader = torch.utils.data.DataLoader(
            data_test, batch_size=args.batch_size, shuffle=False
        )

        with torch.no_grad():
            embeds = {level: [] for level in range(1, len(args.stages) + 1)}
            if args.combine_embeddings and args.fusion_head:
                fused_embeds = []
            for samples in data_loader:
                samples = samples[:, None, :].to(device, non_blocking=True)
                if args.fp16:
                    samples = samples.half()
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    _, preds = model(samples, return_intermediates=True)
                    for i in range(len(preds)):
                        embeds[i + 1].append(preds[i])
                    if args.combine_embeddings and args.fusion_head:
                        preds = preds[: model.q_pool] + preds[-1:]
                        x = 0.0
                        for head, interm_x in zip(fusion_head[:-1], preds):
                            x += apply_fusion_head(head, interm_x.unsqueeze(0))
                        x = fusion_head[-1](x)  # layer norm
                        fused_embeds.append(x.squeeze(0))
                del samples, preds
                torch.cuda.empty_cache()
        embeddings = {
            level: torch.cat(embeds[level], 0)
            for level in range(1, len(args.stages) + 1)
        }
        if args.combine_embeddings and args.fusion_head:
            embeddings["fused"] = torch.cat(fused_embeds, 0)
            del fused_embeds
        del embeds
        torch.cuda.empty_cache()
        t_patch_sizes = np.cumprod(
            np.insert(args.q_strides, 0, args.patch_kernel[0], axis=0)[:, 0]
        )

        for lv in embeddings:

            if lv == "fused":
                tps = t_patch_sizes[-1]
            else:
                tps = t_patch_sizes[lv - 1]

            if args.fast_inference:
                # cut out middle embedding(s)
                embeddings[lv] = embeddings[lv][:, pad // tps, :, :].unsqueeze(1)
                if sliding_window > 1:
                    assert sliding_window <= args.patch_kernel[0]
                    embeddings[lv] = embeddings[lv].repeat_interleave(
                        repeats=sliding_window, dim=1
                    )

            else:

                if args.embedsum:
                    emb_size = embeddings[lv].shape[-1]
                else:
                    emb_size = math.prod(embeddings[lv].shape[2:])

                # for temporal hierarchy!
                embeddings[lv] = embeddings[lv].view(
                    embeddings[lv].shape[0], -1, *embeddings[lv].shape[2:]
                )

                result_embeds = torch.zeros(
                    (full_seq_len + 2 * (sub_seq_length - sliding_window), emb_size),
                    dtype=torch.float16,
                ).to(embeddings[lv].device)

            if args.embedsum:
                # add single animal embeddings up to one embedding
                embs = embeddings[lv]
                embs = embs.view(
                    embs.shape[0], embs.shape[1], -1, embeddings[lv].shape[-1]
                )
                # average pooling
                embeddings[lv] = torch.mean(embs, dim=2)
            else:
                # stack multiple animal embeddings to one embedding
                embeddings[lv] = torch.flatten(embeddings[lv], start_dim=2)

            if args.fast_inference:
                # stacks sub-sequence embeddings back to full sequence embeddings
                embeddings[lv] = np.vstack(embeddings[lv].detach().cpu().numpy())
                if len_diff > 0:
                    embeddings[lv] = embeddings[lv][: len(vec_seq)]
            else:
                # get full sequence embeddings by doing sliding sum
                embeddings[lv] = (
                    averaging_sum(
                        result_embeds,
                        embeddings[lv].repeat_interleave(repeats=tps, dim=1),
                        sliding_window=sliding_window,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

        end_idx = start_idx + full_seq_len
        frame_number_map[name] = (start_idx, end_idx)

        for lv, submission in submissions.items():
            if lv != "combined":
                print(f"start_idx: {start_idx}, end_idx: {end_idx}")
                print(f"Shape of embeddings[{lv + 1}]: {embeddings[lv + 1].shape}")
                print(f"Shape of submission slice: {submission[start_idx:end_idx, :].shape}")
                submission[start_idx:end_idx, :] = embeddings[lv + 1]
            else:
                if args.combine_embeddings:
                    if args.fusion_head:
                        submission[start_idx:end_idx, :] = embeddings["fused"]
                    else:
                        submission[start_idx:end_idx, :] = np.concatenate(
                            [
                                submissions[sub][start_idx:end_idx, :]
                                for sub in range(len(args.stages))
                            ],
                            axis=1,
                        )

        start_idx = end_idx

    while submissions:

        lv, embs = submissions.popitem()
        
        # if constructed frame_embeddings are bigger than mabe evaluation allows, compress it with pca
        if embs.shape[1] > max_frame_emb_size:
            print("Compressing frame embeddings with PCA...")
            if embs.shape[1] < 100:
                pca = PCA(n_components=max_frame_emb_size, svd_solver="full")
                embs_pca = pca.fit_transform(embs)
                ev = sum(pca.explained_variance_ratio_)
            else:
                batch_size = 320000
                ipca = IncrementalPCA(
                    n_components=max_frame_emb_size, batch_size=batch_size
                )
                for i in tqdm(range(0, len(embs), batch_size)):
                    batch = embs[i : i + batch_size]
                    if not isinstance(batch, torch.Tensor):
                        batch = torch.tensor(batch)
                    batch = batch.to(torch.float32)
                    batch = torch.clamp(batch, min=-1e6, max=1e6)  # Prevent extreme values
                    batch = torch.where(torch.isinf(batch), torch.full_like(batch, 1e6), batch)  # Replace inf
                    batch = torch.where(torch.isnan(batch), torch.zeros_like(batch), batch)  # Replace NaN with zeros
                    batch = batch.numpy()
                    ipca.partial_fit(batch)
                ev = sum(ipca.explained_variance_ratio_)
                embs_pca = np.zeros((len(embs), max_frame_emb_size)).astype(np.float16)
                for i in tqdm(range(0, len(embs_pca), batch_size)):
                    batch = embs[i : i + batch_size]
                    if not isinstance(batch, torch.Tensor):
                        batch = torch.tensor(batch)
                    batch = batch.to(torch.float32)
                    batch = torch.clamp(batch, min=-1e6, max=1e6)  # Prevent extreme values
                    batch = torch.where(torch.isinf(batch), torch.full_like(batch, 1e6), batch)  # Replace inf
                    batch = torch.where(torch.isnan(batch), torch.zeros_like(batch), batch)  # Replace NaN with zeros
                    batch = batch.numpy()
                    embs_pca[i : i + batch_size] = ipca.transform(batch)

            print(
                f"transformed shape for level {lv}: from {embs.shape} to {embs_pca.shape}"
            )
            print("explained variance: ", ev)

        else:
            embs_pca = embs

        submission = {"frame_number_map": frame_number_map, "embeddings": embs_pca}

        np.save(os.path.join(args.output_dir, f"test_submission_{lv}.npy"), submission)

        embs.flush()
        del embs
        os.remove(os.path.join(args.output_dir, f"test_submission_TEMP_{lv}.dat"))

    os.remove(temp_file.name)


# HELPER FUNCTIONS


def averaging_sum(results_vector, embeds, sliding_window=1):
    start = 0
    for emb in embeds:
        results_vector[start : start + emb.shape[0]] += emb
        start += sliding_window
    return results_vector[(emb.shape[0] - sliding_window) : start] / emb.shape[0]