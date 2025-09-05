#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Light-weight DVC → hBehaveMAE embedding extractor
-------------------------------------------------
No fancy options, no intermediate-token hacks – just:
    • percentage normalisation (value / 100)
    • B,1,T,1,12   → model(x, mask_ratio=0)  → latent
    • latent.mean(dim=1)                     → frame embedding
Works with any checkpoint that was trained on
input_size = (1440, 1, 12) and patch_kernel = (2,1,12).
"""

import argparse, os, time, datetime, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import sliding_window_view
from iopath.common.file_io import g_pathmgr as pathmgr

# --------------------------------------------------------------------------------------
# repo-local helpers (edit the import paths if you moved things)
# --------------------------------------------------------------------------------------
from data_pipeline.load_dvc import load_dvc_data               # your CSV-loader
from models import models_defs                                   # factory dict
from util import misc                                            # last-checkpoint helper
from util.pos_embed import interpolate_pos_embed                 # input-size fix

# --------------------------------------------------------------------------------------
# command-line
# --------------------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dvc_root",  required=True)
    p.add_argument("--summary_csv",
                   default="summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv")
    p.add_argument("--ckpt_dir",  required=True,
                   help="directory that contains checkpoint-XYZ.pth")
    p.add_argument("--ckpt_name", default="checkpoint-best.pth")
    p.add_argument("--output_dir",   required=True)

    p.add_argument("--num_frames", type=int, default=1440)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device",     default="cuda")
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument("--skip_missing", action="store_true",
                   help="skip sequences that miss any electrode column")
    return p.parse_args()

# --------------------------------------------------------------------------------------
# tiny helpers
# --------------------------------------------------------------------------------------
ELEC_COLS = [f"v_{i}" for i in range(1, 13)]          # ordered list (V1 … V12)

def build_test_list(root:str, summary_csv:str) -> Tuple[Dict[str, List[str]], List[str]]:
    """return mapping {cage:[day,...]} and ordered list of cage_day keys"""
    df = pd.read_csv(os.path.join(root, summary_csv))
    mapping, order = {}, []
    for _, r in df.iterrows():
        if r.get("sets", np.nan) != 1:          # 1 == test
            continue
        mapping.setdefault(r["cage"], []).append(r["day"])
        order.append(f"{r['cage']}_{r['day']}")
    return mapping, order

def load_sequences(root:str, mapping:Dict[str,List[str]], order:List[str],
                   skip_missing:bool) -> List[Tuple[str,np.ndarray]]:
    raw = load_dvc_data(root, mapping)        # {cage: DataFrame}

    seqs = []
    for cage, df in raw.items():
        df["__day"] = pd.to_datetime(df.iloc[:, 13]).dt.date.astype(str)
        for day, sub in df.groupby("__day"):
            key = f"{cage}_{day}"
            if key not in order:
                continue
            # verify electrode columns
            missing = [c for c in ELEC_COLS if c not in sub.columns]
            if missing:
                m = f"[WARN] {key} missing {missing}"
                if skip_missing:
                    print(m);  continue
                raise KeyError(m)
            seqs.append((key, sub[ELEC_COLS].values.astype(np.float32)))
    seqs.sort(key=lambda x: order.index(x[0]))
    return seqs

def load_model(ckpt_dir: str, ckpt_name: str, device: torch.device):
    """
    • finds the right checkpoint
    • converts stored training-args (Namespace → dict)
    • instantiates the model exactly as it was trained
    """
    # ------- pick file -------
    path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(path):
        best = os.path.join(ckpt_dir, "checkpoint-best.pth")
        path  = best if os.path.exists(best) else misc.get_last_checkpoint(ckpt_dir)
        if not path:
            raise FileNotFoundError("no checkpoint found in ckpt_dir")

    print("Loading checkpoint", path)
    ckpt = torch.load(pathmgr.open(path, "rb"), map_location="cpu")

    # ------- training config -------
    tr_args = ckpt.get("args", {})                 # may be Namespace / dict / missing
    if isinstance(tr_args, argparse.Namespace):    # <-- **critical line**
        tr_args = vars(tr_args)                    # Namespace → real dict

    model_name   = tr_args.get("model", "hbehavemae")
    model_ctor   = models_defs.__dict__[model_name]

    model = model_ctor(**tr_args)                  # works because all h/BehaveMAE
                                                   # ctors accept **kwargs
    # ------- load weights -------
    interpolate_pos_embed(model, ckpt["model"])
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:    print("[load_model] missing keys:",    missing)
    if unexpected: print("[load_model] unexpected keys:", unexpected)

    return model.to(device).eval().requires_grad_(False)

# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mapping, order = build_test_list(args.dvc_root, args.summary_csv)
    sequences      = load_sequences(args.dvc_root, mapping, order, args.skip_missing)
    print("Total sequences:", len(sequences))

    model = load_model(args.ckpt_dir, args.ckpt_name, device)

    frame_map : Dict[str, Tuple[int,int]] = {}
    embeds    : List[np.ndarray] = []
    frame_ptr = 0

    for key, mat in tqdm(sequences, desc="Loading sequences", unit="seq"):
        n_frames = mat.shape[0]
        # normalise    (value % → 0-1)
        mat = mat / 100.0

        # sliding windows with stride 1 centred on each frame
        pad = (args.num_frames - 1) // 2
        padded   = np.pad(mat, ((pad,pad),(0,0)), 'edge')
        windows  = sliding_window_view(padded, (args.num_frames, 12), axis=(0,1))[:,0]
        
        # ------------------------------------------------------------
        # build tensor for the model  (windows  →  shape (N,1,T,1,12))
        # ------------------------------------------------------------
        # 'windows' is (N, T, 12)  from the sliding-window code above
        tensor_windows = torch.from_numpy(windows.copy())          # → (N,T,12)
        tensor_windows = tensor_windows.unsqueeze(1)               # add C  : (N,1,T,12)
        tensor_windows = tensor_windows.unsqueeze(3)               # add H  : (N,1,T,1,12)
            
        dl = DataLoader(tensor_windows,   # (N,1,T,1,12) → (N,1,T,12)
                        batch_size=args.batch_size, shuffle=False)


        emb_seq = []
        with torch.no_grad():
            for batch in dl:
                batch = batch.to(device)
                # make sure input and weights have the same dtype
                if batch.dtype != next(model.parameters()).dtype:
                    batch = batch.to(next(model.parameters()).dtype)

                # call the encoder directly (no return_intermediates)
                latent, _mask = model.forward_encoder(batch, mask_ratio=0.0)
                # latent might come back as (B, d1, d2, …, dK, C)
                # flatten all but batch & channel dims:
                B, *spatial, C = latent.shape
                flat = latent.view(B, -1, C)        # (B, N_tokens, C)
                frame_emb = flat.mean(dim=1)        # (B, C)

                emb_seq.append(frame_emb.float().cpu())

        emb_seq = torch.cat(emb_seq).numpy().astype(np.float16)   # (N_frames,C)
        embeds.append(emb_seq)

        frame_map[key] = (frame_ptr, frame_ptr + n_frames)
        frame_ptr     += n_frames

    all_embeds = np.concatenate(embeds, axis=0)
    out_path   = os.path.join(args.output_dir, "test_submission.npy")
    np.save(out_path, {"frame_number_map": frame_map,
                       "embeddings": all_embeds})
    print("Saved", out_path, "shape", all_embeds.shape)
    print("✓ Done in", datetime.timedelta(seconds=int(time.time()-os.path.getmtime(out_path))))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()