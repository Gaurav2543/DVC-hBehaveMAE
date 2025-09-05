
"""
hBehaveMAE embedding extractor
writes:
    test_low.npy   – lowest level     (C_low  )
    test_mid.npy   – middle level     (C_mid  )
    test_high.npy  – highest level    (C_high )
    test_comb.npy  – concat of all 3  (C_low+C_mid+C_high)
"""
import argparse, os, time, datetime, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from iopath.common.file_io import g_pathmgr as pathmgr

from data_pipeline.load_dvc_real_one import load_dvc_data
from models import models_defs
from util import misc
from util.pos_embed import interpolate_pos_embed


# ----------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dvc_root",      required=True)
    p.add_argument("--summary_csv",
                   default="summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv")
    p.add_argument("--ckpt_dir",      required=True)
    p.add_argument("--ckpt_name",     default="checkpoint-best.pth")
    p.add_argument("--output_dir",    required=True)

    p.add_argument("--num_frames",    type=int, default=1440)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--skip_missing",  action="store_true")
    return p.parse_args()


ELEC_COLS = [f"v_{i}" for i in range(1, 13)]


# ----------------------------------------------------------------------------------
def build_test_split(root, csv):
    df, mapping, order = pd.read_csv(os.path.join(root, csv)), {}, []
    for _, r in df.iterrows():
        if r.get("sets", np.nan) != 1:
            continue
        mapping.setdefault(r["cage"], []).append(r["day"])
        order.append(f"{r['cage']}_{r['day']}")
    return mapping, order


def load_sequences(root, mapping, order, skip_missing):
    raw = load_dvc_data(root, mapping)
    seqs = []
    for cage, df in raw.items():
        df["__day"] = pd.to_datetime(df.iloc[:, 13]).dt.date.astype(str)
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
    return seqs


def load_model(ckpt_dir, ckpt_name, device):
    path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(path):
        path = os.path.join(ckpt_dir, "checkpoint-best.pth") \
               if os.path.exists(os.path.join(ckpt_dir, "checkpoint-best.pth")) \
               else misc.get_last_checkpoint(ckpt_dir)
    if not path:
        raise FileNotFoundError("no checkpoint found in ckpt_dir")

    print("Loading checkpoint", path)
    ckpt = torch.load(pathmgr.open(path, "rb"), map_location="cpu")
    tr_args = ckpt.get("args", {})
    if isinstance(tr_args, argparse.Namespace):
        tr_args = vars(tr_args)

    model = models_defs.__dict__[tr_args.get("model", "hbehavemae")](**tr_args)
    interpolate_pos_embed(model, ckpt["model"])
    model.load_state_dict(ckpt["model"], strict=False)
    return model.to(device).eval().requires_grad_(False)


# ----------------------------------------------------------------------------------
def main():
    args   = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mapping, order   = build_test_split(args.dvc_root, args.summary_csv)
    sequences        = load_sequences(args.dvc_root, mapping, order, args.skip_missing)
    print("Total sequences:", len(sequences))

    model = load_model(args.ckpt_dir, args.ckpt_name, device)

    frame_map, ptr = {}, 0
    LOW, MID, HIGH, COMB = [], [], [], []

    pad = (args.num_frames - 1) // 2

    for key, mat in tqdm(sequences, desc="Loading sequences", unit="seq"):
        mat = mat / 100.0                                   # % → [0,1]
        n_frames = mat.shape[0]

        padded  = np.pad(mat, ((pad, pad), (0, 0)), "edge")
        windows = sliding_window_view(padded, (args.num_frames, 12), axis=(0, 1))[:, 0]
        windows = torch.from_numpy(windows.copy()).unsqueeze(1).unsqueeze(3)  # (N,1,T,1,12)

        dl = DataLoader(windows, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

        low_seq, mid_seq, high_seq = [], [], []

        with torch.no_grad():
            for batch in dl:
                batch = batch.to(device, dtype=next(model.parameters()).dtype)
                _, _mask, levels = model.forward_encoder(batch, mask_ratio=0.0,
                                                          return_intermediates=True)
                # 3 levels returned:  low, mid, high  (after q_pool fusion logic)
                l, m, h = levels[0], levels[1], levels[2]

                def pool(feat):                          # (B, T, H, W, C) → (B,C)
                    return feat.flatten(1, -2).mean(1).float().cpu()

                low_seq .append(pool(l))
                mid_seq .append(pool(m))
                high_seq.append(pool(h))

        low_seq  = torch.cat(low_seq ).numpy().astype(np.float16)
        mid_seq  = torch.cat(mid_seq ).numpy().astype(np.float16)
        high_seq = torch.cat(high_seq).numpy().astype(np.float16)
        comb_seq = np.concatenate([low_seq, mid_seq, high_seq], axis=1)

        LOW .append(low_seq)
        MID .append(mid_seq)
        HIGH.append(high_seq)
        COMB.append(comb_seq)

        frame_map[key] = (ptr, ptr + n_frames)
        ptr += n_frames

    def dump(name, lst):
        arr = np.concatenate(lst, axis=0)
        np.save(os.path.join(args.output_dir, f"test_{name}.npy"),
                {"frame_number_map": frame_map, "embeddings": arr})
        print(f"Saved test_{name}.npy  – shape {arr.shape}")

    dump("low",   LOW)
    dump("mid",   MID)
    dump("high",  HIGH)
    dump("comb",  COMB)

    print("✓ Done in", datetime.timedelta(seconds=int(time.time() - os.path.getmtime(
          os.path.join(args.output_dir, 'test_comb.npy')))))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()