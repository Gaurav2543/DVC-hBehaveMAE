#!/usr/bin/env python3
# umap_vis_cli.py
"""
Generic UMAP visualiser for hBehaveMAE (or any) embeddings.

Example
-------
python umap_vis_cli.py \
  --embeddings extracted/test_submission_stage_0.npy \
  --labels     labels/arrays_labels.csv \
  --label-names Age_Days Strain Cage \
  --group-by   Strain
"""
import argparse, warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import umap.umap_ as umap        # pip install umap-learn

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UMAP visualiser for (large) frame-level embeddings")
    p.add_argument("--embeddings", required=True,
                   help="*.npy or *.npz that stores {'embeddings': …}")
    p.add_argument("--labels", required=True,
                   help="*.csv file with samples as rows and label columns")
    p.add_argument("--label-names", nargs="+", default=["avg_age_days_chunk_start"],
                   help="one or more label fields to colour by")
    p.add_argument("--samples", type=int, default=5000,
                   help="random subsample size for the global plot")
    p.add_argument("--group-by", default="Strain",
                   help="label field to split by for per-group plots "
                        "(set to '' or use --no-group to disable)")
    p.add_argument("--max-per-group", type=int, default=5000,
                   help="sub-sample each group to at most this many points")
    p.add_argument("--out-dir", default=".", help="directory for the PDFs")
    p.add_argument("--no-group", action="store_true",
                   help="disable per-group plots entirely")
    return p.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(x: np.ndarray) -> np.ndarray:
    """float64, NaN→mean, clip to float16 range."""
    x = x.astype(np.float32)
    x = np.where(np.isinf(x), np.nan, x)
    x = SimpleImputer(strategy="mean").fit_transform(x)
    return np.clip(x, -65_504, 65_504)

def pca_reduce(x: np.ndarray, n_components: int = 50) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(x)

# ══════════════════════════════════════════════════════════════════════════════
def load_embeddings(path: str) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    return d["embeddings"].astype(np.float32, copy=False)

# ══════════════════════════════════════════════════════════════════════════════
def load_labels(path: str):
    """
    Load labels from a CSV file. Assumes rows are samples and columns are label names.
    Returns a 2D numpy array of shape (n_labels, n_samples) and a list of column names.
    """
    df = pd.read_csv(path)
    label_array = df.T.values
    vocabulary = list(df.columns)
    return label_array, vocabulary

# ══════════════════════════════════════════════════════════════════════════════
def plot_umap(ax, xy, labels, label_name, *, remove_legend=False):
    """Scatter on *ax* coloured by *labels* (auto detects numeric vs categorical)."""

    # attempt numeric conversion
    try:
        labels_num = labels.astype(float)
        numeric = True
    except Exception:
        labels_num = None
        numeric = False

    if numeric:
        # mask out NaNs to compute meaningful vmin/vmax
        mask = np.isfinite(labels_num)
        if not np.any(mask):
            # no valid numeric labels, fallback to grey scatter
            ax.scatter(xy[:, 0], xy[:, 1], color="lightgrey", s=8, alpha=0.8)
        else:
            finite_vals = labels_num[mask]
            vmin, vmax = float(np.min(finite_vals)), float(np.max(finite_vals))
            cmap = plt.cm.viridis
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=finite_vals,
                            cmap=cmap, norm=norm, s=8, alpha=0.8)
            cb = plt.colorbar(sc, ax=ax, pad=0.01)
            # set ticks only between vmin and vmax
            cb.set_ticks(np.linspace(vmin, vmax, 6))
            cb.set_label(f"{label_name}  (min {vmin:.1f} – max {vmax:.1f})")
    else:
        uniq, enc = np.unique(labels, return_inverse=True)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=enc,
                        cmap="tab20", s=8, alpha=0.8)
        if not remove_legend:
            ax.legend(
                handles=[plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor=sc.cmap(sc.norm(i)),
                                    label=str(lbl), markersize=6)
                         for i, lbl in enumerate(uniq)],
                title=label_name, bbox_to_anchor=(1.02, 1), loc="upper left",
                frameon=False, fontsize="x-small")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP coloured by {label_name}")

# ══════════════════════════════════════════════════════════════════════════════
def umap_embed(x: np.ndarray) -> np.ndarray:
    return umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.3).fit_transform(x)

# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    base = Path(args.embeddings).stem
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data …")
    emb = load_embeddings(args.embeddings)
    lab_arr, vocab = load_labels(args.labels)
    n = min(emb.shape[0], lab_arr.shape[1])
    emb, lab_arr = emb[:n], lab_arr[:, :n]
    print(f"[INFO] embeddings {emb.shape} | labels {lab_arr.shape}")

    emb_pca = pca_reduce(preprocess(emb))

    for lbl in args.label_names:
        if lbl not in vocab:
            print(f"[WARN] '{lbl}' not in vocabulary – skipped.")
            continue

        idx = vocab.index(lbl)
        pdf_path = out_dir / f"{base}_{lbl}_umap.pdf"
        print(f"[INFO] → generating {pdf_path}")
        with PdfPages(pdf_path) as pdf:

            sel = np.random.choice(n, min(args.samples, n), replace=False)
            emb2d = umap_embed(emb_pca[sel])

            fig, ax = plt.subplots(figsize=(8, 8))
            plot_umap(ax, emb2d, lab_arr[idx, sel], lbl, remove_legend=True)
            pdf.savefig(fig); plt.close(fig)

            if not args.no_group and args.group_by and args.group_by in vocab:
                g_idx = vocab.index(args.group_by)
                groups = np.unique(lab_arr[g_idx])
                for g in tqdm(groups, desc=f"{lbl} by {args.group_by}"):
                    mask = lab_arr[g_idx] == g
                    if not np.any(mask):
                        continue
                    sel = np.where(mask)[0]
                    if sel.size > args.max_per_group:
                        sel = np.random.choice(sel, args.max_per_group, replace=False)
                    emb2d = umap_embed(emb_pca[sel])

                    fig, ax = plt.subplots(figsize=(7, 7))
                    plot_umap(ax, emb2d, lab_arr[idx, sel],
                              f"{lbl}  ({args.group_by}: {g})",
                              remove_legend=True)
                    pdf.savefig(fig); plt.close(fig)

        print("  ✔ saved", pdf_path)

if __name__ == "__main__":
    main()