#!/usr/bin/env python3
# umap_vis_plotly.py

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import umap.umap_ as umap  # pip install umap-learn
import plotly.express as px

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# ══════════════════════════════════════════════════════════════════════════════
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive UMAP visualiser using Plotly")
    p.add_argument("--embeddings", required=True, help="*.npy or *.npz with {'embeddings': …}")
    p.add_argument("--labels", required=True, help="*.csv file with samples as rows and label columns")
    p.add_argument("--label-names", nargs="+", default=["avg_age_days_chunk_start"],
                   help="one or more label fields to colour by")
    p.add_argument("--samples", type=int, default=5000,
                   help="random subsample size for the global plot")
    p.add_argument("--group-by", default="strain",
                   help="label field to split by for per-group plots")
    p.add_argument("--max-per-group", type=int, default=5000,
                   help="max samples per group for per-group plots")
    p.add_argument("--out-dir", default=".", help="directory for output HTMLs")
    p.add_argument("--no-group", action="store_true", help="disable per-group plots")
    return p.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
def preprocess(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.where(np.isinf(x), np.nan, x)
    x = SimpleImputer(strategy="mean").fit_transform(x)
    return np.clip(x, -65_504, 65_504)

def pca_reduce(x: np.ndarray, n_components: int = 50) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(x)

def load_embeddings(path: str) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    return d["embeddings"].astype(np.float32, copy=False)

def load_labels(path: str):
    df = pd.read_csv(path, low_memory=False)  # Avoid DtypeWarning
    label_array = df.T.values
    vocabulary = list(df.columns)
    return label_array, vocabulary

def umap_embed(x: np.ndarray) -> np.ndarray:
    return umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.3).fit_transform(x)

def plot_umap_plotly(xy, labels, label_name, shapes=None, hover_info=None, title=None):
    df = pd.DataFrame({
        "UMAP-1": xy[:, 0],
        "UMAP-2": xy[:, 1],
        "label": labels,
        "cage_id": hover_info if hover_info is not None else [""] * len(labels),
    })

    try:
        df["label"] = df["label"].astype(float)
        is_numeric = True
    except Exception:
        is_numeric = False

    if is_numeric:
        min_val, max_val = df["label"].min(), df["label"].max()
        fig = px.scatter(
            df,
            x="UMAP-1", y="UMAP-2",
            color="label",
            color_continuous_scale="Turbo",
            range_color=[min_val, max_val],
            symbol="cage_id",  # 🎯 Vary shape by cage_id
            hover_data={
                "label": True,
                "cage_id": True,
                "UMAP-1": False,
                "UMAP-2": False
            },
            title=title or f"UMAP coloured by {label_name}",
            width=800, height=700
        )
        fig.update_coloraxes(colorbar_title=f"Age (days) (min {min_val:.1f} – max {max_val:.1f})")

        # 🧼 Hide symbol (shape) legend
        fig.for_each_trace(lambda t: t.update(showlegend=False))

        # 🏷️ Rename hover fields
        fig.update_traces(
            hovertemplate="<br>".join([
                "Age (days): %{customdata[0]}",
                "Cage ID: %{customdata[1]}"
            ]),
            customdata=df[["label", "cage_id"]].values
        )

    else:
        fig = px.scatter(
            df,
            x="UMAP-1", y="UMAP-2",
            color="label",
            symbol="cage_id",
            hover_data={
                "label": True,
                "cage_id": True,
                "UMAP-1": False,
                "UMAP-2": False
            },
            title=title or f"UMAP coloured by {label_name}",
            width=800, height=700
        )
        fig.for_each_trace(lambda t: t.update(showlegend=False))

    fig.update_layout(
        legend=dict(
            title=label_name,
            x=1.05,
            y=1,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='black',
            borderwidth=1,
        ),
        title_x=0.5
    )

    return fig





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
        print(f"[INFO] → generating UMAPs for label: {lbl}")

        # Combined plot
        print("[INFO] Creating combined UMAP plot...")
        sel = np.random.choice(n, min(args.samples, n), replace=False)
        emb2d_combined = umap_embed(emb_pca[sel])
        cage_labels = lab_arr[vocab.index("cage_id"), sel] if "cage_id" in vocab else None

        fig = plot_umap_plotly(
            emb2d_combined,
            labels=lab_arr[idx, sel],
            label_name=lbl,
            shapes=cage_labels,
            hover_info=cage_labels,
            title=f"{lbl} (All strains Combined)"
        )
        html_path = out_dir / f"{base}_{lbl}_combined.html"
        fig.write_html(str(html_path))
        print(f"  ✔ saved {html_path}")

        # Per-strain plots
        if not args.no_group and args.group_by and args.group_by in vocab:
            g_idx = vocab.index(args.group_by)
            groups = np.unique(lab_arr[g_idx])
            print(f"[INFO] Found {len(groups)} unique {args.group_by}s: {groups}")

            for g in tqdm(groups, desc=f"Processing {args.group_by}s"):
                strain_mask = lab_arr[g_idx] == g
                if not np.any(strain_mask):
                    continue
                strain_indices = np.where(strain_mask)[0]

                if len(strain_indices) < 10:
                    print(f"[WARN] Skipping {args.group_by} '{g}' - too few samples")
                    continue
                if len(strain_indices) > args.max_per_group:
                    strain_indices = np.random.choice(strain_indices, args.max_per_group, replace=False)

                strain_emb_pca = emb_pca[strain_indices]
                emb2d_strain = umap_embed(strain_emb_pca)
                cage_labels = lab_arr[vocab.index("cage_id"), strain_indices] if "cage_id" in vocab else None

                fig = plot_umap_plotly(
                    emb2d_strain,
                    labels=lab_arr[idx, strain_indices],
                    label_name=lbl,
                    shapes=cage_labels,
                    hover_info=cage_labels,
                    title=f"{lbl} ({args.group_by}: {g})"
                )
                safe_g = str(g).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
                html_path = out_dir / f"{base}_{lbl}_{args.group_by}_{safe_g}.html"
                fig.write_html(str(html_path))
        else:
            print("[INFO] Per-strain plotting disabled or group column not found")

if __name__ == "__main__":
    main()
