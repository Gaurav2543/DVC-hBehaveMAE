
import argparse, warnings
from pathlib import Path
import gc
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA, IncrementalPCA
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
    p.add_argument("--labels", default="test-outputs/arrays_sub20_with_cage_complete_correct_strains.npy",
                   help="*.npy containing {'label_array','vocabulary'}")
    p.add_argument("--label-names", nargs="+", default=["Age_Days"],
                   help="one or more label fields to colour by")
    p.add_argument("--samples", type=int, default=10000,
                   help="random subsample size for the global plot")
    p.add_argument("--group-by", default="Strain",
                   help="label field to split by for per-group plots "
                        "(set to '' or use --no-group to disable)")
    p.add_argument("--max-per-group", type=int, default=10000,
                   help="sub-sample each group to at most this many points")
    p.add_argument("--out-dir", default=".", help="directory for the PDFs")
    p.add_argument("--no-group", action="store_true",
                   help="disable per-group plots entirely")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="process embeddings in chunks of this size")
    p.add_argument("--pca-batch-size", type=int, default=5000,
                   help="batch size for incremental PCA")
    return p.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_chunk(x: np.ndarray) -> np.ndarray:
    """Process a chunk: float32, NaN→mean, clip to float16 range."""
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    
    # Handle inf values
    x = np.where(np.isinf(x), np.nan, x)
    
    # Impute NaN values
    if np.any(np.isnan(x)):
        imputer = SimpleImputer(strategy="mean")
        x = imputer.fit_transform(x)
    
    # Clip to avoid overflow
    return np.clip(x, -65_504, 65_504, out=x)

def pca_reduce_incremental(x: np.ndarray, n_components: int = 50, batch_size: int = 10000) -> np.ndarray:
    """
    Runs PCA in small batches to avoid memory issues.
    """
    print(f"[INFO] Running Incremental PCA with batch_size={batch_size}")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Fit in batches
    n_samples = x.shape[0]
    for i in tqdm(range(0, n_samples, batch_size), desc="PCA fitting"):
        end_idx = min(i + batch_size, n_samples)
        chunk = x[i:end_idx]
        chunk = preprocess_chunk(chunk)
        ipca.partial_fit(chunk)
        del chunk  # Explicit cleanup
        if i % (batch_size * 5) == 0:  # Every 5 batches
            gc.collect()
    
    # Transform in batches
    result = np.empty((n_samples, n_components), dtype=np.float32)
    for i in tqdm(range(0, n_samples, batch_size), desc="PCA transform"):
        end_idx = min(i + batch_size, n_samples)
        chunk = x[i:end_idx]
        chunk = preprocess_chunk(chunk)
        result[i:end_idx] = ipca.transform(chunk)
        del chunk
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    return result

def load_embeddings(path: str) -> np.ndarray:
    print(f"[INFO] Loading embeddings from {path}")
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    emb = d["embeddings"]
    print(f"[INFO] Original embeddings shape: {emb.shape}, dtype: {emb.dtype}")
    
    # Convert to float32 immediately to save memory
    if emb.dtype != np.float32:
        print("[INFO] Converting to float32...")
        emb = emb.astype(np.float32, copy=False)
    
    return emb

def load_labels(path: str):
    print(f"[INFO] Loading labels from {path}")
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    return np.array(d["label_array"]), d["vocabulary"]

# ══════════════════════════════════════════════════════════════════════════════
def plot_umap(ax, xy, labels, label_name, *, remove_legend=False):
    """Scatter on *ax* coloured by *labels* (auto detects numeric vs categorical)."""

    numeric = False
    try:
        labels_num = labels.astype(float)
        numeric = True
    except Exception:
        labels_num = None

    if numeric:
        vmin, vmax = labels_num.min(), labels_num.max()
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels_num,
                        cmap=cmap, norm=norm, s=6, alpha=0.7)  # Smaller points
        cb = plt.colorbar(sc, ax=ax, pad=0.01)
        cb.set_ticks(np.linspace(vmin, vmax, 6))
        cb.set_label(f"{label_name}  (min {vmin:.1f} – max {vmax:.1f})")
    else:
        uniq, enc = np.unique(labels, return_inverse=True)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=enc,
                        cmap="tab20", s=6, alpha=0.7)  # Smaller points
        if not remove_legend and len(uniq) <= 20:  # Only show legend if reasonable number
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
    """Memory-optimized UMAP embedding."""
    print(f"[INFO] Running UMAP on {x.shape[0]} samples...")
    
    # Use memory-efficient UMAP settings
    reducer = umap.UMAP(
        random_state=42, 
        n_neighbors=min(15, x.shape[0] // 3),  # Adaptive neighbors
        min_dist=0.3,
        low_memory=True,  # Enable low memory mode
        n_jobs=1,  # Avoid parallel processing memory overhead
        verbose=False
    )
    
    result = reducer.fit_transform(x)
    
    # Cleanup
    del reducer
    gc.collect()
    
    return result

def smart_subsample(indices: np.ndarray, max_samples: int, chunk_size: int = 1000) -> np.ndarray:
    """
    Smart subsampling that processes in chunks to get exactly max_samples.
    If we need 10k samples but have 100k, we'll sample in chunks to get representative data.
    """
    if len(indices) <= max_samples:
        return indices
    
    if max_samples <= chunk_size:
        # Simple random sampling for small requests
        return np.random.choice(indices, max_samples, replace=False)
    
    # For larger requests, sample in chunks to ensure representation
    n_chunks = max_samples // chunk_size
    remainder = max_samples % chunk_size
    
    # Shuffle indices first
    np.random.shuffle(indices)
    
    # Split into roughly equal segments
    segment_size = len(indices) // n_chunks
    selected = []
    
    for i in range(n_chunks):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(indices))
        if start < end:
            segment = indices[start:end]
            n_from_segment = chunk_size if i < n_chunks - 1 or remainder == 0 else chunk_size + remainder
            n_from_segment = min(n_from_segment, len(segment))
            selected.extend(np.random.choice(segment, n_from_segment, replace=False))
    
    return np.array(selected[:max_samples])

# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    base = Path(args.embeddings).stem      # remove .npy
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # -- load ------------------------------------------------------------------
    print("[INFO] Loading data …")
    emb = load_embeddings(args.embeddings)
    lab_arr, vocab = load_labels(args.labels)
    n = min(emb.shape[0], lab_arr.shape[1])
    emb, lab_arr = emb[:n], lab_arr[:, :n]
    print(f"[INFO] Final data shapes: embeddings {emb.shape} | labels {lab_arr.shape}")

    # -- run PCA once ----------------------------------------------------------
    print("[INFO] Running PCA preprocessing...")
    start_time = time.time()
    emb_pca = pca_reduce_incremental(emb, n_components=50, batch_size=args.pca_batch_size)
    print(f"[INFO] PCA completed in {time.time() - start_time:.1f}s, shape: {emb_pca.shape}")
    
    # Clear original embeddings to save memory
    del emb
    gc.collect()
    print(f"[INFO] Memory cleanup completed")

    # -- iterate over requested label fields -----------------------------------
    for lbl in args.label_names:
        if lbl not in vocab:
            print(f"[WARN] '{lbl}' not in vocabulary – skipped.")
            continue

        idx = vocab.index(lbl)
        pdf_path = out_dir / f"{base}_{lbl}_umap.pdf"
        print(f"[INFO] → generating {pdf_path}")
        
        with PdfPages(pdf_path) as pdf:

            # ---------- global subsample plot ----------
            print(f"[INFO] Creating global plot with {args.samples} samples...")
            all_indices = np.arange(n)
            sel = smart_subsample(all_indices, args.samples, args.chunk_size)
            print(f"[INFO] Selected {len(sel)} samples for global plot")
            
            emb2d = umap_embed(emb_pca[sel])

            fig, ax = plt.subplots(figsize=(10, 8))
            plot_umap(ax, emb2d, lab_arr[idx, sel], lbl, remove_legend=True)
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Cleanup
            del emb2d, sel
            gc.collect()

            # ---------- per-group breakdown -------------
            if not args.no_group and args.group_by and args.group_by in vocab:
                g_idx = vocab.index(args.group_by)
                groups = np.unique(lab_arr[g_idx])
                print(f"[INFO] Processing {len(groups)} groups for {args.group_by}")
                
                for g in tqdm(groups, desc=f"{lbl} by {args.group_by}"):
                    mask = lab_arr[g_idx] == g
                    if mask.sum() == 0:
                        continue
                        
                    group_indices = np.where(mask)[0]
                    if len(group_indices) < 10:  # Skip very small groups
                        continue
                        
                    # Smart subsample for this group
                    sel = smart_subsample(group_indices, args.max_per_group, args.chunk_size)
                    
                    if len(sel) < 10:  # Still too small after sampling
                        continue
                    
                    print(f"[INFO] Group {g}: {len(group_indices)} total, {len(sel)} selected")
                    
                    emb2d = umap_embed(emb_pca[sel])

                    fig, ax = plt.subplots(figsize=(9, 7))
                    plot_umap(ax, emb2d, lab_arr[idx, sel],
                              f"{lbl}  ({args.group_by}: {g})",
                              remove_legend=True)
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Cleanup after each group
                    del emb2d, sel, group_indices
                    gc.collect()

        print(f"  ✔ saved {pdf_path}")
        
        # Final cleanup after each label
        gc.collect()

    print("[INFO] All visualizations completed!")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()