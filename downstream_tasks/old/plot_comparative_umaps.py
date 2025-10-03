import os
import argparse
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import umap
import itertools
import json
import warnings
import math

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden.*", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*", category=UserWarning)

# Set matplotlib to use dark style
plt.style.use('dark_background')

# =============================
# Argument Parsing
# =============================
parser = argparse.ArgumentParser(description="UMAP Visualization Script with Strain Filtering")
parser.add_argument('--metadata_path', required=True)
parser.add_argument('--results_json', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--embedding_paths', required=True, nargs='+', 
                    help='List of embedding file paths')
parser.add_argument('--embedding_labels', required=True, nargs='+',
                    help='List of embedding labels (must match number of embedding_paths)')
parser.add_argument('--embedding_lengths', required=True, help='JSON string: {"level1":"(15mins)",...}')
parser.add_argument('--umap_name_prefix', required=True)
parser.add_argument('--embedding_dimensions', type=int, default=1440)
parser.add_argument('--strains_filter', nargs='+', default=None,
                    help='List of specific strains to plot (if not provided, plots all strains)')
args = parser.parse_args()

# Validate that embedding_paths and embedding_labels have same length
if len(args.embedding_paths) != len(args.embedding_labels):
    raise ValueError("Number of embedding_paths must match number of embedding_labels")

# =============================
# Configuration
# =============================
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
embedding_lengths = json.loads(args.embedding_lengths)

# Build prediction_results[ds_label] = json content
prediction_results = {}
for label in args.embedding_labels:
    for method in ["mean", "median", "percentiles"]:
        # Look in the age_strain_predictor output directories
        subfolder = os.path.join(os.path.dirname(args.results_json), f"age_strain_predictor_{label}")
        json_filename = f"4320_3days_345_{label}.json"
        json_path = os.path.join(subfolder, json_filename)
        key = f"{label}_{method}"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                prediction_results[key] = json.load(f)
                preview = list(prediction_results[key].items())[:3]
                print(f"[INFO] Loaded {key} from {json_path} | First entries: {preview}")
        else:
            prediction_results[key] = {}
            print(f"[WARNING] Missing JSON for {key}: {json_path}")

# =============================
# Grid Layout Function
# =============================
def get_grid_layout(num_embeddings):
    """Determine grid layout based on number of embeddings."""
    if num_embeddings <= 2:
        return (1, num_embeddings)  # 1 row, up to 2 columns
    elif num_embeddings <= 4:
        return (2, 2)  # 2x2 grid
    elif num_embeddings <= 6:
        return (2, 3)  # 2 rows, 3 columns
    else:
        return (2, 4)  # 2 rows, 4 columns (max 8 embeddings)

# =============================
# Utilities
# =============================
def load_embeddings(path):
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    return d["embeddings"], d["frame_number_map"]

def plot_umaps_to_pdf(X, y_age, y_strain, cage_ids, embedding_label):
    strains = sorted(set(y_strain))
    
    # Filter strains if specified
    if args.strains_filter:
        strains = [s for s in strains if s in args.strains_filter]
        if not strains:
            print(f"[WARNING] No matching strains found for filter: {args.strains_filter}")
            return
    
    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, f"{args.umap_name_prefix}_{embedding_label}.pdf")

    with PdfPages(pdf_path) as pdf:
        for strain in tqdm(strains, desc=f"[{embedding_label.upper()}] Strains", disable=not sys.stdout.isatty()):
            mask = y_strain == strain
            if np.sum(mask) < 5:
                continue

            X_s = X[mask]
            y_age_s = y_age[mask]
            cage_s = cage_ids[mask]

            reducer = umap.UMAP(random_state=42)
            X_scaled = StandardScaler().fit_transform(X_s)
            X_umap = reducer.fit_transform(X_scaled)

            # Create figure with dark background
            fig, ax = plt.subplots(figsize=(7, 6), facecolor='black')
            ax.set_facecolor('black')
            
            unique_cages = sorted(set(cage_s))
            marker_map = {c: m for c, m in zip(unique_cages, itertools.cycle(markers))}

            for cage in unique_cages:
                cage_mask = cage_s == cage
                ax.scatter(X_umap[cage_mask, 0], X_umap[cage_mask, 1], c=y_age_s[cage_mask],
                           cmap="viridis", s=12, alpha=0.7, marker=marker_map[cage], label=f"Cage {cage}")

            sm = plt.cm.ScalarMappable(cmap="viridis")
            sm.set_array(y_age_s)
            cbar = fig.colorbar(sm, ax=ax, label="Average Age (days)")
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')

            # Get RMSE for title
            duration = embedding_lengths.get(embedding_label.split("_")[0], "")
            strain_key = str(strain).strip()
            label_key = embedding_label

            rmse_val = prediction_results.get(label_key, {}).get(strain_key, {}).get("rmse_age_prediction", None)
            rmse_str = f"RMSE: {rmse_val:.2f}" if rmse_val is not None else "RMSE: N/A"

            ax.set_title(f"{label_key.upper()} | {strain_key}\n{duration}\n{rmse_str}", 
                        fontsize=9, color='white')
            ax.set_xlabel("UMAP-1", color='white')
            ax.set_ylabel("UMAP-2", color='white')
            ax.tick_params(colors='white')
            
            legend = ax.legend(fontsize=6, loc='best')
            legend.get_frame().set_facecolor('black')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
                
            fig.tight_layout()
            pdf.savefig(fig, facecolor='black')
            plt.close(fig)

    print(f"[SAVED] Full PDF: {pdf_path}")

def plot_comparative_umaps(datasets, method):
    """Create comparative UMAP plots with flexible grid layout."""
    variant_labels = [f"{label}_{method}" for label in args.embedding_labels]
    
    # Get all strains from the first available dataset
    all_strains = None
    for label in variant_labels:
        if label in datasets:
            all_strains = sorted(set(datasets[label][2]))
            break
    
    if all_strains is None:
        print(f"[WARNING] No datasets found for method {method}")
        return
    
    # Filter strains if specified
    if args.strains_filter:
        all_strains = [s for s in all_strains if s in args.strains_filter]
        if not all_strains:
            print(f"[WARNING] No matching strains found for filter: {args.strains_filter}")
            return
    
    pdf_path = os.path.join(args.output_dir, f"{args.umap_name_prefix}_comparison_{method}.pdf")
    
    # Determine grid layout
    num_embeddings = len(variant_labels)
    rows, cols = get_grid_layout(num_embeddings)
    
    with PdfPages(pdf_path) as pdf:
        for strain in tqdm(all_strains, desc=f"[{method.upper()} COMPARISON] Strains", disable=not sys.stdout.isatty()):
            # Create figure with dark background
            fig = plt.figure(figsize=(5*cols, 4*rows), facecolor='black')
            
            # Create a single colorbar axis
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            added_colorbar = False
            
            for idx, label in enumerate(variant_labels):
                if label not in datasets:
                    continue
                    
                ax = plt.subplot(rows, cols, idx + 1, facecolor='black')
                
                X, y_age, y_strain, cage_ids = datasets[label]
                mask = (y_strain == strain)
                
                if np.sum(mask) < 5:
                    ax.set_title(f"{label.upper()} - N/A", color='white')
                    ax.axis('off')
                    continue

                X_s = X[mask]
                y_age_s = y_age[mask]
                cage_s = cage_ids[mask]
                X_scaled = StandardScaler().fit_transform(X_s)
                X_umap = umap.UMAP(random_state=42).fit_transform(X_scaled)

                unique_cages = sorted(set(cage_s))
                marker_map = {c: m for c, m in zip(unique_cages, itertools.cycle(markers))}

                for cage in unique_cages:
                    cage_mask = cage_s == cage
                    ax.scatter(X_umap[cage_mask, 0], X_umap[cage_mask, 1], c=y_age_s[cage_mask],
                               cmap="viridis", s=10, alpha=0.7, marker=marker_map[cage], label=f"Cage {cage}")

                # Add colorbar only once
                if not added_colorbar:
                    sm = plt.cm.ScalarMappable(cmap="viridis")
                    sm.set_array(y_age_s)
                    cbar = plt.colorbar(sm, cax=cbar_ax, label="Average Age (days)")
                    cbar.ax.yaxis.label.set_color('white')
                    cbar.ax.tick_params(colors='white')
                    added_colorbar = True

                # Get RMSE for subplot title
                base_label = label.split("_")[0]
                duration = embedding_lengths.get(base_label, "")
                strain_key = str(strain).strip()
                
                rmse_val = prediction_results.get(label, {}).get(strain_key, {}).get("rmse_age_prediction", None)
                rmse_str = f"\nRMSE: {rmse_val:.2f}" if rmse_val is not None else "\nRMSE: N/A"
                
                ax.set_title(f"{label.upper()}\n{duration}{rmse_str}", fontsize=8, color='white')
                ax.set_xlabel("UMAP-1", color='white')
                ax.set_ylabel("UMAP-2", color='white')
                ax.tick_params(colors='white')
                
                # Add legend for first subplot only to avoid clutter
                if idx == 0:
                    legend = ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1, 1))
                    legend.get_frame().set_facecolor('black')
                    legend.get_frame().set_edgecolor('white')
                    legend.get_frame().set_alpha(0.8)
                    for text in legend.get_texts():
                        text.set_color('white')

            # Get strain information
            strain_cages = []
            for label in variant_labels:
                if label in datasets:
                    strain_cages = sorted(set(datasets[label][3][datasets[label][2] == strain]))
                    break
            
            num_cages = len(strain_cages)

            # Find the best RMSE across all embedding-method combinations for this strain
            best_rmse = None
            best_key = None
            for key, result_dict in prediction_results.items():
                if strain in result_dict:
                    val = result_dict[strain].get("rmse_age_prediction", None)
                    if val is not None and (best_rmse is None or val < best_rmse):
                        best_rmse = val
                        best_key = key

            if best_rmse is not None:
                rmse_str = f" | Best RMSE: {best_rmse:.2f} ({best_key})"
            else:
                rmse_str = ""

            fig.suptitle(f"Strain: {strain} | Cages ({num_cages}): {', '.join(str(c) for c in strain_cages)}{rmse_str}",
                         fontsize=14, color='white')
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            pdf.savefig(fig, facecolor='black')
            plt.close(fig)

    print(f"[SAVED] Combined {method.upper()} PDF: {pdf_path}")

# =============================
# Metadata
# =============================
metadata = pd.read_csv(args.metadata_path, low_memory=False)
metadata["from_tpt"] = pd.to_datetime(metadata["from_tpt"])
metadata["key"] = metadata.apply(lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1)

# =============================
# Load embeddings dynamically
# =============================
embedding_types = {}
for path, label in zip(args.embedding_paths, args.embedding_labels):
    embedding_types[label] = path

datasets = {}

for label, path in tqdm(embedding_types.items(), desc="[LOOP] Embedding types", disable=not sys.stdout.isatty()):
    print(f"\n[INFO] Processing {label} embeddings...")
    embeddings, frame_map = load_embeddings(path)

    for method in ["mean", "median", "percentiles"]:
        X, y_age, y_strain, cage_ids = [], [], [], []
        for row in metadata.itertuples():
            key = row.key
            if key not in frame_map: 
                continue
            start, end = frame_map[key]
            if end <= start or end > embeddings.shape[0]: 
                continue
            emb = embeddings[start:end]
            if emb.shape[0] == 0 or np.isnan(emb).any(): 
                continue
            strain = str(row.strain).strip()
            if strain == '' or strain.lower() == 'nan': 
                continue

            if method == "mean":
                flat = emb.mean(axis=0)
            elif method == "median":
                flat = np.median(emb, axis=0)
            elif method == "percentiles":
                flat = np.percentile(emb, [25, 50, 75], axis=0).flatten()
            else:
                continue

            if np.isnan(flat).any(): 
                continue

            X.append(flat)
            y_age.append(row.avg_age_days_chunk_start)
            y_strain.append(strain)
            cage_ids.append(row.cage_id)

        X = np.array(X)
        y_age = np.array(y_age)
        y_strain = np.array(y_strain)
        cage_ids = np.array(cage_ids)

        ds_label = f"{label}_{method}"
        datasets[ds_label] = (X, y_age, y_strain, cage_ids)
        
        # Plot individual embedding type
        plot_umaps_to_pdf(X, y_age, y_strain, cage_ids, embedding_label=ds_label)

# =============================
# Combined Comparison PDFs
# =============================
# for method in ["mean", "edian", "percentiles"]:
for method in ["mean"]:
    plot_comparative_umaps(datasets, method)

print("\n[COMPLETE] All UMAP PDFs saved with strain filtering and flexible layouts.")