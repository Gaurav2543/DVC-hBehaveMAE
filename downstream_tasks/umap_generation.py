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

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden.*", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*", category=UserWarning)

# =============================
# Argument Parsing
# =============================
parser = argparse.ArgumentParser(description="UMAP Visualization Script")
parser.add_argument('--metadata_path', required=True)
parser.add_argument('--results_json', required=True)
parser.add_argument('--output_dir', required=True)
# parser.add_argument('--embedding_level1', required=True)
# parser.add_argument('--embedding_level2', required=True)
parser.add_argument('--embedding_level3', required=True)
parser.add_argument('--embedding_level4', required=True)
parser.add_argument('--embedding_level5', required=True)
parser.add_argument('--embedding_comb', required=True)
parser.add_argument('--embedding_lengths', required=True, help='JSON string: {"level1":"(15mins)",...}')
parser.add_argument('--umap_name_prefix', required=True)
parser.add_argument('--embedding_dimensions', type=int, default=1440)
args = parser.parse_args()

# =============================
# Configuration
# =============================
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
# with open(args.results_json, "r") as f:
#     prediction_results = json.load(f)
embedding_lengths = json.loads(args.embedding_lengths)

# Build prediction_results[ds_label] = json content
prediction_results = {}
# for label in ["level1", "level2", "level3", "level4", "level5", "comb"]:
# for label in ["level3", "level4", "level5", "comb"]:
#     for method in ["mean", "median", "percentiles"]:
#         subfolder = os.path.join(os.path.dirname(args.results_json), f"3days_{label}", method)
#         json_filename = f"results_predictions_{label}_{method}.json"
#         json_path = os.path.join(subfolder, json_filename)
#         key = f"{label}_{method}"
#         if os.path.exists(json_path):
#             with open(json_path, "r") as f:
#                 prediction_results[key] = json.load(f)
for label in ["level3", "level4", "level5", "comb"]:
    for method in ["mean", "median", "percentiles"]:
        # Look in the age_strain_predictor output directories
        subfolder = os.path.join(os.path.dirname(args.results_json), f"age_strain_predictor_{label}")
        json_filename = f"4320_3days_345_{label}.json"  # Use your actual JSON naming pattern
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
# Utilities
# =============================
def load_embeddings(path):
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    return d["embeddings"], d["frame_number_map"]

def plot_umaps_to_pdf(X, y_age, y_strain, cage_ids, embedding_label):
    strains = sorted(set(y_strain))
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

            fig, ax = plt.subplots(figsize=(7, 6))
            unique_cages = sorted(set(cage_s))
            marker_map = {c: m for c, m in zip(unique_cages, itertools.cycle(markers))}

            for cage in unique_cages:
                cage_mask = cage_s == cage
                ax.scatter(X_umap[cage_mask, 0], X_umap[cage_mask, 1], c=y_age_s[cage_mask],
                           cmap="viridis", s=12, alpha=0.7, marker=marker_map[cage], label=f"Cage {cage}")

            sm = plt.cm.ScalarMappable(cmap="viridis")
            sm.set_array(y_age_s)
            fig.colorbar(sm, ax=ax, label="Average Age (days)")

            # -------------------------------
            # RMSE + Strain title debug logic
            # -------------------------------
            duration = embedding_lengths.get(embedding_label.split("_")[0], "")
            strain_key = str(strain).strip()
            # label_key = embedding_label.lower()  # enforce lowercase matching
            label_key = embedding_label

            # Debug: print available keys
            if label_key not in prediction_results:
                print(f"[DEBUG] Label key '{label_key}' NOT found in prediction_results keys: {list(prediction_results.keys())[:5]}")
            elif strain_key not in prediction_results[label_key]:
                print(f"[DEBUG] Strain '{strain_key}' NOT found in prediction_results['{label_key}']")
            else:
                print(f"[DEBUG] Found RMSE for label '{label_key}' and strain '{strain_key}': {prediction_results[label_key][strain_key].get('rmse_age_prediction')}")

            rmse_val = prediction_results.get(label_key, {}).get(strain_key, {}).get("rmse_age_prediction", None)
            rmse_str = f"RMSE: {rmse_val:.2f}" if rmse_val is not None else "RMSE: N/A"

            ax.set_title(f"{label_key.upper()} | {strain_key}\n{duration}\n{rmse_str}", fontsize=9)
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.legend(fontsize=6, loc='best')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[SAVED] Full PDF: {pdf_path}")


# =============================
# Metadata
# =============================
metadata = pd.read_csv(args.metadata_path, low_memory=False)
metadata["from_tpt"] = pd.to_datetime(metadata["from_tpt"])
metadata["key"] = metadata.apply(lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1)

# =============================
# Embedding types and loop
# =============================
embedding_types = {
    # "level1": args.embedding_level1,
    # "level2": args.embedding_level2,
    "level3": args.embedding_level3,
    "level4": args.embedding_level4,
    "level5": args.embedding_level5,
    "comb": args.embedding_comb
}

datasets = {}

for label, path in tqdm(embedding_types.items(), desc="[LOOP] Embedding types", disable=not sys.stdout.isatty()):
    print(f"\n[INFO] Processing {label} embeddings...")
    embeddings, frame_map = load_embeddings(path)

    for method in ["mean", "median", "percentiles"]:
        X, y_age, y_strain, cage_ids = [], [], [], []
        for row in metadata.itertuples():
            key = row.key
            if key not in frame_map: continue
            start, end = frame_map[key]
            if end <= start or end > embeddings.shape[0]: continue
            emb = embeddings[start:end]
            if emb.shape[0] == 0 or np.isnan(emb).any(): continue
            strain = str(row.strain).strip()
            if strain == '' or strain.lower() == 'nan': continue

            if method == "mean":
                flat = emb.mean(axis=0)
            elif method == "median":
                flat = np.median(emb, axis=0)
            elif method == "percentiles":
                flat = np.percentile(emb, [25, 50, 75], axis=0).flatten()
            else:
                continue

            if np.isnan(flat).any(): continue

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
        plot_umaps_to_pdf(X, y_age, y_strain, cage_ids, embedding_label=ds_label)

# =============================
# Combined Comparison PDFs
# =============================
for method in ["mean", "median", "percentiles"]:
    # variant_labels = [f"{t}_{method}" for t in ['level1', 'level2', 'level3', 'level4', 'level5', 'comb']]
    variant_labels = [f"{t}_{method}" for t in ['level3', 'level4', 'level5', 'comb']]
    all_strains = sorted(set(datasets[variant_labels[-1]][2]))  # from comb
    pdf_path = os.path.join(args.output_dir, f"{args.umap_name_prefix}_comparison_{method}.pdf")

    with PdfPages(pdf_path) as pdf:
        for strain in tqdm(all_strains, desc=f"[{method.upper()} COMPARISON] Strains", disable=not sys.stdout.isatty()):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            added = False

            for ax, label in zip(axes, variant_labels):
                X, y_age, y_strain, cage_ids = datasets[label]
                mask = (y_strain == strain)
                if np.sum(mask) < 5:
                    ax.set_title(f"{label.upper()} - N/A")
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

                if not added:
                    sm = plt.cm.ScalarMappable(cmap="viridis")
                    sm.set_array(y_age_s)
                    plt.colorbar(sm, cax=cbar_ax, label="Average Age (days)")
                    added = True

                duration = embedding_lengths.get(label.split("_")[0], "")
                ax.set_title(f"{label.upper()}\n{duration}")

            strain_cages = sorted(set(datasets[variant_labels[-1]][3][datasets[variant_labels[-1]][2] == strain]))
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
                         fontsize=14)
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[SAVED] Combined {method.upper()} PDF: {pdf_path}")

print("\n[COMPLETE] All UMAP PDFs saved.")
