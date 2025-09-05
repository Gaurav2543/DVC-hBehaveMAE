import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================
# Argument Parser
# =============================
parser = argparse.ArgumentParser(description="UMAP + Clustering colored by strain")
parser.add_argument('--metadata_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--embedding_dimensions', type=int, default=1440)
parser.add_argument('--method', type=str, default="mean", choices=["mean", "median", "percentiles"])
parser.add_argument('--embedding_level1', required=True)
parser.add_argument('--embedding_level2', required=True)
parser.add_argument('--embedding_level3', required=True)
parser.add_argument('--embedding_level4', required=True)
parser.add_argument('--embedding_level5', required=True)
parser.add_argument('--embedding_comb', required=True)
args = parser.parse_args()

# =============================
# Load Metadata
# =============================
metadata = pd.read_csv(args.metadata_path, low_memory=False)
metadata["from_tpt"] = pd.to_datetime(metadata["from_tpt"])
metadata["key"] = metadata.apply(lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1)

# =============================
# Strain Grouping Function
# =============================
def map_strain_to_group(strain_name):
    if strain_name.startswith("BXD"):
        return "BXD"
    elif strain_name.startswith("CC"):
        return "CC"
    else:
        return strain_name  # Keep original name for other strains

# =============================
# Helper to Load + Process Embedding
# =============================
def process_embedding(path, label):
    d = np.load(path, allow_pickle=True)
    if isinstance(d, np.ndarray) and d.size == 1:
        d = d.item()
    embeddings = d["embeddings"]
    frame_map = d["frame_number_map"]

    X_all, y_strain_all, y_group_all = [], [], []

    for row in metadata.itertuples():
        key = row.key
        if key not in frame_map: continue
        start, end = frame_map[key]
        if end <= start or end > embeddings.shape[0]: continue
        emb = embeddings[start:end]
        if emb.shape[0] == 0 or np.isnan(emb).any(): continue
        strain = str(row.strain).strip()
        if strain == '' or strain.lower() == 'nan': continue

        if args.method == "mean":
            flat = emb.mean(axis=0)
        elif args.method == "median":
            flat = np.median(emb, axis=0)
        elif args.method == "percentiles":
            flat = np.percentile(emb, [25, 50, 75], axis=0).flatten()
        else:
            continue

        if np.isnan(flat).any(): continue

        X_all.append(flat)
        y_strain_all.append(strain)
        y_group_all.append(map_strain_to_group(strain))

    X_all = np.array(X_all)
    y_strain_all = np.array(y_strain_all)
    y_group_all = np.array(y_group_all)
    return X_all, y_strain_all, y_group_all

# =============================
# Process Each Embedding Type
# =============================
embedding_paths = {
    "level1": args.embedding_level1,
    "level2": args.embedding_level2,
    "level3": args.embedding_level3,
    "level4": args.embedding_level4,
    "level5": args.embedding_level5,
    "comb": args.embedding_comb
}

os.makedirs(args.output_dir, exist_ok=True)

for label, path in embedding_paths.items():
    print(f"[INFO] Processing {label} embedding")
    X, y_strain, y_group = process_embedding(path, label)

    if len(X) == 0:
        print(f"[SKIP] No valid data for {label}")
        continue

    # UMAP
    X_scaled = StandardScaler().fit_transform(X)
    X_umap = umap.UMAP(random_state=42).fit_transform(X_scaled)

    # Clustering
    k = len(set(y_strain))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_umap)
    sil_score = silhouette_score(X_umap, clusters)

    # Create PDF with both strain and group plots
    pdf_path = os.path.join(args.output_dir, f"umap_strain_and_group_{label}_{args.method}.pdf")
    with PdfPages(pdf_path) as pdf:

        # Plot by exact strain
        plt.figure(figsize=(16, 16))
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_strain,
                        palette="tab20", s=5, alpha=0.8, edgecolor="none")
        plt.title(f"{label.upper()} - UMAP by Strain (method={args.method}, silhouette={sil_score:.2f})")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(loc='best', bbox_to_anchor=(1.02, 1), title='Strain', fontsize=8)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Plot by strain group
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_group,
                        palette="tab10", s=10, alpha=0.8, edgecolor="none")
        plt.title(f"{label.upper()} - UMAP by Strain Group (method={args.method})")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(loc='best', bbox_to_anchor=(1.02, 1), title='Strain Group', fontsize=10)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"[SAVED] PDF with strain + group plots: {pdf_path}")

print("[COMPLETE] All plots saved.")
