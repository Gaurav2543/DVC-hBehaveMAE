import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
# import umap.umap_ as umap
from umap import UMAP
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib as mpl
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# CONFIG
EMBEDDINGS_PATH = "extracted_embeddings_z/test_submission_stage_0.npy"
# EMBEDDINGS_PATH = "extracted_embeddings_per/test_submission_stage_0.npy"
# EMBEDDINGS_PATH = "extracted_embeddings_per/test_submission_stage_1.npy"
# EMBEDDINGS_PATH = "extracted_embeddings2_per/test_submission2.npy"
# EMBEDDINGS_PATH = "extracted_embeddings2_z/test_submission.npy"

LABELS_PATH = "test-outputs/arrays_sub20_with_cage_complete_correct_strains.npy"
OUTPUT_DIR = "./"
LABEL_NAMES = ["Age_Days"]
N_SAMPLES = 10000
MAX_PER_STRAIN = 10000  # Maximum samples per strain for UMAP

def preprocess_data(X):
    X = X.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X = np.clip(X, -65504, 65504)
    return X

def reduce_dimensions(X, n_components=50):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def load_data():
    print("[INFO] Loading embeddings and label arrays...")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data.item()
    embeddings = data["embeddings"]  # (N_frames, D)

    labels_np = np.load(LABELS_PATH, allow_pickle=True)
    if isinstance(labels_np, np.ndarray) and labels_np.size == 1:
        labels_np = labels_np.item()
    label_array = np.array(labels_np["label_array"])  # shape (n_labels, N_frames)
    vocabulary  = labels_np["vocabulary"]

    # truncate to common length
    N_embed = embeddings.shape[0]
    N_label = label_array.shape[1]
    L = min(N_embed, N_label)
    embeddings = embeddings[:L]
    label_array = label_array[:, :L]  # now shape (n_labels, L)

    print(f"[INFO] Loaded embeddings of shape {embeddings.shape}")
    print(f"[INFO] Loaded {label_array.shape[1]} labels over {label_array.shape[0]} fields")
    print(f"[INFO] Available labels: {vocabulary}")
    return embeddings, label_array, vocabulary

def plot_umap(embeddings, label_array, vocabulary, label_name, pdf, n_samples=None, remove_legend=False, title_suffix=""):

    if label_name not in vocabulary:
        raise ValueError(f"Label '{label_name}' not found in vocabulary. Available: {vocabulary}")

    label_idx = vocabulary.index(label_name)
    labels = label_array[label_idx]

    if n_samples and len(labels) > n_samples:
        n = embeddings.shape[0]
        sampled_idx = np.random.choice(n, size=n_samples, replace=False)
        # sampled_idx = np.random.choice(len(labels), size=n_samples, replace=False)
        embeddings = embeddings[sampled_idx]
        labels = labels[sampled_idx]

    # reducer = umap.UMAP(random_state=42, low_memory=True, n_neighbors=15, min_dist=0.3)
    reducer = UMAP(random_state=42, low_memory=True, n_neighbors=15, min_dist=0.3)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))

    # if labels.dtype.kind in 'iu':  # Integer/continuous-like
    if labels.dtype.kind in 'if':    # treat both ints *and* floats as continuous
        norm = mpl.colors.Normalize(vmin=30, vmax=2500) if label_name == "Age_Days" else None
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='viridis',
            norm=norm,
            s=8,
            alpha=0.8
        )
        # if label_name == "Age_Days":
        #     cbar = plt.colorbar(scatter)
        #     cbar.set_label("Age (Days)")
        #     cbar.set_ticks([30, 500, 1000, 1500, 2000, 2500])
        # else:
        #     plt.colorbar(scatter, label=label_name)
        # only one nice colorbar
        cbar = plt.colorbar(scatter)
        if label_name == "Age_Days":
            cbar.set_label("Age (Days)")
            # auto–spread 6 ticks from min→max
            mn, mx = float(labels.min()), float(labels.max())
            cbar.set_ticks(np.linspace(mn, mx, 6))
        else:
            cbar.set_label(label_name)
    else:
        unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=encoded_labels,
            cmap='Spectral',
            s=8,
            alpha=0.8
        )

        if not remove_legend:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           label=str(label),
                           markerfacecolor=scatter.cmap(scatter.norm(i)),
                           markersize=6)
                for i, label in enumerate(unique_labels)
            ]
            plt.legend(handles=handles, title=label_name,
                       bbox_to_anchor=(1.02, 1), loc='upper left',
                       borderaxespad=0., fontsize='small', ncol=1, frameon=False)

    plt.title(f"UMAP projection colored by {label_name}{title_suffix}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.axis('equal')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    pdf.savefig()
    plt.close()
    print(f"[✓] UMAP for {label_name} added to PDF.")

if __name__ == "__main__":
    embeddings, label_array, vocabulary = load_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, "umap_z_test_submission_stage_0.pdf")
    print(f"[INFO] Starting UMAP visualizations. Output: {pdf_path}")
    
    # print("[INFO] Age_Days summary per cage:\
    # [Cage: EHDP-00777] N = 269280 | Age_Days range: 60.0 – 582.0\
    # [Cage: EHDP-00779] N = 252000 | Age_Days range: 57.0 – 573.0\
    # [Cage: EHDP-00780] N = 263520 | Age_Days range: 53.0 – 575.0\
    # [Cage: EHDP-00783] N = 241920 | Age_Days range: 55.0 – 573.0\
    # [Cage: EHDP-00812] N = 254880 | Age_Days range: 62.0 – 588.0\
    # [Cage: EHDP-00813] N = 262080 | Age_Days range: 63.0 – 588.0\
    # [Cage: EHDP-00814] N = 267840 | Age_Days range: 60.0 – 579.0\
    # [Cage: EHDP-00815] N = 260640 | Age_Days range: 57.0 – 582.0\
    # [Cage: EHDP-00816] N = 253440 | Age_Days range: 56.0 – 566.1\
    # [Cage: EHDP-00817] N = 257760 | Age_Days range: 57.9 – 565.0\
    # [Cage: EHDP-00818] N = 273600 | Age_Days range: 57.0 – 582.0\
    # [Cage: EHDP-00890] N = 285120 | Age_Days range: 50.0 – 571.0\
    # [Cage: EHDP-00891] N = 262080 | Age_Days range: 51.0 – 575.0\
    # [Cage: EHDP-01107] N = 256320 | Age_Days range: 32.0 – 564.0\
    # [Cage: EHDP-01108] N = 289440 | Age_Days range: 32.0 – 567.0\
    # [Cage: EHDP-01321] N = 282240 | Age_Days range: 65.0 – 589.0\
    # [Cage: EHDP-01327] N = 257760 | Age_Days range: 57.0 – 581.0\
    # [Cage: EHDP-01328] N = 285120 | Age_Days range: 52.0 – 568.0\
    # [Cage: EHDP-01480] N = 277920 | Age_Days range: 54.0 – 572.0\
    # [Cage: EHDP-01481] N = 269280 | Age_Days range: 47.0 – 563.9\
    # [INFO] Age_Days summary per strain:\
    # [Strain: B6CAST-129SPWK-F2] N = 547200 | Age_Days range: 47.0 – 572.0\
    # [Strain: BXD102] N = 542880 | Age_Days range: 52.0 – 581.0\
    # [Strain: BXD62] N = 545760 | Age_Days range: 32.0 – 567.0\
    # [Strain: BXD84] N = 282240 | Age_Days range: 65.0 – 589.0\
    # [Strain: CAROLI/EiJ] N = 241920 | Age_Days range: 55.0 – 573.0\
    # [Strain: CC005/TauUnc] N = 269280 | Age_Days range: 60.0 – 582.0\
    # [Strain: CC042/Unc] N = 254880 | Age_Days range: 62.0 – 588.0\
    # [Strain: CC059/TauUnc] N = 262080 | Age_Days range: 51.0 – 575.0\
    # [Strain: CC078/TauUncJ] N = 529920 | Age_Days range: 60.0 – 588.0\
    # [Strain: DBA/2J] N = 285120 | Age_Days range: 50.0 – 571.0\
    # [Strain: LG/J] N = 514080 | Age_Days range: 56.0 – 582.0\
    # [Strain: MSM/MsJ] N = 515520 | Age_Days range: 53.0 – 575.0\
    # [Strain: PANCEVO/EiJ] N = 531360 | Age_Days range: 57.0 – 582.0)")

    # Preprocess and apply PCA once
    print("[INFO] Preprocessing and reducing all embeddings...")
    preprocessed_embeddings = preprocess_data(embeddings)
    pca_embeddings = reduce_dimensions(preprocessed_embeddings)

    with PdfPages(pdf_path) as pdf:
        for label in LABEL_NAMES:
            if label == "Age_Days":
                print(f"\n[INFO] Plotting UMAP for all samples colored by Age_Days (subsampled to {N_SAMPLES})...")
                plot_umap(pca_embeddings, label_array, vocabulary, label_name=label, pdf=pdf, n_samples=N_SAMPLES, remove_legend=True)

                print("[INFO] Now plotting Age_Days UMAPs grouped by strain...")

                strain_idx = vocabulary.index("Strain")
                strains = label_array[strain_idx]
                unique_strains = np.unique(strains)

                for strain in tqdm(unique_strains, desc="Processing strains"):
                    strain_mask = (strains == strain)
                    num_samples = strain_mask.sum()
                    print(f"  [Strain: {strain}] {num_samples} samples \n")

                    filtered_embeddings = pca_embeddings[strain_mask]
                    filtered_label_array = [la[strain_mask] for la in label_array]

                    # Subsample if too many samples
                    if num_samples > MAX_PER_STRAIN:
                        idx = np.random.choice(num_samples, MAX_PER_STRAIN, replace=False)
                        filtered_embeddings = filtered_embeddings[idx]
                        filtered_label_array = [la[idx] for la in filtered_label_array]

                    plot_umap(
                        filtered_embeddings,
                        filtered_label_array,
                        vocabulary,
                        label_name="Age_Days",
                        pdf=pdf,
                        n_samples=None,
                        remove_legend=False,
                        title_suffix=f" (Strain: {strain})"
                    )

            else:
                print(f"\n[INFO] Plotting UMAP for label: {label} (subsampled to {N_SAMPLES})...")
                plot_umap(pca_embeddings, label_array, vocabulary, label_name=label, pdf=pdf, n_samples=N_SAMPLES)

    print(f"\nAll UMAP plots saved to: {pdf_path}")