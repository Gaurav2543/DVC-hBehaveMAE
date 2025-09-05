import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import umap.umap_ as umap
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib as mpl
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# CONFIG
EMBEDDINGS_PATH = "extracted_embeddings_z/test_submission_stage_0.npy"
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
    embeddings = data['embeddings']

    labels = np.load(LABELS_PATH, allow_pickle=True)
    if isinstance(labels, np.ndarray) and labels.size == 1:
        labels = labels.item()
    label_array = labels['label_array']
    vocabulary = labels['vocabulary']

    print(f"[INFO] Loaded embeddings of shape {embeddings.shape}")
    print(f"[INFO] Available labels: {vocabulary}")
    return embeddings, label_array, vocabulary

def plot_umap(embeddings, label_array, vocabulary, label_name, pdf, n_samples=None, remove_legend=False, title_suffix=""):

    if label_name not in vocabulary:
        raise ValueError(f"Label '{label_name}' not found in vocabulary. Available: {vocabulary}")

    label_idx = vocabulary.index(label_name)
    labels = label_array[label_idx]

    if n_samples and len(labels) > n_samples:
        sampled_idx = np.random.choice(len(labels), size=n_samples, replace=False)
        embeddings = embeddings[sampled_idx]
        labels = labels[sampled_idx]

    reducer = umap.UMAP(random_state=42, low_memory=True, n_neighbors=15, min_dist=0.3)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))

    if labels.dtype.kind in 'iu':  # Integer/continuous-like
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
        if label_name == "Age_Days":
            cbar = plt.colorbar(scatter)
            cbar.set_label("Age (Days)")
            cbar.set_ticks([30, 500, 1000, 1500, 2000, 2500])
        else:
            plt.colorbar(scatter, label=label_name)
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
    pdf_path = os.path.join(OUTPUT_DIR, "z_test_submission_stage_0.pdf")
    print(f"[INFO] Starting UMAP visualizations. Output: {pdf_path}")

    # Preprocess and apply PCA once
    print("[INFO] Preprocessing and reducing all embeddings...")
    preprocessed_embeddings = preprocess_data(embeddings)
    pca_embeddings = reduce_dimensions(preprocessed_embeddings)

    with PdfPages(pdf_path) as pdf:
        for label in tqdm(LABEL_NAMES, desc="Processing labels"):
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
                    print(f"  [Strain: {strain}] {num_samples} samples")

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
                        remove_legend=True,
                        title_suffix=f" (Strain: {strain})"
                    )

            else:
                print(f"\n[INFO] Plotting UMAP for label: {label} (subsampled to {N_SAMPLES})...")
                plot_umap(pca_embeddings, label_array, vocabulary, label_name=label, pdf=pdf, n_samples=N_SAMPLES)

    print(f"\nAll UMAP plots saved to: {pdf_path}")
