#!/usr/bin/env python3
# evaluate_embeddings.py

import argparse
import warnings
import pandas as pd
import numpy as np
import zarr
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, ConfusionMatrixDisplay, root_mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate learned embeddings with linear models for regression or classification."
    )
    p.add_argument("--embeddings", required=True, help="Path to embeddings directory (contains aggregation subdirs)")
    p.add_argument("--labels", required=True, help="Path to labels CSV file")
    p.add_argument("--task", required=True, choices=["regression", "classification"],
                   help="The evaluation task to perform.")
    p.add_argument("--target-label", required=True,
                   help="The column name in the CSV file to predict (e.g., 'avg_age_days_chunk_start').")
    p.add_argument("--group-label", default="strain",
                   help="[For Regression] The column name to group by for Leave-One-Out validation (e.g., 'Strain').")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="[For Classification] Proportion of the dataset for the test split (default: 0.25 for a 75-25 split).")
    p.add_argument("--plot-confusion-matrix", action="store_true",
                   help="[For Classification] If set, saves the confusion matrix as a PNG file.")
    p.add_argument("--aggregation", default="1day", help="Aggregation level to use (e.g., '1day', '1.5h')")
    p.add_argument("--embedding-level", default="combined", 
                   help="Which embedding level to use ('level_1_pooled', 'combined', etc.)")
    return p.parse_args()

# def load_data_from_zarr(emb_path: str, labels_path: str, target_label: str, task: str, 
#                         group_label: str, aggregation: str, embedding_level: str) -> tuple:
#     """Loads embeddings from Zarr stores and labels, aligns them, and removes missing values."""
#     print(f"[INFO] Loading embeddings from {emb_path}/{aggregation}/core_embeddings/...")
    
#     emb_path = Path(emb_path)
#     core_emb_dir = emb_path / aggregation / 'core_embeddings'
    
#     if not core_emb_dir.exists():
#         raise FileNotFoundError(f"Embeddings directory not found: {core_emb_dir}")
    
#     print(f"[INFO] Loading labels from {labels_path}...")
#     labels_df = pd.read_csv(labels_path)
    
#     # Define required columns based on the task
#     required_cols = [target_label]
#     if task == 'regression':
#         required_cols.append(group_label)
    
#     for col in required_cols:
#         if col not in labels_df.columns:
#             raise ValueError(f"Required label '{col}' not found in the CSV file.")
    
#     # Prepare metadata keys
#     if 'from_tpt' in labels_df.columns:
#         labels_df['from_tpt'] = pd.to_datetime(labels_df['from_tpt'])
#         labels_df['key'] = labels_df.apply(
#             lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
#         )
    
#     # Load embeddings from individual Zarr files
#     embeddings_list = []
#     valid_indices = []
    
#     print("[INFO] Loading embeddings from Zarr stores...")
#     for idx, row in labels_df.iterrows():
#         key = row.get('key', f"{row['cage_id']}_{str(row['from_tpt']).split()[0]}")
#         zarr_path = core_emb_dir / f"{key}.zarr"
        
#         if zarr_path.exists():
#             try:
#                 store = zarr.open(str(zarr_path), mode='r')
                
#                 # Get the specified embedding level
#                 if embedding_level not in store:
#                     print(f"[WARN] Level '{embedding_level}' not found in {zarr_path}, trying 'combined'")
#                     embedding_level = 'combined'
                
#                 embedding = store[embedding_level][:]
                
#                 # Average over time dimension to get daily representation
#                 if len(embedding.shape) > 1:
#                     embedding = embedding.mean(axis=0)
                
#                 embeddings_list.append(embedding)
#                 valid_indices.append(idx)
#             except Exception as e:
#                 print(f"[WARN] Failed to load {zarr_path}: {e}")
    
#     if len(embeddings_list) == 0:
#         raise ValueError("No valid embeddings could be loaded from Zarr stores")
    
#     embeddings = np.array(embeddings_list)
#     labels_df = labels_df.iloc[valid_indices].copy()
    
#     print(f"[INFO] Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    
#     # Create a working copy and handle missing values
#     work_df = labels_df[required_cols].copy()
#     work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     work_df.dropna(inplace=True)
    
#     if work_df.empty:
#         raise ValueError(f"No valid data remains for targets after removing NaNs.")
    
#     aligned_embeddings = embeddings[work_df.index - valid_indices[0]]  # Adjust indices
#     y = work_df[target_label].values
#     groups = work_df[group_label].values if task == 'regression' else None
#     class_names = sorted(list(np.unique(y))) if task == 'classification' else None
    
#     print(f"[INFO] Data loaded. Found {len(y)} valid samples for evaluation.")
#     return aligned_embeddings, y, groups, class_names

def load_data_from_zarr(embeddings_dir, labels_path, aggregation, embedding_level, 
                        task, target_label, group_label=None):
    """Load embeddings from Zarr stores and align with labels"""
    print(f"[INFO] Loading embeddings from {embeddings_dir}/{aggregation}/core_embeddings/...")
    print(f"[INFO] Loading labels from {labels_path}...")
    
    # IMPORTANT: Use the aggregation parameter, NOT target_label
    core_emb_dir = Path(embeddings_dir) / aggregation / 'core_embeddings'
    
    if not core_emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {core_emb_dir}")
    
    # Load labels
    df = pd.read_csv(labels_path)
    
    # Prepare metadata keys
    if 'from_tpt' in df.columns:
        df['from_tpt'] = pd.to_datetime(df['from_tpt'])
        df['key'] = df.apply(
            lambda row: f"{row['cage_id']}_{row['from_tpt'].strftime('%Y-%m-%d')}", axis=1
        )
    
    print("[INFO] Loading embeddings from Zarr stores...")
    
    # Load embeddings into a dictionary first
    embeddings_dict = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings"):
        key = row['key']
        zarr_path = core_emb_dir / f"{key}.zarr"
        
        if zarr_path.exists():
            try:
                store = zarr.open(str(zarr_path), mode='r')
                
                # Get the specified embedding level
                if embedding_level not in store:
                    level_to_use = 'combined'
                else:
                    level_to_use = embedding_level
                
                seq_embeddings = store[level_to_use][:]
                
                # Handle both 1D and 2D embeddings
                if seq_embeddings.ndim == 1:
                    daily_embedding = seq_embeddings
                else:
                    daily_embedding = seq_embeddings.mean(axis=0)
                
                if not np.isnan(daily_embedding).any():
                    embeddings_dict[key] = daily_embedding
                    
            except Exception as e:
                print(f"[WARN] Failed to load {zarr_path}: {e}")
    
    if not embeddings_dict:
        raise ValueError(f"No embeddings loaded from {core_emb_dir}. Check that Zarr files exist and match metadata keys.")
    
    print(f"[INFO] Loaded {len(embeddings_dict)} embeddings with {next(iter(embeddings_dict.values())).shape[0]} dimensions")
    
    # Filter dataframe to only rows with valid embeddings
    df_with_embeddings = df[df['key'].isin(embeddings_dict.keys())].copy()
    
    print(f"[INFO] Filtered to {len(df_with_embeddings)} samples with both embeddings and labels")
    
    # Build aligned arrays
    aligned_embeddings = []
    aligned_labels = []
    aligned_groups = []
    
    for _, row in df_with_embeddings.iterrows():
        key = row['key']
        
        # Get embedding
        embedding = embeddings_dict[key]
        
        # Get label - skip if missing
        if target_label not in row or pd.isna(row[target_label]):
            continue
        
        label = row[target_label]
        
        # Get group if needed
        if group_label and group_label in row:
            group = row[group_label]
            if pd.isna(group):
                continue
            aligned_groups.append(group)
        else:
            group = None
        
        aligned_embeddings.append(embedding)
        aligned_labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(aligned_embeddings)
    y = np.array(aligned_labels)
    groups = np.array(aligned_groups) if aligned_groups else None
    
    print(f"[INFO] Final dataset: {X.shape[0]} samples with {X.shape[1]} features")
    
    if X.shape[0] == 0:
        raise ValueError(f"No valid samples found with target_label='{target_label}'. Check that this column exists and has non-NaN values.")
    
    # Handle task-specific processing
    class_names = None
    
    if task == 'classification':
        # Encode string labels to integers
        unique_classes = np.unique(y)
        class_names = unique_classes.tolist()
        label_map = {label: idx for idx, label in enumerate(unique_classes)}
        y = np.array([label_map[label] for label in y])
        print(f"[INFO] Classification task: {len(class_names)} classes - {class_names[:10]}")
    
    elif task == 'regression':
        # Ensure labels are numeric
        y = y.astype(float)
        print(f"[INFO] Regression task: target range [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, groups, class_names

def evaluate_regression_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Performs Leave-One-Group-Out regression evaluation."""
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    unique_groups = np.unique(groups)
    print(f"\n[INFO] Starting Leave-One-Strain-Out validation across {n_splits} strains...")

    maes, r2s, baseline_maes, rmses = [], [], [], []

    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        held_out_strain = unique_groups[i]
        print(f"  -> Fold {i+1}/{n_splits}: Holding out strain '{held_out_strain}'...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
        rmses.append(root_mean_squared_error(y_test, y_pred))

        baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
        baseline_maes.append(mean_absolute_error(y_test, baseline_pred))

    mean_mae, std_mae = np.mean(maes), np.std(maes)
    mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
    mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
    mean_baseline_mae = np.mean(baseline_maes)

    print("\n" + "="*60)
    print("      LEAVE-ONE-STRAIN-OUT REGRESSION REPORT")
    print("="*60)
    print(f"  Mean Absolute Error (MAE): {mean_mae:.2f} (± {std_mae:.2f})")
    print(f"  Mean RMSE:                 {mean_rmse:.2f} (± {std_rmse:.2f})")
    print(f"  Mean R-squared (R²):       {mean_r2:.3f} (± {std_r2:.3f})")
    print("-"*60)
    print(f"  Mean Baseline MAE:         {mean_baseline_mae:.2f}")
    print("="*60)
    print("\nInterpretation:")
    print("-> The model was tested on each strain after being trained on all other strains.")
    print(f"-> On average, when predicting age for a new strain, the error is {mean_mae:.2f} days.")
    print("-> The (±) value shows the standard deviation, indicating how much performance varies across strains.")
    
def evaluate_classification(X: np.ndarray, y: np.ndarray, test_size: float, class_names: list, plot_cm: bool):
    """Performs classification evaluation using a 75-25 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[INFO] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, target_names=class_names)
    
    print("\n" + "="*50)
    print("    CLASSIFICATION EVALUATION REPORT (75-25 Split)")
    print("="*50)
    print(report)
    print("="*50)

    if plot_cm:
        print("\n[INFO] Generating confusion matrix plot...")
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation='vertical', 
                                               values_format='d', cmap='Blues')
        plt.tight_layout()
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        print(f"  ✔ Saved confusion matrix to {plot_path}")

def main():
    """Main function to run the evaluation."""
    args = get_args()
    try:
        X, y, groups, class_names = load_data_from_zarr(
            args.embeddings, args.labels, args.target_label, args.task, 
            args.group_label, args.aggregation, args.embedding_level
        )
        
        if args.task == "regression":
            if groups is None:
                raise ValueError("Group labels are required for Leave-One-Strain-Out regression.")
            evaluate_regression_logo(X, y, groups)
        elif args.task == "classification":
            evaluate_classification(X, y, args.test_size, class_names, args.plot_confusion_matrix)

    except (ValueError, FileNotFoundError) as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()