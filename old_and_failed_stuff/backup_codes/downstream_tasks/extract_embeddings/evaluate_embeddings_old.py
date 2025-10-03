#!/usr/bin/env python3
# evaluate_embeddings.py

import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, ConfusionMatrixDisplay, root_mean_squared_error
import matplotlib.pyplot as plt

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate learned embeddings with linear models for regression or classification."
    )
    p.add_argument("--embeddings", required=True, help="Path to embeddings file (*.npy or *.npz)")
    p.add_argument("--labels", required=True, help="Path to labels CSV file")
    p.add_argument("--task", required=True, choices=["regression", "classification"],
                   help="The evaluation task to perform.")
    p.add_argument("--target-label", required=True,
                   help="The column name in the CSV file to predict (e.g., 'avg_age_days_chunk_start').")
    # Add a new argument for the grouping variable in the regression task
    p.add_argument("--group-label", default="strain",
                   help="[For Regression] The column name to group by for Leave-One-Out validation (e.g., 'Strain').")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="[For Classification] Proportion of the dataset for the test split (default: 0.25 for a 75-25 split).")
    p.add_argument("--plot-confusion-matrix", action="store_true",
                   help="[For Classification] If set, saves the confusion matrix as a PNG file.")
    return p.parse_args()

def load_data(emb_path: str, labels_path: str, target_label: str, task: str, group_label: str) -> tuple:
    """Loads embeddings and labels, aligns them, and removes missing values."""
    print(f"[INFO] Loading embeddings from {emb_path}...")
    # Load the file, which could be a standard array or a wrapped object
    loaded_obj = np.load(emb_path, allow_pickle=True)

    # Check if the loaded object is a 0-d array containing a dictionary
    if loaded_obj.shape == () and isinstance(loaded_obj.item(), dict):
        embeddings = loaded_obj.item()['embeddings']
    # Check if it's an NPZ file with an 'embeddings' key
    elif isinstance(loaded_obj, np.lib.npyio.NpzFile):
        embeddings = loaded_obj['embeddings']
    # Otherwise, assume it's a standard numpy array
    else:
        embeddings = loaded_obj

    print(f"[INFO] Loading labels from {labels_path}...")
    labels_df = pd.read_csv(labels_path)
    
    # Define required columns based on the task
    required_cols = [target_label]
    if task == 'regression':
        required_cols.append(group_label)

    for col in required_cols:
        if col not in labels_df.columns:
            raise ValueError(f"Required label '{col}' not found in the CSV file.")

    n_samps = min(len(embeddings), len(labels_df))
    embeddings = embeddings[:n_samps]
    labels_df = labels_df.iloc[:n_samps].copy()

    # Create a working copy and handle missing values
    work_df = labels_df[required_cols].copy()
    work_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    work_df.dropna(inplace=True)

    if work_df.empty:
        raise ValueError(f"No valid data remains for targets after removing NaNs.")

    aligned_embeddings = embeddings[work_df.index]
    y = work_df[target_label].values
    groups = work_df[group_label].values if task == 'regression' else None
    class_names = sorted(list(np.unique(y))) if task == 'classification' else None

    print(f"[INFO] Data loaded. Found {len(y)} valid samples for evaluation.")
    return aligned_embeddings, y, groups, class_names

def evaluate_regression_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Performs Leave-One-Group-Out regression evaluation."""
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    unique_groups = np.unique(groups)
    print(f"\n[INFO] Starting Leave-One-Strain-Out validation across {n_splits} strains...")

    maes, r2s, baseline_maes, rmses = [], [], [], [] # <-- Add rmses list

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

        # Calculate metrics for this fold
        maes.append(mean_absolute_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
        # Calculate RMSE using mean_squared_error with squared=False
        rmses.append(root_mean_squared_error(y_test, y_pred))

        baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
        baseline_maes.append(mean_absolute_error(y_test, baseline_pred))

    # Calculate final mean and std dev of scores
    mean_mae, std_mae = np.mean(maes), np.std(maes)
    mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
    mean_rmse, std_rmse = np.mean(rmses), np.std(rmses) # <-- Calculate mean/std for RMSE
    mean_baseline_mae = np.mean(baseline_maes)

    print("\n" + "="*60)
    print("      LEAVE-ONE-STRAIN-OUT REGRESSION REPORT")
    print("="*60)
    print(f"  Mean Absolute Error (MAE): {mean_mae:.2f} (± {std_mae:.2f})")
    print(f"  Mean RMSE:                 {mean_rmse:.2f} (± {std_rmse:.2f})") # <-- Add RMSE to report
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
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, xticks_rotation='vertical', values_format='d', cmap='Blues')
        plt.tight_layout()
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        print(f"  ✔ Saved confusion matrix to {plot_path}")

def main():
    """Main function to run the evaluation."""
    args = get_args()
    try:
        X, y, groups, class_names = load_data(args.embeddings, args.labels, args.target_label, args.task, args.group_label)
        
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