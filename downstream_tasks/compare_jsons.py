import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True, help='Folder containing 3days_* subfolders')
parser.add_argument('--output_dir', required=True, help='Output folder for plots and CSVs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

variants = ['level1', 'level2', 'level3', 'level4', 'level5', 'comb']
methods = ['mean', 'median', 'percentiles']
rmse_table = {}

# Load all JSON files
for variant in variants:
    for method in methods:
        subdir = os.path.join(args.base_dir, f"3days_{variant}", method)
        json_file = f"results_predictions_{variant}_{method}.json"
        path = os.path.join(subdir, json_file)
        key = f"{variant}_{method}"

        if not os.path.exists(path):
            print(f"[WARNING] Missing: {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        for strain, metrics in data.items():
            rmse = metrics.get("rmse_age_prediction")
            if strain not in rmse_table:
                rmse_table[strain] = {}
            rmse_table[strain][key] = rmse

# Convert to DataFrame
df = pd.DataFrame(rmse_table).T  # rows = strains, cols = variant_method
df.to_csv(os.path.join(args.output_dir, "rmse_matrix.csv"))
print("\n[INFO] RMSE matrix saved.")

# Best variant per strain
best_variants = df.idxmin(axis=1)
best_rmses = df.min(axis=1)
summary = pd.DataFrame({
    "best_variant": best_variants,
    "best_rmse": best_rmses
})
summary = summary.sort_values(by="best_rmse")
summary.to_csv(os.path.join(args.output_dir, "best_rmse_summary.csv"))
print("[INFO] Best RMSE summary saved.")

# Plot: Barplot of best RMSEs
plt.figure(figsize=(22, 10))
sns.set_palette("Blues")
bar = sns.barplot(x=summary.index, y=summary["best_rmse"], hue=summary["best_variant"], dodge=False)
bar.set_xticklabels(bar.get_xticklabels(), rotation=90)
for i, val in enumerate(summary["best_rmse"]):
    bar.text(i, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=6)
plt.ylabel("Best RMSE")
plt.title("Best RMSE per Strain")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "barplot_best_rmse.pdf"))
plt.close()

# Plot: Heatmap of all RMSEs
plt.figure(figsize=(22, 12))
sns.set_palette("coolwarm")
sns.heatmap(df, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.4, cbar_kws={"label": "RMSE"})
plt.title("RMSE Heatmap by Variant and Strain")
plt.xlabel("Variant_Method")
plt.ylabel("Strain")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "heatmap_rmse.pdf"))
plt.close()

print("\n[INFO] All plots and summaries saved to:", args.output_dir)
