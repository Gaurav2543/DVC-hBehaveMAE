import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Input/output directories
input_dir = Path("/scratch/bhole/dvc_data/smoothed") / "cage_activations_full_data_final"
output_dir = Path("/scratch/bhole/dvc_data/smoothed") / "cage_activations_full_data_fixed"
output_dir.mkdir(parents=True, exist_ok=True)

# max workers (safe default)
MAX_WORKERS = min(8, (os.cpu_count() or 1))

def fix_csv(file: Path):
    """
    Read possibly-broken CSVs, normalize header, coerce v_1..v_12 to numeric
    (empty strings -> NaN), drop rows with any NaN in v_1..v_12, save result.
    Returns (filename, status, rows_kept, rows_dropped)
    """
    try:
        # Read raw as strings so empty fields remain as raw ""
        raw = pd.read_csv(file, header=None, dtype=str)

        # If file already parsed into multiple columns, use it; otherwise split the single-column string
        if raw.shape[1] == 1:
            df = raw[0].str.split(",", expand=True)
        else:
            df = raw.copy()

        # Use first row as header (strip whitespace)
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.drop(0).reset_index(drop=True)

        before_count = len(df)

        # v columns we care about
        v_cols = [f"v_{i}" for i in range(1, 13)]
        existing_v = [c for c in v_cols if c in df.columns]

        if not existing_v:
            # nothing to filter on — save file as-is (or mark it)
            out_file = output_dir / file.name
            df.to_csv(out_file, index=False)
            return file.name, "no_v_columns_found", before_count, 0

        # Convert empty/whitespace strings to NA and coerce to numeric (non-numeric -> NaN)
        df[existing_v] = df[existing_v].replace(r'^\s*$', pd.NA, regex=True)
        df[existing_v] = df[existing_v].apply(pd.to_numeric, errors="coerce")

        # Drop rows with any NaN in the v columns
        df_clean = df.dropna(subset=existing_v)

        after_count = len(df_clean)
        dropped_count = before_count - after_count

        # Save cleaned CSV
        out_file = output_dir / file.name
        df_clean.to_csv(out_file, index=False)

        return file.name, "success", after_count, dropped_count

    except Exception as e:
        return file.name, f"failed: {e}", 0, 0


if __name__ == "__main__":
    files = list(input_dir.glob("*.csv"))
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fix_csv, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fixing CSVs", unit="file"):
            results.append(future.result())

    # Print summary and write a report CSV
    report = []
    for fname, status, kept, dropped in results:
        if status == "success":
            print(f"✅ {fname} | Rows kept: {kept} | Rows dropped: {dropped}")
        elif status == "no_v_columns_found":
            print(f"⚠️ {fname} | No v_1..v_12 columns found — saved unchanged ({kept} rows)")
        else:
            print(f"❌ {fname}: {status}")
        report.append({"filename": fname, "status": status, "rows_kept": kept, "rows_dropped": dropped})

    report_df = pd.DataFrame(report)
    report_df.to_csv(output_dir / "fix_report.csv", index=False)
    print("🎉 All files processed. Report saved to", str(output_dir / "fix_report.csv"))