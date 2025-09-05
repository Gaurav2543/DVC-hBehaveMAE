import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tempfile

def load_dvc_data(folder_path, cage_list):
    data_concat = {}
    for cage, days in tqdm(cage_list.items(), desc="Loading & imputing DVC data"):
        path = os.path.join(folder_path, f"{cage}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")

        # --- AUTO-CLEAN CSV LINES (strip leading/trailing quotes) ---
        with open(path, "r") as fin, tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as fout:
            for line in fin:
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                fout.write(line + "\n")
            clean_path = fout.name

        # Now load the cleaned file as normal
        df = pd.read_csv(clean_path)

        print(f"Loaded {len(df)} rows from {path} for cage {cage}")
        # print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")
        # print(f"Head:\n{df.head()}")

        # parse timestamp & filter
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"].dt.strftime('%Y-%m-%d').isin(days)].copy()
        data_concat[cage] = df

        # Remove temp file after use (cleanup)
        try:
            os.remove(clean_path)
        except Exception as e:
            print(f"Warning: could not remove temp file {clean_path}: {e}")

    return data_concat