import os 
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_dvc_data(folder_path, cage_list):
    data_concat = {}
    for cage, days in tqdm(cage_list.items(), desc="Loading DVC data"):
        path = os.path.join(folder_path, f"{cage}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")

        df = pd.read_csv(path, low_memory=False)
        # print(f"Loaded {len(df)} rows from {path} for cage {cage}")
        # print(f"Shape of the dataframe: {df.shape}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"].dt.strftime('%Y-%m-%d').isin(days)].copy()
        data_concat[cage] = df

    return data_concat
