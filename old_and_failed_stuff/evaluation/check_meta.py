import os
import pandas as pd

def main(path):
    meta_data = pd.read_csv(path, sep = "\t")
    print(meta_data.columns)
    print(meta_data.head())
    print(meta_data["event"].unique())


if __name__ == "__main__":
    metadatapath = r"C:\Users\andyb\OneDrive\EPFL\PhD\DVC_project\DVC_data\event_periods_meta_parsed.tsv"
    main(metadatapath)