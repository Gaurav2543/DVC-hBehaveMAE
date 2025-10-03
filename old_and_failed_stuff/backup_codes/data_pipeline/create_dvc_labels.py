import numpy as np
import pandas as pd

def create_day_night_pattern(repeats: int) -> np.ndarray:
    """Create an array with a custom day/night pattern repeated a specified number of times."""
    pattern = np.concatenate([
        np.ones(420, dtype=int),
        np.zeros(720, dtype=int),
        np.ones(300, dtype=int)
    ])
    return np.tile(pattern, repeats)

def count_lines_in_csv(file_path: str) -> int:
    """Count the number of rows in a CSV where 'sets' == 1 (excluding NAs)."""
    df = pd.read_csv(file_path, usecols=['sets'])
    df = df.dropna(subset=['sets'])
    count = df[df['sets'] == 1].shape[0]
    del df
    return count

def extract_and_repeat_column(file_path: str, column_name: str, repeat_times: int = 1440) -> np.ndarray:
    """Extract values of a given column where 'sets' == 1 and repeat each value a fixed number of times."""
    df = pd.read_csv(file_path, usecols=['sets', column_name])
    df = df.dropna(subset=['sets'])
    values = df[df['sets'] == 1][column_name].values
    repeated_values = np.repeat(values, repeat_times)
    del df
    return repeated_values

# File path to the CSV
csv_file_path = "/scratch/bhole/data/summary_table_imputed_with_sets_sub_20_CompleteAge_Strains.csv"

# Count valid 'sets' == 1 rows
ndays = count_lines_in_csv(csv_file_path)

# Define column metadata
columns = ["strain", "Age_Days", "cage"]
vocabularies = ["Strain", "Age_Days", "Cage"]
task_types = ["Discrete", "Continuous", "Discrete"]

# Create all label arrays
label_arrays = [create_day_night_pattern(ndays)] + [extract_and_repeat_column(csv_file_path, col) for col in columns]
vocabularies = ["Day_night"] + vocabularies
task_types = ["Discrete"] + task_types

# Construct label dictionary
labels = {
    "label_array": label_arrays,
    "vocabulary": vocabularies,
    "task_type": task_types
}

print("Arrays: \n", labels)
print("Label arrays created successfully.")

# Save to .npy file
np.save("test-outputs/arrays_sub20_with_cage_complete_correct_strains.npy", labels)
