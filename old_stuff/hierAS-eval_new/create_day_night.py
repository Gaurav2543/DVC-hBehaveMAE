import numpy as np
import pandas as pd

def create_day_night_pattern(repeats: int) -> np.ndarray:
    """Create an array with a custom pattern repeated a specified number of times.
    """
    # Define the pattern
    pattern = np.concatenate([
        np.ones(420, dtype=int),
        np.zeros(720, dtype=int),
        np.ones(300, dtype=int)
    ])
    
    # Repeat the pattern
    pattern_array = np.tile(pattern, repeats)
    
    return pattern_array

def count_lines_in_csv(file_path: str) -> int:
    """Count the number of lines in a CSV file where the 'sets' column has a value of 1, excluding rows with NA."""
    df = pd.read_csv(file_path, usecols=['sets'])
    df = df.dropna(subset=['sets'])
    count = df[df['sets'] == 1].shape[0]
    del df
                
    return count

# Example usage
csv_file_path = "dvc-data/Imputed_single_cage/summary_table_imputed_with_sets_sub.csv"
ndays = count_lines_in_csv(csv_file_path)

day_night_array = create_day_night_pattern(ndays)  

def extract_strain_for_testing(file_path: str) -> np.ndarray:
    """Extract the 'strain' values for rows where the 'sets' column is 1 
    and repeat each strain value 1440 times."""
    
    df = pd.read_csv(file_path, usecols=['sets', 'strain_id'])
    df = df.dropna(subset=['sets'])  # Exclude rows with NA in 'sets' column
    
    # Extract strain values where sets == 1
    strain_values = df[df['sets'] == 1]['strain_id'].values

    # Repeat each strain value 1440 times
    repeated_strain_values = np.repeat(strain_values, 1440)
    
    del df
    return repeated_strain_values

strain_array = extract_strain_for_testing(csv_file_path)
# Example vocabulary
vocabulary_dn = "Day_night"
vocabulary_s  = "Strain"
# Example task types
task_type = "Discrete"

# Create the label object
labels = {
    "label_array": [day_night_array, strain_array],  # Wrap in a list to match expected format
    "vocabulary": [vocabulary_dn, vocabulary_s],
    "task_type": [task_type, task_type]
}


# Save the label object to a .npy file
np.save("dvc-data/day_night_strain_sub.npy", labels)
