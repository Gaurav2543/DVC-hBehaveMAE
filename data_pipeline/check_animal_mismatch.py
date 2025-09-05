import pandas as pd
from collections import defaultdict

# --- CONFIGURATION ---
# Set the path to your summary file here
SUMMARY_FILE_PATH = "/scratch/bhole/dvc_data/smoothed/1440/final_summary_metadata_1440.csv" 
OUTPUT_FILE_PATH = "animal_count_mismatches_1440.txt"

def analyze_animal_counts(summary_df):
    """
    Analyzes a summary DataFrame for animal count inconsistencies.

    Args:
        summary_df (pd.DataFrame): The loaded summary data.
    
    Returns:
        A list of formatted strings describing any mismatches found.
    """
    all_mismatch_reports = []
    
    # Get unique strains, ignoring any potential NaN values
    strains = summary_df['strain'].dropna().unique()
    
    print(f"Analyzing {len(strains)} unique strains...")

    for strain in sorted(strains):
        strain_df = summary_df[summary_df['strain'] == strain]
        cages = strain_df['cage_id'].unique()
        
        strain_mismatches = []
        cage_to_animal_count = {}

        # --- 1. Intra-Cage Consistency Check (within each cage's history) ---
        for cage_id in cages:
            cage_df = strain_df[strain_df['cage_id'] == cage_id]
            # Find all unique animal counts recorded for this cage
            unique_counts = cage_df['nb_animals_all'].unique()
            
            if len(unique_counts) > 1:
                # Mismatch found within this cage's history
                report = (f"  [!] Intra-Cage Mismatch in '{cage_id}':"
                          f" Found multiple counts: {sorted(list(unique_counts))}")
                strain_mismatches.append(report)
            
            # Store the first recorded count for the inter-cage check
            if len(unique_counts) > 0:
                cage_to_animal_count[cage_id] = unique_counts[0]

        # --- 2. Inter-Cage Consistency Check (between cages of the same strain) ---
        if len(cages) > 1:
            # Check if all cages within the strain have the same animal count
            unique_strain_counts = set(cage_to_animal_count.values())
            if len(unique_strain_counts) > 1:
                report = (f"  [!] Inter-Cage Mismatch for Strain '{strain}':"
                          f" Cages have different animal counts: {cage_to_animal_count}")
                strain_mismatches.append(report)
        
        # If any mismatches were found for this strain, format the report
        if strain_mismatches:
            header = f"\n{'='*20} Strain: {strain} {'='*20}"
            all_mismatch_reports.append(header)
            all_mismatch_reports.extend(strain_mismatches)
            
    return all_mismatch_reports

def main():
    """
    Main function to run the analysis and save the report.
    """
    print(f"Loading summary file from: {SUMMARY_FILE_PATH}")
    
    try:
        # Load the summary file, ensuring required columns exist
        summary_df = pd.read_csv(SUMMARY_FILE_PATH, low_memory=False)
        required_cols = ['strain', 'cage_id', 'nb_animals_all']
        if not all(col in summary_df.columns for col in required_cols):
            raise ValueError(f"CSV must contain the columns: {required_cols}")

        # Run the analysis
        mismatch_report = analyze_animal_counts(summary_df)
        
        # Save the report to a text file
        with open(OUTPUT_FILE_PATH, 'w') as f:
            if mismatch_report:
                print(f"\n[INFO] Mismatches found. Saving report to: {OUTPUT_FILE_PATH}")
                f.write("Animal Count Mismatch Report\n")
                f.write("="*40 + "\n")
                for line in mismatch_report:
                    f.write(line + "\n")
            else:
                print("\n[SUCCESS] No animal count mismatches found across all strains.")
                f.write("No animal count mismatches found.")
                
    except FileNotFoundError:
        print(f"[ERROR] The file was not found at: {SUMMARY_FILE_PATH}")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()
