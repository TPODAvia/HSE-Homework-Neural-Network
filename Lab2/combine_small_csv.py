import glob
import pandas as pd
import os

def combine_csv_files(directory, output_file):
    # Find all CSV files in the directory
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    all_files.sort()  # Ensure files are combined in the correct order

    # Read and concatenate each file
    combined_data = pd.concat((pd.read_csv(f) for f in all_files))

    # Write the combined data to a new file
    combined_data.to_csv(output_file, index=False)

# Usage
combine_csv_files('D:\\Coding_AI\\HW2\\Lab2\\train_dataset_combine', 'D:\\Coding_AI\HW2\\Lab2\\train.csv')