import pandas as pd
import os

def split_large_csv(file_path, max_size_mb=24):
    chunk_size =  5 **  6  # Size of each chunk in bytes (1MB)
    max_size = max_size_mb * chunk_size  # Max size in bytes
    output_file_index =  1
    output_file_path = f"{file_path.rsplit('.',  1)[0]}_part{output_file_index}.csv"

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk.to_csv(output_file_path, index=False, mode='a')  # Append mode
        if os.path.getsize(output_file_path) > max_size:  # Check if file size exceeds limit
            output_file_index +=  1
            output_file_path = f"{file_path.rsplit('.',  1)[0]}_part{output_file_index}.csv"
        else:
            # Ensure the file is closed after writing to avoid memory issues
            with open(output_file_path, 'w') as f:
                chunk.to_csv(f, index=False)

# Usage
split_large_csv('D:\\CodingAI\\HW2\\Lab2\\train.csv')