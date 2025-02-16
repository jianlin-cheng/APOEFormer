import pandas as pd
import glob
import os

# Set the directory where your CSV files are located
directory_path = 'Users/thongnguyen/Downloads/New'  # Update this path accordingly

# Use glob to get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

for file_path in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Select numeric columns for normalization
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Apply Z-score normalization: (x - mean) / std for each numeric column
    df[numeric_columns] = df[numeric_columns].apply(lambda x: (x - x.mean()) / x.std())
    
    # Save the normalized DataFrame back to the same file
    df.to_csv(file_path, index=False)
    
    print(f"Normalized data saved to: {file_path}")

print("All files have been processed and normalized.")
