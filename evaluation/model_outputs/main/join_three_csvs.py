#!/usr/bin/env python3
"""
Script to join three CSV files by row order.
"""

import pandas as pd
import sys

def join_three_csvs_by_row_order(file1_path, file2_path, file3_path, output_path):
    """
    Join three CSV files by row order.
    
    Args:
        file1_path: Path to first CSV file (llama3.csv)
        file2_path: Path to second CSV file (scicomma_joined.csv)
        file3_path: Path to third CSV file (scicomma_dpo.csv)
        output_path: Path for output file
    """
    
    # Read the CSV files
    print(f"Reading {file1_path}...")
    df1 = pd.read_csv(file1_path)
    
    print(f"Reading {file2_path}...")  
    df2 = pd.read_csv(file2_path)
    
    print(f"Reading {file3_path}...")
    df3 = pd.read_csv(file3_path)
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    print(f"File 3 shape: {df3.shape}")
    
    # Reset indices to ensure proper row-by-row joining
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)
    
    # Since df2 (scicomma_joined) already has the Question column, 
    # we only need to add the columns from df1 and df3, not their Question columns again
    
    # Extract only the answer columns from df1 and df3 (skip the Question column)
    df1_answers = df1.iloc[:, 1:]  # Skip first column (Question)
    df3_answers = df3.iloc[:, 1:]  # Skip first column (Question)
    
    # Join all dataframes by concatenating along columns (axis=1)
    # Order: df1 (llama3), df2 (scicomma_joined), df3 (scicomma_dpo)
    joined_df = pd.concat([df1, df2.iloc[:, 1:], df3_answers], axis=1)
    
    print(f"Joined data shape: {joined_df.shape}")
    print(f"Columns: {list(joined_df.columns)}")
    
    # Save the result
    print(f"Saving to {output_path}...")
    joined_df.to_csv(output_path, index=False)
    
    print("Join completed successfully!")
    
    # Show first few rows
    print("\nFirst 3 rows of joined data:")
    print(joined_df.head(3))
    
    return joined_df

if __name__ == "__main__":
    # File paths in the order specified by user
    file1_path = "llama3.csv"          # First
    file2_path = "scicomma_joined.csv"  # Second  
    file3_path = "scicomma_dpo.csv"     # Third
    output_path = "all_models_joined.csv"
    
    # Join the files
    result = join_three_csvs_by_row_order(file1_path, file2_path, file3_path, output_path)
