#!/usr/bin/env python3
"""
Filter out control questions from experiment_b data.
Control questions are identified by having null/empty SOURCE columns.
"""

import pandas as pd
from pathlib import Path


def clean_question_text(question: str) -> str:
    """
    Clean question text by removing appended answer text.
    Some questions have answers appended after '\nExplanation A'.
    """
    if pd.isna(question):
        return question
    question_str = str(question)
    # Split by '\nExplanation A' and keep only the question part
    if '\nExplanation A' in question_str:
        question_str = question_str.split('\nExplanation A')[0]
    return question_str.strip()


def filter_control_questions(input_path: Path, output_path: Path = None, experiment_id: int = None) -> pd.DataFrame:
    """
    Filter out control questions (rows with null sources) from experiment data.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Optional path to save filtered data. If None, uses input_path with '_filtered' suffix.
        experiment_id: Optional experiment identifier (1 or 2) to add as a column
    
    Returns:
        Filtered DataFrame with only real questions (non-control)
    """
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Clean question text (remove appended answers)
    if 'question' in df.columns:
        df['question'] = df['question'].apply(clean_question_text)
    
    # Count original rows
    original_count = len(df)
    
    # Filter out rows where both SOURCE columns are null/empty
    # Control questions have empty source columns
    source_cols = ['explanation_a_SOURCE', 'explanation_b_SOURCE']
    
    # Check which source columns exist in the dataframe
    existing_source_cols = [col for col in source_cols if col in df.columns]
    
    if not existing_source_cols:
        print(f"Warning: No SOURCE columns found in {input_path}")
        return df
    
    # A row is a control question if ALL source columns are null/empty
    # Keep rows where at least one source column has a value
    mask = df[existing_source_cols].notna().any(axis=1) & (df[existing_source_cols] != '').any(axis=1)
    df_filtered = df[mask].copy()
    
    # Add experiment_id column if provided
    if experiment_id is not None:
        df_filtered['experiment'] = experiment_id
    
    # Count filtered rows
    filtered_count = len(df_filtered)
    control_count = original_count - filtered_count
    
    print(f"File: {input_path.name}")
    print(f"  Original rows: {original_count}")
    print(f"  Control questions removed: {control_count}")
    print(f"  Remaining rows: {filtered_count}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    
    # Save filtered data
    df_filtered.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df_filtered


def merge_datasets(dataframes: list[pd.DataFrame], output_path: Path) -> pd.DataFrame:
    """
    Merge multiple dataframes into one combined dataset.
    
    Args:
        dataframes: List of DataFrames to merge
        output_path: Path to save the merged dataset
    
    Returns:
        Merged DataFrame
    """
    # Concatenate all dataframes
    df_merged = pd.concat(dataframes, ignore_index=True)
    
    # Save merged data
    df_merged.to_csv(output_path, index=False)
    
    print(f"Merged dataset:")
    print(f"  Total rows: {len(df_merged)}")
    print(f"  Saved to: {output_path}")
    
    return df_merged


def main():
    # Define paths
    data_dir = Path(__file__).parent / "data"
    
    # Process all CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))
    
    # Exclude already filtered files, merged files, and pivoted files
    csv_files = [f for f in csv_files if "_filtered" not in f.name 
                 and "_merged" not in f.name 
                 and "questions_by_model" not in f.name]
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Processing {len(csv_files)} file(s)...\n")
    
    # Collect filtered dataframes for merging
    filtered_dfs = []
    
    # Map filenames to experiment IDs
    experiment_ids = {
        'first_exp.csv': 1,
        'sec_exp.csv': 2,
    }
    
    for csv_file in csv_files:
        # Get experiment ID from filename, default to index+1 if not in map
        exp_id = experiment_ids.get(csv_file.name, csv_files.index(csv_file) + 1)
        df_filtered = filter_control_questions(csv_file, experiment_id=exp_id)
        filtered_dfs.append(df_filtered)
        print()
    
    # Merge all filtered datasets into one
    if len(filtered_dfs) > 1:
        print("-" * 50)
        merged_output = data_dir / "experiment_b_merged.csv"
        merge_datasets(filtered_dfs, merged_output)
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()

