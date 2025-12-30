import os
import fire
import pandas as pd

def truncate_parquet(input_file, start_row=0, num_rows=None, columns=None):
    """
    Truncate a Parquet file by selecting a specific subset of rows and columns.
    Save the truncated file with a name that includes the row range.
    
    Parameters:
    - input_file: The path to the input Parquet file.
    - start_row: The starting row index to begin truncating from (default is 0).
    - num_rows: The number of rows to include from the starting point (default is None).
    - columns: A list of column names to select from the Parquet file (default is None).
    """
    # Read the Parquet file into a pandas DataFrame, selecting the specified columns if provided
    df = pd.read_parquet(input_file, columns=columns)
    
    # If num_rows is specified, slice the DataFrame to include only the desired rows
    if num_rows is not None:
        end_row = start_row + num_rows
        df = df.iloc[start_row:end_row]
    else:
        # If num_rows is not specified, take all rows from start_row onwards
        df = df.iloc[start_row:]
        end_row = len(df)  # If num_rows is None, set end_row to the length of the remaining data

    # Get the original file's directory and name
    original_dir = os.path.dirname(input_file)
    original_filename = os.path.basename(input_file)

    # Create a new file name including the row range
    output_filename = f"{os.path.splitext(original_filename)[0]}_row_{start_row}-{end_row}{os.path.splitext(original_filename)[1]}"
    
    # Create the full path for the new file
    output_file = os.path.join(original_dir, output_filename)
    
    # Save the truncated DataFrame to the new Parquet file
    df.to_parquet(output_file, index=False)
    
    return output_file


if __name__ == "__main__":
    fire.Fire(truncate_parquet)