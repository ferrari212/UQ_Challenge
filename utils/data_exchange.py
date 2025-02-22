import pandas as pd 

def read_output_file(path:str)->pd.DataFrame:
    """
    Read a CSV file with simulation output data into a pandas DataFrame.
    
    Parameters:
    path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The DataFrame containing the simulation output data.
    """
    df = pd.read_csv(path, header=None)
    print(f'Simulation output data loaded from {path}\n')

    return df

def extract_and_remove_sample_indices(df: pd.DataFrame) -> (pd.DataFrame, int):
    """
    Extract unique sample indices from the DataFrame and remove the sample index column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the simulation output data.
    
    Returns:
    pd.DataFrame: The DataFrame with the sample index column removed.
    int: The number of unique sample indices.
    """
    sample_indices = df[6].unique()  # Extract unique sample indices
    num_samples = len(sample_indices)
    df = df.drop(columns=[6])  # Drop the sample index column
    
    return df, num_samples
