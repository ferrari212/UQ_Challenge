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

def return_response(output_file_path: str):
    """
    Loads simulation output data from a CSV file, processes it, and returns it in a reshaped format.
    Args:
        output_file_path (str): The file path to the CSV file containing the simulation output data.
    Returns:
        numpy.ndarray: A 3D NumPy array of shape (num_time_steps, num_features, num_samples) containing the processed simulation output data.
    """
    # Step 1: Load the output data
    # Load the output data from the CSV file into a pandas DataFrame
    df = pd.read_csv(output_file_path, header=None)
    print(f'Simulation output data loaded from {output_file_path}')

    # Step 2: Extract unique sample indices (number "N" if using batch input) and remove that column (Column 7 contains sample indices)
    df, num_samples = extract_and_remove_sample_indices(df)

    # Step 3: Reshape the Output Data (6 features, 60 timesteps per simulation)
    # Convert DataFrame to a NumPy array and reshape/transpose to the desired 3D shape (num_time_steps x num features x num_samples)
    Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)

    return Y_out