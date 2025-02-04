import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Example usage
if __name__ == "__main__":

    # Step 1: Load the output data
    # Load the output data from the CSV file into a pandas DataFrame
    output_file_path = '../output_data/seed_variation/Y_multiple_seeds.csv'
    df = pd.read_csv(output_file_path, header=None)
    print(f'Simulation output data loaded from {output_file_path}')

    # Step 2: Extract unique sample indices (number "N" if using batch input) and remove that column (Column 7 contains sample indices)
    sample_indices = df[6].unique()  # Extract unique sample indices
    num_samples = len(sample_indices)
    df = df.drop(columns=[6])  # Drop the sample index column

    # Step 3: Reshape the Output Data (6 features, 60 timesteps per simulation)
    # Convert DataFrame to a NumPy array and reshape/transpose to the desired 3D shape (num_time_steps x num features x num_samples)
    Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)
        
    # Create variables for the number of time steps, features, and samples
    num_time_steps = Y_out.shape[0]
    num_features = Y_out.shape[1]
    num_samples = Y_out.shape[2]

    # Animation frames
    frames = []

    # Create a multiplot for each feature variation across time steps
    for i in range(num_time_steps):  # for each time step
        fig, axs = plt.subplots(3, 2, figsize=(24, 18), sharex=True)
        fig.suptitle(f'Time Stamp Index: {i+1}', y=0.95)  # Adjust the y position of the title
        for j in range(num_features):  # for each feature
            row = j % 3
            col = j // 3
            axs[row, col].plot(range(num_samples), Y_out[i, j, :])
            axs[row, col].set_ylabel(f'Feature: {j+1}')
            
        plt.xlabel('Sample Index')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
        plt.show()
