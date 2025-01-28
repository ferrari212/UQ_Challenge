import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Example usage
if __name__ == "__main__":

    # Generate random data for demonstration
    N = 6
    num_time_steps = 10
    num_features = N
    num_samples = 3

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

    # Plot Y in function of X
    for j in range(Y_out.shape[1]): # for each feature

        plt.figure(figsize=(10, 6))

        time_steps = range(Y_out.shape[0])

        for k in range(Y_out.shape[2]): # for each sample

            plt.plot(time_steps, Y_out[:, j, k])

        plt.title(f'Feature {j+1} over Time Stamp; each line is a specific timestamp')
        plt.xlabel('Time Stamp [i]')
        plt.ylabel('Y')
        plt.legend()
        plt.show()