import numpy as np

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt

from data_exchange import read_output_file, extract_and_remove_sample_indices

import pandas as pd
import time
import os

# Example usage
if __name__ == "__main__":

    # Make the object for the linear spaces
    array_data_name = [
            # "aleatory_one", 
            # "aleatory_two", 
            # "epistemic_one", 
            # "epistemic_two", 
            # "controller_one", 
            # "controller_two", 
            # "controller_three",
            "seed_variation"
        ]

    for data_name in array_data_name:

        # Step 1: Load the output data
        # Load the output data from the CSV file into a pandas DataFrame
        output_file_path = f'../output_data/{data_name}/Y_multiple_{data_name}.csv'
        df = read_output_file(output_file_path)

        # Step 2: Extract unique sample indices (number "N" if using batch input) and remove that column (Column 7 contains sample indices)
        df, num_samples = extract_and_remove_sample_indices(df)
        
        # Step 3: Reshape the Output Data (6 features, 60 timesteps per simulation)
        # Convert DataFrame to a NumPy array and reshape/transpose to the desired 3D shape (num_time_steps x num features x num_samples)
        Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)
            
        # Create variables for the number of time steps, features, and samples
        num_time_steps = Y_out.shape[0]
        num_features = Y_out.shape[1]
        num_samples = Y_out.shape[2]

        # Create directory for plots if it does not exist
        plot_dir = f'../output_data/{data_name}/plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Animation frames
        frames = []

        # Plot Y in function of X
        for j in range(Y_out.shape[1]): # for each feature

            plt.figure(figsize=(10, 6))

            time_steps = range(Y_out.shape[0])

            for k in range(Y_out.shape[2]): # for each sample

                plt.plot(time_steps, Y_out[:, j, k])

            plt.title(f'Feature {j} line across time')
            plt.xlabel('Time Stamp [i]')
            plt.ylabel('Y')
            plt.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
            
            output_correlation_plot_path = f'{plot_dir}/correlation_feature_{j}_plot_{data_name}_across_time.png'
            plt.savefig(output_correlation_plot_path)
            plt.close()
            end_time = time.time()  # End timing
           