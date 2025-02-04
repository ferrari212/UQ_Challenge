# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 07:08:37 2025

@author: felip
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Example usage
if __name__ == "__main__":

    # Make the object for the linear spaces
    array_data_name = [
            "aleatory_one", 
            "aleatory_two", 
            "epistemic_one", 
            "epistemic_two", 
            "controller_one", 
            "controller_two", 
            "controller_three"
        ]

    for data_name in array_data_name:



        # Step 1: Load the output data
        # Load the output data from the CSV file into a pandas DataFrame
        output_file_path = f'../output_data/{data_name}/Y_multiple_{data_name}.csv'
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

        print("\nNOTE: First timestamp notation is always relatively good for the first timestamp:")
        print(np.corrcoef(Y_out[0, :, :]))

        # Take the average correlation for each time stamp across all features
        correlation_feature_time_stamp = np.zeros((num_time_steps, num_features, num_features))
        average_correlation = np.zeros((num_features, num_features))

        for i in range(num_time_steps):
            correlation_feature_time_stamp[i, :] = np.corrcoef(Y_out[i, :, :])
        
        average_correlation = np.mean(correlation_feature_time_stamp, axis=0)

        fig, axs = plt.subplots(3, 2, figsize=(24, 18), sharex=True)
        data_title = data_name.replace("_", " ")
        fig.suptitle(f'Plot for all the timestamps across the samples, variation on the {data_title}', y=0.95, fontweight='bold')  # Adjust the y position of the title
        
        # Create a multiplot for each feature variation across time steps
        for i in range(num_time_steps):  # for each time step
            for j in range(num_features):  # for each feature
                row = j % 3
                col = j // 3

                axs[row, col].text(0.60, 0.95, f'Avg Correlation: {average_correlation[j, :].round(2)}', 
                                horizontalalignment='right', 
                                verticalalignment='center', 
                                transform=axs[row, col].transAxes,
                                fontsize=18)
                
                axs[row, col].plot(range(num_samples), Y_out[i, j, :])
                axs[row, col].set_ylabel(f'Feature: {j+1}')
                
        
        
                
        output_plot_path = f'../output_data/{data_name}/correlation_plot_{data_name}.png'
        plt.xlabel('Sample Index')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
        plt.savefig(output_plot_path)
        plt.close()
        print(f'Plot saved to {output_plot_path}')
        
        
        correlation_feature_time_stamp[:, :, 0]
        # Plot the correlation feature across the time steps
        plt.figure(figsize=(18, 12))

        axes = plt.axes()
        axes.set_ylim([-1.05, 1.05])
        
        for feature in range(num_features):
            plt.plot(range(num_time_steps), correlation_feature_time_stamp[:, feature, 0], label=f'Feature {feature + 1}')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Correlation with Feature 1')
        plt.title('Correlation of Each Feature with Feature 1 Across Time Steps')
        plt.legend()
        plt.grid(True)
        
        output_correlation_plot_path = f'../output_data/{data_name}/correlation_feature_plot_{data_name}.png'
        plt.savefig(output_correlation_plot_path)
        plt.close()
        print(f'Correlation feature plot saved to {output_correlation_plot_path}')

    
    
