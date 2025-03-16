# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:18:22 2025

@author: felipe

Generate histogram from results
"""



# Importing sys for importing the utils and visualizations_functions folders
import sys
sys.path.append('../utils')
sys.path.append('../visualizations_functions')

import numpy as np

import matplotlib.pyplot as plt
from data_exchange import read_output_file, extract_and_remove_sample_indices
from plot_shapes import plot_3d_shapes

from scipy.stats import truncnorm

import pandas as pd
import time
import os

import seaborn as sns

# Example usage
if __name__ == "__main__":

    # Make the object for the linear spaces
    array_data_name = [
            "aleatory_one", 
            # "aleatory_two", 
            # "epistemic_one", 
            # "epistemic_two", 
            # "epistemic_three", 
            # "controller_one", 
            # "controller_two", 
            # "controller_three",
            # "seed_variation"
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
            
        df_arr = []
            
        for i in range(num_time_steps):
            
            var_value = np.linspace(0, 1, num_samples)
            arr = Y_out[i, 5, :]

            df = pd.DataFrame(arr, columns=["Value"])
            df["Variable Value"] = var_value
            df["Variable Name"] = data_name

            # create quartiles for the value
            df["Quartile"] = pd.qcut(var_value, q=4, labels=["Q1", "Q2", "Q3", "Q4"])

            df_arr.append(df)

            # Show timer for each iteration
            print(f"Value {i} processed")

        df_final = pd.concat(df_arr)
        df_final.reset_index(inplace=True)
        
        # Define bin edges manually
        bin_edges = np.linspace(df_final["Value"].min(), df_final["Value"].max(), num=6)  # 5 bins
        
        # sns.histplot(data=df_final, x="Value", bins=bin_edges, kde=True)
        sns.histplot(data=df_final, x="Value", hue="Quartile", element="poly")
        

        plt.title(f"Histogram of the array for {data_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.show()
        
        sns.histplot(data=df_final, x="Value")
        plt.show()
        
        lim_inf = 0.0
        lim_sup = 1.0
        
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(truncnorm.ppf(0.01, lim_inf, lim_sup),
                truncnorm.ppf(0.99, lim_inf, lim_sup), 100)
        ax.plot(x, truncnorm.pdf(x, lim_inf, lim_sup),
               'r-', lw=5, alpha=0.6, label='truncnorm pdf')