import subprocess
import numpy as np
import pandas as pd
import os
from pathlib import Path

## Section 1: Creating the input and calling the executable

# # Step 1: Define input parameters to the model
# # Define the input vector X_input with the required values
# X_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 42.0])
# # Alternatively, the model also supports a batch of N input vectors (as an example, "N" copies of X_input)
# N = 3
# X_input_batch = np.tile(X_input, (N, 1))  # Shape will be (N, 9)


# Step 2: Write the input data to a local File (input.txt)
# input_file_path = './input_two.txt'
input_file_path = './input_two.txt'
# np.savetxt(input_file_path, X_input_batch, delimiter=',')
# print(f'Input data written to {input_file_path}')


# Step 3: Run the executable and capture its output
# !!!!! Important: The executable must have the local_model_windows.exe name and be in the same folder as this script !!!!!
# Find the file in the challenge repository folder
exe_path = os.path.abspath(Path(__file__).parent / "local_model_windows.exe")  # Path to the executable in the current folder
command = [exe_path, input_file_path]

print('Simulation executable has been called.')
result = subprocess.run(command, capture_output=True, text=True)


# Print the output from the executable
print(result.stdout)

# Section 2: Loading and post-processing the output

# Step 1: Load the output data
# Load the output data from the CSV file into a pandas DataFrame
output_file_path = 'Y_out.csv'
df = pd.read_csv(output_file_path, header=None)
print(f'Simulation output data loaded from {output_file_path}')

# Step 2: Extract unique sample indices (number "N" if using batch input) and remove that column (Column 7 contains sample indices)
sample_indices = df[6].unique()  # Extract unique sample indices
num_samples = len(sample_indices)
df = df.drop(columns=[6])  # Drop the sample index column

# Step 3: Reshape the Output Data (6 features, 60 timesteps per simulation)
# Convert DataFrame to a NumPy array and reshape/transpose to the desired 3D shape (num_time_steps x num features x num_samples)
Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)

# Step 4: As an example, Print the First Time Step for All 6 Output Features of Sample 3
print('Output values for the first timestep for all 6 features of sample 3:')
print(Y_out[0, :, 2])



























