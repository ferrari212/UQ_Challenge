import subprocess
import shutil
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time

## Section 1: Creating the input and calling the executable

# # Step 1: Define input parameters to the model
# # Define the input vector X_input with the required values
X_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0])  # Shape will be (9,)
# # Alternatively, the model also supports a batch of N input vectors (as an example, "N" copies of X_input)
N = 100
X_input_batch = np.tile(X_input, (N, 1))  # Shape will be (N, 9)

# Make the random seed a variable
np.random.seed(0)

# Add random noise to the input data
random_seeds = np.random.randint(low=0.0, high=1000, size=N)

X_input_batch[:, -1] = random_seeds

# Step 2: Write the input data to a local File (input.txt)
input_file_path = './input_data/input_multiple_random_seeds.txt'
np.savetxt(input_file_path, X_input_batch, delimiter=',')
print(f'Input data written to {input_file_path}')

# Step 3: Run the executable and capture its output
# !!!!! Important: The executable must have the local_model_windows.exe name and be in the same folder as this script !!!!!
# Find the file in the challenge repository folder
exe_path = os.path.abspath(Path(__file__).parent / "local_model_windows.exe")  # Path to the executable in the current folder
command = [exe_path, input_file_path]

start_time = time.time()
print('Simulation executable has been called.')
result = subprocess.run(command, capture_output=True, text=True)

# Check for errors and print the appropriate output
if result.stderr:
    print("\nError occurred during the execution, error message below:\n")
    print(result.stderr)
    raise Exception("\nError occurred during the execution, process terminated.\n")


# Step 4: Define the output folder and the name of the output file
print('Simulation completed successfully.')
output_folder = "./output_data/seed_variation"  # Target folder for renamed file
renamed_file = "Y_multiple_seeds.csv"

# Define paths for renaming and moving
current_output_file = os.path.join(os.getcwd(), "Y_out.csv")
target_file_path = os.path.join(output_folder, renamed_file)

# Check if Y_out.csv exists
if os.path.exists(current_output_file):
    # Rename and move the file
    shutil.move(current_output_file, target_file_path)
    print(f"File 'Y_out.csv' renamed to '{renamed_file}' and moved to '{output_folder}'.")
else:
    print("Error: Y_out.csv file was not found.")


end_time = time.time()

# Compute and print the time span of the process
time_span = end_time - start_time
print(f'Time span of the process: {time_span:.2f} seconds')
