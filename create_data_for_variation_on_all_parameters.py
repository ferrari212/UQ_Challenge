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
X_input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 42.0])  # Shape will be (9,)
# # Alternatively, the model also supports a batch of N input vectors (as an example, "N" copies of X_input)
# N = 1001
N = 101
X_input_batch_original = np.tile(X_input, (N, 1))  # Shape will be (N, 9)

# Make the object for the linear spaces
array_data_name = [
        "aleatory_one", 
        "aleatory_two", 
        "epistemic_one", 
        "epistemic_two", 
        "controller_one", 
        "controller_two", 
        "controller_three",
        "seed_variation"
    ]

# Add random noise to the input data
X_Var = np.linspace(0, 1, N)
X_Var_seed = np.linspace(0, 100, N)

for i, data_name in enumerate(array_data_name):
    X_input_batch = X_input_batch_original.copy()

    X_input_batch[:, i] = X_Var if i != 7 else X_Var_seed

    # Step 2: Write the input data to a local File (input.txt)
    input_file_path = f'./input_data/input_multiple_{data_name}.txt'
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
    output_folder = f"./output_data/{data_name}"  # Target folder for renamed file
    renamed_file = f"Y_multiple_{data_name}.csv"

    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created.")

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
