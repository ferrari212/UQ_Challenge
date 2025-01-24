%% Section 1: Creating the Input and Calling the Executable

close all; clear;

% Step 1: Define Input Parameters
X_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 42];

% Step 2: Create a Batch of N Inputs (as an example, just copies of X_input)
N = 3;
X_input_batch = repmat(X_input, N, 1);

% Step 3: Write Input Data to a File
writematrix(X_input_batch, 'input.txt', 'Delimiter', ',');

% Step 4: Call the Executable to Run the Simulation with this file as input
disp('Simulation executable has been called.');
system('model.exe input.txt');



%% Section 2: Loading and Post-processing the Results

% Step 1: Load the Output Data
Y_output = readmatrix('Y_out.csv');
disp('Simulation output data loaded.');


% Step 2: Extract Unique Sample Indices and Remove the Sample Index Column (Column 7)
sample_indices = unique(Y_output(:, 7));  % Extract unique sample indices from column 7
num_samples = numel(sample_indices);     % Determine the number of unique samples
Y_output(:, 7) = [];                     % Remove the sample index column from the output matrix


% Step 3: Reshape the Output Data (6 features, 60 timesteps per simulation)
% into the desired 3D shape (num_time_steps x num features x num_samples)
Y_out_reshaped = reshape(Y_output', 6, 60, num_samples);
Y_out_reshaped = permute(Y_out_reshaped, [2, 1, 3]);

% Step 4: As an example, display the First Time Step for All Features of Sample 3
disp('Example: First timestep for all features of sample 3:');
disp(Y_out_reshaped(1, :, 3));
