# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:05:06 2025

@author: felip
"""

import matplotlib.pyplot as plt

import sys
sys.path.append('../utils')

import numpy as np

from numpy.fft import fft, fftfreq

from data_exchange import read_output_file, extract_and_remove_sample_indices

def reconstruct_signal(A, freq, t, N):
    """
    Reconstructs a real-valued periodic signal from the real and imaginary parts
    of the FFT coefficients, properly scaled by the number of samples N.

    Parameters:
        A (numpy.ndarray): Real FFT coefficients.
        freq (numpy.ndarray): Frequencies corresponding to the FFT coefficients.
        t (numpy.ndarray): Time points where the reconstructed signal will be evaluated.
        N (int): Number of original data points used in the rFFT.

    Returns:
        numpy.ndarray: Reconstructed signal values at the specified time points.
    """
    # Initialize the reconstructed signal with the DC component (properly scaled)
    y_reconstructed = (A[0].real / N)  # DC term

    # Add the cosine and sine terms
    for k in range(1, len(A)):
        y_reconstructed += 2 * (A[k].real / N) * np.cos(2 * np.pi * freq[k] * t)
        y_reconstructed -= 2 * (A[k].imag / N) * np.sin(2 * np.pi * freq[k] * t) 

    return y_reconstructed

# Example usage
if __name__ == "__main__":
    
    data_name = "controller_one"
    output_file_path = f'../output_data/{data_name}/Y_multiple_{data_name}.csv'
    
    
    df = read_output_file(output_file_path)
    
    df, num_samples = extract_and_remove_sample_indices(df)
    
    Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)
    Y = Y_out[:,3,0] - Y_out[:,3,0].mean() # Taking the third feature, sample 0 averaged by the mean   
    N = len(Y)
    t_original = np.linspace(0, 60, N, endpoint=False)
    
    A = fft(Y)
    freq = fftfreq(N, d=1.0)  # Assuming unit sampling interval
    
    # Create a finer time grid for better visualization
    t_fine = np.linspace(0, N, 5000, endpoint=False)
    
    # Reconstruct the signal over the fine grid
    reconstructed_signal = reconstruct_signal(A, freq, t_fine, N)
    
    # Plot the original points and the reconstructed signal
    plt.figure(figsize=(12, 7))
    plt.plot(t_fine, reconstructed_signal, label='Reconstructed Signal (Real Terms Only)', color='orange')
    plt.scatter(t_original, Y, color='red', s=10, label='Original Measurement Points')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Original Measurement and Reconstructed Periodic Function')
    plt.legend()
    plt.grid(True)
    plt.show()