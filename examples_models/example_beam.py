import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

import time

def structural_deflection_model(W, F, L, c):

    return F * L**3/(48 * c) + 5 *W * L**4/(384 * c)

def return_montecarlo_values(W, W_sigma, F, F_sigma, L, L_sigma, c, n_samples=1000):
    '''
    Inputs:
    W: float, beam weight average
    F: float, force average
    L: float, length average
    c: float, constant equivalent to E*I (modulus of elasticity * moment of inertia)
    '''
    np.random.seed(42)

    W_samples = np.random.normal(W, W_sigma, n_samples)
    F_samples = np.random.normal(F, F_sigma, n_samples)
    L_samples = np.random.normal(L, L_sigma, n_samples)

    deflection_samples = structural_deflection_model(W_samples, F_samples, L_samples, c)

    return deflection_samples

# Example usage
if __name__ == "__main__":

    # start time to run the model
    initial_time = time.time()

    W = 250
    W_sigma = 12.5
    F = 1000
    F_sigma = 75
    L = 36
    L_sigma = 0.25
    c = 1.33 * 10 ** 7

    # Maximun allowable deflection
    threshold = 0.52

    deflections = return_montecarlo_values(W, W_sigma, F, F_sigma, L, L_sigma, c, n_samples=100000)
    print("Total time to run the model: ", time.time() - initial_time)


    df_final = pd.DataFrame({"Value": deflections})

    plt.figure(figsize=[15,7])
    sns.histplot(data=df_final, x="Value", bins=100, label="bellow Threshold", kde=True)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label="Threshold")
    plt.text(threshold, plt.ylim()[1] * 0.95, f'Threshold: {threshold}', color="red", fontsize=12, ha="center", fontweight="bold")

    # plt.hist(deflections, bins = 100, density=True, label='Deflection')
    plt.xlabel('Deflection (in)')
    plt.ylabel('Frequency')
    plt.title('Deflection of a beam')
    plt.legend()
    plt.show()