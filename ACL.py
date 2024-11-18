#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm

# Constants for the FM signal
fs = 10e6      # Sampling rate (Hz), 10 MHz for high resolution
fc = 100e6     # Carrier frequency (Hz), 100 MHz (FM radio band)
fm = 1e3       # Modulating frequency (Hz), audio signal frequency (1 kHz)
delta_f = 75e3 # Frequency deviation (Hz), typical for FM radio
duration = 1   # Duration of the signal in seconds

# True parameter values used to generate the FM signal (for comparison)
d_true = 1.0   # True scaling factor for the sine wave
e_true = 2 * np.pi * fm  # True frequency of the sine wave (modulating frequency)
f_true = 0.0   # True phase offset

# Time vector for the FM signal
t = np.arange(0, duration, 1/fs)

# Modulating signal (audio signal, e.g., a sine wave)
audio_signal = np.sin(2 * np.pi * fm * t)

# FM Modulated signal
fm_signal = np.cos(2 * np.pi * fc * t + delta_f * np.sin(2 * np.pi * fm * t))

# --- MCMC Section ---

# Define the model for the data
def model(x, d, e, f):
    return d * np.sin(e * x + f)

# Define the log-likelihood function (negative log of likelihood)
def log_likelihood(params, x, data, noise_std=3):
    d, e, f = params
    model_vals = model(x, d, e, f)
    return -0.5 * np.sum((data - model_vals)**2 / noise_std**2)

# Proposal function to generate new parameter values for the MCMC walk
def proposal(current_position):
    return current_position + np.random.normal(0, 0.5, size=current_position.shape)  # Small random walk step

# MCMC function for multiple walkers
def mcmc_walkers(n_walkers, n_iterations, x, data):
    walkers = np.zeros((n_walkers, 3, n_iterations))  # Array to store d, e, f for each walker
    current_positions = np.random.uniform(1, 10, size=(n_walkers, 3))  # Initialize d, e, f randomly

    # Initial likelihoods
    log_likelihoods = np.array([log_likelihood(pos, x, data) for pos in current_positions])

    for i in tqdm.tqdm(range(n_iterations)):
        for j in range(n_walkers):
            # Propose new positions for the walker
            new_position = proposal(current_positions[j])

            # Calculate the new log-likelihood
            new_log_likelihood = log_likelihood(new_position, x, data)

            # Calculate acceptance probability
            accept_prob = min(1, np.exp(new_log_likelihood - log_likelihoods[j]))

            # Accept or reject the new position
            if np.random.rand() < accept_prob:
                current_positions[j] = new_position
                log_likelihoods[j] = new_log_likelihood

            # Store the current position of the walker
            walkers[j, :, i] = current_positions[j]

    return walkers

# --- Simulation data for MCMC ---
# x values from the FM signal (time vector)
x = t

# Run MCMC for 500 walkers and 10000 iterations (use the FM signal as the data)
n_walkers = 10
n_iterations = 3000
walkers = mcmc_walkers(n_walkers, n_iterations, x, fm_signal)

# --- Calculate the Correlation Length ---

# Function to compute the autocorrelation of a walker’s path
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]  # Keep only the non-negative lags
    result /= result[0]  # Normalize so that the first element is 1
    return result

# Calculate the autocorrelation for each parameter (d, e, f)
corr_d = np.array([autocorr(walkers[i, 0, :]) for i in range(n_walkers)])  # For parameter d
corr_e = np.array([autocorr(walkers[i, 1, :]) for i in range(n_walkers)])  # For parameter e
corr_f = np.array([autocorr(walkers[i, 2, :]) for i in range(n_walkers)])  # For parameter f

# Average autocorrelation over all walkers for each parameter
avg_corr_d = np.mean(corr_d, axis=0)
avg_corr_e = np.mean(corr_e, axis=0)
avg_corr_f = np.mean(corr_f, axis=0)

# --- Estimation of Correlation Length ---

# Function to estimate correlation length by finding where the autocorrelation falls below 1/e
def estimate_corr_length(corr):
    return np.where(corr < np.exp(-1))[0][0]

# Estimate correlation lengths for d, e, f
corr_length_d = estimate_corr_length(avg_corr_d)
corr_length_e = estimate_corr_length(avg_corr_e)
corr_length_f = estimate_corr_length(avg_corr_f)

# Print the estimated correlation lengths
print(f"Correlation length for parameter d: {corr_length_d} iterations")
print(f"Correlation length for parameter e: {corr_length_e} iterations")
print(f"Correlation length for parameter f: {corr_length_f} iterations")

# --- 3D Plot of Autocorrelation ---

# Create a 3D plot to display the autocorrelation for d, e, f
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate lag values (x-axis)
lags = np.arange(len(avg_corr_d))

# Plot the autocorrelation for each parameter (d, e, f) in 3D
ax.plot(lags, avg_corr_d, zs=0, zdir='y', label="Autocorrelation (d)", color='r')
ax.plot(lags, avg_corr_e, zs=1, zdir='y', label="Autocorrelation (e)", color='g')
ax.plot(lags, avg_corr_f, zs=2, zdir='y', label="Autocorrelation (f)", color='b')

# Mark the correlation length for each parameter with a vertical line
ax.scatter(corr_length_d, 0, avg_corr_d[corr_length_d], color='r', s=100, label=f"Corr Length (d) = {corr_length_d}")
ax.scatter(corr_length_e, 1, avg_corr_e[corr_length_e], color='g', s=100, label=f"Corr Length (e) = {corr_length_e}")
ax.scatter(corr_length_f, 2, avg_corr_f[corr_length_f], color='b', s=100, label=f"Corr Length (f) = {corr_length_f}")

# Labels and title
ax.set_xlabel('Lag')
ax.set_ylabel('Parameter')
ax.set_zlabel('Autocorrelation')
ax.set_title('3D Autocorrelation Plot for Parameters d, e, f')

# Legend
ax.legend()

# Show plot
plt.tight_layout()
plt.savefig('ACL.png')
