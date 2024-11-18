#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import emcee

# Constants for the FM signal
fs = 10e6      # Sampling rate (Hz)
fc = 100e6     # Carrier frequency (Hz)
fm = 1e3       # Modulating frequency (Hz)
delta_f = 75e3 # Frequency deviation (Hz)
duration = 1   # Duration of the signal in seconds

# True parameter values used to generate the FM signal
d_true = 1.0   # True scaling factor for the sine wave (Amplitude)
e_true = 2 * np.pi * fm  # True frequency of the sine wave (Modulating frequency)
f_true = 0.0   # True phase offset

# Time vector for the FM signal
t = np.arange(0, duration, 1/fs)

# Modulating signal (audio signal, e.g., a sine wave)
audio_signal = np.sin(2 * np.pi * fm * t)

# Function to generate the FM signal based on parameters (d, e, f)
def generate_fm_signal(d, e, f, t):
    """Generate an FM signal based on scaling factor d, modulating frequency e, and phase f."""
    return d * np.cos(2 * np.pi * fc * t + delta_f * np.sin(2 * np.pi * e * t + f))

# Generate the true FM signal using the true parameters
fm_signal = generate_fm_signal(d_true, e_true, f_true, t)

# Function to calculate the likelihood
def log_likelihood(theta, data):
    d, e, f = theta
    # Generate the FM signal based on the parameters
    fm_signal_sim = generate_fm_signal(d, e, f, t)
    # Calculate the likelihood (Gaussian likelihood)
    diff = data - fm_signal_sim
    return -0.5 * np.sum(diff**2)

# Function to calculate the prior (sine-based prior)
def log_prior(theta):
    d, e, f = theta
    # Define a sine prior on the modulating frequency (e) and phase (f), and uniform prior on d
    if 0 < e < 2 * np.pi and 0 <= f <= np.pi:  # e in range [0, 2pi] and f in [0, pi]
        prior_e = np.sin(e)  # Prior for e
        prior_f = np.sin(f)  # Prior for f
        prior_d = 1  # Uniform prior for d
        return np.log(prior_e * prior_f * prior_d)  # Prior for all parameters
    return -np.inf  # If outside valid range, return a very low probability

# Log probability combining the likelihood and prior
def log_probability(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)

# Function to calculate correlation length (just an example based on the FM signal)
def correlation_length(walkers_positions):
    """Calculate the correlation length by measuring the average distance between walkers."""
    distances = np.linalg.norm(walkers_positions[:, np.newaxis] - walkers_positions, axis=-1)
    return np.mean(distances)

# MCMC simulation function using emcee
def run_mcmc(n_walkers, n_iterations, data):
    # Initialize walkers with a larger spread
    pos = np.random.randn(n_walkers, 3) * 10  # Scale the random numbers to spread them out

    # Set up the sampler using emcee
    ndim = 3  # Number of parameters to estimate (d, e, f)
    nsteps = n_iterations  # Number of iterations to run the MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=[data])

    # Run the sampler
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Store the walker positions (all positions for all iterations)
    all_walkers_positions = sampler.get_chain()

    # Calculate the correlation lengths at each step
    correlation_lengths = []
    for i in range(nsteps):
        # Flatten walker positions at iteration i (all walkers)
        walkers_pos_at_iter = all_walkers_positions[i, :, :].reshape(-1, 3)
        correlation_lengths.append(correlation_length(walkers_pos_at_iter))

    return all_walkers_positions, correlation_lengths

# User prompt for number of walkers and iterations
n_walkers = int(input("Enter the number of walkers: "))
n_iterations = int(input("Enter the number of iterations: "))

# Run MCMC simulation
all_walkers_positions, correlation_lengths = run_mcmc(n_walkers, n_iterations, fm_signal)

# Plot the correlation length vs. iterations
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations), correlation_lengths, label='Correlation Length')
plt.xlabel('Iterations')
plt.ylabel('Correlation Length')
plt.title(f"Correlation Length vs. Iterations ({n_walkers} walkers, {n_iterations} iterations)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations_correlation.png")

# 3D Plot of all walkers' positions throughout the iterations
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot all walkers' positions over iterations with grey lines
for i in range(n_walkers):
    # Extract the path of walker i across all iterations
    path = all_walkers_positions[:, i, :]
    
    # Plot the trajectory of each walker (grey line connecting all positions)
    ax.plot(path[:, 1], path[:, 2], path[:, 0], color='grey', alpha=0.6)  # e, f, d

# Plot the final positions of the walkers in 3D (parameters d, e, f)
final_positions = all_walkers_positions[-1, :, :]
ax.scatter(final_positions[:, 1], final_positions[:, 2], final_positions[:, 0], 
           c='blue', marker='o', label="Final Position")

# Set axis labels
ax.set_xlabel('Modulating Frequency (e)')
ax.set_ylabel('Phase Shift (f)')
ax.set_zlabel('Amplitude (d)')
ax.set_title(f"Walkers' Positions Over Iterations (3D)")

# Plot the true value as a red point (d_true, e_true, f_true)
ax.scatter(d_true, e_true, f_true, c='red', marker='x', label='True Value')

# Show the legend to distinguish final positions and true values
plt.legend()

# Layout adjustments and saving the plot
plt.tight_layout()
plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations_walkers_paths.png")

# --- New Plot: Average of Walkers' Positions every 10 iterations ---

# Create a list to store the averages
avg_positions = []

# Iterate over the iterations, taking the average every 10 iterations
for i in range(0, n_iterations, 10):
    # Extract the walker positions at iteration i (all walkers)
    walkers_at_iter = all_walkers_positions[i, :, :]
    
    # Calculate the average of the positions for each parameter (d, e, f)
    avg_d = np.mean(walkers_at_iter[:, 0])
    avg_e = np.mean(walkers_at_iter[:, 1])
    avg_f = np.mean(walkers_at_iter[:, 2])
    
    # Append the averages as a tuple (d, e, f)
    avg_positions.append((avg_d, avg_e, avg_f))

# Convert avg_positions to a numpy array for easier slicing
avg_positions = np.array(avg_positions)

# Create the plot for the average walker positions over every 10 iterations
plt.figure(figsize=(10, 6))

# Plot the average values for each parameter
plt.plot(range(0, n_iterations, 10), avg_positions[:, 0], label="Average d (Amplitude)", color='blue')
plt.plot(range(0, n_iterations, 10), avg_positions[:, 1], label="Average e (Frequency)", color='green')
plt.plot(range(0, n_iterations, 10), avg_positions[:, 2], label="Average f (Phase)", color='red')

# Adding labels and title
plt.xlabel('Iterations (every 10 steps)')
plt.ylabel('Parameter Value')
plt.title(f"Average Walker Positions Every 10 Iterations")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations_avg_positions.png")
