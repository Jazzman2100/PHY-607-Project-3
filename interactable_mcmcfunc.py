#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Constants for the FM signal
fs = 10e6      # Sampling rate (Hz), 10 MHz for high resolution
fc = 100e6     # Carrier frequency (Hz), 100 MHz (FM radio band)
fm = 1e3       # Modulating frequency (Hz), audio signal frequency (1 kHz)
delta_f = 75e3 # Frequency deviation (Hz), typical for FM radio
duration = 1   # Duration of the signal in seconds

# True parameter values used to generate the FM signal (these will be used for comparison)
d_true = 1.0   # True scaling factor for the sine wave
e_true = 2 * np.pi * fm  # True frequency of the sine wave (modulating frequency)
f_true = 0.0   # True phase offset

# Time vector for the FM signal
t = np.arange(0, duration, 1/fs)

# Modulating signal (audio signal, e.g., a sine wave)
audio_signal = np.sin(2 * np.pi * fm * t)

# FM Modulated signal
fm_signal = np.cos(2 * np.pi * fc * t + delta_f * np.sin(2 * np.pi * fm * t))

# Plot the audio signal (modulating signal)
plt.figure(figsize=(10, 6))

# Plot the first 1000 samples of the audio signal
plt.subplot(3, 1, 1)
plt.plot(t[:1000], audio_signal[:1000])  
plt.title("Modulating Signal (Audio Signal)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the FM modulated signal (carrier with modulation)
plt.subplot(3, 1, 2)
plt.plot(t[:1000], fm_signal[:1000])  
plt.title("FM Modulated Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the frequency spectrum of the FM signal using FFT
fft_signal = np.fft.fft(fm_signal)
frequencies = np.fft.fftfreq(len(t), 1/fs)

plt.subplot(3, 1, 3)
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_signal)[:len(frequencies)//2])
plt.title("Frequency Spectrum of FM Signal")
plt.tight_layout()
plt.show()

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

# Get user input for the number of walkers and iterations
n_walkers = int(input("Enter the number of walkers: "))
n_iterations = int(input("Enter the number of iterations: "))

# Run MCMC for user-defined walkers and iterations (use the FM signal as the data)
walkers = mcmc_walkers(n_walkers, n_iterations, x, fm_signal)

# Extract the final positions for d, e, and f
final_d = walkers[:, 0, -1]  # Final values of d
final_e = walkers[:, 1, -1]  # Final values of e
final_f = walkers[:, 2, -1]  # Final values of f

# Plot the joint distribution of d, e, and f with the walker trajectories
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the final joint distribution of d, e, f (scatter plot of final walker positions)
ax.scatter(final_d, final_e, final_f, color='b', alpha=0.7, label="Final Positions of Walkers")

# Add the trajectories (path of each walker across iterations)
for i in range(n_walkers):
    ax.plot(walkers[i, 0, :], walkers[i, 1, :], walkers[i, 2, :], color='gray', alpha=0.5, lw=1)

# Add labels and title
ax.set_xlabel('d')
ax.set_ylabel('e')
ax.set_zlabel('f')
ax.set_title('Joint Distribution of Parameters d, e, f with Walkers\' Trajectories')

# Plot true parameter values as red dots for comparison
ax.scatter(d_true, e_true, f_true, color='red', s=100, label="True Values", marker='x')

# Show the plot
ax.legend()
plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations.png")
plt.savefig('interaction.png')
