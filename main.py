#!/usr/bin/env python3

import numpy as np
from fm_mcmc.signal import generate_fm_signal, generate_audio_signal
from fm_mcmc.mcmc import run_mcmc
from fm_mcmc.plotting import plot_correlation_length, plot_walkers_paths, plot_avg_positions
from fm_mcmc.cor_len import correlation_length

def main():
    # Constants for the FM signal
    fs = 10e6      # Sampling rate (Hz)
    fc = 100e6     # Carrier frequency (Hz)
    fm = 1e3       # Modulating frequency (Hz)
    duration = 1   # Duration of the signal in seconds

    # Time vector for the FM signal
    t = np.arange(0, duration, 1/fs)

    # Generate the true FM signal using the true parameters
    d_true = 1.0   # True scaling factor for the sine wave (Amplitude)
    e_true = 2 * np.pi * fm  # True frequency of the sine wave (Modulating frequency)
    f_true = 0.0   # True phase offset

    audio_signal = generate_audio_signal(t)
    fm_signal = generate_fm_signal(d_true, e_true, f_true, t)

    # User prompt for number of walkers and iterations
    n_walkers = int(input("Enter the number of walkers: "))
    n_iterations = int(input("Enter the number of iterations: "))

    # Run MCMC simulation
    all_walkers_positions = run_mcmc(n_walkers, n_iterations, fm_signal, t)

    # Calculate correlation lengths
    correlation_lengths = []
    for i in range(n_iterations):
        walkers_pos_at_iter = all_walkers_positions[i, :, :].reshape(-1, 3)
        correlation_lengths.append(correlation_length(walkers_pos_at_iter))

    # Plotting
    plot_correlation_length(correlation_lengths, n_walkers, n_iterations)
    plot_walkers_paths(all_walkers_positions, d_true, e_true, f_true, n_walkers, n_iterations)

    # Calculate and plot average walker positions every 10 iterations
    avg_positions = []
    for i in range(0, n_iterations, 10):
        walkers_at_iter = all_walkers_positions[i, :, :]
        avg_d = np.mean(walkers_at_iter[:, 0])
        avg_e = np.mean(walkers_at_iter[:, 1])
        avg_f = np.mean(walkers_at_iter[:, 2])
        avg_positions.append((avg_d, avg_e, avg_f))

    avg_positions = np.array(avg_positions)
    plot_avg_positions(avg_positions, n_walkers, n_iterations)

if __name__ == "__main__":
    main()
