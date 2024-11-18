#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_length(correlation_lengths, n_walkers, n_iterations):
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iterations), correlation_lengths, label='Correlation Length')
    plt.xlabel('Iterations')
    plt.ylabel('Correlation Length')
    plt.title(f"Correlation Length vs. Iterations ({n_walkers} walkers, {n_iterations} iterations)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations_correlation.png")

def plot_walkers_paths(all_walkers_positions, d_true, e_true, f_true, n_walkers, n_iterations):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_walkers):
        path = all_walkers_positions[:, i, :]
        ax.plot(path[:, 1], path[:, 2], path[:, 0], color='grey', alpha=0.6)  # e, f, d
    final_positions = all_walkers_positions[-1, :, :]
    ax.scatter(final_positions[:, 1], final_positions[:, 2], final_positions[:, 0], 
               c='blue', marker='o', label="Final Position")
    ax.scatter(d_true, e_true, f_true, c='red', marker='x', label='True Value')
    ax.set_xlabel('Modulating Frequency (e)')
    ax.set_ylabel('Phase Shift (f)')
    ax.set_zlabel('Amplitude (d)')
    ax.set_title(f"Walkers' Positions Over Iterations (3D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations_walkers_paths.png")

def plot_avg_positions(avg_positions, n_walkers, n_iterations):
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, n_iterations, 10), avg_positions[:, 0], label="Average d (Amplitude)", color='blue')
    plt.plot(range(0, n_iterations, 10), avg_positions[:, 1], label="Average e (Frequency)", color='green')
    plt.plot(range(0, n_iterations, 10), avg_positions[:, 2], label="Average f (Phase)", color='red')
    plt.xlabel('Iterations (every 10 steps)')
    plt.ylabel('Parameter Value')
    plt.title(f"Average Walker Positions Every 10 Iterations")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{n_walkers}walker(s), {n_iterations}iterations_avg_positions.png")

