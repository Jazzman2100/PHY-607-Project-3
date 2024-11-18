#!/usr/bin/env python3

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# True values for the parameters
a_true, b_true, c_true = 1.0, 2.0, 3.0

# Define a simple likelihood function
def log_likelihood(theta, x, y, yerr):
    a, b, c = theta
    model = a * x**2 + b * x + c
    return -0.5 * np.sum(((y - model) / yerr)**2)

# Define the prior (uniform prior for each parameter)
def log_prior(theta):
    a, b, c = theta
    if -10.0 < a < 10.0 and -10.0 < b < 10.0 and -10.0 < c < 10.0:
        return 0.0  # Uniform prior
    return -np.inf  # Outside prior range

# Define the log-probability function (prior + likelihood)
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

# Generate synthetic data
x = np.linspace(0, 10, 50)
y = a_true * x**2 + b_true * x + c_true + np.random.normal(0, 1, size=x.shape)
yerr = np.ones_like(x)

# Set up the sampler (500 walkers, 1000 steps, 3 parameters)
ndim, nwalkers, nsteps = 3, 500, 1000
initial = np.random.randn(nwalkers, ndim)

# Create the MCMC sampler using emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))

# Run the MCMC
sampler.run_mcmc(initial, nsteps)

# Flatten the chain into 2D array (n_samples, n_parameters)
flat_samples = sampler.get_chain(flat=True)

# Plot the corner plot
figure = corner.corner(flat_samples, labels=["a", "b", "c"], truths=[a_true, b_true, c_true])

# Show the plot
plt.savefig('test2.jpg')
