#!/usr/bin/env python3

import numpy as np
import emcee
from .signal import generate_fm_signal

def log_likelihood(theta, data, t):
    d, e, f = theta
    fm_signal_sim = generate_fm_signal(d, e, f, t)
    diff = data - fm_signal_sim
    return -0.5 * np.sum(diff**2)

def log_prior(theta):
    d, e, f = theta
    if 0 < e < 2 * np.pi and 0 <= f <= np.pi:
        prior_e = np.sin(e)
        prior_f = np.sin(f)
        prior_d = 1
        return np.log(prior_e * prior_f * prior_d)
    return -np.inf

def log_probability(theta, data, t):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data, t)

def run_mcmc(n_walkers, n_iterations, data, t):
    ndim = 3  # Number of parameters to estimate (d, e, f)
    pos = np.random.randn(n_walkers, ndim) * 10  # Initialize walkers
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=[data, t])
    sampler.run_mcmc(pos, n_iterations, progress=True)
    all_walkers_positions = sampler.get_chain()
    return all_walkers_positions
