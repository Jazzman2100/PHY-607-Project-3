#!/usr/bin/env python3

import numpy as np

def correlation_length(walkers_positions):
    """Calculate the correlation length by measuring the average distance between walkers."""
    distances = np.linalg.norm(walkers_positions[:, np.newaxis] - walkers_positions, axis=-1)
    return np.mean(distances)
