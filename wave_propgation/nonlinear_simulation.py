# wave_propagation/nonlinear_simulation.py
import numpy as np
from .nonlinear_wave import NonlinearUltrasoundWave
from .propagation import Medium


def simulate_nonlinear_wave_propagation(wave, medium, distances, times):
    """
    Simulate nonlinear wave propagation in a medium.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param distances: 1D array of distances over which the wave propagates.
    :type distances: numpy.ndarray
    :param times: 1D array of time points at which the wave is observed.
    :type times: numpy.ndarray
    :return: A 2D array where each row represents the wave amplitude at a given distance and time.
    :rtype: numpy.ndarray
    """
    results = np.zeros((len(distances), len(times)))
    for i, distance in enumerate(distances):
        for j, time in enumerate(times):
            results[i, j] = wave.propagate(distance, time)
    return results
