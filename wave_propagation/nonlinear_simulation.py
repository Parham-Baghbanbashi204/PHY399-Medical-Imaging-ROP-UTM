"""
Nonlinear Simulation Module
============================
This module defines functions for simulating nonlinear wave propagation using the leapfrog method.
"""

import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium
from numba import njit, prange


@njit(parallel=True)
def compute_initial_conditions(nx, nz, distances, initial_amplitude, sigma):
    """
    Compute the initial conditions for the wave propagation simulation.

    :param nx: Number of points in the x-dimension.
    :type nx: int
    :param nz: Number of points in the z-dimension.
    :type nz: int
    :param distances: 2D array of distances from the scatterer for all spatial points.
    :type distances: numpy.ndarray
    :param initial_amplitude: Initial amplitude of the wave.
    :type initial_amplitude: float
    :param sigma: Standard deviation of the Gaussian pulse.
    :type sigma: float
    :return: 2D array of initial wave amplitudes.
    :rtype: numpy.ndarray
    """
    # Initialize the results array for the initial condition
    results = np.zeros((nx, nz))
    # Loop over each point in the spatial grid
    for i in prange(nx):
        for j in prange(nz):
            # Compute the Gaussian envelope for the initial amplitude
            results[i, j] = initial_amplitude * \
                np.exp(-distances[i, j]**2 / (2 * sigma**2))
    return results


@njit(parallel=True)
def leapfrog_first_step(results, wave_speed, dt, dx):
    """
    Perform the first step of the leapfrog method for wave propagation.

    :param results: 3D array of wave amplitudes over time and space.
    :type results: numpy.ndarray
    :param wave_speed: Speed of the wave in the medium.
    :type wave_speed: float
    :param dt: Time step.
    :type dt: float
    :param dx: Spatial step.
    :type dx: float
    :return: 2D array of wave amplitudes after the first time step.
    :rtype: numpy.ndarray
    """
    # Get the number of points in the x and z dimensions
    nx, nz = results.shape[1], results.shape[2]
    # Initialize a new results array for the first step
    new_results = np.copy(results[0])
    # Loop over each point in the spatial grid, excluding the boundaries
    for i in prange(1, nx - 1):
        for j in prange(1, nz - 1):
            # Apply the leapfrog method for the first time step
            new_results[i, j] = results[0, i, j] + 0.5 * wave_speed**2 * (
                (results[0, i - 1, j] + results[0, i + 1, j] - 2 * results[0, i, j]) +
                (results[0, i, j - 1] + results[0,
                 i, j + 1] - 2 * results[0, i, j])
            ) * (dt**2 / dx**2)
    return new_results


@njit(parallel=True)
def leapfrog_steps(results, wave_speed, dt, dx):
    """
    Perform the subsequent steps of the leapfrog method for wave propagation.

    :param results: 3D array of wave amplitudes over time and space.
    :type results: numpy.ndarray
    :param wave_speed: Speed of the wave in the medium.
    :type wave_speed: float
    :param dt: Time step.
    :type dt: float
    :param dx: Spatial step.
    :type dx: float
    """
    # Get the number of time steps and points in the x and z dimensions
    nt, nx, nz = results.shape
    # Loop over each time step, excluding the first step
    for k in prange(1, nt - 1):
        # Loop over each point in the spatial grid, excluding the boundaries
        for i in prange(1, nx - 1):
            for j in prange(1, nz - 1):
                # Apply the leapfrog method for subsequent time steps
                results[k + 1, i, j] = 2 * results[k, i, j] - results[k - 1, i, j] + wave_speed**2 * (
                    (results[k, i - 1, j] + results[k, i + 1, j] - 2 * results[k, i, j]) +
                    (results[k, i, j - 1] + results[k,
                     i, j + 1] - 2 * results[k, i, j])
                ) * (dt**2 / dx**2)


def simulate_nonlinear_wave_propagation(wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude=0.1, num_cycles=3):
    """
    Simulate nonlinear wave propagation in a medium using the leapfrog method.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param x_points: 1D array of spatial points along x-dimension.
    :type x_points: numpy.ndarray
    :param z_points: 1D array of spatial points along z-dimension.
    :type z_points: numpy.ndarray
    :param times: 1D array of time points.
    :type times: numpy.ndarray
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
    :param initial_amplitude: Initial amplitude of the wave (representing voltage).
    :type initial_amplitude: float
    :param num_cycles: Number of cycles in the ultrasound pulse.
    :type num_cycles: int
    :return: A 3D array of wave amplitudes over time and space.
    :rtype: numpy.ndarray
    """
    # Determine the number of points along each dimension
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)

    # Initialize a 3D array to store the results
    results = np.zeros((nt, nx, nz))

    # Create a meshgrid of spatial points
    xx, zz = np.meshgrid(x_points, z_points, indexing='ij')

    # Calculate the distances from the scatterer for all spatial points
    distances = np.sqrt(
        (xx - scatterer_pos[0])**2 + (zz - scatterer_pos[1])**2)

    # Calculate the wavelength based on the wave frequency and medium's sound speed
    wavelength = medium.sound_speed / wave.frequency  # in m
    wavelength *= 1e3  # Convert to mm

    # Set the pulse length to be a few wavelengths (e.g., num_cycles wavelengths)
    pulse_length = num_cycles * wavelength

    # Calculate the Gaussian pulse width (standard deviation)
    sigma = pulse_length / 2.355

    # Initial wave condition: Gaussian pulse centered at the scatterer
    results[0] = compute_initial_conditions(
        nx, nz, distances, initial_amplitude, sigma)

    # Time step (dt) and spatial step (dx)
    dx = x_points[1] - x_points[0]
    dt = 0.707 * dx / medium.sound_speed  # Satisfy CFL condition

    # First time step using leapfrog method
    results[1] = leapfrog_first_step(results, wave.speed, dt, dx)

    # Subsequent time steps
    leapfrog_steps(results, wave.speed, dt, dx)

    return results
