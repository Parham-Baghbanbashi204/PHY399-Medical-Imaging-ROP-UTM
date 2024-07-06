"""
Nonlinear Simulation Module
============================
This module defines functions for simulating nonlinear wave propagation.
"""
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


def simulate_nonlinear_wave_propagation(wave, medium, x_points, z_points, times, scatterer_pos, inital_amplitude=0.1, num_cycles=3):
    """ 
    Use Scipy to solve the pressure wave using coupled ODE's 

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

    pass


def simulate_nonlinear_wave_propagation_leapfrog(wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude=0.1, num_cycles=3):
    """
    Simulate nonlinear wave propagation in a medium using the leapfrog method. This function helps us understand how scipy will solve ode's

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
    # convert to mm for the simulations order of magnitude:
    wavelength = wavelength*1e3  # in mm

    # Set the pulse length to be a few wavelengths (e.g., num_cycles wavelengths)
    pulse_length = num_cycles * wavelength

    # Calculate the Gaussian pulse width (standard deviation)
    sigma = pulse_length / 2.355

    # Initial wave condition: Gaussian pulse centered at the scatterer with an automatically adjusted width
    results[0, :, :] = initial_amplitude * \
        np.exp(-distances**2 / (2 * sigma**2))

    # Time step (dt) and spatial step (dx)
    dx = x_points[1] - x_points[0]
    dt = 0.707 * dx / medium.sound_speed  # Satisfy CFL condition

    # First time step using leapfrog method
    results[1, 1:-1, 1:-1] = results[0, 1:-1, 1:-1] + 0.5 * wave.speed**2 * (
        (results[0, :-2, 1:-1] + results[0, 2:, 1:-1] - 2 * results[0, 1:-1, 1:-1]) +
        (results[0, 1:-1, :-2] + results[0, 1:-1, 2:] -
         2 * results[0, 1:-1, 1:-1])
    ) * (dt**2 / dx**2)

    # all subsiquent time steps
    for t_idx in range(1, nt-1):
        results[t_idx + 1, 1:-1, 1:-1] = -results[t_idx - 1, 1:-1, 1:-1] + 2 * results[t_idx, 1:-1, 1:-1] + wave.speed**2 * (
            (results[t_idx, :-2, 1:-1] + results[t_idx, 2:, 1:-1] - 2 * results[t_idx, 1:-1, 1:-1]) +
            (results[t_idx, 1:-1, :-2] + results[t_idx,
             1:-1, 2:] - 2 * results[t_idx, 1:-1, 1:-1])
        ) * (dt**2 / dx**2)

    return results


def simulate_using_steps(wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude=0.1):
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
    :return: A 3D array of wave amplitudes over time and space.
    :rtype: numpy.ndarray
    """
    # Parameters
    # Number of spatial points in x and z directions
    nx, nz = len(x_points), len(z_points)
    nt = len(times)  # Number of time steps
    dx = x_points[1] - x_points[0]
    dz = z_points[1] - z_points[0]
    dt = times[1] - times[0]
    c = 1.0  # Wave speed
    s = np.zeros((nx, nz, nt))  # Source term initialized to zero

    # # Ensure CFL condition for numerical stability
    # if c * dt > np.sqrt(dx**2 + dz**2):
    #     raise ValueError("CFL condition not met. Reduce dt or increase dx/dz.")

    # # Include scatterer in the source term if scatterer_pos is provided
    # for pos in scatterer_pos:
    #     i, j = int(pos[0] / dx), int(pos[1] / dz)
    #     if 0 <= i < nx and 0 <= j < nz:
    #         # Add wave amplitude to source term at scatterer position
    #         s[i, j, :] = wave.amplitude
    # Return the pressure field in the shape (nt, nx, nz)
    # Parameters
    nx, nz = 100, 100  # Number of spatial points in x and z directions
    nt = 500  # Number of time steps
    dx, dz, dt = 0.01, 0.01, 0.001  # Spatial and time step sizes
    c = 1.0  # Wave speed
    s = np.zeros((nx, nz, nt))  # Source term

    # Initial pressure wave function
    p = np.zeros((nx, nz, nt))

    # Initialize p at t=0 (n=0) with some initial condition, e.g., a Gaussian pulse
    x = np.linspace(0, (nx-1)*dx, nx)
    z = np.linspace(0, (nz-1)*dz, nz)
    X, Z = np.meshgrid(x, z)
    p[:, :, 0] = np.exp(-((X-nx*dx/2)**2 + (Z-nz*dz/2)**2) / (2*(0.1**2)))

    # Time stepping
    for n in range(1, nt-1):
        p[1:-1, 1:-1, n+1] = (
            2 * p[1:-1, 1:-1, n]
            - p[1:-1, 1:-1, n-1]
            + c**2 * dt**2 * (
                (p[2:, 1:-1, n] - 2 * p[1:-1, 1:-1, n] + p[:-2, 1:-1, n]) / dx**2
                + (p[1:-1, 2:, n] - 2 * p[1:-1, 1:-1, n] + p[1:-1, :-2, n]) / dz**2
            )
            + s[1:-1, 1:-1, n] * dt**2
        )
    return np.transpose(p, (2, 0, 1))
