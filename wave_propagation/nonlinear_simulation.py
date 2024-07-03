"""
Nonlinear Simulation Module
============================
This module defines functions for simulating nonlinear wave propagation.
"""
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


def solve_2d_wave_equation(c, x_points, y_points, times, s, initial_p=None, initial_dpdt=None):
    """
    Solve the 2D wave equation using solve_ivp.

    :param c: Wave speed.
    :type c: float
    :param x_points: 1D array of spatial points along x-dimension.
    :type x_points: numpy.ndarray
    :param y_points: 1D array of spatial points along y-dimension.
    :type y_points: numpy.ndarray
    :param times: 1D array of time points.
    :type times: numpy.ndarray
    :param s: Source term function s(x, y, t).
    :type s: function
    :param initial_p: Initial condition for p(x, y).
    :type initial_p: numpy.ndarray, optional
    :param initial_dpdt: Initial condition for dp/dt(x, y).
    :type initial_dpdt: numpy.ndarray, optional
    :return: A 3D array of wave amplitudes over time and space.
    :rtype: numpy.ndarray
    """
    nx = len(x_points)
    ny = len(y_points)
    nt = len(times)
    dx = x_points[1] - x_points[0]
    dy = y_points[1] - y_points[0]

    # Initialize p and dp/dt
    if initial_p is None:
        initial_p = np.zeros((nx, ny))
    if initial_dpdt is None:
        initial_dpdt = np.zeros((nx, ny))

    # Flatten the initial conditions
    u0 = np.concatenate([initial_p.ravel(), initial_dpdt.ravel()])

    def wave_equation(t, u):
        p = u[:nx*ny].reshape((nx, ny))
        dpdt = u[nx*ny:].reshape((nx, ny))

        d2pdt2 = np.zeros_like(p)
        laplacian_p = np.zeros_like(p)

        # Compute the Laplacian using finite differences
        laplacian_p[1:-1, 1:-1] = (
            (p[:-2, 1:-1] + p[2:, 1:-1] - 2 * p[1:-1, 1:-1]) / dx**2 +
            (p[1:-1, :-2] + p[1:-1, 2:] - 2 * p[1:-1, 1:-1]) / dy**2
        )

        # Update d2pdt2 with the wave equation
        d2pdt2[1:-1, 1:-1] = c**2 * laplacian_p[1:-1, 1:-1] + \
            s(x_points[1:-1][:, None], y_points[1:-1][None, :], t)

        # Flatten the derivatives
        du_dt = np.concatenate([dpdt.ravel(), d2pdt2.ravel()])
        return du_dt

    # Solve the wave equation using solve_ivp
    sol = solve_ivp(
        wave_equation, [times[0], times[-1]], u0, t_eval=times, method='RK45')

    # Extract the results
    results = np.zeros((nt, nx, ny))
    for t_idx, t in enumerate(sol.t):
        results[t_idx] = sol.y[:nx*ny, t_idx].reshape((nx, ny))

    return results

# Define the source term s(x, y, t)


def source_term(x, y, t):
    return np.sin(x) * np.cos(y) * np.exp(-t)


# Example usage
x_points = np.linspace(0, 1, 50)
y_points = np.linspace(0, 1, 50)
times = np.linspace(0, 1, 100)
c = 1.0  # Wave speed

# Solve the wave equation
results = solve_2d_wave_equation(c, x_points, y_points, times, source_term)

# The results array contains the wave amplitudes over time and space
