"""
Nonlinear Simulation Module
============================
This module defines functions for simulating nonlinear wave propagation.
"""
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import pandas as pd


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

    # calculate the distances from the scatterer for all spatial points
    distances = np.sqrt(
        (xx - scatterer_pos[0])**2 + (zz - scatterer_pos[1])**2)

    # calculate the wavelength based on the wave frequency and medium's sound speed
    wavelength = medium.sound_speed / wave.frequency  # in m
    # convert to mm for the simulations order of magnitude:
    wavelength = wavelength*1e3  # in mm

    # set the pulse length to be a few wavelengths (e.g., num_cycles wavelengths)
    pulse_length = num_cycles * wavelength

    # calculate the gaussian pulse width (standard deviation)
    sigma = pulse_length / 2.355

    # initial wave condition: gaussian pulse centered at the scatterer with an automatically adjusted width
    results[0, :, :] = initial_amplitude * \
        np.exp(-distances**2 / (2 * sigma**2))

    # time step (dt) and spatial step (dx)
    dx = x_points[1] - x_points[0]
    dt = 0.707 * dx / medium.sound_speed  # satisfy cfl condition

    # first time step using leapfrog method
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
    def discretize_pressure_wave(p, ndt, idx, jdz):
        """
        Discretize the pressure wave function into discrete values of time and position.

        Equation:
        p(x, z, t) → p_{i,j}^n = p(ndt, idx, jdz)

        :param p: Pressure wave function.
        :type p: function
        :param ndt: Discrete time value.
        :type ndt: float
        :param idx: Discrete x position value.
        :type idx: float
        :param jdz: Discrete z position value.
        :type jdz: float
        :return: Discretized pressure wave value.
        :rtype: float
        """
        # Call the pressure wave function with discrete time and position values
        return p(ndt, idx, jdz)

    def partial_derivative_t(p_next, p_current, p_prev, dt):
        """
        Compute the second partial derivative of the pressure wave function with respect to time.

        Equation:
        ∂²p(x, t)/∂t² ≈ (p_{i,j}^{n+1} - 2p_{i,j}^n + p_{i,j}^{n-1}) / dt²

        :param p_next: Pressure wave value at next time step.
        :type p_next: float
        :param p_current: Pressure wave value at current time step.
        :type p_current: float
        :param p_prev: Pressure wave value at previous time step.
        :type p_prev: float
        :param dt: Time step size.
        :type dt: float
        :return: Second partial derivative with respect to time.
        :rtype: float
        """
        # Compute the second partial derivative with respect to time
        return (p_next - 2 * p_current + p_prev) / dt**2

    def partial_derivative_x(p_next_x, p_current, p_prev_x, dx):
        """
        Compute the second partial derivative of the pressure wave function with respect to the x coordinate.

        Equation:
        ∂²p/∂x² ≈ (p_{i+1,j}^n - 2p_{i,j}^n + p_{i-1,j}^n) / dx²

        :param p_next_x: Pressure wave value at next x position.
        :type p_next_x: float
        :param p_current: Pressure wave value at current x position.
        :type p_current: float
        :param p_prev_x: Pressure wave value at previous x position.
        :type p_prev_x: float
        :param dx: x position step size.
        :type dx: float
        :return: Second partial derivative with respect to x.
        :rtype: float
        """
        # Compute the second partial derivative with respect to x
        return (p_next_x - 2 * p_current + p_prev_x) / dx**2

    def partial_derivative_z(p_next_z, p_current, p_prev_z, dz):
        """
        Compute the second partial derivative of the pressure wave function with respect to the z coordinate.

        Equation:
        ∂²p/∂z² ≈ (p_{i,j+1}^n - 2p_{i,j}^n + p_{i,j-1}^n) / dz²

        :param p_next_z: Pressure wave value at next z position.
        :type p_next_z: float
        :param p_current: Pressure wave value at current z position.
        :type p_current: float
        :param p_prev_z: Pressure wave value at previous z position.
        :type p_prev_z: float
        :param dz: z position step size.
        :type dz: float
        :return: Second partial derivative with respect to z.
        :rtype: float
        """
        # Compute the second partial derivative with respect to z
        return (p_next_z - 2 * p_current + p_prev_z) / dz**2

    def main_equation(p_next, p_current, p_prev, c, partial_x, partial_z, s, dt):
        """
        Combine the partial derivatives into the main equation for the numerical method.

        Equation:
        (p_{i,j}^{n+1} - 2p_{i,j}^n + p_{i,j}^{n-1}) / dt² = c²(∂²p/∂x² + ∂²p/∂z²) + s_{i,j}^n

        :param p_next: Pressure wave value at next time step.
        :type p_next: float
        :param p_current: Pressure wave value at current time step.
        :type p_current: float
        :param p_prev: Pressure wave value at previous time step.
        :type p_prev: float
        :param c: Wave speed constant.
        :type c: float
        :param partial_x: Second partial derivative with respect to x.
        :type partial_x: float
        :param partial_z: Second partial derivative with respect to z.
        :type partial_z: float
        :param s: Source term.
        :type s: float
        :param dt: Time step size.
        :type dt: float
        :return: Left side and right side of the main equation.
        :rtype: tuple
        """
        # Compute the left side of the main equation
        left_side = (p_next - 2 * p_current + p_prev) / dt**2
        # Compute the right side of the main equation
        right_side = c**2 * (partial_x + partial_z) + s
        # Return the left and right sides of the main equation
        return left_side, right_side

    # Define the initial pressure wave function
    def initial_pressure_wave(x, z):
        # Example: A Gaussian pulse as the initial condition
        return np.exp(-((x-5)**2 + (z-5)**2))

    # Define parameters
    c = 1.0      # Wave speed constant
    dt = 0.01    # Time step size
    dx = 0.1     # x position step size
    dz = 0.1     # z position step size
    nx = len(x_points)     # Number of x points
    nz = len(z_points)     # Number of z points
    nt = len(times)     # Number of time steps
    # Initialize pressure wave array to store all time steps
    p = np.zeros((nt, nx, nz))

    # Set initial conditions
    for i in range(nx):
        for j in range(nz):
            p[0, i, j] = initial_pressure_wave(i * dx, j * dz)
            p[1, i, j] = p[0, i, j]  # Initial condition for the second time step

    # Run the simulation
    for n in range(1, nt-1):
        print("running time step", n)
        for i in range(1, nx-1):
            for j in range(1, nz-1):
                # Compute partial derivatives
                partial_x = partial_derivative_x(
                    p[n, i+1, j], p[n, i, j], p[n, i-1, j], dx)
                partial_z = partial_derivative_z(
                    p[n, i, j+1], p[n, i, j], p[n, i, j-1], dz)
                s = 0  # Source term, could be defined as needed

                # Compute the next pressure wave value using the main equation
                left_side, right_side = main_equation(
                    p[n+1, i, j], p[n, i, j], p[n-1, i, j],
                    c, partial_x, partial_z, s, dt
                )
                p[n+1, i, j] = right_side * dt**2 + \
                    2 * p[n, i, j] - p[n-1, i, j]

    # p now contains the pressure wave values for all time steps and spatial points

    return p
