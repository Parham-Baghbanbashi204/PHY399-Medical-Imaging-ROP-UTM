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
from tqdm import tqdm
import math
from utils.run_on_gpu import run_on_gpu


@run_on_gpu
def simulate_reciver(wave_data, x_points, z_points, times, listen_pos):
    """
    Listen for the wave amplitude at a specified position over time(acts as the reciver).

    :param wave_data: 3D numpy array where each slice is the wave amplitude at a given time.
    :type wave_data: numpy.ndarray
    :param x_points: 1D numpy array representing the x-dimension.
    :type x_points: numpy.ndarray
    :param z_points: 1D numpy array representing the z-dimension (depth).
    :type z_points: numpy.ndarray
    :param times: 1D numpy array representing the time domain.
    :type times: numpy.ndarray
    :param listen_pos: Tuple of the listening position (x, z) in meters.
    :type listen_pos: tuple
    :return: 1D numpy array of wave amplitudes at the listening position over time.
    :rtype: numpy.ndarray
    """
    x_idx = np.argmin(np.abs(x_points - listen_pos[0]))
    z_idx = np.argmin(np.abs(z_points - listen_pos[1]))

    return wave_data[:, x_idx, z_idx], times


@run_on_gpu
def simulate_using_steps_optimized(wave, medium, x_start, x_stop, x_steps, z_start, z_stop, z_steps, t_start, t_stop, t_steps, scatterer_pos):
    """
    Simulate pressure wave propagation using partial derivatives, optimized version of the simulate_using_steps() using vectorization, with no source term.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param x_start: Start value for the x-dimension.
    :type x_start: float
    :param x_stop: Stop value for the x-dimension.
    :type x_stop: float
    :param x_steps: Number of steps in the x-dimension.
    :type x_steps: int
    :param z_start: Start value for the z-dimension.
    :type z_start: float
    :param z_stop: Stop value for the z-dimension.
    :type t_start: float
    :param t_start: Start value for time doimain
    :param t_stop: Stop value for the time domain.
    :type t_stop: float
    :param t_steps: Number of steps in the time domain.
    :type t_steps: int
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
    :param initial_amplitude: Initial amplitude of the wave (representing voltage).
    :type initial_amplitude: float
    :return: A 3D array of wave amplitudes over time and space, all the x points, all the z_points and all the times
    :rtype: numpy.ndarray
    """
    v = medium.sound_speed  # m/s
    # Calculate dx, dz, and dt based on the provided start, stop, and step
    x_points, dx = np.linspace(
        x_start, x_stop, x_steps, retstep=True, endpoint=True)
    z_points, dz = np.linspace(
        z_start, z_stop, z_steps, retstep=True, endpoint=True)
    times, dt = np.linspace(t_start, t_stop, t_steps,
                            retstep=True, endpoint=True)
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)
    amplitude = wave.amplitude
    frequency = wave.frequency
    width = medium.density * medium.sound_speed  # acoustic impidance in kg/m^3
    c = v / (dx / dt)
    cfl_number = c * dt / min(dx, dz)

    if cfl_number > 1:
        print("CFL Number", cfl_number)
        raise ValueError(
            "CFL condition not satisfied. Reduce the time step size or increase the spatial step size.")

    center_x, center_z = scatterer_pos
    p = np.zeros((nt, nx, nz))

    X, Z = np.meshgrid(x_points, z_points, indexing='ij')
    sine_wave = np.cos(2 * np.pi * frequency * 0)
    gaussian_envelope = np.exp(-((X - center_x) **
                                 2 + (Z - center_z)**2) / (2 * width**2))
    p[0] = amplitude * sine_wave * gaussian_envelope  # without sin wave
    p[1] = p[0]

    # loop over the times
    for n in tqdm(range(1, nt - 1), desc="Simulation Progress"):
        partial_x = (p[n, 2:, 1:-1] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, :-2, 1:-1]) / dx**2
        partial_z = (p[n, 1:-1, 2:] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, 1:-1, :-2]) / dz**2
        right_side = c**2 * (partial_x + partial_z)
        p[n + 1, 1:-1, 1:-1] = right_side * dt**2 + \
            2 * p[n, 1:-1, 1:-1] - p[n - 1, 1:-1, 1:-1]

    return p


@run_on_gpu  # RUN THE SIM USING THE GPU
def simulate_using_steps(wave, medium, x_start, x_stop, x_steps, z_start, z_stop, z_steps, t_start, t_stop, t_steps, scatterer_pos):
    """
    Simulate preasure wave propgation using partial dirivitives.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param x_start: Start value for the x-dimension.
    :type x_start: float
    :param x_stop: Stop value for the x-dimension.
    :type x_stop: float
    :param x_steps: Number of steps in the x-dimension.
    :type x_steps: int
    :param z_start: Start value for the z-dimension.
    :type z_start: float
    :param z_stop: Stop value for the z-dimension.
    :type t_start: float
    :param t_start: Start value for time doimain
    :param t_stop: Stop value for the time domain.
    :type t_stop: float
    :param t_steps: Number of steps in the time domain.
    :type t_steps: int
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
    :param initial_amplitude: Initial amplitude of the wave (representing voltage).
    :type initial_amplitude: float
    :return: A 3D array of wave amplitudes over time and space, all the x points, all the z_points and all the times
    :rtype: numpy.ndarray
    """
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
    def initial_pressure_wave(x, z, t, amplitude, frequency, width, center_x, center_z):
        """
        Generate an initial pressure wave with control over amplitude, frequency, and pulsing in cycles.

        :param x: x position.
        :type x: float
        :param z: z position.
        :type z: float
        :param t: Time.
        :type t: float
        :param amplitude: Amplitude of the wave.
        :type amplitude: float
        :param frequency: Frequency of the wave.
        :type frequency: float
        :param cycles: Number of cycles in the pulse.
        :type cycles: int
        :param width: Width of the Gaussian envelope.
        :type width: float
        :param center_x: Center position of the Gaussian envelope in x.
        :type center_x: float
        :param center_z: Center position of the Gaussian envelope in z.
        :type center_z: float
        :return: Initial pressure wave value.
        :rtype: float
        """
        # Sinusoidal function for the pulsing behavior
        sine_wave = np.cos(2 * np.pi * frequency * t)

        # Gaussian envelope for the spatial distribution
        gaussian_envelope = np.exp(-((x - center_x) **
                                   2 + (z - center_z)**2) / (2 * width**2))  # pulled out / (2 * width**2)

        # Combine both to form the initial pressure wave
        # return amplitude * sine_wave * gaussian_envelope
        return amplitude * sine_wave * gaussian_envelope

    v = medium.sound_speed
    # Calculate dx, dz, and dt based on the provided start, stop, and step
    x_points, dx = np.linspace(
        x_start, x_stop, x_steps, retstep=True, endpoint=True)
    z_points, dz = np.linspace(
        z_start, z_stop, z_steps, retstep=True, endpoint=True)
    times, dt = np.linspace(t_start, t_stop, t_steps,
                            retstep=True, endpoint=True)
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)
    amplitude = wave.amplitude
    frequency = wave.frequency
    width = medium.density * medium.sound_speed  # acoustic impidance
    c = v / (dx / dt)
    cfl_number = c * dt / min(dx, dz)

    if cfl_number > 1:
        raise ValueError(
            "CFL condition not satisfied. Reduce the time step size or increase the spatial step size.")

    center_x, center_z = scatterer_pos
    p = np.zeros((nt, nx, nz))

   # Set initial conditions
    for i in range(nx):
        for j in range(nz):
            p[0, i, j] = initial_pressure_wave(
                i * dx, j*dz, 0, amplitude, frequency, width, center_x*10**-1, center_z*10**-1)
            p[1, i, j] = p[0, i, j]  # Initial condition for the second time step

    # Run the simulation
    for n in tqdm(range(1, nt-1), desc="Simulation Progress"):
        # print("running time step", n)
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
                # take only the right side for our data
                p[n+1, i, j] = right_side * dt**2 + \
                    2 * p[n, i, j] - p[n-1, i, j]

    # p now contains the pressure wave values for all time steps and spatial points
    print("finished processing")
    return p, x_points, z_points, times


@ run_on_gpu
def simulate_using_steps_optimized_with_cont_src(wave, medium, x_start, x_stop, x_steps, z_start, z_stop, z_steps, t_start, t_stop, t_steps, scatterer_pos):
    """
    Simulate pressure wave propagation using partial derivatives, optimized version of the original function using vectorization, with a point source term.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param x_start: Start value for the x-dimension.
    :type x_start: float
    :param x_stop: Stop value for the x-dimension.
    :type x_stop: float
    :param x_steps: Number of steps in the x-dimension.
    :type x_steps: int
    :param z_start: Start value for the z-dimension.
    :type z_start: float
    :param z_stop: Stop value for the z-dimension.
    :type t_start: float
    :param t_start: Start value for time doimain
    :param t_stop: Stop value for the time domain.
    :type t_stop: float
    :param t_steps: Number of steps in the time domain.
    :type t_steps: int
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
    :param initial_amplitude: Initial amplitude of the wave (representing voltage).
    :type initial_amplitude: float
    :return: A 3D array of wave amplitudes over time and space, all the x points, all the z_points and all the times
    :rtype: numpy.ndarray
    """
    v = medium.sound_speed
    # Calculate dx, dz, and dt based on the provided start, stop, and step
    x_points, dx = np.linspace(
        x_start, x_stop, x_steps, retstep=True, endpoint=True)
    z_points, dz = np.linspace(
        z_start, z_stop, z_steps, retstep=True, endpoint=True)
    times, dt = np.linspace(t_start, t_stop, t_steps,
                            retstep=True, endpoint=True)
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)
    amplitude = wave.amplitude
    frequency = wave.frequency
    width = medium.density * medium.sound_speed  # acoustic impidance
    c = v / (dx / dt)
    cfl_number = c * dt / min(dx, dz)

    if cfl_number > 1:
        raise ValueError(
            "CFL condition not satisfied. Reduce the time step size or increase the spatial step size.")

    center_x, center_z = scatterer_pos
    p = np.zeros((nt, nx, nz))

    X, Z = np.meshgrid(x_points, z_points, indexing='ij')
    sine_wave = np.cos(2 * np.pi * frequency * 0)
    gaussian_envelope = np.exp(-((X - center_x) **
                                 2 + (Z - center_z)**2) / (2 * width**2))
    p[0] = amplitude * sine_wave * gaussian_envelope  # without sin wave
    p[1] = p[0]

    for n in tqdm(range(1, nt - 1), desc="Simulation Progress"):
        partial_x = (p[n, 2:, 1:-1] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, :-2, 1:-1]) / dx**2
        partial_z = (p[n, 1:-1, 2:] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, 1:-1, :-2]) / dz**2
        right_side = c**2 * (partial_x + partial_z)

        # Final Equation without source term
        p[n + 1, 1:-1, 1:-1] = right_side * dt**2 + \
            2 * p[n, 1:-1, 1:-1] - p[n - 1, 1:-1, 1:-1]

        # Add source term
        source_term = amplitude * np.cos(2 * np.pi * frequency * times[n])
        p[n + 1, center_x, center_z] += source_term

    return p, x_points, z_points, times


@ run_on_gpu
def simulate_using_steps_optimized_with_pulse_source(wave, medium, x_start, x_stop, x_steps, z_start, z_stop, z_steps, t_start, t_stop, t_steps, scatterer_pos):
    """
    Simulate pressure wave propagation using partial derivatives, optimized version of the original function using vectorization, with a gaussian pulse source term.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param x_start: Start value for the x-dimension.
    :type x_start: float
    :param x_stop: Stop value for the x-dimension.
    :type x_stop: float
    :param x_steps: Number of steps in the x-dimension.
    :type x_steps: int
    :param z_start: Start value for the z-dimension.
    :type z_start: float
    :param z_stop: Stop value for the z-dimension.
    :type t_start: float
    :param t_start: Start value for time doimain
    :param t_stop: Stop value for the time domain.
    :type t_stop: float
    :param t_steps: Number of steps in the time domain.
    :type t_steps: int
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
    :param initial_amplitude: Initial amplitude of the wave (representing voltage).
    :type initial_amplitude: float
    :return: A 3D array of wave amplitudes over time and space, all the x points, all the z_points and all the times
    :rtype: numpy.ndarray
    """
    v = medium.sound_speed
    # Calculate dx, dz, and dt based on the provided start, stop, and step
    x_points, dx = np.linspace(
        x_start, x_stop, x_steps, retstep=True, endpoint=True)
    z_points, dz = np.linspace(
        z_start, z_stop, z_steps, retstep=True, endpoint=True)
    times, dt = np.linspace(t_start, t_stop, t_steps,
                            retstep=True, endpoint=True)
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)
    amplitude = wave.amplitude
    frequency = wave.frequency
    width = medium.density * medium.sound_speed  # acoustic impidance
    c = v / (dx / dt)
    cfl_number = c * dt / min(dx, dz)

    if cfl_number > 1:
        raise ValueError(
            "CFL condition not satisfied. Reduce the time step size or increase the spatial step size.")

    center_x, center_z = scatterer_pos
    p = np.zeros((nt, nx, nz))

    X, Z = np.meshgrid(x_points, z_points, indexing='ij')
    sine_wave = np.cos(2 * np.pi * frequency * 0)
    gaussian_envelope = np.exp(-((X - center_x) **
                                 2 + (Z - center_z)**2) / (2 * width**2))
    p[0] = amplitude * sine_wave * gaussian_envelope  # without sin wave
    p[1] = p[0]
    # Creating the Gaussian pulse source term
    src = amplitude * \
        np.exp(-((np.arange(nt) - nt // 2)**2) / (2 * width**2))
    sine_wave = np.sin(2 * np.pi * frequency * times)
    pulse = src

    for n in tqdm(range(1, nt - 1), desc="Simulation Progress"):
        partial_x = (p[n, 2:, 1:-1] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, :-2, 1:-1]) / dx**2
        partial_z = (p[n, 1:-1, 2:] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, 1:-1, :-2]) / dz**2
        right_side = c**2 * (partial_x + partial_z)

        # Final Equation without source term
        p[n + 1, 1:-1, 1:-1] = right_side * dt**2 + \
            2 * p[n, 1:-1, 1:-1] - p[n - 1, 1:-1, 1:-1]

        # Add pulse source term
        p[n + 1, center_x, center_z] += pulse[n]

    return p, x_points, z_points, times
