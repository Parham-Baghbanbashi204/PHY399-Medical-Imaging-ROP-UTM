import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium


def simulate_nonlinear_wave_propagation(wave, medium, x_points, z_points, times):
    """
    Simulate nonlinear wave propagation in a medium.

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
    :return: A 3D array of wave amplitudes over time and space.
    :rtype: numpy.ndarray
    """
    # Determine the number of points along each dimension
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)

    # Initialize a 3D array to store the results
    results = np.zeros((nt, nx, nz))

    # Loop over each time step
    for t_idx, time in enumerate(times):
        # Loop over each point in the x-dimension
        for x_idx, x in enumerate(x_points):
            # Loop over each point in the z-dimension
            for z_idx, z in enumerate(z_points):
                # Calculate the distance from the origin for the wave propagation
                distance = np.sqrt(x**2 + z**2)
                # Calculate the wave amplitude at the current time and position
                results[t_idx, x_idx, z_idx] = wave.propagate(distance, time)

    return results
