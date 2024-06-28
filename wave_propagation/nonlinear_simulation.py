import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium


def simulate_nonlinear_wave_propagation(wave, medium, x_points, z_points, times, scatterer_pos):
    """
    Simulate nonlinear wave propagation in a medium.

    :param wave: An instance of NonlinearUltrasoundWave.
    :type wave: NonlinearUltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param x_points: 1D array of spatial points along x-dimension.
    :type x_points: numpy.ndarray
    :param z_points: 1D array of spatial points along z-dimension.
    :param times: 1D array of time points.
    :type times: numpy.ndarray
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
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

    # Precompute constants
    angular_frequency = 2 * np.pi * wave.frequency

    # Calculate wave amplitude for all points at all time steps
    for t_idx, time in enumerate(times):
        phase = angular_frequency * (time - distances / wave.speed)
        attenuation = np.exp(-wave.nonlinearity * distances)
        results[t_idx, :, :] = wave.amplitude * np.sin(phase) * attenuation

    return results
