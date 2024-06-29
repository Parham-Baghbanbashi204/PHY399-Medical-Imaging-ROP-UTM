import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium


def simulate_nonlinear_wave_propagation(wave, medium, x_points, z_points, times, scatterer_pos):
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
    :return: A 3D array of wave amplitudes over time and space.
    :rtype: numpy.ndarray
    """
    # Determine the number of points along each dimension
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)

    # Initialize a 3D array to store the results
    results = np.zeros((nt, nx, nz, 3))

    # Create a meshgrid of spatial points
    xx, zz = np.meshgrid(x_points, z_points, indexing='ij')

    # Calculate the distances from the scatterer for all spatial points
    distances = np.sqrt(
        (xx - scatterer_pos[0])**2 + (zz - scatterer_pos[1])**2)

    # Initial wave condition: Gaussian pulse centered at the scatterer
    w = 0.05  # Width of the Gaussian pulse
    results[:, :, 0] = wave.amplitude * \
        np.exp(-((xx - scatterer_pos[0])**2 +
               (zz - scatterer_pos[1])**2) / (2 * w**2))

    # Time step (dt) and spatial step (dx)
    dx = x_points[1] - x_points[0]
    dt = 0.707 * dx / medium.sound_speed  # Satisfy CFL condition

    # First time step using leapfrog method
    results[1:-1, 1:-1, 1] = results[1:-1, 1:-1, 0] + 0.5 * wave.speed**2 * (
        (results[:-2, 1:-1, 0] + results[2:, 1:-1, 0] - 2 * results[1:-1, 1:-1, 0]) +
        (results[1:-1, :-2, 0] + results[1:-1,
         2:, 0] - 2 * results[1:-1, 1:-1, 0])
    ) * (dt**2 / dx**2)

    for t_idx in range(1, nt-1):
        results[1:-1, 1:-1, 2] = -results[1:-1, 1:-1, 0] + 2 * results[1:-1, 1:-1, 1] + wave.speed**2 * (
            (results[:-2, 1:-1, 1] + results[2:, 1:-1, 1] - 2 * results[1:-1, 1:-1, 1]) +
            (results[1:-1, :-2, 1] + results[1:-1,
             2:, 1] - 2 * results[1:-1, 1:-1, 1])
        ) * (dt**2 / dx**2)

        # Shift results for the next iteration
        results[:, :, 0] = results[:, :, 1]
        results[:, :, 1] = results[:, :, 2]

    return results[:, :, :, 1]
