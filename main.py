from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_simulation import simulate_nonlinear_wave_propagation
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from utils.data_visualization import animate_wave
import numpy as np


def main():
    """
    Main function to run the ultrasound simulation and visualization.
    """
    # Define medium properties
    medium = Medium(density=1000, sound_speed=1500)

    # Define simulation parameters
    x_points = np.linspace(0, 500, 500)  # in mm
    z_points = np.linspace(0, 500, 500)  # in mm
    times = np.linspace(0, 1e-6, 130)    # in seconds

    # Define scatterer and receiver positions
    scatterer_pos = (200, 200)  # in mm
    receiver_pos = (300, 300)   # in mm

    # Generate nonlinear ultrasound wave propagation data
    wave = NonlinearUltrasoundWave(
        frequency=5e6, amplitude=1.0, speed=medium.sound_speed, nonlinearity=0.01)
    # Simulate the wave propagation and get the results as a 3D array
    propagation_results = simulate_nonlinear_wave_propagation(
        wave, medium, x_points, z_points, times)

    # Ensure propagation_results has the correct shape
    assert propagation_results.shape == (len(times), len(x_points), len(
        z_points)), "Shape of propagation_results is incorrect"

    # Animate the wave propagation
    animate_wave(propagation_results, x_points, z_points,
                 times, scatterer_pos, receiver_pos)


if __name__ == '__main__':
    main()
