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
    distances = np.linspace(0, 100, 500)  # in mm
    times = np.linspace(0, 1e-6, 1000)    # in seconds

    # Generate nonlinear ultrasound wave propagation data
    wave = NonlinearUltrasoundWave(
        frequency=5e6, amplitude=1.0, speed=medium.sound_speed, nonlinearity=0.01)
    propagation_results = simulate_nonlinear_wave_propagation(
        wave, medium, distances, times)

    # Reshape the propagation_results to match the expected shape for imshow
    propagation_results_reshaped = propagation_results.reshape(
        len(times), len(distances), len(distances))

    # Animate the wave propagation
    animate_wave(propagation_results_reshaped, distances, times)


if __name__ == '__main__':
    main()
