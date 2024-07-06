"""
Main Module
============
This module runs the ultrasound simulation and visualization.
"""
import numpy as np
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from utils.data_visualization import animate_wave
from wave_propagation.nonlinear_simulation import simulate_using_steps


def main():
    """
    Main function to run the ultrasound simulation and visualization.
    """
    # Define medium properties
    medium = Medium(density=1, sound_speed=1500)

    # Define simulation parameters
    x_points = np.linspace(0, 100, 100)  # 500 points along x-dimension in mm
    # 500 points along z-dimension (depth) in mm
    z_points = np.linspace(0, 100, 100)
    times = np.linspace(0, 1e-6, 500)    # 500 time steps up to 1 microsecond

    # Define scatterer and receiver positions
    scatterer_pos = [(50, 50)]  # in mm
    receiver_pos = (75, 75)   # in mm

    # Initial amplitude (representing voltage)
    initial_amplitude = 3.0  # Adjust this value to change wave strength

    # Frequency of the ultrasound wave
    frequency = 5e6  # 5 MHz

    # Generate nonlinear ultrasound wave propagation data
    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=initial_amplitude, speed=medium.sound_speed, nonlinearity=0.01)

    # Simulate the wave propagation and get the results as a 3D array
    propagation_results = simulate_using_steps(
        wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude)

    # Ensure propagation_results has the correct shape
    # assert propagation_results.shape == (len(times), len(x_points), len(
    #     z_points)), "Shape of propagation_results is incorrect"

    # Debugging: Print propagation results summary
    print("Propagation Results Summary:")
    print("Min value:", np.min(propagation_results))
    print("Max value:", np.max(propagation_results))
    print("Mean value:", np.mean(propagation_results))
    print(propagation_results)

    # Animate the wave propagation
    animate_wave(propagation_results, x_points, z_points,
                 times, scatterer_pos, receiver_pos)


if __name__ == '__main__':
    main()
