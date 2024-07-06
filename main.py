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
    times = np.linspace(0, 1e-6, 300)    # 500 time steps up to 1 microsecond

    # Define scatterer and receiver positions
    scatterer_pos = [(50, 50)]  # in mm
    receiver_pos = (75, 75)   # in mm

    # Initial amplitude (representing voltage)
    initial_amplitude = 30  # Adjust this value to change wave strength

    # Frequency of the ultrasound wave
    frequency = 5e6  # 5 MHz

    # Generate nonlinear ultrasound wave propagation data
    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=initial_amplitude, speed=medium.sound_speed, nonlinearity=0.01)

    # Simulate the wave propagation and get the results as a 3D array
    wavefield = simulate_using_steps(
        wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude)

    # Ensure the wavefield has the correct shape
    # assert wavefield.shape == (len(times), len(z_points), len(
    #     x_points)), "Shape of wavefield is incorrect"

    # Debugging: Print wavefield summary
    print("Wavefield Summary:")
    print("Min value:", np.min(wavefield))
    print("Max value:", np.max(wavefield))
    print("Mean value:", np.mean(wavefield))
    print("RAW Values:", wavefield)

    # Animate the wave propagation
    print("Rendering Animation")
    animate_wave(wavefield, x_points, z_points,
                 times, scatterer_pos, receiver_pos)


if __name__ == '__main__':
    main()
