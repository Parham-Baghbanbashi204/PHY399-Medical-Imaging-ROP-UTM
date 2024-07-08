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
# import tensorflow as tf


def main():
    """
    Main function to run the ultrasound simulation and visualization.
    """

    # # Ensure GPU is used
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #     print("GPU is available and will be used.")
    # else:
    #     print("No GPU available, using CPU.")
    # Define medium properties
    medium = Medium(density=1, sound_speed=1500)

    # Define simulation parameters
    start_time = 0
    end_time = 1e-6  # in sec
    total_time_steps = 600

    # TODO Make this part happen in theh simulator itself this way everything scales well
    x_points = np.linspace(0, 200, 200)  # in mm
    # 500 points along z-dimension (depth) in mm
    z_points = np.linspace(0, 200, 200)
    # 0.1 seconds per time step
    times = np.linspace(0, end_time, total_time_steps)

    # Define scatterer and receiver positions
    # RN the way this is is (x,z)
    scatterer_pos = (60, 60)  # in mm
    receiver_pos = (100, 100)   # in mm

    # Initial amplitude (representing voltage)
    initial_amplitude = 10  # Adjust this value to change wave strength

    # Frequency of the ultrasound wave
    frequency = 5e5  # 5 MHz

    # Get GPU to run calculations remove this if you want the CPU to run calculations
    # with tf.device('/GPU:0'):

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
    # Animate the wave propagation
    print("Rendering Animation")
    animate_wave(wavefield, x_points, z_points,
                 times, scatterer_pos, receiver_pos)


if __name__ == '__main__':
    main()
