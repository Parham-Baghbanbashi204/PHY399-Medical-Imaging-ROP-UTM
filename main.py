"""
Main Module
============
This module runs the ultrasound simulation and visualization.
"""
import numpy as np
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from utils.data_visualization import animate_wave
from wave_propagation.nonlinear_simulation import simulate_using_steps, simulate_using_steps_optimized
from utils.run_on_gpu import run_on_gpu
import time
# import tensorflow as tf


def main():
    """
    Main function to run the ultrasound simulation and visualization.
    """

    # Define medium properties
    medium = Medium(density=1, sound_speed=1500)

    # Define simulation parameters
    start_time = 0
    end_time = 1  # in sec
    total_time_steps = 1000

    x_points = np.linspace(0, 200, 200)  # in mm
    z_points = np.linspace(0, 200, 200)  # in mm
    times, step = np.linspace(
        0, end_time, total_time_steps, retstep=True, endpoint=True)

    scatterer_pos = (60, 60)  # in mm
    receiver_pos = (100, 100)   # in mm

    initial_amplitude = 10  # Adjust this value to change wave strength
    frequency = 5e5  # 5 MHz

    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=initial_amplitude, speed=medium.sound_speed, nonlinearity=0.01)

    # OPTIMIZATION TESTING
    # start_time = time.time()
    # result_original = simulate_using_steps(
    #     wave, medium, x_points, z_points, times, scatterer_pos)
    # end_time = time.time()
    # print(
    #     f"Original function execution time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    result_optimized = simulate_using_steps_optimized(
        wave, medium, x_points, z_points, times, scatterer_pos)
    end_time = time.time()
    print(
        f"Optimized function execution time: {end_time - start_time:.2f} seconds")

    # try:
    #     np.testing.assert_allclose(
    #         result_original, result_optimized, rtol=1e-5)
    #     print("The original and optimized functions return the same output.")
    # except AssertionError as e:
    #     print("The outputs of the original and optimized functions do not match.")
    #     print(e)

    wavefield = result_optimized
    print("Wavefield Summary:")
    print("Min value:", np.min(wavefield))
    print("Max value:", np.max(wavefield))
    print("Mean value:", np.mean(wavefield))

    print("Rendering Animation")
    animate_wave(wavefield, x_points, z_points,
                 times, scatterer_pos, receiver_pos)


if __name__ == '__main__':
    main()
