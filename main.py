"""
Main Module
============
This module runs the ultrasound simulation and visualization.
"""
import numpy as np
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from utils.data_visualization import animate_wave
from wave_propagation.nonlinear_simulation import simulate_using_steps_optimized_with_pulse_source, simulate_using_steps_optimized_no_src, simulate_using_steps_optimized_no_src_auto
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
    end_time = 0.3  # in sec
    total_time_steps = 600

    # TODO pass the step as the dx,dz and dt for the sim
    x_points, dx = np.linspace(
        0, 200, 200, retstep=True, endpoint=True)  # in mm
    z_points, dz = np.linspace(
        0, 200, 200, retstep=True, endpoint=True)  # in mm
    times, dt = np.linspace(
        0, end_time, total_time_steps, retstep=True, endpoint=True)

    scatterer_pos = (60, 60)  # in mm
    receiver_pos = (80, 80)   # in mm

    initial_amplitude = 7e-6  # Adjust this value to change wave strength
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
    # NO PULSE SOURCE FOR THIS PHASE SINCE WE ARE ONLY SIMULATING THE ECHO thus no src term since the echo is just an intal pulse back aka just the inital conditions.
    result_optimized = simulate_using_steps_optimized_no_src(
        wave, medium, x_points, z_points, times, scatterer_pos, dx, dz, dt)
    # result_optimized = simulate_using_steps_optimized_no_src_auto(
    # wave, medium, 0, 200, 200, 0, 200, 200, 0, 0.3, 500, scatterer_pos)
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
