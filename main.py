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
    medium = Medium(density=1.081e-3, sound_speed=1700)

    # Define simulation parameters
    t_start = 0
    end_time = 5  # in sec
    total_time_steps = 1000

    # TODO pass the step as the dx,dz and dt for the sim
    x_points, dx = np.linspace(
        0, 100, 100, retstep=True, endpoint=True)  # in mm
    z_points, dz = np.linspace(
        0, 100, 100, retstep=True, endpoint=True)  # in mm
    times, dt = np.linspace(
        0, end_time, total_time_steps, retstep=True, endpoint=True)

    print("orignal dt", dt)

    scatterer_pos = (60, 60)  # in mm
    receiver_pos = (65, 65)   # in mm

    initial_amplitude = 7e-6  # Adjust this value to change wave strength
    frequency = 5e5  # 5 MHz

    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=initial_amplitude, speed=medium.sound_speed, nonlinearity=0.01)

    start_timer = time.time()
    # NO PULSE SOURCE FOR THIS PHASE SINCE WE ARE ONLY SIMULATING THE ECHO thus no src term since the echo is just an intal pulse back aka just the inital conditions.
    result_optimized = simulate_using_steps_optimized_no_src_auto(
        wave, medium, 0, 100, 100, 0, 100, 100, t_start, end_time, total_time_steps, scatterer_pos)
    # result_optimized = simulate_using_steps_optimized_no_src_auto(
    # wave, medium, 0, 200, 200, 0, 200, 200, 0, 0.3, 500, scatterer_pos)
    end_time = time.time()
    print(
        f"Optimized function execution time: {end_time - start_timer:.2f} seconds")
    wavefield = result_optimized
    print("Wavefield Summary:")
    print("Min value:", np.min(wavefield))
    print("Max value:", np.max(wavefield))
    print("Mean value:", np.mean(wavefield))

    # TODO MAKE THIS WORK WITH THE AUTO FUNCTION
    print("Rendering Animation")
    animate_wave(wavefield, x_points, z_points,
                 times, scatterer_pos, receiver_pos)


if __name__ == '__main__':
    main()
