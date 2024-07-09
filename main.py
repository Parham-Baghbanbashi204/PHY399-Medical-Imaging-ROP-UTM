"""
Main Module
============
This module runs the ultrasound simulation and visualization. Here we configure the simulation and run it.
"""
import numpy as np
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from utils.data_visualization import animate_wave
from wave_propagation.nonlinear_simulation import simulate_reciver, simulate_using_steps, simulate_using_steps_optimized, simulate_using_steps_optimized_with_pulse_source, simulate_using_steps_optimized_with_cont_src
from utils.run_on_gpu import run_on_gpu


@run_on_gpu
def main():
    """
    Main function to run the ultrasound simulation and visualization.
    """

    # Define simulation parameters
    t_start = 0
    t_stop = 5  # in sec - the longer the time the better the resolution
    # Controls simulation time, basicly how many timesteps we want to see bigger this is the better the simulation(better aproximation)
    # temporal resolution (dt = 0.005) - larger this number better the aproximation
    t_steps = 1000
    x_start = 0
    """ 
    The resolution of the grid will be stop/steps in metres ie, x_stop = 1000 x_step = 100 => 10m resolution
    """
    # in m - standard unit of measure for the scaliablity
    x_stop = 100
    # determines resolution - currently set to 0.5m resolution(dx=0.5m)
    x_steps = 200
    z_start = 0
    z_stop = 100  # in m - since the standard unit of measure is m currently
    # determines z resolution - currently 0.5m resolution (dz = 0.5m)
    z_steps = 200
    "formated in (z,x)"
    scatterer_pos = (50, 50)  # in m
    receiver_pos = (70, 70)   # in m
    initial_amplitude = 7e-6  # Adjust this value to change wave strength
    frequency = 5e5  # 5 MHz
    # Define medium properties
    medium = Medium(density=1.081e-3, sound_speed=1530)

    # Build the wave
    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=initial_amplitude, speed=medium.sound_speed, nonlinearity=0.01)

    # SIMULATE
    wavefield = simulate_using_steps_optimized(
        wave, medium, x_start, x_stop, x_steps, z_start, z_stop, z_steps, t_start, t_stop, t_steps, scatterer_pos)

    # SIMULATION RESULTS
    print("Wavefield Summary:")
    print("Min value:", np.min(wavefield))
    print("Max value:", np.max(wavefield))
    print("Mean value:", np.mean(wavefield))

    # Create the grid and timespacing for the animate function:
    # TODO find out why i cant output the preasurewave and all the xpoints zpoints and times
    x_points, dx = np.linspace(
        x_start, x_stop, x_steps, retstep=True, endpoint=True)
    z_points, dz = np.linspace(
        z_start, z_stop, z_steps, retstep=True, endpoint=True)
    times, dt = np.linspace(t_start, t_stop, t_steps,
                            retstep=True, endpoint=True)

    # Create the sizemogram from the reciver
    rf_signal, times = simulate_reciver(
        wavefield, x_points, z_points, times, receiver_pos)

    # Works  with auto function
    print("Rendering Animation")
    animate_wave(wavefield, rf_signal, x_points, z_points,
                 times, scatterer_pos, receiver_pos, file="using_steps_no_source")


if __name__ == '__main__':
    main()
