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


def main():
    """
    Main function to run the ultrasound simulation and visualization.
    """

    # Define simulation parameters
    """ 
    The resolution of the grid will be stop/steps in metres ie, x_stop = 1000 x_step = 100 => 10m resolution
        - best resolution seeems to be 0.5 for the x and z steps, having an index every 0.5m
    """
    t_start = 0
    # controls total simulation time - ie are we simulating 3 sec or whatnot must be an integer
    t_stop = 4  # in sec - the longer the time the better the resolution
    # temporal resolution (dt = 0.005) - smaller this number better the aproximation - can never model real time due to calculs constraint(were using descrete math)
    t_steps = 2*t_stop*100  # Best temporal resolution with computing constraints also allows for constant temporal resolution with changes in time interval
    x_start = 0

    # in m - standard unit of measure for the scaliablity
    x_stop = 100
    # determines resolution - currently set to 0.5m, basicly setting the tickmarks every 50cm resolution(dx=0.5m)
    x_steps = 2*x_stop
    z_start = 0
    z_stop = 100  # in m - since the standard unit of measure is m currently
    # determines z resolution - currently 0.5m resolution (dz = 0.5m)
    z_steps = 2*z_stop
    "formated in (z,x)"
    scatterer_pos = (20, 20)  # in m
    receiver_pos = (70, 70)   # in m
    initial_amplitude = 7e-6  # Adjust this value to change wave strength
    frequency = 5e5  # 5 MHz
    # Define medium properties - note that souond speed will be wave speed in this instance
    # TODO move the speed to the wave class insted
    medium = Medium(density=1.081e-3, sound_speed=1530)

    # Build the wave
    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=initial_amplitude, speed=medium.sound_speed, nonlinearity=0.01)

    # SIMULATE
    wavefield, x_points, z_points, times = simulate_using_steps_optimized(
        wave, medium, x_start, x_stop, x_steps, z_start, z_stop, z_steps, t_start, t_stop, t_steps, scatterer_pos)

    # SIMULATION RESULTS
    print("Wavefield Summary:")
    print("Min value:", np.min(wavefield))
    print("Max value:", np.max(wavefield))
    print("Mean value:", np.mean(wavefield))

    # Create the sizemogram from the reciver
    rf_signal, times = simulate_reciver(
        wavefield, x_points, z_points, times, receiver_pos)

    # Works  with auto function
    print("Rendering Animation")
    animate_wave(wavefield, rf_signal, x_points, z_points,
                 times, scatterer_pos, receiver_pos, file="using_steps_no_source")


if __name__ == '__main__':
    main()
