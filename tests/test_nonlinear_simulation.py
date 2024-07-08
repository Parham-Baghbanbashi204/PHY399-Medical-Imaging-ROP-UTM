from wave_propagation.nonlinear_simulation import simulate_using_steps, simulate_using_steps_optimized
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
import numpy as np


def test_simulate_using_steps():
    # Define test parameters
    wave = NonlinearUltrasoundWave(
        frequency=5e6, amplitude=3, speed=1530, nonlinearity=0.1)
    medium = Medium(sound_speed=1500, density=1)
    x_points = np.linspace(0, 10, 100)
    z_points = np.linspace(0, 10, 100)
    times = np.linspace(0, 1, 100)
    scatterer_pos = (5, 5)
    initial_amplitude = 0.1
    pulse_width = 0.5
    cycles = 1

    # Run original function
    p_original = simulate_using_steps(
        wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude, pulse_width, cycles)

    # Run optimized function
    p_optimized = simulate_using_steps_optimized(
        wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude, pulse_width, cycles)

    # Compare the results
    assert np.allclose(
        p_original, p_optimized), "The optimized function does not match the original function."


# Call the test function
test_simulate_using_steps()
