import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_simulation import simulate_nonlinear_wave_propagation


def test_simulate_nonlinear_wave_propagation():
    """
    Test the simulate_nonlinear_wave_propagation function.
    """
    # Set up the parameters for the test
    density = 1000
    sound_speed = 1500
    frequency = 5e6
    amplitude = 1.0
    speed = sound_speed
    nonlinearity = 0.01
    x_points = np.linspace(0, 500, 250)
    z_points = np.linspace(0, 500, 250)
    times = np.linspace(0, 1e-6, 170)
    scatterer_pos = (200, 200)
    initial_amplitude = 0.2
    pulse_radius = 10.0

    # Create instances of Medium and NonlinearUltrasoundWave
    medium = Medium(density=density, sound_speed=sound_speed)
    wave = NonlinearUltrasoundWave(
        frequency=frequency, amplitude=amplitude, speed=speed, nonlinearity=nonlinearity)

    # Run the simulation
    results = simulate_nonlinear_wave_propagation(
        wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude, pulse_radius)

    # Check the shape of the results
    assert results.shape == (len(times), len(x_points), len(
        z_points)), "Shape of the results is incorrect."

    # Check initial condition
    pulse_width = pulse_radius / np.sqrt(2 * np.log(2))
    xx, zz = np.meshgrid(x_points, z_points, indexing='ij')
    distances = np.sqrt(
        (xx - scatterer_pos[0])**2 + (zz - scatterer_pos[1])**2)
    expected_initial_condition = initial_amplitude * \
        np.exp(-distances**2 / (2 * pulse_width**2))
    np.testing.assert_allclose(
        results[0, :, :], expected_initial_condition, rtol=1e-5, atol=1e-8)

    # Check that the propagation is non-trivial (i.e., results change over time)
    assert not np.allclose(
        results[0, :, :], results[-1, :, :]), "Results do not change over time."


# This part allows the test to be run from the command line
if __name__ == "__main__":
    test_simulate_nonlinear_wave_propagation()
    print("All tests passed.")
