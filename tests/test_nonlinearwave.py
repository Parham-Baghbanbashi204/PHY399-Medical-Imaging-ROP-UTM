import numpy as np
from wave_propagation.nonlinear_wave import NonlinearUltrasoundWave
from wave_propagation.propagation import Medium
from wave_propagation.nonlinear_simulation import simulate_nonlinear_wave_propagation


def test_nonlinear_ultrasound_wave_propagate():
    """
    Test the propagate method of the NonlinearUltrasoundWave class.
    """
    # Create an instance of NonlinearUltrasoundWave with specific parameters
    wave = NonlinearUltrasoundWave(
        frequency=1e6, amplitude=1.0, speed=1500, nonlinearity=0.01)

    # Call the propagate method with a specific distance and time
    distance = 0.001
    time = 0.000001
    amplitude = wave.propagate(distance, time)

    # Calculate the expected amplitude using the wave equation with nonlinearity
    wavelength = wave.speed / wave.frequency
    expected_amplitude = wave.amplitude * \
        np.sin(2 * np.pi * (distance / wavelength - wave.frequency * time)
               ) * np.exp(-wave.nonlinearity * distance)

    # Display the actual and expected values
    print(
        f"Computed amplitude: {amplitude}, Expected amplitude: {expected_amplitude}")

    # Check if the result is close to the expected value
    assert np.isclose(amplitude, expected_amplitude, atol=1e-6)


def test_simulate_nonlinear_wave_propagation():
    """
    Test the simulate_nonlinear_wave_propagation function.
    """
    # Create an instance of NonlinearUltrasoundWave with specific parameters
    wave = NonlinearUltrasoundWave(
        frequency=1e6, amplitude=1.0, speed=1500, nonlinearity=0.01)

    # Create an instance of Medium with specific parameters
    medium = Medium(density=1000, sound_speed=1500)

    # Create an array of distances for the simulation
    distances = np.linspace(0, 0.1, 100)

    # Create an array of times for the simulation
    times = np.linspace(0, 0.0001, 100)

    # Call the simulate_nonlinear_wave_propagation function with the wave, medium, distances, and times
    results = simulate_nonlinear_wave_propagation(
        wave, medium, distances, times)

    # Calculate the expected results using the wave equation with nonlinearity
    wavelength = wave.speed / wave.frequency
    expected_results = wave.amplitude * \
        np.sin(2 * np.pi * (distances[:, None] / wavelength - wave.frequency *
               times[None, :])) * np.exp(-wave.nonlinearity * distances[:, None])

    # Display the actual and expected results
    print(
        f"Computed results:\n{results}\nExpected results:\n{expected_results}")

    # Check if the results are close to the expected values
    assert np.allclose(results, expected_results, atol=1e-6)
