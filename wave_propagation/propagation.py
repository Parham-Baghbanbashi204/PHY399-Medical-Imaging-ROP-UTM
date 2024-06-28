"""
Basic ultrasound wave behavior with ability to modify propgated medium
"""
import numpy as np


class UltrasoundWave:
    """
    Class to represent an ultrasound wave.

    :param frequency: The frequency of the wave in Hz.
    :type frequency: float
    :param amplitude: The amplitude of the wave.
    :type amplitude: float
    :param speed: The speed of the wave in the medium.
    :type speed: float
    """

    def __init__(self, frequency, amplitude, speed):
        # Initialize the ultrasound wave parameters
        self.frequency = frequency
        self.amplitude = amplitude
        self.speed = speed

    def propagate(self, distance, time):
        """
        Simulate the propagation of the wave.

        :param distance: The distance over which the wave propagates.
        :type distance: float
        :param time: The time duration of the propagation.
        :type time: float
        :return: The wave amplitude at the given distance and time.
        :rtype: float
        """
        # Calculate the wavelength
        wavelength = self.speed / self.frequency
        # Calculate the phase of the wave
        phase = 2 * np.pi * (distance / wavelength - self.frequency * time)
        # Calculate and return the amplitude of the wave at the given distance and time
        return self.amplitude * np.sin(phase)


class Medium:
    """
    Class to represent a medium through which the wave propagates.

    :param density: The density of the medium.
    :type density: float
    :param sound_speed: The speed of sound in the medium.
    :type sound_speed: float
    """

    def __init__(self, density, sound_speed):
        # Initialize the medium parameters
        self.density = density
        self.sound_speed = sound_speed


def simulate_wave_propagation(wave, medium, distances, times):
    """
    Simulate wave propagation in a medium.

    :param wave: An instance of UltrasoundWave.
    :type wave: UltrasoundWave
    :param medium: An instance of Medium.
    :type medium: Medium
    :param distances: 1D array of distances over which the wave propagates.
    :type distances: numpy.ndarray
    :param times: 1D array of time points at which the wave is observed.
    :type times: numpy.ndarray
    :return: A 2D array where each row represents the wave amplitude at a given distance and time.
    :rtype: numpy.ndarray
    """
    # Initialize the results array with zeros
    results = np.zeros((len(distances), len(times)))
    # Loop over each distance
    for i, distance in enumerate(distances):
        # Loop over each time point
        for j, time in enumerate(times):
            # Calculate the wave amplitude at the current distance and time
            results[i, j] = wave.propagate(distance, time)
    return results
