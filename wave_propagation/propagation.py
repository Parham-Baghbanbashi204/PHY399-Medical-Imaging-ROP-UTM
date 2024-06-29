"""
Propagation Module
===================
This module defines classes for the medium and base ultrasound wave.
"""


class Medium:
    """
    Class to represent the medium in which the ultrasound wave propagates.

    :param density: Density of the medium. in kg/m^3
    :type density: float
    :param sound_speed: Speed of sound in the medium. in m/s
    :type sound_speed: float
    """

    def __init__(self, density, sound_speed):
        self.density = density
        self.sound_speed = sound_speed


class UltrasoundWave:
    """
    Base class to represent an ultrasound wave.

    :param frequency: The frequency of the wave in Hz.
    :type frequency: float
    :param amplitude: The amplitude of the wave.
    :type amplitude: float
    :param speed: The speed of the wave in the medium.
    :type speed: float
    """

    def __init__(self, frequency, amplitude, speed):
        self.frequency = frequency
        self.amplitude = amplitude
        self.speed = speed

    def propagate(self, distance, time):
        """
        Simulate the propagation of the wave. To be implemented by subclasses.

        :param distance: The distance over which the wave propagates.
        :type distance: float
        :param time: The time duration of the propagation.
        :type time: float
        :return: The wave amplitude at the given distance and time.
        :rtype: float
        """
        raise NotImplementedError("Subclasses should implement this!")
