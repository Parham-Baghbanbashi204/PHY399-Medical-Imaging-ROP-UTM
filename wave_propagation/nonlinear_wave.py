"""
Nonlinear Wave Module
======================
This module defines classes for nonlinear ultrasound waves.
"""

import numpy as np
from .propagation import UltrasoundWave


class NonlinearUltrasoundWave(UltrasoundWave):
    """
    Class to represent a nonlinear Gaussian ultrasound wave.

    :param frequency: The frequency of the wave in Hz.
    :type frequency: float
    :param amplitude: The amplitude of the wave.
    :type amplitude: float
    :param speed: The speed of the wave in the medium.
    :type speed: float
    :param nonlinearity: Parameter representing the degree of nonlinearity.
    :type nonlinearity: float
    """

    def __init__(
        self,
        frequency,
        amplitude,
        speed,
        nonlinearity,
    ):
        # Validate parameters
        if (
            not isinstance(
                frequency, (int, float)
            )
            or frequency <= 0
        ):
            raise ValueError(
                "Frequency must be a positive number."
            )
        if (
            not isinstance(
                amplitude, (int, float)
            )
            or amplitude <= 0
        ):
            raise ValueError(
                "Amplitude must be a positive number."
            )
        if (
            not isinstance(speed, (int, float))
            or speed <= 0
        ):
            raise ValueError(
                "Speed must be a positive number."
            )
        if (
            not isinstance(
                nonlinearity, (int, float)
            )
            or nonlinearity < 0
        ):
            raise ValueError(
                "Nonlinearity must be a non-negative number."
            )

        # Initialize superclass with frequency, amplitude, and speed
        super().__init__(
            frequency, amplitude, speed
        )
        # Initialize the nonlinearity parameter
        self.nonlinearity = nonlinearity

    def propagate(self, distance, time):
        """
        Simulate the propagation of the nonlinear wave over a fixed distance and time.

        :param distance: The distance over which the wave propagates.
        :type distance: float
        :param time: The time duration of the propagation.
        :type time: float
        :return: The wave amplitude at the given distance and time.
        :rtype: float
        """
        # Validate parameters
        if (
            not isinstance(distance, (int, float))
            or distance < 0
        ):
            raise ValueError(
                "Distance must be a non-negative number."
            )
        if (
            not isinstance(time, (int, float))
            or time < 0
        ):
            raise ValueError(
                "Time must be a non-negative number."
            )

        # Calculate the wavelength
        wavelength = self.speed / self.frequency
        # Calculate the Gaussian envelope
        envelope = np.exp(
            -((distance - self.speed * time) ** 2)
            / (2 * (wavelength**2))
        )
        # Apply the nonlinearity effect
        nonlinear_effect = np.exp(
            -self.nonlinearity * distance
        )
        # Return the resulting amplitude
        return (
            self.amplitude
            * envelope
            * nonlinear_effect
        )
