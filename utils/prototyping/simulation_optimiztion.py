import time
import numpy as np
from tqdm import tqdm

# Define your original function here for comparison


def simulate_using_steps_original(wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude=0.1, pulse_width=0.5, cycles=1):
    # Original function code
    def discretize_pressure_wave(p, ndt, idx, jdz):
        return p(ndt, idx, jdz)

    def partial_derivative_t(p_next, p_current, p_prev, dt):
        return (p_next - 2 * p_current + p_prev) / dt**2

    def partial_derivative_x(p_next_x, p_current, p_prev_x, dx):
        return (p_next_x - 2 * p_current + p_prev_x) / dx**2

    def partial_derivative_z(p_next_z, p_current, p_prev_z, dz):
        return (p_next_z - 2 * p_current + p_prev_z) / dz**2

    def main_equation(p_next, p_current, p_prev, c, partial_x, partial_z, s, dt):
        left_side = (p_next - 2 * p_current + p_prev) / dt**2
        right_side = c**2 * (partial_x + partial_z) + s
        return left_side, right_side

    def initial_pressure_wave(x, z, t, amplitude, frequency, cycles, width, center_x, center_z):
        sine_wave = np.cos(2 * np.pi * frequency * t)
        gaussian_envelope = np.exp(-((x - center_x)
                                   ** 2 + (z - center_z)**2) / (2 * width**2))
        return amplitude * sine_wave * gaussian_envelope

    v = medium.sound_speed
    dt = 0.001
    dx = 0.1
    dz = 0.1
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)
    amplitude = 7e-8
    frequency = wave.frequency
    width = 1.0 / medium.density
    c = v / (dx / dt)
    cfl_number = c * dt / min(dx, dz)

    if cfl_number > 1:
        raise ValueError(
            "CFL condition not satisfied. Reduce the time step size or increase the spatial step size.")

    center_x, center_z = scatterer_pos
    p = np.zeros((nt, nx, nz))

    for i in range(nx):
        for j in range(nz):
            p[0, i, j] = initial_pressure_wave(
                i * dx, j * dz, 0, amplitude, frequency, cycles, width, center_x * 10**-1, center_z * 10**-1)
            p[1, i, j] = p[0, i, j]

    for n in tqdm(range(1, nt - 1), desc="Simulation Progress"):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                partial_x = partial_derivative_x(
                    p[n, i + 1, j], p[n, i, j], p[n, i - 1, j], dx)
                partial_z = partial_derivative_z(
                    p[n, i, j + 1], p[n, i, j], p[n, i, j - 1], dz)
                s = 0
                left_side, right_side = main_equation(
                    p[n + 1, i, j], p[n, i, j], p[n - 1, i, j], c, partial_x, partial_z, s, dt)
                p[n + 1, i, j] = right_side * dt**2 + \
                    2 * p[n, i, j] - p[n - 1, i, j]

    return p

# Define the optimized function (as provided earlier)


def simulate_using_steps_optimized(wave, medium, x_points, z_points, times, scatterer_pos, initial_amplitude=0.1, pulse_width=0.5, cycles=1):
    v = medium.sound_speed
    dt = 0.001
    dx = 0.1
    dz = 0.1
    nx = len(x_points)
    nz = len(z_points)
    nt = len(times)
    amplitude = 7e-8
    frequency = wave.frequency
    width = 1.0 / medium.density
    c = v / (dx / dt)
    cfl_number = c * dt / min(dx, dz)

    if cfl_number > 1:
        raise ValueError(
            "CFL condition not satisfied. Reduce the time step size or increase the spatial step size.")

    center_x, center_z = scatterer_pos
    p = np.zeros((nt, nx, nz))

    X, Z = np.meshgrid(x_points * dx, z_points * dz, indexing='ij')
    sine_wave = np.cos(2 * np.pi * frequency * 0)
    gaussian_envelope = np.exp(-((X - center_x * 10**-1)
                               ** 2 + (Z - center_z * 10**-1)**2) / (2 * width**2))
    p[0] = amplitude * sine_wave * gaussian_envelope
    p[1] = p[0]

    for n in tqdm(range(1, nt - 1), desc="Simulation Progress"):
        partial_x = (p[n, 2:, 1:-1] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, :-2, 1:-1]) / dx**2
        partial_z = (p[n, 1:-1, 2:] - 2 * p[n, 1:-1, 1:-1] +
                     p[n, 1:-1, :-2]) / dz**2
        s = 0
        right_side = c**2 * (partial_x + partial_z) + s
        p[n + 1, 1:-1, 1:-1] = right_side * dt**2 + \
            2 * p[n, 1:-1, 1:-1] - p[n - 1, 1:-1, 1:-1]

    return p

# Setup test parameters


class NonlinearUltrasoundWave:
    def __init__(self, frequency):
        self.frequency = frequency


class Medium:
    def __init__(self, sound_speed, density):
        self.sound_speed = sound_speed
        self.density = density


wave = NonlinearUltrasoundWave(frequency=5e6)
medium = Medium(sound_speed=1500, density=1000)
x_points = np.linspace(0, 1, 100)
z_points = np.linspace(0, 1, 100)
times = np.linspace(0, 1, 1000)
scatterer_pos = (0.5, 0.5)

# Measure the execution time of the original function
start_time = time.time()
result_original = simulate_using_steps_original(
    wave, medium, x_points, z_points, times, scatterer_pos)
end_time = time.time()
print(f"Original function execution time: {end_time - start_time:.2f} seconds")

# Measure the execution time of the optimized function
start_time = time.time()
result_optimized = simulate_using_steps_optimized(
    wave, medium, x_points, z_points, times, scatterer_pos)
end_time = time.time()
print(
    f"Optimized function execution time: {end_time - start_time:.2f} seconds")

# Compare the results
try:
    np.testing.assert_allclose(result_original, result_optimized, rtol=1e-5)
    print("The original and optimized functions return the same output.")
except AssertionError as e:
    print("The outputs of the original and optimized functions do not match.")
    print(e)
