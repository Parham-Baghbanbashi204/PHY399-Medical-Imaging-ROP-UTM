import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Step 1: 2D Grid Setup
nx, nz = 500, 500  # Grid size
dx, dz = 1, 1  # Spatial step (in meters)
nt = 300  # Number of time steps
dt = 1e-7  # Time step (in seconds)
c = 1500  # Speed of sound (m/s)

# Initialize pressure field
pressure = np.zeros((nx, nz))
pressure_new = np.zeros((nx, nz))
pressure_old = np.zeros((nx, nz))

# Scatterer and Receiver positions
scatterer_pos = (250, 250)
receiver_pos = (350, 350)
pressure[scatterer_pos] = 1  # Initial pulse at the scatterer

# Storage for the receiver signal
receiver_signal = []

# Define the convolution kernel for the Laplacian (second derivative)
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

# Step 2: Simulate Wave Propagation
# Problem here is that the simulation can be simplifed if we just solve the ode's
for t in range(nt):
    laplacian_pressure = convolve(pressure, kernel, mode='constant', cval=0)
    pressure_new = 2 * pressure - pressure_old + \
        (c**2 * dt**2 / dx**2) * laplacian_pressure  # type:ignore

    # Update the pressure fields
    pressure_old = pressure.copy()
    pressure = pressure_new.copy()

    # Store receiver signal
    receiver_signal.append(pressure[receiver_pos])

# Convert receiver signal to numpy array for plotting
receiver_signal = np.array(receiver_signal)

# Step 3: Calculate the Green's Function (reference signal)

# This stuff could be good I just need to see how that plays with the presure wave


def greens_function(t, c=1500, d=100):
    return np.exp(-((t - d/c)**2) / (2 * (0.1 / c)**2))


time = np.arange(nt) * dt
reference_signal = greens_function(time)

# # Step 4: Visualize the Results
# already implimented
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Plot pressure wave at a specific time step
# im = axs[0].imshow(pressure, cmap='seismic', extent=[
#                    0, nx*dx, 0, nz*dz], origin='lower')
# axs[0].scatter(*scatterer_pos, color='red', label='Scatterer')
# axs[0].scatter(*receiver_pos, color='black', label='Receiver')
# axs[0].set_title(f'Time Step (nt) = {nt}')
# axs[0].set_xlabel('x-dimension (m)')
# axs[0].set_ylabel('z-dimension (depth) (m)')
# axs[0].legend()
# fig.colorbar(im, ax=axs[0], label='Amplitude (Pa)')

# # Plot radiofrequency signal
# axs[1].plot(time, reference_signal, 'r--', label='Green\'s function')
# axs[1].plot(time, receiver_signal, 'b-', label='Finite Difference')
# axs[1].set_title('Radiofrequency Signal')
# axs[1].set_xlabel('Time [s]')
# axs[1].set_ylabel('Amplitude [Pa]')
# axs[1].legend()

# plt.tight_layout()
# plt.show()
