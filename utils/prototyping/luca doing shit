import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Step 1: Define the domain and discretize it
nx, nz = 500, 500
dx, dz = 0.01, 0.01
nt =1000
dt = 0.001
x = np.linspace(0, (nx-1)*dx, nx)
z = np.linspace(0, (nz-1)*dz, nz)

# Step 2: Initialize the wavefield and other variables
p = np.zeros((nz, nx))
pold = np.zeros((nz, nx))
pnew = np.zeros((nz, nx))
d2px = np.zeros((nz, nx))
d2pz = np.zeros((nz, nx))
seis = np.zeros(nt)
src = np.exp(-((np.arange(nt) - 50)**2) / (2 * 10**2))  # Gaussian source

# Source and receiver positions
isz, isx = nz // 2, nx // 2
irz, irx = isz, isx

# Velocity
c = 1.0

# Step 3: Time-stepping loop
fig, ax = plt.subplots()
cax = ax.imshow(p, vmin=-0.01, vmax=0.01, interpolation="nearest", cmap=plt.cm.hsv)
fig.colorbar(cax)

def update_wavefield(it):
    global p, pold, pnew, d2px, d2pz, seis
    # Compute second-order spatial derivatives
    for i in range(1, nx-1):
        d2px[:, i] = (p[:, i-1] - 2 * p[:, i] + p[:, i+1]) / dx**2
    for j in range(1, nz-1):
        d2pz[j, :] = (p[j-1, :] - 2 * p[j, :] + p[j+1, :]) / dz**2

    # Update wavefield
    pnew = 2 * p - pold + (c**2) * (dt**2) * (d2pz + d2px)
    
    # Add source term
    pnew[isz, isx] = pnew[isz, isx] + src[it] / (dx * dz) * (dt**2)
    
    # Update previous wavefields
    pold = p
    p = pnew
    
    # Record seismogram
    seis[it] = p[irz, irx]
    
    # Visualization every 10 time steps
    if it % 10 == 0:
        ax.set_title(f'Time Step (nt) = {it}')
        cax.set_data(p)
    
    return cax,

# Animate the wavefield
ani = animation.FuncAnimation(fig, update_wavefield, frames=nt, blit=True, interval=30)
plt.show()

# Step 4: Plot seismogram
time = np.arange(nt) * dt
plt.figure(figsize=(10, 6))
plt.plot(time, seis)
plt.title("Seismogram at Receiver Position")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig('output.png')""\"""