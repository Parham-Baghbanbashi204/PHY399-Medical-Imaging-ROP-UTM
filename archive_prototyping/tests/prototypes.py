import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Initialize physical and numerical constants
L = 1.0  # set the length of the system
dx = 0.01  # set the discrete spatial stepsize
c = 1.0  # define the wave speed

dt = .707 * dx / c  # choose a time step to satisfy the CFL condition

x = np.arange(0, L + dx, dx)  # define an array to store x position data
y = np.arange(0, L + dx, dx)  # define an array to store y position data

xx, yy = np.meshgrid(x, y)

npts = len(x)  # this is the number of spatial points along x
nsteps = 60  # set the number of time steps to get a 2 second animation at 30 FPS

f = np.zeros((npts, npts, 3))

xc = 0.5  # define the center of the system to locate a Gaussian pulse
w = 0.05  # define the width of the Gaussian wave pulse

f[:, :, 0] = np.exp(-(xx - xc) ** 2 / (w ** 2)) * np.exp(-(yy - xc)
                                                         ** 2 / (w ** 2))  # initial condition for a Gaussian

# First time step in the leap frog algorithm
f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * c ** 2 * (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2. * f[1:-1, 1:-1, 0]) * (dt ** 2 / dx ** 2) \
    + 0.5 * c ** 2 * (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] -
                      2. * f[1:-1, 1:-1, 0]) * (dt ** 2 / dx ** 2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def update(frame):
    global f
    # For all additional time steps
    for _ in range(1):
        f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2 * f[1:-1, 1:-1, 1] \
            + c ** 2 * (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2. * f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2) \
            + c ** 2 * (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2. *
                        f[1:-1, 1:-1, 1]) * (dt ** 2 / dx ** 2)

        # Push the data back for the leapfrogging
        f[:, :, 0] = f[:, :, 1]
        f[:, :, 1] = f[:, :, 2]

    ax.clear()
    ax.plot_surface(xx, yy, f[:, :, 2], rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.plot_wireframe(xx, yy, f[:, :, 2], rstride=10,
                      cstride=10, color='green')
    plt.title(f't= {frame * dt:.2f}')
    ax.set_zlim(-.25, 1)


# Create the animation
ani = FuncAnimation(fig, update, frames=nsteps, repeat=False)

# Save the animation to a video file
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save("wave_animation.mp4", writer=writer)
