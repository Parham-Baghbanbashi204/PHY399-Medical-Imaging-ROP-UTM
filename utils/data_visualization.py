import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def animate_wave(wave_data, x_points, z_points, times, interval=20, title='Wave Animation'):
    """
    Animates the wave data over time.

    :param wave_data: 3D numpy array where each slice is the wave amplitude at a given time.
    :type wave_data: numpy.ndarray
    :param x_points: 1D numpy array representing the x-dimension.
    :type x_points: numpy.ndarray
    :param z_points: 1D numpy array representing the z-dimension (depth).
    :type z_points: numpy.ndarray
    :param times: 1D numpy array representing the time domain.
    :type times: numpy.ndarray
    :param interval: Time between frames in milliseconds.
    :type interval: int
    :param title: Title of the animation.
    :type title: str
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    # Display the initial frame of the wave data using imshow
    cax = ax.imshow(wave_data[0, :, :], cmap='seismic', extent=[
                    x_points.min(), x_points.max(), z_points.min(), z_points.max()], origin='lower')
    # Add a color bar to indicate the amplitude values
    fig.colorbar(cax, ax=ax, label='Amplitude')

    def update(frame):
        # Clear the previous frame's data
        ax.clear()
        # Display the current frame of the wave data using imshow
        cax = ax.imshow(wave_data[frame, :, :], cmap='seismic', extent=[
                        x_points.min(), x_points.max(), z_points.min(), z_points.max()], origin='lower')
        # Set the title to show the current time step
        ax.set_title(f'Time Step (nt) = {frame}')
        # Label the x and y axes
        ax.set_xlabel('x-dimension (m)')
        ax.set_ylabel('z-dimension (depth in m)')
        # Add a color bar to indicate the amplitude values
        fig.colorbar(cax, ax=ax, label='Amplitude')

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=len(
        times), interval=interval, repeat=False)

    # Save the animation to a video file
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("animations/wave_animation.mp4", writer=writer)


def animate_rf_signal(rf_data, time, interval=20, title='RF Signal Animation'):
    """
    Animates the RF signal data over time.

    :param rf_data: 2D numpy array where each row is the RF signal at a given time.
    :type rf_data: numpy.ndarray
    :param time: 1D numpy array representing the time domain.
    :type time: numpy.ndarray
    :param interval: Time between frames in milliseconds.
    :type interval: int
    :param title: Title of the animation.
    :type title: str
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    # Initialize a line object to be updated in the animation
    line, = ax.plot(time, rf_data[0, :])

    def update(frame):
        # Update the line data for the current frame
        line.set_ydata(rf_data[frame, :])
        return line,

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=range(
        rf_data.shape[0]), interval=interval, blit=True)
    # Set the title and labels for the plot
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    # Save the animation to a video file
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("animaitons/rf_animation.mp4", writer=writer)
