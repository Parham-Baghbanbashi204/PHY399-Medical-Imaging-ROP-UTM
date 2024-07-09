"""
Data Visualization Module
==========================
This module defines functions for visualizing wave and RF signal data.
"""
# from vispy import app, gloo, scene
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as mcolors
from utils.run_on_gpu import run_on_gpu
import numpy as np
from tqdm import tqdm


# @run_on_gpu  # runs calculations on the gpu - no calcuations being made here
def animate_wave(wave_data, rfsignal, x_points, z_points, times, scatterer_pos, receiver_pos, interval=20, title='Wave Animation', file="wave_animation"):
    """
    Animates the wave data over time, representing the amplitude as a colorbar and the radio frequency signal as a sizemograph.

    :param wave_data: 3D numpy array where each slice is the wave amplitude at a given time.
    :type wave_data: numpy.ndarray
    :param x_points: 1D numpy array representing the x-dimension.
    :type x_points: numpy.ndarray
    :param z_points: 1D numpy array representing the z-dimension (depth).
    :type z_points: numpy.ndarray
    :param times: 1D numpy array representing the time domain.
    :type times: numpy.ndarray
    :param scatterer_pos: Tuple of the scatterer's position (x, z).
    :type scatterer_pos: tuple
    :param receiver_pos: Tuple of the receiver's position (x, z).
    :type receiver_pos: tuple
    :param interval: Time between frames in milliseconds.
    :type interval: int
    :param title: Title of the animation.
    :param file: File Name without extension
    :type title: str
    """
    # Create the figure and axis objects
    fig, (ax_wave, ax_signal) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    # Define the colormap with a center value of white
    cmap = plt.get_cmap('seismic_r')

    # Determine the max amplitude for the color scale
    max_val = np.max(np.abs(wave_data))
    vmin, vmax = -max_val, max_val
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Create the initial plot with a color bar
    cax = ax_wave.imshow(wave_data[0], cmap=cmap, norm=norm, extent=[
                         x_points.min(), x_points.max(), z_points.min(), z_points.max()], origin='lower')
    colorbar = fig.colorbar(cax, ax=ax_wave, label='Amplitude')

    # Add scatterer and receiver markers in (z,x) - conveerts (x,z) to (z,x)
    ax_wave.plot(scatterer_pos[1], scatterer_pos[0], 'ro', label='Scatterer')
    ax_wave.plot(receiver_pos[1], receiver_pos[0], 'ks', label='Receiver')

    # Add legend
    ax_wave.legend()

    # Plot settings for seismogram
    ax_signal.set_title('Radiofrequency Signal')
    ax_signal.set_xlabel('Time [s]')
    ax_signal.set_ylabel('Amplitude')
    ax_signal.grid(True)

    # Initialize the seismogram line
    seismogram, = ax_signal.plot([], [], 'b-', label='Finite Difference')

    # Initialize the progress bar
    progress_bar = tqdm(total=len(times), desc="Rendering Animation")

    # set sizmogram max and min
    siz_max = np.max(rfsignal)
    siz_min = np.min(rfsignal)

    def update(frame):
        """
        Update function for each frame of the animation.
        :param frame: Current frame number.
        :type frame: int
        """
        # Update the seismograph
        seismogram.set_data(
            times[:frame], rfsignal[:frame])
        ax_signal.set_xlim(0, times[frame])
        ax_signal.set_ylim(siz_min, siz_max)

        # Update the wave plot
        cax.set_array(wave_data[frame])
        ax_wave.set_title(
            f'Time Step (nt) = {frame}')
        ax_wave.set_xlabel('x-dimension (m)')
        ax_wave.set_ylabel('z-dimension (depth in m)')

        # Update the progress bar
        progress_bar.update(1)
        return cax, seismogram

    # Create the animation object
    ani = FuncAnimation(fig, update, frames=len(
        times), interval=interval, repeat=False)

    # Save the animation to a video file
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(f"animations/{file}.mp4", writer=writer)

    # Close the progress bar
    progress_bar.close()
