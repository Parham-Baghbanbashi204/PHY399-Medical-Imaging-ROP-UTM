import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.colors as mcolors
from matplotlib.patches import Circle


def animate_wave(wave_data, x_points, z_points, times, scatterer_pos, receiver_pos, interval=20, title='Wave Animation'):
    """
    Animates the wave data over time, highlighting the first peak and first trough as expanding rings.

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
    :type title: str
    """
    fig, ax = plt.subplots()

    # Define the colormap with a center value of white
    cmap = plt.get_cmap('seismic')
    norm = mcolors.TwoSlopeNorm(
        vmin=-wave_data.max(), vcenter=0, vmax=wave_data.max())

    # Create the initial plot with a color bar
    cax = ax.imshow(wave_data[0, :, :], cmap=cmap, norm=norm, extent=[
                    x_points.min(), x_points.max(), z_points.min(), z_points.max()], origin='lower')
    colorbar = fig.colorbar(cax, ax=ax, label='Amplitude')

    # Add scatterer and receiver markers
    ax.plot(scatterer_pos[0], scatterer_pos[1], 'ro', label='Scatterer')
    ax.plot(receiver_pos[0], receiver_pos[1], 'ks', label='Receiver')

    # Add legend
    ax.legend()

    # Find the first peak (maximum value)
    peak_idx = np.unravel_index(
        np.argmax(wave_data[0, :, :]), wave_data[0, :, :].shape)
    peak_x = x_points[peak_idx[1]]
    peak_z = z_points[peak_idx[0]]
    peak_radius = np.sqrt(
        (peak_x - scatterer_pos[0])**2 + (peak_z - scatterer_pos[1])**2)

    # Find the first trough (minimum value)
    trough_idx = np.unravel_index(
        np.argmin(wave_data[0, :, :]), wave_data[0, :, :].shape)
    trough_x = x_points[trough_idx[1]]
    trough_z = z_points[trough_idx[0]]
    trough_radius = np.sqrt(
        (trough_x - scatterer_pos[0])**2 + (trough_z - scatterer_pos[1])**2)

    peak_circle = Circle(scatterer_pos, peak_radius, color='r',
                         fill=False, linestyle='-', linewidth=2, label='Initial Peak')
    trough_circle = Circle(scatterer_pos, trough_radius, color='b',
                           fill=False, linestyle='-', linewidth=2, label='Initial Trough')

    ax.add_patch(peak_circle)
    ax.add_patch(trough_circle)

    def update(frame):
        # Update the data for the current frame
        cax.set_array(wave_data[frame, :, :])

        # Update the radii of the circles
        current_time = times[frame]
        wave_speed = (x_points[1] - x_points[0]) / (times[1] - times[0])
        peak_circle.set_radius(peak_radius + wave_speed * current_time)
        trough_circle.set_radius(trough_radius + wave_speed * current_time)

        # Check for interaction with the figure boundary
        for circle in [peak_circle, trough_circle]:
            if circle.radius > max(x_points.max(), z_points.max()):
                # Hide the circle if it goes beyond the boundary
                circle.set_radius(0)

        ax.set_title(f'Time Step (nt) = {frame}')
        ax.set_xlabel('x-dimension (m)')
        ax.set_ylabel('z-dimension (depth in m)')

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
