import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update_lines(num, dataLines, lines):
    # print(num)
    # print(dataLines)
    # print(lines)
    for line, data in zip(lines, dataLines):
        # print(line, data)
        # print(data[0:2, :3])
        # exit()
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

def show_swarm_flight(
    old_hive_position, new_hive_position, recorded_positions,
    space_dimensions,
    total_scout,
    ):

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Plot hives
    ax.scatter(old_hive_position[0], old_hive_position[1], old_hive_position[2])
    ax.scatter(new_hive_position[0], new_hive_position[1], new_hive_position[2])

    # Recorded positions:
    # (Iterations + 1) x Bees x Dimensions
    iterations = len(recorded_positions)
    recording_shape = recorded_positions[0].shape
    # Data
    # Bees x Dimensions x (Iterations + 1)
    data = [np.zeros((recording_shape[1], len(recorded_positions))) for _ in range(recording_shape[0])]
    for it in range(len(recorded_positions)):
        for b_i in range(len(data)):
            for d_i in range(recording_shape[1]):
                data[b_i][d_i,it] = recorded_positions[it][b_i,d_i]

    # Initialization of lines with first points.
    lines = []
    for i, dat in enumerate(data):
        if i < total_scout:
            lines.append(ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], color='gold')[0])
        else:
            lines.append(ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], color='black')[0])

    # Setting the axes properties
    ax.set_xlim3d([0.0, space_dimensions[0]])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, space_dimensions[1]])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, space_dimensions[2]])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, iterations, fargs=(data, lines),
                                       interval=50, blit=False, repeat=False)

    plt.show()

def animate_scatter(iteration, data, scatters, total_scout):
    scatters[0]._offsets3d = (data[iteration][:total_scout,0], data[iteration][:total_scout,1], data[iteration][:total_scout,2])
    scatters[1]._offsets3d = (data[iteration][total_scout:,0], data[iteration][total_scout:,1], data[iteration][total_scout:,2])
    return scatters

def show_swarm_flight_scatter(
    old_hive_position, new_hive_position, recorded_positions,
    space_dimensions,
    total_scout,
    save=False, save_filename='swarm'
    ):

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Plot hives
    ax.scatter(old_hive_position[0], old_hive_position[1], old_hive_position[2])
    ax.scatter(new_hive_position[0], new_hive_position[1], new_hive_position[2])

    # Initialize scatter
    scatters = [
        ax.scatter(recorded_positions[0][:total_scout,0], recorded_positions[0][:total_scout,1], recorded_positions[0][:total_scout,2], color='gold'),
        ax.scatter(recorded_positions[0][total_scout:,0], recorded_positions[0][total_scout:,1], recorded_positions[0][total_scout:,2], color='black'),
    ]

    ax.legend(['Old Hive', 'New Hive', 'Scouts', 'Uninformed Bees'])

    # Number of iterations
    iterations = len(recorded_positions)

    # Setting the axes properties
    ax.set_xlim3d([0.0, space_dimensions[0]])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, space_dimensions[1]])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, space_dimensions[2]])
    ax.set_zlabel('Z')

    ax.set_title('Swarm Flight Demonstration')

    ax.view_init(25, 10)

    ani = animation.FuncAnimation(fig, animate_scatter, iterations, fargs=(recorded_positions, scatters, total_scout),
                                       interval=50, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(f'{save_filename}.mp4', writer=writer)

    plt.show()









