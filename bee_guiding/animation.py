import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
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