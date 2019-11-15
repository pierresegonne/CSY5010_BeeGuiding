import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import os

from utils import *

SHOW_FIGURES = True

# ============================================
SWARM_FILENAME = 'recordings/swarm_object 2019-11-15 19-24.pickle'
analysis_filename = SWARM_FILENAME.split('/')[1].split('.')[0]
directory = 'analysis/{}'.format(analysis_filename)
if not os.path.exists(directory):
    os.makedirs(directory)
# ============================================

swarm = read_pickled(SWARM_FILENAME)

# ============================================
# =========== Plot of positions ==============
barycenters = swarm.get_barycenters()
# Y,X
plt.figure(figsize=(7,2.5))
plt.plot(barycenters[:,1], barycenters[:,0], '-', color='slategray')
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Trajectory of Barycenter (Y,X)')
plt.tight_layout()
plt.savefig(f'{directory}/barycenter_yx.png')

# X,Z
plt.figure(figsize=(3,3))
plt.plot(barycenters[:,0], barycenters[:,2], '-', color='slategray')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Trajectory of Barycenter (X,Z)')
plt.tight_layout()
plt.savefig(f'{directory}/barycenter_xz.png')

# Y,Z
plt.figure(figsize=(7,2.5))
plt.plot(barycenters[:,1], barycenters[:,2], '-', color='slategray')
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('Trajectory of Barycenter (Y,Z)')
plt.tight_layout()
plt.savefig(f'{directory}/barycenter_yz.png')

# 3D
fig = plt.figure()
ax = p3.Axes3D(fig)
plt.title('3D Trajectory of Barycenter')

# Setting the axes properties
ax.view_init(10, -10)
ax.set_xlim3d([0.0, swarm.end_hive_position[0] + 200])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, swarm.end_hive_position[1] + 200])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, swarm.end_hive_position[2] + 200])
ax.set_zlabel('Z')

# Plot hives
ax.scatter(swarm.start_hive_position[0], swarm.start_hive_position[1], swarm.start_hive_position[2], color='midnightblue', marker='h')
ax.scatter(swarm.end_hive_position[0], swarm.end_hive_position[1], swarm.end_hive_position[2], color='darkgreen', marker='h')

# Plot trajectory
ax.scatter(barycenters[:,0], barycenters[:,1], barycenters[:,2], s=2, color='slategray')

plt.savefig(f'{directory}/barycenter_3d.png')

# ============================================
# ============= Plot of speed ================
speeds = swarm.get_average_speeds()
plt.figure(figsize=(7,2.5))
plt.plot(speeds, '-', color='slategray')
plt.xlabel('Iterations')
plt.ylabel('Average Speed')
plt.title('Average speed of swarm during its flight')
plt.tight_layout()
plt.savefig(f'{directory}/avg_speed.png')


# ============================================
# =========== Print convergence ==============
text_output = ''
if swarm.has_converged:
    text_output += '-- Convergence: {0}\n   Distance to target_hive: {1:.2f}'.format(
        barycenters[-1],
        np.linalg.norm(swarm.end_hive_position - barycenters[-1].reshape(-1,1))
        )
    text_output += '\n   Converged at iteration {}'.format(swarm.has_converged_iteration)
else:
    text_output += '-- No Convergence: {0}\n   Distance to target_hive: {1:.2f}'.format(
        barycenters[-1],
        np.linalg.norm(swarm.end_hive_position - barycenters[-1].reshape(-1,1))
        )


# ============================================
# ============= Print density ================
text_output += '\n\n-- Density at the end of the simulation: {0:.6f}'.format(swarm.get_density())

print(text_output)
with open(f'{directory}/data.txt', 'w+') as f:
    f.write(text_output)
f.close()

if SHOW_FIGURES:
    plt.show()
