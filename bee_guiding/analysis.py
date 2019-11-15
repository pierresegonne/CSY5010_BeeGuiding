import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from utils import *


# ============================================
SWARM_FILENAME = 'recordings/swarm_object 2019-11-15 14-28.pickle'
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

# X,Z
plt.figure(figsize=(3,3))
plt.plot(barycenters[:,0], barycenters[:,2], '-', color='slategray')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Trajectory of Barycenter (X,Z)')

# Y,Z
plt.figure(figsize=(7,2.5))
plt.plot(barycenters[:,1], barycenters[:,2], '-', color='slategray')
plt.xlabel('Y')
plt.ylabel('Z')
plt.title('Trajectory of Barycenter (Y,Z)')

# 3D
fig = plt.figure()
ax = p3.Axes3D(fig)

# Setting the axes properties
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


# ============================================
# ============= Plot of speed ================
speeds = swarm.get_average_speeds()
plt.figure(figsize=(7,2.5))
plt.plot(speeds, '-', color='slategray')
plt.xlabel('Iterations')
plt.ylabel('Average Speed')
plt.title('Average speed of swarm during its flight')


# ============================================
# =========== Print convergence ==============
if swarm.has_converged:
    print('-- Convergence: {0}\n   Distance to target_hive: {1:.2f}'.format(
        barycenters[-1],
        np.linalg.norm(swarm.end_hive_position - barycenters[-1].reshape(-1,1))
        ))
else:
    print('-- No Convergence.')


# ============================================
# ============= Print density ================
print('-- Density at the end of the simulation: {0:.6f}'.format(swarm.get_density()))


plt.show()
