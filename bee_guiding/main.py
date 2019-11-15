import numpy as np

from animation import show_swarm_flight, show_swarm_flight_scatter
from bee import Bee, populate_swarm, Swarm
from params import *
from utils import serialize_object

# ======================================================================
# ======================================================================
# ======================================================================

swarm = Swarm(
    OLD_HIVE_POSITION, NEW_HIVE_POSITION, BRAKING_MODE, POSITION_MODE, SCOUT_BEHAVIOUR,
    pheromones_initial_intensity=PHEROMONES_INITIAL_INTENSITY, pheromones_max_radius=PHEROMONES_MAX_RADIUS,
    depop_radius=BRAKING_DEPOP_RADIUS, depop_probability=BRAKING_DEPOP_PROBABILITY,
    slowdown_radius=BRAKING_SLOWDOWN_RADIUS, slowdown_coefficient=BRAKING_SLOWDOWN_COEFFICIENT,
    stationary_radius=BRAKING_STATIONARY_RADIUS
    )
swarm = populate_swarm(
    swarm,
    OLD_HIVE_POSITION, NEW_HIVE_POSITION,
    SWARM_SIZE, PROPORTION_SCOUT,
    15*np.ones((3,1)),
    VISION_RADIUS, V_MAX, SCOUT_SPEED,
    POSITION_MODE,
    d_min=D_MIN, alpha=ALPHA, v_max=V_MAX, a_max=A_MAX, w_ws=W_WS, w_decay=W_DECAY
)


for iteration in range(MAX_ITERATIONS):
    converged = swarm.step(TIME_STEP)
    if converged:
        print('\n----- Swarm converged to new hive before reaching max iterations [{}/{}]-----'.format(
            iteration + 1, MAX_ITERATIONS))
        print(f'Position of the barycenter of the swarm at convergence: {swarm.get_barycenters()[-1]}')
        break
    if ((iteration + 1) % 100 == 0):
        print(f'[{iteration + 1}/{MAX_ITERATIONS}] Iterations')

if SAVE_DATA:
    serialize_object(swarm, SAVE_DATA_NAME)

# ==== DEBUG

# ===========

if GENERATE_VIDEO:
    show_swarm_flight_scatter(
        OLD_HIVE_POSITION, NEW_HIVE_POSITION, swarm.recorded_positions,
        SPACE_DIMENSIONS,
        int(SWARM_SIZE * PROPORTION_SCOUT / 100),
        save=SAVE_FLIGHT, save_filename=SAVE_FLIGHT_NAME
    )
