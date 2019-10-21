import numpy as np

from animation import show_swarm_flight, show_swarm_flight_scatter
from bee import Bee, generate_swarm


# ======================================================================
# ======================= PARAMETERS ===================================
# ======================================================================

# Run details
MAX_ITERATIONS = 100
TIME_STEP = 1

# Swarm Info
SWARM_SIZE = 100
PROPORTION_SCOUT = 30 # %

# Space
SPACE_DIMENSIONS = (400, 400, 400)
OLD_HIVE_POSITION = (200, 200, 200)
NEW_HIVE_POSITION = (200, 4000, 200)

# Bee specific
VISION_RADIUS = 30
SCOUT_SPEED = 1.05

# Bee speed update
D_MIN = 15
ALPHA = 3/4
V_MAX = 1.55
A_MAX = 0.3
W_WS = 0.3
W_DECAY = 0.8

# Save file
SAVE_FLIGHT = True
SAVE_FLIGHT_NAME = 'swarm_basic'


# ======================================================================
# ======================================================================
# ======================================================================


swarm = generate_swarm(
    OLD_HIVE_POSITION, NEW_HIVE_POSITION,
    SWARM_SIZE, PROPORTION_SCOUT,
    15*np.ones((3,1)),
    VISION_RADIUS, V_MAX, SCOUT_SPEED,
    d_min=D_MIN, alpha=ALPHA, v_max=V_MAX, a_max=A_MAX, w_ws=W_WS, w_decay=W_DECAY
)

for iteration in range(MAX_ITERATIONS):
    swarm.step(TIME_STEP)

# ==== DEBUG

# for bee in swarm.bees.values():
#     print(bee.speed)

# print(swarm.recorded_positions[-1])

# ===========

show_swarm_flight_scatter(
    OLD_HIVE_POSITION, NEW_HIVE_POSITION, swarm.recorded_positions,
    SPACE_DIMENSIONS,
    int(SWARM_SIZE * PROPORTION_SCOUT / 100),
    save=SAVE_FLIGHT, save_filename=SAVE_FLIGHT_NAME
)
