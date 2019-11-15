import numpy as np
import time

from datetime import datetime

# ======================================================================
# ======================= PARAMETERS ===================================
# ======================================================================

# Run details ==========================
MAX_ITERATIONS = 2500
TIME_STEP = 1

# Swarm Info ==========================
SWARM_SIZE = 200
PROPORTION_SCOUT = 5 # %

# Space ==========================
SPACE_DIMENSIONS = (400, 500, 400)
OLD_HIVE_POSITION = (200, 200, 200)
NEW_HIVE_POSITION = (200, 500, 200)

# Bee specific ==========================
VISION_RADIUS = 30
SCOUT_SPEED = 1.55

# Bee speed update ==========================
D_MIN = 15
ALPHA = 3/4
V_MAX = 1.55
A_MAX = 0.3
W_WS = 0.3
W_DECAY = 0.8

# Braking ==========================
# Modes
BRAKING_MODE = '' # 'depop', 'slowdown', 'stationary' or 'pheromones'
SCOUT_BEHAVIOUR = 'streak' # 'streak' or 'guide'
POSITION_MODE = 'everywhere' # 'everywhere' or 'top'

# Radii
BRAKING_PAPER_RADIUS = 100 * np.linalg.norm(
    np.array(OLD_HIVE_POSITION).reshape(-1,1)
    - np.array(NEW_HIVE_POSITION).reshape(-1,1)
    ) / 3800
BRAKING_DEPOP_RADIUS = BRAKING_PAPER_RADIUS * 10
BRAKING_SLOWDOWN_RADIUS = BRAKING_PAPER_RADIUS * 5
BRAKING_STATIONARY_RADIUS = 1

# Misc
BRAKING_DEPOP_PROBABILITY = 0.2
BRAKING_SLOWDOWN_COEFFICIENT = 0.95

# Pheromones ==========================
PHEROMONES_MAX_RADIUS = 100
PHEROMONES_INITIAL_INTENSITY = 10

# Show Flight video or not ==========================
GENERATE_VIDEO = False

# Save file
SAVE_FLIGHT = True
SAVE_FLIGHT_NAME = 'recordings/swarm_flight {}'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M'))
SAVE_DATA = True
SAVE_DATA_NAME = 'recordings/swarm_object {}.pickle'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M'))
