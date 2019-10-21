import numpy as np

class Bee:

    def __init__(self, id, is_scout, vision, position, speed,
        d_min=4, alpha=0.9, v_max=5, a_max=2, w_ws=2, w_decay=0.9):

        self.id = id

        self.is_scout = is_scout

        self.vision = vision

        self.position = position
        self.speed = speed

        self.d_min = d_min
        self.alpha = alpha
        self.v_max = v_max
        self.a_max = a_max
        self.w_ws = w_ws
        self.w_decay = w_decay


    def align(self, neighbours):
        v_align = np.zeros(self.speed.shape)

        if len(neighbours) > 0:
            for neighbour in neighbours:
                v_align = v_align + neighbour.speed
            v_align = v_align * (1/self.v_max) * (1/len(neighbours))

        return v_align

    def cohere(self, neighbours):
        v_cohere = np.zeros(self.speed.shape)

        if len(neighbours) > 0:
            for neighbour in neighbours:
                v_cohere = v_cohere + (neighbour.position - self.position)
            v_cohere = v_cohere * (1/self.vision) * (1/len(neighbours))

        return v_cohere

    def avoid(self, neighbours):
        v_avoid = np.zeros(self.speed.shape)

        neighbours_avoid = []
        for neighbour in neighbours:
            if np.linalg.norm(self.position - neighbour.position) <= self.d_min:
                neighbours_avoid.append(neighbour)

        if len(neighbours_avoid) > 0:
            for neighbour in neighbours_avoid:
                v_avoid = v_avoid + ((self.position - neighbour.position)*((self.d_min/(np.abs(self.position - neighbour.position) + np.ones(self.position.shape)*1e-16)) - 1))
            v_avoid = v_avoid * (1/self.d_min) * (1/len(neighbours_avoid))

            # Non-linearity
            v_avoid = v_avoid/((np.abs(v_avoid)+np.ones(v_avoid.shape)*1e-16)**(self.alpha))

        return v_avoid

    def random_speed(self):
        lbd = 2
        beta = min(1, max(np.random.exponential(1/lbd, 1), 0)) # Numpy uses scale, inverse rate.
        v_random = np.random.randint(2, size=3).reshape((-1,1)) * 2 - 1 # Random from [-1, 1]**2
        v_random = beta * (v_random/np.abs(v_random))
        return v_random

    def step(self, time_step, neighbours):
        if not self.is_scout:
            v_align = self.align(neighbours)
            v_cohere = self.cohere(neighbours)
            v_avoid = self.avoid(neighbours)
            v_random = self.random_speed()

            v_new = self.w_ws*v_align + self.w_ws*v_cohere + self.w_ws*v_avoid + self.w_ws*v_random
            if np.linalg.norm(v_new) > self.a_max:
                # v_new = (self.a_max)**(1/v_new.size) * (v_new/abs(v_new)) # In paper
                v_new = v_new * self.a_max / np.linalg.norm(v_new) # My take

            self.speed = self.speed * self.w_decay + v_new

        else:
            # Basic model, scouts don't update their speed.
            pass

        self.position = self.position + (self.speed * time_step)

class Swarm:
    def __init__(self, bees=dict()):
        self.bees = bees
        self.recorded_positions = []

    def initialize_recording(self):
        bee_keys = list(self.bees.keys())
        new_recording = np.zeros((len(bee_keys), self.bees[bee_keys[0]].position.size))
        new_recording_index = 0
        for bee_id, bee in self.bees.items():
            new_recording[new_recording_index] = bee.position.reshape((1,-1))
            new_recording_index += 1
        self.recorded_positions.append(new_recording)

    def step(self, time_step):
        bee_keys = list(self.bees.keys())
        new_recording = np.zeros((len(bee_keys), self.bees[bee_keys[0]].position.size))
        new_recording_index = 0
        for bee_id, bee in self.bees.items():
            neighbours = self.get_neighbours(bee)
            bee.step(time_step, neighbours)
            new_recording[new_recording_index] = bee.position.reshape((1,-1))
            new_recording_index += 1
        self.recorded_positions.append(new_recording)

    def get_neighbours(self, bee):
        neighbours = []
        for neighbour_bee_id, neighbour_bee in self.bees.items():
            if neighbour_bee_id != bee.id:
                if np.linalg.norm(bee.position - neighbour_bee.position) <= bee.vision:
                    neighbours.append(neighbour_bee)
        return neighbours

def generate_swarm(
        start_hive_position, end_hive_position,
        swarm_size, proportion_scout,
        swarm_deviation,
        bee_vision, max_speed, scout_speed,
        distribution='',
        **kwargs
    ):
    dim = len(start_hive_position)

    # Positions of all bees
    if distribution == 'gaussian':
        positions = np.array(list(map(np.random.normal, start_hive_position, swarm_deviation, [swarm_size] * dim))).T
    else:
        start_position_low = np.array(start_hive_position) - ((swarm_size / 3) / 2) # total cube side n/3
        start_position_high = np.array(start_hive_position) + ((swarm_size / 3) / 2)
        positions = np.array(list(map(np.random.uniform, start_position_low, start_position_high, [swarm_size] * dim))).T
    # Ids of all bees
    ids = np.arange(1, swarm_size + 1)

    # Speed of uninformed bees
    speed_mean = [0 for _ in range(dim)]
    speed_deviation = [3 for _ in range(dim)]
    speeds = np.array(list(map(np.random.normal, speed_mean, speed_deviation, [swarm_size] * dim))).T
    speeds = max_speed * speeds / np.linalg.norm(speeds, axis=1).reshape(-1, 1)

    # Speed of scouts
    number_scouts = int(swarm_size * proportion_scout / 100)
    end_positions = np.tile(end_hive_position, number_scouts).reshape((number_scouts, -1))
    scout_speeds = end_positions - positions[:number_scouts]
    scout_speeds = scout_speed * scout_speeds / np.linalg.norm(scout_speeds, axis=1).reshape(-1, 1)

    swarm = Swarm()

    for index, id in enumerate(ids):
        # Scouts
        if (index < number_scouts):
            swarm.bees[id] = Bee(id, True, bee_vision, positions[index].reshape((-1,1)), scout_speeds[index].reshape((-1,1)), **kwargs)
        # Uninformed
        else:
            swarm.bees[id] = Bee(id, False, bee_vision, positions[index].reshape((-1,1)), speeds[index].reshape((-1,1)), **kwargs)

    swarm.initialize_recording()

    return swarm
