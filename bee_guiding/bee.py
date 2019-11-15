import numpy as np

from math import atan, cos, log, pi, sin

# ======================================================================
# ============================= BEES ===================================
# ======================================================================

class Bee:

    def __init__(self, end_hive_position, id, position, speed, vision):

        self.end_hive_position = np.array(end_hive_position).reshape(-1,1)
        self.id = id
        self.iteration = 0
        self.position = position.reshape((-1,1)) # Dimension: SpaceDim*1
        self.reached_end_hive = False
        self.speed = speed.reshape((-1,1)) # Dimension: SpaceDim*1
        self.vision = vision

    def step(self):
        if self.reached_end_hive:
            return
        self.iteration += 1


# ======================================================================


class UninformedBee(Bee):
    def __init__(self, end_hive_position, id, position, speed, vision,
        d_min=4, alpha=0.9, v_max=5, a_max=2, w_ws=2, w_decay=0.9):
        super().__init__(end_hive_position, id, position, speed, vision)

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


    def pheromone_speed(self, pheromones_intensity_from_radius):
        hive_direction = self.end_hive_position - self.position
        return (pheromones_intensity_from_radius(np.linalg.norm(hive_direction))
            * hive_direction / np.linalg.norm(hive_direction))



    def step(self, time_step, neighbours, pheromones_intensity_from_radius=(lambda x: 1)):
        super().step()
        v_align = self.align(neighbours)
        v_cohere = self.cohere(neighbours)
        v_avoid = self.avoid(neighbours)
        v_random = self.random_speed()
        v_pheromone = self.pheromone_speed(pheromones_intensity_from_radius)

        v_new = (self.w_ws*v_align
            + self.w_ws*v_cohere
            + self.w_ws*v_avoid
            + self.w_ws*v_random
            + self.w_ws*v_pheromone)
        if np.linalg.norm(v_new) > self.a_max:
            # v_new = (self.a_max)**(1/v_new.size) * (v_new/abs(v_new)) # In paper
            v_new = v_new * self.a_max / np.linalg.norm(v_new) # My take

        self.speed = self.speed * self.w_decay + v_new

        self.position = self.position + (self.speed * time_step)


# ======================================================================


class Scout(Bee):

    def __init__(self, behaviour, end_hive_position, id, position, speed, speed_norm, start_hive_position, vision):
        super().__init__(end_hive_position, id, position, speed, vision)

        self.behaviour = behaviour
        self.number_scouts = None
        self.speed_norm = speed_norm
        self.start_hive_position = np.array(start_hive_position).reshape(-1,1)
        self.started_moving = False
        self.swarm_size = None

        # `streak` or `around`
        self.__streak_mode = 'streak'
        self.__around_mode = 'around'
        self.__mode = self.__streak_mode

        self.__around_direction = None
        if np.random.random() < 0.5:
            self.__around_direction = -1
        else:
            self.__around_direction = 1

        self.__around_step = 0
        self.__max_around_step = None
        self.__around_angle = None
        self.__center_around_circle = None
        self.__theta = None

        self.__finished_around = 0
        self.__min_iteration_two_around = 30


    def inform_about_swarm(self, swarm_size, number_scouts):
        """
        Adds information about the swarm (total number of scouts & swarm size.
        It is required that scouts have this information to follow the streaking behavior.
        """
        self.swarm_size = swarm_size
        self.number_scouts = number_scouts


    def compute_around_angle_properties(self):
        self.__around_angle = self.speed_norm / (self.swarm_size/2) * 2 # Hoping the swarm span is swarmsize/2
        self.__max_around_step = int(pi / self.__around_angle) + 1 # 1 probably not necessary


    def recompute_speed_towards_end_hive(self):
        new_direction = self.end_hive_position - self.position
        self.speed = self.speed_norm * new_direction / np.linalg.norm(new_direction)


    def step(self, time_step, neighbours, swarm_barycenter):
        super().step()
        swarm_spread = 100 # From observation

        # If streaking behaviour is exhibited, implement the streak and back around
        if self.behaviour == 'streak':

            # Determine if scout is in front of the hive.
            main_axis = np.argmax(self.end_hive_position - self.start_hive_position)
            in_front_of_swarm = (self.position[main_axis] > swarm_barycenter[main_axis])
            circling_radius = swarm_spread/2

            if (
                (len(neighbours) < self.number_scouts * 1.2) # 1.2 is arbitrary
                and (self.__mode == self.__streak_mode)
                and (self.iteration - self.__finished_around >= self.__min_iteration_two_around)
                and in_front_of_swarm
            ):
                # Go back to the back of the swarm
                self.__mode = self.__around_mode
                self.__center_around_circle = self.position - (self.speed/np.linalg.norm(self.speed)) * (circling_radius)
                self.__theta = atan((self.position[2] - self.__center_around_circle[2]) / (self.position[1] - self.__center_around_circle[1]))
                self.__around_step = 0

            if self.__mode == self.__around_mode:
                # If reach the end of the circular trajectory, go back to streak
                if self.__around_step >= self.__max_around_step:
                    self.__mode = self.__streak_mode
                    self.recompute_speed_towards_end_hive()
                    self.__finished_around = self.iteration
                    self.__around_step = 0
                # Pursue
                else:
                    self.position = np.array(
                        [self.__center_around_circle[0] + (circling_radius)*cos((pi/2) + self.__around_direction * (self.__around_step*self.__around_angle)),
                        self.__center_around_circle[1] + (circling_radius)*sin((pi/2) + self.__around_direction * (self.__around_step*self.__around_angle))*cos(self.__theta),
                        self.__center_around_circle[2] + (circling_radius)*sin((pi/2) + self.__around_direction * (self.__around_step*self.__around_angle))*sin(self.__theta)]
                    ).reshape(-1,1)
                    self.__around_step += 1


            if self.__mode == self.__streak_mode:
                # self.recompute_speed_towards_end_hive() # this causes the bees to go stationary. (even though more precise)
                self.position = self.position + (self.speed * time_step)

        # If guiding behaviour, simply goes towards target
        if self.behaviour == 'guide':
            self.recompute_speed_towards_end_hive()
            self.position = self.position + (self.speed * time_step)


# ======================================================================
# ============================ SWARM ===================================
# ======================================================================

class Swarm:
    def __init__(
        self, start_hive_position, end_hive_position, braking_mode, position_mode, scout_behaviour,
        bees=dict(), number_scouts=0, pheromones_initial_intensity=0, pheromones_max_radius=0,
        depop_radius=0, depop_probability=0, slowdown_radius=0, slowdown_coefficient=0, stationary_radius=0
        ):

        self.bees = bees
        self.braking_mode = braking_mode # Braking mode from scouts.
        self.end_hive_position = np.array(end_hive_position).reshape(-1,1)
        self.iteration = 0
        self.has_converged = False
        self.has_converged_iteration = 0
        self.number_scouts = number_scouts
        self.pheromones_initial_intensity = pheromones_initial_intensity
        self.pheromones_max_radius = pheromones_max_radius
        self.pheromones_radius = 0
        self.pheromones_spreading = False
        self.pheromones_spreading_start_iteration = 0
        self.position_mode = position_mode
        self.recorded_positions = [] # memory of all position for all bees. Each element is an array #Bees*SpaceDimension.
        self.scout_behaviour = scout_behaviour # Whether scouts will streak and go around or go straight to new hive.
        self.size = len(bees.keys())
        self.space_dimension = np.array(start_hive_position).size
        self.start_hive_position = np.array(start_hive_position).reshape(-1,1)

        # Braking parameters
        self.depop_radius = depop_radius
        self.depop_probability = depop_probability
        self.slowdown_radius = slowdown_radius
        self.slowdown_coefficient = slowdown_coefficient
        self.stationary_radius = stationary_radius


    def add_bee(self, bee):
        self.size += 1
        self.bees[bee.id] = bee


    def initialize_recording(self):
        new_recording = np.zeros((self.size, self.space_dimension))
        new_recording_index = 0
        for bee_id, bee in self.bees.items():
            new_recording[new_recording_index] = bee.position.reshape((1,-1))
            new_recording_index += 1
        self.recorded_positions.append(new_recording)


    def step(self, time_step):
        iterations_before_start = 50
        # If position mode is top, attribute the scouts their top position after the swarm is formed.
        if self.position_mode == 'top' and self.iteration == iterations_before_start - 1:
            self.reposition_scouts_on_top()

        bee_keys = list(self.bees.keys())
        new_recording = np.zeros((self.size, self.space_dimension))
        new_recording_index = 0
        for bee_id, bee in self.bees.items():
            neighbours = self.get_neighbours(bee)
            if isinstance(bee, Scout):
                if not bee.started_moving: # Allow formation of the swarm before scouts move.
                    if self.iteration > iterations_before_start - 1: # Space out beginning of scouts mouvement.
                        if self.scout_behaviour == 'guide':
                            bee.started_moving = True
                        if np.random.rand() < 0.01: # After 20 steps they should start.
                            bee.started_moving = True
                if bee.started_moving:
                    bee.step(time_step, neighbours, self.get_barycenters()[-1])
            if isinstance(bee, UninformedBee):
                bee.step(time_step, neighbours, pheromones_intensity_from_radius=self.get_pheromones_intensity())
            new_recording[new_recording_index] = bee.position.reshape((1,-1))
            new_recording_index += 1

            # If scout, implement braking.
            if isinstance(bee, Scout):
                self.scout_braking(bee)

        if self.pheromones_spreading:
            self.update_pheromones_radius()

        self.recorded_positions.append(new_recording)

        self.iteration += 1

        # Check if the swarm has reached the target hive
        has_converged = self.has_crossed_target_plane()
        self.has_converged = has_converged
        if has_converged:
            self.has_converged_iteration = self.iteration
        return has_converged


    def get_neighbours(self, bee):
        neighbours = []
        for neighbour_bee_id, neighbour_bee in self.bees.items():
            if neighbour_bee_id != bee.id:
                if np.linalg.norm(bee.position - neighbour_bee.position) <= bee.vision:
                    neighbours.append(neighbour_bee)
        return neighbours


    def inform_scouts_number(self):
        """
        Provides information about the swarm to all scouts.
        """
        # Count total number
        for bee_id, bee in self.bees.items():
            if isinstance(bee, Scout):
                self.number_scouts += 1

        # Inform all scouts
        for bee_id, bee in self.bees.items():
            if isinstance(bee, Scout):
                bee.inform_about_swarm(self.size, self.number_scouts)
                bee.compute_around_angle_properties()


    def get_barycenters(self):
        """
        Get barycenter of the swarm along its flight.
        """
        return np.mean(np.array(self.recorded_positions), axis=1)


    def get_average_speeds(self):
        """
        Get the average speed vector, and its norm of the swarm along its flight.
        """
        speeds = np.zeros(len(self.recorded_positions))
        barycenters = self.get_barycenters()
        for i in range(len(self.recorded_positions)-1):
            speeds[i+1] = np.linalg.norm(barycenters[i+1] - barycenters[i])
        return speeds


    def get_density(self):
        """
        Get density of the swarm. Does so by computing the volume of the smallest
        rectangle parallelepiped that contains the swarm.
        """
        lower_corner = np.inf * np.ones((self.space_dimension,1))
        upper_corner = - np.inf * np.ones((self.space_dimension,1))

        for bee in self.bees.values():
            for dimension_index in range(lower_corner.shape[0]):
                if bee.position[dimension_index] > upper_corner[dimension_index]:
                    upper_corner[dimension_index] = bee.position[dimension_index]
                if bee.position[dimension_index] < lower_corner[dimension_index]:
                    lower_corner[dimension_index] = bee.position[dimension_index]

        volume = np.prod([upper_corner[i]-lower_corner[i] for i in range(lower_corner.shape[0])])

        return self.size / volume


    def has_crossed_target_plane(self):
        """
        Verifies whether the barycenter of the swarm has crossed the plane
        passing through the target hive, perpendicular to the direction from
        the origin hive to the target hive.
        """
        barycenter = self.get_barycenters()[-1].reshape(-1,1)
        direction = self.end_hive_position - self.start_hive_position
        distance_to_plane = (
            np.sum(direction * (self.end_hive_position - barycenter))
            / np.sqrt(np.sum(direction**2))
        )
        return (distance_to_plane < 1e-5)


    def reposition_scouts_on_top(self, proportion=0.50):
        """
        Go through randomly attributed bee positions and reposition the scouts in the top (z axis) ~10%
        """
        scout_ids = []
        z_positions = []
        for bee_id, bee in self.bees.items():
            if isinstance(bee, Scout):
                scout_ids.append(bee_id)
            z_positions.append(bee.position[2])

        z_positions = np.array(z_positions)
        max_z_positions = np.max(z_positions)
        median_z_positions = np.median(z_positions)

        for scout_id in scout_ids:
            new_z_position = (
                median_z_positions
                + (np.random.rand() * proportion * (max_z_positions - median_z_positions))
                + ((1 - proportion) * (max_z_positions - median_z_positions))
            )
            self.bees[scout_id].position[2] = new_z_position


    def scout_braking(self, scout):

        if self.braking_mode == '':
            return

        if self.braking_mode == 'depop':
            if np.linalg.norm(scout.position - self.end_hive_position) < self.depop_radius:
                if np.random.rand() <= self.depop_probability:
                   scout.position = self.end_hive_position
        if self.braking_mode == 'slowdown':
            if np.linalg.norm(scout.position - self.end_hive_position) < self.slowdown_radius:
                scout.speed = scout.speed * self.slowdown_coefficient
        if (self.braking_mode == 'stationary'
            or self.braking_mode == 'slowdown'
            or self.braking_mode == 'depop'
            or self.braking_mode == 'pheromones'):
            if np.linalg.norm(scout.position - self.end_hive_position) < self.stationary_radius:
                scout.speed = np.zeros(scout.speed.shape)
                scout.reached_end_hive = True
        if self.braking_mode == 'pheromones':
            # First scout to reach hive will start the pheromones spread.
            if np.linalg.norm(scout.position - self.end_hive_position) < self.stationary_radius:
                if not self.pheromones_spreading:
                    self.pheromones_spreading = True
                    self.pheromones_spreading_start_iteration = self.iteration
                    print(f'\n-- Pheromones spread start, Iteration {self.iteration}')


    def update_pheromones_radius(self):
        self.pheromones_radius = self.pheromones_max_radius * log(
            ((self.pheromones_max_radius/2) + (self.iteration - self.pheromones_spreading_start_iteration))
            /(self.pheromones_max_radius/2),
            10)


    def get_pheromones_intensity(self):
        """
        Returns a function that computes the intensity of the pheromones based on a radius
        """
        def pheromones_intensity_from_radius(r, i_i, p_r):
            if r > p_r:
                return 0
            else:
                return i_i / (r**2)
        return (lambda r: pheromones_intensity_from_radius(r, self.pheromones_initial_intensity, self.pheromones_radius))




# ======================================================================
# ======================= SWARM GENERATION =============================
# ======================================================================

def populate_swarm(
        swarm,
        start_hive_position, end_hive_position,
        swarm_size, proportion_scout,
        swarm_deviation,
        bee_vision, max_speed, scout_speed,
        position_mode,
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

    # Add all bees to swarm
    for index, id in enumerate(ids):
        # Scouts
        if (index < number_scouts):
            swarm.add_bee(Scout(
                swarm.scout_behaviour, end_hive_position, id, positions[index], scout_speeds[index],
                scout_speed, start_hive_position, bee_vision
                ))
        # Uninformed
        else:
            swarm.add_bee(UninformedBee(
                end_hive_position, id, positions[index], speeds[index], bee_vision, **kwargs
                ))

    swarm.initialize_recording()
    swarm.inform_scouts_number()

    return swarm
