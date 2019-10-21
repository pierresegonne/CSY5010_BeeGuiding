import numpy as np

class Bee:

    def __init__(self, id, is_scout, vision, position, speed,
        d_min=4, alpha=0.9, beta=0.6, v_max=5, a_max=2, w_ws=2, w_decay=0.9):

        self.id = id

        self.is_scout = is_scout

        self.vision = vision

        self.position = position
        self.speed = speed

        self.d_min = d_min
        self.alpha = alpha
        self.beta = beta
        self.v_max = v_max
        self.a_max = a_max
        self.w_ws = w_ws
        self.w_decay = w_decay


    def align(self, neighbours):
        v_align = np.zeros((3,1))
        for neighbour in neighbours:
            v_align = v_align + neighbour.speed
        v_align = v_align * (1/self.v_max) * (1/len(neighbours))
        return v_align

    def cohere(self, neighbours):
        v_cohere = np.zeros((3,1))
        for neighbour in neighbours:
            v_cohere = v_cohere + (self.position - neighbour.position)
        v_cohere = v_cohere * (1/self.vision) * (1/len(neighbours))
        return v_cohere

    def avoid(self, neighbours):
        neighbours_avoid = []
        for neighbour in neighbours:

            if np.linalg.norm(self.position - neighbour.position) <= self.d_min:
                neighbours_avoid.append(neighbour)

        v_avoid = np.zeros((3,1))
        for neighbour in neighbours_avoid:
            v_avoid = v_avoid + ((self.position - neighbour.position)*((self.d_min/(np.abs(self.position - neighbour.position) + np.ones(self.position.shape)*1e-16)) - 1))
        v_avoid = v_avoid * (1/self.d_min) * (1/len(neighbours_avoid))

        v_avoid = v_avoid/((np.abs(v_avoid)+np.ones(v_avoid.shape)*1e-16)**(self.alpha))
        return v_avoid

    def random_speed(self):
        lbd = 2
        v_random = np.random.exponential(1/lbd, (3,1)) # Numpy uses scale, inverse rate.
        v_random = self.beta * (v_random/np.abs(v_random))
        return v_random

    def step(self, time_step, neighbours=[]):
        if not self.is_scout:
            # neighbours = THING.get_neighbours(self.position, self.vision)
            v_align = self.align(neighbours)
            print('align', v_align)
            v_cohere = self.cohere(neighbours)
            print('cohere', v_cohere)
            v_avoid = self.avoid(neighbours)
            print('avoid', v_avoid)
            v_random = self.random_speed()
            print('random', v_random)

            v_new = self.w_ws*v_align + self.w_ws*v_cohere + self.w_ws*v_avoid + self.w_ws*v_random
            print('combined', v_new)
            if np.linalg.norm(v_new) > self.a_max:
                # v_new = (self.a_max)**(1/v_new.size) * (v_new/abs(v_new)) # In paper
                v_new = v_new * self.a_max / np.linalg.norm(v_new) # My take

            self.speed = self.speed * self.w_decay + v_new
            print('new', v_new)
            print('new new', self.speed)
        else:
            self.speed = self.speed # TODO

        self.position = self.position + (self.speed * time_step)
        print('pos', self.position)
