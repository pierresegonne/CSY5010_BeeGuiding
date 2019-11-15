# INTRODUCTION

This folder provides a python implementation of the bee swarm evolution model proposed by Janson et Al. in "Honeybee swarms: how do scouts guide a swarm of uninformed bees?".

Some additions are present, such as the possibility to choose between different braking mechanisms.

### Requirements

* Numpy

* Matplotlib

# INSTRUCTIONS

Running the main file ```main.py``` allows to start the simulation. There is no argument parser, due to the sheer number of different parameters that the model rely on. There instead a parameter file, ```params.py```, which contains all parameters of the model and allow different simulations.

# TESTING

Before running experiments, play around with the parameters for a small hive size (<50). Please try to find what is the critical swarm size below which the guiding does not work.

# EXPERIMENTS

To be able to obtain generalized experimental results, every setting tested must be ran three different times.
I would advise to generate a video for the first run (```GENERATE_VIDEO=True```) to make sure that everyhing is going according to what is planned, and then remove it to speed up the process.
To test:

1. Streak behaviour versus subtle guide:

    For both experiments choose swarm ```SWARM_SIZE = 200```, ```OLD_HIVE_POSITION = (200, 200, 200)```, ```NEW_HIVE_POSITION = (200, 500, 200)```, no braking mechanism (leave blank ```BRAKING_MODE=''```) and position everywhere ```POSITION_MODE = 'everywhere'``` and 2 500 iterations (if it converges, it should converge before that: ~2000 iterations)

    To set the streaking behaviour use ```SCOUT_BEHAVIOUR = 'streak'``` for scout behaviour

    To set the subtle guide use ```SCOUT_BEHAVIOUR = 'guide'``` for behaviour and reduce speed to match the speed of other bees. It will probably not converge with 5% scouts. Try to find the % for which there is convergence.

2. Top position vs everywhere:

    To study the influence of starting position, run experiments with previous streak settings, while comparing previous streaking results with new streaking results where position is 'top'

3. Braking Mechanisms:

    *   Depop: set ```BRAKING_MODE='depop'``` with a satisfactory depop radius, I would start with ```BRAKING_DEPOP_RADIUS``` around a 100

    *   Slow down: set ```BRAKING_MODE='slowdown'``` with a satisfactory slowdown radius and coefficient, I would start with ```BRAKING_SLOWDOWN_RADIUS``` around a 100 and ```BRAKING_SLOWDOWN_COEFFICIENT``` around 0.99

    *   Stationary: set ```BRAKING_MODE='stationary'```. Nothing else to set.

    *   Pheromones: set ```BRAKING_MODE='pheromones'``` with appropriate ```PHEROMONES_MAX_RADIUS``` and ```PHEROMONES_INITIAL_INTENSITY```. (To test yourselves).


The runs will save all the swarm data if ```SAVE_DATA=True```. You can then extract all relevant information running the ```analysis.py``` file while providing the swarm object filename (ex: *recordings/swarm_object 2019-11-14.pickle*).

It will plot the trajectory of the barycenter in the planes x,y, x,z and y,z, as well as the evolution of the norm of the speed of the swarm. It will also show in the console the coordinates of the convergence point, if any, with the distance to the old hive.






