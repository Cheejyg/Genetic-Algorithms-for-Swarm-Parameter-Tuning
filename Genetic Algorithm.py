"""Genetic Algorithm.py

Swarming behaviour is based on aggregation of simple drones exhibiting basic instinctive reactions to stimuli. However, 
to achieve overall balanced/interesting behaviour the relative importance of these instincts, as well their internal
parameters, must be tuned. In this project, you will learn how to apply Genetic Programming as means of such tuning, 
and attempt to achieve a series of non-trivial swarm-level behaviours.
"""

from __future__ import barry_as_FLUFL

__all__ = None
__author__ = "#CHEE JUN YUAN GLENN#"
__copyright__ = "Copyright © 2019, Cheejyg"
__email__ = "CHEE0124@e.ntu.edu.sg"
__license__ = "MIT"
__maintainer__ = "#CHEE JUN YUAN GLENN#"
__status__ = "Development"
__version__ = "1.0"

# import argparse
# import json
# import matplotlib
# import matplotlib.animation
# import matplotlib.pyplot
# from mpl_toolkits.mplot3d import Axes3D
import numpy
# import os
import random
# import scipy
# import sys
# import tensorflow
# import time

GeneticAlgorithmsForSwarmParameterTuning = __import__("Genetic Algorithms for Swarm Parameter Tuning")

random.seed(24)
numpy.random.seed(24)

parameters = None


def __main__() -> None:
	global parameters
	
	parameters = {
		"boidSize": 0.10922,
		"radii": {
			"separation": 2.0,
			"alignment": 4.0,
			"cohesion": 8.0,
			"predator": 4.0,
			"prey": 8.0
		},
		"weights": {
			"separation": 1.0,
			"alignment": 1.0,
			"cohesion": 1.0,
			"predator": 1.0, 
			"predatorBoost": 2.0,
			"prey": 1.0,
			"preyBoost": 2.0
		},
		"maximumSpeed": 5.4934993590917414392968320820363
	}
	
	fitness = GeneticAlgorithmsForSwarmParameterTuning.__run__(parameters, "scene.json")
	
	print(fitness)
	
	return


if __name__ == "__main__":
	__main__()
