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
# import math
# import matplotlib
# import matplotlib.animation
# import matplotlib.pyplot
# from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import numpy
# import os
import random
# import scipy
# import sys
# import tensorflow
import time

GeneticAlgorithmsForSwarmParameterTuning = __import__("Genetic Algorithms for Swarm Parameter Tuning")

random.seed(24)
numpy.random.seed(24)
search_space = 100
crossover_type = 2  # [0 = Uniform, 1 = Single-point, 2 = Two-point, k = k-point]
mutation_type = 0  # [0 = Bit, 1 = Flip, 2 = Boundary, 3 = Non-Uniform, 4 = Uniform, 5 = Gaussian, 6 = Shrink]
n = 24
properties = 12
nSpecialisations = 6
ε = 0.1
generations = 1000

scenes = None

population = None
populationFitness = None
populationSpecialisation = None

parameters = None

# multiprocessing
process = None
processReturn = None


def __main__() -> None:
	global population
	global populationFitness
	global populationSpecialisation
	global process
	global processReturn
	
	process = []
	processReturn = multiprocessing.Manager().list()
	for x in range(2 * n):
		processReturn.append([])
	
	__initialise__()
	
	print(str(populationFitness).replace("],", "], \n"))
	
	return


def __initialise__() -> None:
	global population
	global populationFitness
	global populationSpecialisation
	global process
	global processReturn
	
	population = numpy.random.rand(n, properties) * search_space
	
	# "seed" initial population
	population[0][1] = 2
	population[0][2] = 4
	population[0][3] = 8
	population[0][4] = 4
	population[0][5] = 8
	population[0][6] = 1
	population[0][7] = 1
	population[0][8] = 1
	population[0][9] = 1
	population[0][10] = 1
	population[0][11] = 5.4934993590917414392968320820363
	
	for x in range(n):
		p = multiprocessing.Process(
			target=__fitness_multiprocessing__, 
			args=(x, population[x], "scene/scene.json", processReturn)
		)
		process.append(p)
		p.start()
	for x in range(n):
		process[x].join()
	populationFitness = numpy.array(processReturn[:n], dtype=float, copy=True, order=None, subok=False, ndmin=0)
	
	populationSpecialisation = numpy.random.randint(0, nSpecialisations, n, dtype=int)
	
	process = []
	
	return


def __fitness__(candidate_solution: list, scene_file: str) -> (tuple, [float]):
	global parameters
	
	parameters = {
		"boidSize": 0.10922,
		"radii": {
			"separation": candidate_solution[1], 
			"alignment": candidate_solution[2], 
			"cohesion": candidate_solution[3], 
			"predator": candidate_solution[4], 
			"prey": candidate_solution[5]
		},
		"weights": {
			"separation": candidate_solution[6], 
			"alignment": candidate_solution[7], 
			"cohesion": candidate_solution[8], 
			"predator": candidate_solution[9], 
			"predatorBoost": 2.0, 
			"prey": candidate_solution[10], 
			"preyBoost": 2.0
		},
		"maximumSpeed": candidate_solution[11]  # 42
	}
	
	return GeneticAlgorithmsForSwarmParameterTuning.__run__(parameters, scene_file)


def __fitness_multiprocessing__(x: int, candidate_solution: list, scene_file: str, process_return: list):
	process_return[x] = __fitness__(candidate_solution, scene_file)[1]
	
	return


def crossover(a: numpy.ndarray, b: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
	size = min(len(a), len(b))
	
	if crossover_type < 1:
		lhs = numpy.random.randint(1 + 1, size=size) > 0
		rhs = lhs < 1
		
		children = (a * lhs + b * rhs, b * lhs + a * rhs)
	elif crossover_type == 1:
		start_point = random.randint(1, size - 2)
		
		children = (
			numpy.concatenate((a[:start_point], b[start_point:])), numpy.concatenate((b[:start_point], a[start_point:]))
		)
	elif crossover_type == 2:
		start_point = random.randint(1, size - 3)
		mid_point = random.randint(start_point + 1, size - 2)
		
		children = (
			numpy.concatenate((a[0:start_point], b[start_point:mid_point], a[mid_point:])), 
			numpy.concatenate((b[0:start_point], a[start_point:mid_point], b[mid_point:]))
		)
	else:
		return None
	
	return children


def mutation(a: numpy.ndarray) -> numpy.ndarray:
	size = len(a)
	
	if mutation_type == 0:
		bit = numpy.random.rand(size) < (1 / size)
		
		a = (numpy.random.rand(size) * search_space) * (bit > 0) + a * (bit < 1)
	elif mutation_type == 1:
		a = numpy.array(search_space) - a
	elif mutation_type == 2:
		boundary = numpy.random.rand()
		
		if boundary < (1 / 3):
			a = numpy.clip(a, random.random() * search_space, None)  # lower bound
		elif boundary < (2 / 3):
			a = numpy.clip(a, None, random.random() * search_space)  # upper bound
		else:
			a = numpy.clip(a, random.random() * search_space, random.random() * search_space)  # lower and upper bound
	else:
		a = numpy.random.rand(size) * search_space
	
	return a


if __name__ == "__main__":
	__main__()
