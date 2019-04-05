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
import json
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
n = 100
nParents = 12
genes = 12
nSpecialisations = 6
α = 0.5
β = 0.1
generations = 10000

outputFilename = "Genetic Algorithm.log.json"
verbosity = 1
outputFile = None
output = None

scenes = None
specialisationScenes = [
	"scene/scene_velocities.json", "scene/scene_predators.json", "scene/scene_preys.json", 
	"scene/scene_distances.json", "scene/scene_distancesPredator.json", "scene/scene_distancesPrey.json"
]

population = None
populationFitness = None
populationSpecialisation = None
childrenFitness = None
childrenSpecialisation = None

parameters = None

# multiprocessing
batch_size = int(multiprocessing.cpu_count() / 2)
process = None
processReturn = None


def __main__() -> None:
	global outputFilename
	global verbosity
	global outputFile
	global output
	global population
	global populationFitness
	global populationSpecialisation
	global childrenFitness
	global childrenSpecialisation
	global process
	global processReturn
	
	children = []
	childrenFitness = []
	childrenSpecialisation = []
	
	process = []
	processReturn = multiprocessing.Manager().list()
	for x in range(n + (2 * nParents)):
		processReturn.append([])
	
	output = []
	outputFile = open(outputFilename, "at")
	outputFile.write("[")
	
	__initialise__()
	
	for generation in range(generations):
		population_fitness = numpy.copy(populationFitness)
		population_fitness[numpy.arange(len(population)), populationSpecialisation] = 0
		population_fitness = populationFitness - population_fitness
		population_fitness = population_fitness / numpy.sum(population_fitness, axis=0)  # normalise
		
		alpha_normalisation = numpy.zeros(population_fitness.shape, dtype=float)
		for specialisation in range(nSpecialisations):
			specialisation_population = numpy.where(populationSpecialisation == specialisation)[0]
			alpha_normalisation[specialisation_population, specialisation] += 1 / len(specialisation_population)
		population_fitness = (α * alpha_normalisation) + (1 - α) * population_fitness
		
		for parents in range(nParents):
			a_specialisation, b_specialisation = (
				random.randint(0, nSpecialisations - 1), random.randint(0, nSpecialisations - 1)
			)
			
			if a_specialisation != b_specialisation:
				a = numpy.random.choice(len(population), 1, p=population_fitness[:, a_specialisation])[0]
				b = numpy.random.choice(len(population), 1, p=population_fitness[:, b_specialisation])[0]
			else:
				a, b = numpy.random.choice(len(population), 2, False, p=population_fitness[:, a_specialisation])
			
			a, b = population[a], population[b]
			
			a, b = crossover(a, b)
			a, b = mutation(a), mutation(b)
			
			children.append(a), children.append(b)
			
			a_specialisation, b_specialisation = numpy.random.choice(
				numpy.array([random.randint(0, nSpecialisations - 1), a_specialisation, b_specialisation]), 2, True, 
				p=[β, (1 - β)/2, (1 - β)/2]
			)
			
			childrenSpecialisation.append(a_specialisation), childrenSpecialisation.append(b_specialisation)
			
			p1, p2 = multiprocessing.Process(
				target=__fitness_multiprocessing__, 
				args=(len(children) - 2, a, specialisationScenes[a_specialisation], processReturn)
			), multiprocessing.Process(
				target=__fitness_multiprocessing__, 
				args=(len(children) - 1, b, specialisationScenes[b_specialisation], processReturn)
			)
			process.append(p1), process.append(p2)
			p1.start(), p2.start()
			
			if (2 * parents) % batch_size == 0:
				for p in process:
					p.join()
		
		population = numpy.concatenate((population, children))
		populationSpecialisation = numpy.concatenate((populationSpecialisation, childrenSpecialisation))
		for p in process:
			p.join()
		childrenFitness = numpy.array(processReturn[:len(children)], copy=True)
		populationFitness = numpy.concatenate((populationFitness, childrenFitness))
		
		specialisation_remove = (numpy.unique(
			populationSpecialisation, return_index=False, return_inverse=False, return_counts=True
		)[1] * ((len(population) - n) / len(population))).astype(int)
		population_remove = []
		for specialisation in range(nSpecialisations):
			specialisation_population = numpy.where(populationSpecialisation == specialisation)[0]
			fitness_sort = numpy.argsort(populationFitness[specialisation_population, specialisation], axis=0)
			population_remove += (specialisation_population[fitness_sort][0:specialisation_remove[specialisation]])\
				.tolist()
		population = numpy.delete(population, population_remove, axis=0)
		populationFitness = numpy.delete(populationFitness, population_remove, axis=0)
		populationSpecialisation = numpy.delete(populationSpecialisation, population_remove, axis=0)
		
		children = []
		childrenFitness = []
		childrenSpecialisation = []
		
		output.append({
			"population": population.tolist(), 
			"specialisation": populationSpecialisation.tolist(), 
			"fitness": populationFitness.tolist()
		})
		
		if generation % 20 == 0:
			outputFile.write(json.dumps(output)[1:-1])
			outputFile.write(",")
			outputFile.flush()
			output = []
		
		if verbosity > 0:
			if verbosity == 1:
				print(
					"generation: %d, n: %d" % (generation, len(population))
				)
			elif verbosity == 2:
				print(
					"generation: %d, n: %d\npopulation: \t\t%s\nspecialisation: \t%s\nfitness: \t\t\t%s\n" 
					% (
						generation, len(population), population.tolist(), populationSpecialisation.tolist(), 
						populationFitness.tolist()
					)
				)
	outputFile.write(json.dumps(output)[1:-1])
	outputFile.write("]")
	outputFile.flush()
	outputFile.close()
	
	return


def __initialise__() -> None:
	global population
	global populationFitness
	global populationSpecialisation
	global process
	global processReturn
	
	population = numpy.random.rand(n, genes) * search_space
	
	# "seed" initial population
	population[0][0] = 2
	population[0][1] = 4
	population[0][2] = 8
	population[0][3] = 4
	population[0][4] = 8
	population[0][5] = 1
	population[0][6] = 1
	population[0][7] = 1
	population[0][8] = 1
	population[0][9] = 2
	population[0][10] = 1
	population[0][11] = 2
	
	populationSpecialisation = numpy.random.randint(0, nSpecialisations, n, dtype=int)
	
	for x in range(n):
		p = multiprocessing.Process(
			target=__fitness_multiprocessing__, 
			args=(x, population[x], specialisationScenes[populationSpecialisation[x]], processReturn)
		)
		process.append(p)
		p.start()
		
		if x % batch_size == 0:
			for p in process:
				p.join()
	
	for p in process:
		p.join()
	
	populationFitness = numpy.array(processReturn[:n], dtype=float, copy=True, order=None, subok=False, ndmin=0)
	
	
	process = []
	
	return


def __fitness__(candidate_solution: list, scene_file: str) -> (tuple, [float]):
	global parameters
	
	parameters = {
		"boidSize": 0.10922,
		"radii": {
			"separation": candidate_solution[0], 
			"alignment": candidate_solution[1], 
			"cohesion": candidate_solution[2], 
			"predator": candidate_solution[3], 
			"prey": candidate_solution[4]
		},
		"weights": {
			"separation": candidate_solution[5], 
			"alignment": candidate_solution[6], 
			"cohesion": candidate_solution[7], 
			"predator": candidate_solution[8], 
			"predatorBoost": candidate_solution[9], 
			"prey": candidate_solution[10], 
			"preyBoost": candidate_solution[11]
		},
		"maximumSpeed": 42  # candidate_solution[11]
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
		start_point = random.randint(0, size)
		
		children = (
			numpy.concatenate((a[:start_point], b[start_point:])), numpy.concatenate((b[:start_point], a[start_point:]))
		)
	elif crossover_type == 2:
		start_point = random.randint(0, size)
		mid_point = random.randint(start_point, size)
		
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
		
		child = (numpy.random.rand(size) * search_space) * (bit > 0) + a * (bit < 1)
	elif mutation_type == 1:
		child = numpy.array(search_space) - a
	elif mutation_type == 2:
		boundary = numpy.random.rand()
		
		if boundary < (1 / 3):
			child = numpy.clip(a, random.random() * search_space, None)  # lower bound
		elif boundary < (2 / 3):
			child = numpy.clip(a, None, random.random() * search_space)  # upper bound
		else:
			child = numpy.clip(a, random.random() * search_space, random.random() * search_space)  # lower and upper bound
	else:
		child = numpy.random.rand(size) * search_space
	
	return child


if __name__ == "__main__":
	__main__()
