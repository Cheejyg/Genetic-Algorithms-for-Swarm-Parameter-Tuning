"""Genetic Algorithms for Swarm Parameter Tuning.py

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

import argparse
import json
# import matplotlib
import matplotlib.animation
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
# import os
import random
# import scipy
# import sys
# import tensorflow
import time

random.seed(24)
numpy.random.seed(24)
dT = 0.1
graph = True
measure = False

inputFilename = None
sceneFilename = None
outputFilename = None
verbosity = None
outputFile = None
output = {}

canvas = None
ax = None
scatter = None

boidSize = None
radiusSeparation = None
radiusAlignment = None
radiusCohesion = None
weightSeparation = None
weightAlignment = None
weightCohesion = None
maximumSpeed = None

dimension = None
width = None
height = None
depth = None
ticks = None
n = None
positons = None
rotations = None
velocities = None

separations = None
alignments = None
cohesions = None


def __main__() -> None:
	global verbosity
	global canvas
	global ax
	global scatter
	global dimension
	global ticks
	global n
	
	__argparse__()
	__global__()
	
	if graph:
		if 0 < dimension < 4:
			canvas = matplotlib.pyplot.figure(1)
			
			if dimension == 1:
				matplotlib.pyplot.xlabel("x")
				matplotlib.pyplot.xlim(-width, width)
				matplotlib.pyplot.ylabel("y")
				matplotlib.pyplot.ylim(-0, 0)
				
				scatter = matplotlib.pyplot.scatter(
					positions[:, 0], numpy.zeros((n, dimension), dtype=float, order=None), s=8, c="Blue", marker="o"
				)
			if dimension == 2:
				matplotlib.pyplot.xlabel("x")
				matplotlib.pyplot.xlim(-width, width)
				matplotlib.pyplot.ylabel("y")
				matplotlib.pyplot.ylim(-height, height)
				
				scatter = matplotlib.pyplot.scatter(
					positions[:, 0], positions[:, 1], s=8, c="Blue", marker="o"
				)
			if dimension == 3:
				ax = canvas.add_subplot(111, projection="3d")
				ax.set_xlabel("x")
				ax.set_ylabel("y")
				ax.set_zlabel("z")
				
				scatter = ax.scatter(
					positions[:, 0], positions[:, 1], positions[:, 2], s=8, c="Blue", marker="o"
				)
			
			animation = matplotlib.animation.FuncAnimation(
				fig=canvas, func=__update__, frames=ticks, init_func=__global__, fargs=(), save_count=(ticks % 1024), 
				interval=100, repeat=False
			)
			
			matplotlib.pyplot.show()
	else:
		for tick in range(ticks):
			__update__(tick)
	
	if verbosity > 0:
		if verbosity > 1:
			outputFile.write(json.dumps(output))
			outputFile.close()
	
	return


def __argparse__() -> None:
	global inputFilename
	global sceneFilename
	global outputFilename
	global verbosity
	
	parser = argparse.ArgumentParser(
		description="Swarming behaviour is based on aggregation of simple drones exhibiting basic instinctive reactions"
		"to stimuli. However, to achieve overall balanced/interesting behaviour the relative importance of these"
		"instincts, as well their internalparameters, must be tuned. In this project, you will learn how to apply"
		"Genetic Programming as means of such tuning, and attempt to achieve a series of non-trivial swarm-level"
		"behaviours.", 
		add_help=True
	)
	parser.add_argument(
		"-i", "--input", type=str, default="parameters.json", 
		help="json parameters file"
	)
	parser.add_argument(
		"-s", "--scene", type=str, default="scene.json", 
		help="json scene file"
	)
	parser.add_argument(
		"-o", "--output", type=str, default="Genetic Algorithms for Swarm Parameter Tuning.log", 
		help="output debug file"
	)
	parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0, help="verbosity")
	
	args = parser.parse_args()
	
	inputFilename = str(args.input)
	sceneFilename = str(args.scene)
	outputFilename = str(args.output)
	verbosity = args.verbosity if args.verbosity else 0
	
	return


def __global__() -> None:
	global inputFilename
	global sceneFilename
	global outputFilename
	global verbosity
	global outputFile
	global output
	global boidSize
	global radiusSeparation
	global radiusAlignment
	global radiusCohesion
	global weightSeparation
	global weightAlignment
	global weightCohesion
	global maximumSpeed
	global dimension
	global width
	global height
	global depth
	global ticks
	global n
	global positions
	global rotations
	global velocities
	global separations
	global alignments
	global cohesions
	
	try:
		with open(inputFilename, "rt") as input_file:
			inputs = json.load(input_file)
		
		boidSize = inputs["boidSize"]
		radiusSeparation = inputs["radii"]["separation"]
		radiusAlignment = inputs["radii"]["alignment"]
		radiusCohesion = inputs["radii"]["cohesion"]
		weightSeparation = inputs["weights"]["separation"]
		weightAlignment = inputs["weights"]["alignment"]
		weightCohesion = inputs["weights"]["cohesion"]
		maximumSpeed = inputs["maximumSpeed"]
	except (FileNotFoundError, json.decoder.JSONDecodeError, UnboundLocalError, KeyError, IndexError, TypeError) as e:
		print("EXCEPTION: %s" % (str(e)))
		
		boidSize = random.random()
		radiusSeparation = random.random()
		radiusAlignment = random.random()
		radiusCohesion = random.random()
		weightSeparation = random.random()
		weightAlignment = random.random()
		weightCohesion = random.random()
		maximumSpeed = random.random()
	radiusSeparation += (radiusSeparation * boidSize) + boidSize
	radiusAlignment += (radiusAlignment * boidSize) + boidSize
	radiusCohesion += (radiusCohesion * boidSize) + boidSize
	
	try:
		with open(sceneFilename, "rt") as scene_file:
			scene = json.load(scene_file)
		
		ticks = scene["ticks"]
		positions = numpy.array(scene["boids"]["positions"], dtype=float, copy=False, order=None, subok=False, ndmin=0)
		rotations = numpy.array(scene["boids"]["rotations"], dtype=float, copy=False, order=None, subok=False, ndmin=0)
		velocities = numpy.array(scene["boids"]["velocities"], dtype=float, copy=False, order=None, subok=False, ndmin=0)
		n = positions.shape[0]
		dimension = positions.shape[1]
		width = scene["window"]["width"]
		height = scene["window"]["height"]
		if dimension > 2:
			depth = scene["window"]["depth"]
		
		if positions.shape[0] != n or positions.shape[1] != dimension:
			raise IndexError
	except (FileNotFoundError, json.decoder.JSONDecodeError, UnboundLocalError, KeyError, IndexError, TypeError) as e:
		print("EXCEPTION: %s" % (str(e)))
		
		dimension = random.randint(1, 3) \
			if dimension is None else dimension
		width = random.randint(10, 300) \
			if width is None else width
		height = random.randint(10, 300) \
			if height is None else height
		if dimension > 2 and depth is None:
			depth = random.randint(10, 300)  # if depth is None else depth
		ticks = random.randint(1, 1024) \
			if ticks is None else ticks
		n = random.randint(2, 2048) \
			if n is None else n
		positions = (min(width, height) / (40 / 9)) * numpy.random.randn(n, dimension) \
			if positions is None else positions
		rotations = numpy.random.rand(n, dimension) \
			if rotations is None else rotations
		velocities = numpy.random.randn(n, dimension) \
			if velocities is None else velocities
		
		positions[0] = numpy.array(
			numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		rotations[0] = numpy.array(
			numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		velocities[0] = numpy.array(
			numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
	
	# OVERRIDE
	verbosity = 1
	dimension = 2
	# n = 4
	ticks = 1000
	if positions.shape[0] != n or positions.shape[1] != dimension:
		print("OVERRIDE")
		
		dimension = random.randint(1, 3) \
			if dimension is None else dimension
		width = random.randint(10, 300) \
			if width is None else width
		height = random.randint(10, 300) \
			if height is None else height
		if dimension > 2 and depth is None:
			depth = random.randint(10, 300)  # if depth is None else depth
		ticks = random.randint(1, 1024) \
			if ticks is None else ticks
		n = random.randint(2, 2048) \
			if n is None else n
		positions = (min(width, height) / (40 / 9)) * numpy.random.randn(n, dimension) \
			if positions is None or positions.shape[0] != n or positions.shape[1] != dimension else positions
		rotations = numpy.random.rand(n, dimension) \
			if rotations is None or rotations.shape[0] != n or rotations.shape[1] != dimension else rotations
		velocities = numpy.random.randn(n, dimension) \
			if velocities is None or velocities.shape[0] != n or velocities.shape[1] != dimension else velocities
		
		positions[0] = numpy.array(
			numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		rotations[0] = numpy.array(
			numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		velocities[0] = numpy.array(
			numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
	
	if verbosity > 0:
		print(
			"------------------------------------------------------------"
			"------------------------------------------------------------"
		)
		print("{")
		print(
			"\t\"n\": %d, "
			"\"ticks\": %d, " % (n, ticks)
		)
		print(
			"\t\"boids\": {\n"
			"\t\t\"positions\": %s, \n"
			"\t\t\"velocities\": %s \n"
			"\t}, "
			% (str(positions.tolist()), str(velocities.tolist()))
		)
		print(
			"\t\"parameters\": {\n"
			"\t\t\"radiuses\": {\n"
			"\t\t\t\"separation\": %f, \n"
			"\t\t\t\"alignment\": %f, \n"
			"\t\t\t\"cohesion\": %f \n"
			"\t\t}, \n"
			"\t\t\"weights\": {\n"
			"\t\t\t\"separation\": %f, \n"
			"\t\t\t\"alignment\": %f, \n"
			"\t\t\t\"cohesion\": %f \n"
			"\t\t}\n"
			"\t}" % (
				radiusSeparation, radiusAlignment, radiusCohesion, 
				weightSeparation, weightAlignment, weightCohesion
			)
		)
		print("}")
		print(
			"------------------------------------------------------------"
			"------------------------------------------------------------"
		)
		if verbosity > 1:
			outputFile = open(outputFilename, "at")
			
			output["n"] = n
			output["ticks"] = ticks
			output["boids"] = {
				"positions": str(positions).replace('\n', ""), 
				"rotations": str(rotations).replace('\n', ""), 
				"velocities": str(velocities).replace('\n', "")
			}
	
	# Normalise
	total_weights = weightSeparation + weightAlignment + weightCohesion
	weightSeparation /= total_weights
	weightAlignment /= total_weights
	weightCohesion /= total_weights
	
	separations = numpy.zeros((n, dimension), dtype=float, order=None)
	alignments = numpy.zeros((n, dimension), dtype=float, order=None)
	cohesions = numpy.zeros((n, dimension), dtype=float, order=None)
	
	return


def __update__(tick: int) -> None:
	global canvas
	global scatter
	global dimension
	global n
	global positions
	global velocities
	global separations
	global alignments
	global cohesions
	
	'''position = None
	position_other = None
	neighbours = None
	c = None'''
	
	# Separation
	for boid in range(n):
		position = positions[boid, :]
		c = numpy.zeros(dimension, dtype=float, order=None)
		for other in range(n):
			if other != boid:
				position_other = positions[other]
				if numpy.linalg.norm(position_other - position) < radiusSeparation:
					c -= position_other - position
		separations[boid] = c
	# Alignment
	for boid in range(n):
		position = positions[boid, :]
		c = numpy.zeros(dimension, dtype=float, order=None)
		neighbours = 0.0
		for other in range(n):
			if other != boid:
				position_other = positions[other]
				velocity_other = velocities[other]
				if numpy.linalg.norm(position_other - position) < radiusAlignment:
					c += velocity_other
					neighbours += 1.0
		alignments[boid] = c if neighbours < 1 else c / neighbours
	# Cohesion
	for boid in range(n):
		position = positions[boid, :]
		c = numpy.copy(position)
		neighbours = 1.0
		for other in range(n):
			if other != boid:
				position_other = positions[other]
				if numpy.linalg.norm(position_other - position) < radiusCohesion:
					c += position_other
					neighbours += 1.0
		cohesions[boid] = c if neighbours < 1 else (c / neighbours) - position
	
	target = (weightSeparation * separations) + (weightAlignment * alignments) + (weightCohesion * cohesions)
	
	velocities += target
	
	velocities = numpy.clip(velocities, -maximumSpeed, maximumSpeed)
	
	positions += dT * velocities
	
	velocities *= 0.9
	
	if graph:
		matplotlib.pyplot.figure(1).clear()
		matplotlib.pyplot.title("Tick %d" % tick)
		matplotlib.pyplot.xlim(-width, width)
		matplotlib.pyplot.ylim(-height, height)
		
		if dimension == 1:
			scatter = matplotlib.pyplot.scatter(
				positions[:, 0], numpy.zeros((n, dimension), dtype=float, order=None), s=8, marker="o"
			)
		if dimension == 2:
			scatter = matplotlib.pyplot.scatter(
				positions[:, 0], positions[:, 1], s=8, marker="o"
			)
		if dimension == 3:
			scatter = ax.scatter(
				positions[:, 0], positions[:, 1], positions[:, 2], s=8, marker="o"
			)
	
	return


if __name__ == "__main__":
	__main__()
