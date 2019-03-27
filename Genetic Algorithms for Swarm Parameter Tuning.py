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
# import time

random.seed(24)
numpy.random.seed(24)
dT = 0.1
boundary_type = 1  # [1 = Bound (Velocity), 2 = Wrap (Position)]
animation = True
animation_type = 1 if animation else None  # [1 = Normal (Standard), 2 = History (Path)]
measure = True

inputFilename = None
sceneFilename = None
outputFilename = None
verbosity = None
outputFile = None
output = None

canvas = None
ax = None
scatter = None
scatterPredators = None
scatterPreys = None

boidSize = None
radiusSeparationSquared = None
radiusAlignmentSquared = None
radiusCohesionSquared = None
radiusPredatorSquared = None
radiusPreySquared = None
weightSeparation = None
weightAlignment = None
weightCohesion = None
weightPredator = None
weightPredatorBoost = None
weightPrey = None
weightPreyBoost = None
maximumSpeed = None
maximumSpeedSquared = None

dimension = None
width = None
height = None
depth = None
ticks = None
n = None
nPredators = None
nPreys = None
positions = None
positionsPredator = None
positionsPrey = None
rotations = None
rotationsPredator = None
rotationsPrey = None
velocities = None
velocitiesPredator = None
velocitiesPrey = None
typePredator = None
typePrey = None
waypointsPredator = None

matrixSeparations = None
matrixAlignments = None
matrixCohesions = None
matrixPredators = None
matrixPreys = None

separations = None
alignments = None
cohesions = None
predator = None
prey = None

differences = None
differencesPredator = None
differencesPrey = None
distances = None
distancesPredator = None
distancesPrey = None

predatorsWaypoint = None

measurement = None


def __main__() -> None:
	global verbosity
	global canvas
	global ax
	global scatter
	global scatterPredators
	global scatterPreys
	global dimension
	global ticks
	global n
	
	__argparse__()
	__global__()
	
	if animation:
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
				scatterPredators = matplotlib.pyplot.scatter(
					positionsPredator[:, 0], numpy.zeros((nPredators, dimension), dtype=float, order=None), 
					s=8, c="Red", marker="s"
				)
				scatterPreys = matplotlib.pyplot.scatter(
					positionsPrey[:, 0], numpy.zeros((nPreys, dimension), dtype=float, order=None), 
					s=8, c="Green", marker="^"
				)
			elif dimension == 2:
				matplotlib.pyplot.xlabel("x")
				matplotlib.pyplot.xlim(-width, width)
				matplotlib.pyplot.ylabel("y")
				matplotlib.pyplot.ylim(-height, height)
				
				scatter = matplotlib.pyplot.scatter(
					positions[:, 0], positions[:, 1], s=8, c="Blue", marker="o"
				)
				scatterPredators = matplotlib.pyplot.scatter(
					positionsPredator[:, 0], positionsPredator[:, 1], s=8, c="Red", marker="s"
				)
				scatterPreys = matplotlib.pyplot.scatter(
					positionsPrey[:, 0], positionsPrey[:, 1], s=8, c="Green", marker="^"
				)
			elif dimension == 3:
				ax = canvas.add_subplot(111, projection="3d")
				ax.set_xlabel("x")
				ax.set_xlim(-width, width)
				ax.set_ylabel("y")
				ax.set_ylim(-height, height)
				ax.set_zlabel("z")
				ax.set_zlim(-depth, depth)
				
				scatter = ax.scatter(
					positions[:, 0], positions[:, 1], positions[:, 2], s=8, c="Blue", marker="o"
				)
				scatterPredators = ax.scatter(
					positionsPredator[:, 0], positionsPredator[:, 1], positionsPredator[:, 2], s=8, c="Red", marker="s"
				)
				scatterPreys = ax.scatter(
					positionsPrey[:, 0], positionsPrey[:, 1], positionsPrey[:, 2], s=8, c="Green", marker="^"
				)
			
			func_animation = matplotlib.animation.FuncAnimation(
				fig=canvas, func=__update__, frames=ticks, init_func=__global__, fargs=(), save_count=(1024 % ticks), 
				interval=1, repeat=False
			)
			
			matplotlib.pyplot.show()
	else:
		for tick in range(ticks):
			__update__(tick)
	
	try:
		print(str((
			(
				"velocities", 
				"predators", 
				"preys", 
				"distances", 
				"distancesPredator", 
				"distancesPrey"
			), 
			output["fitness"]
		)).replace("), [", "), \n["))
	except (UnboundLocalError, KeyError, IndexError) as e:
		print(str((
			(
				"velocities", 
				"predators", 
				"preys", 
				"distances", 
				"distancesPredator", 
				"distancesPrey"
			), 
			e
		)).replace("), [", "), \n["))
	
	if verbosity > 0:
		if verbosity > 1:
			json.dump(output, outputFile)
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
		"-s", "--scene", type=str, default="scene/scene.json", 
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
	global radiusSeparationSquared
	global radiusAlignmentSquared
	global radiusCohesionSquared
	global radiusPredatorSquared
	global radiusPreySquared
	global weightSeparation
	global weightAlignment
	global weightCohesion
	global weightPredator
	global weightPredatorBoost
	global weightPrey
	global weightPreyBoost
	global maximumSpeed
	global maximumSpeedSquared
	global dimension
	global width
	global height
	global depth
	global ticks
	global n
	global nPredators
	global nPreys
	global positions
	global positionsPredator
	global positionsPrey
	global rotations
	global rotationsPredator
	global rotationsPrey
	global velocities
	global velocitiesPredator
	global velocitiesPrey
	global typePredator
	global typePrey
	global waypointsPredator
	global separations
	global alignments
	global cohesions
	global predator
	global prey
	global predatorsWaypoint
	global measurement
	
	output = {}
	
	try:
		with open(inputFilename, "rt") as input_file:
			inputs = json.load(input_file)
		
		boidSize = inputs["boidSize"]
		radius_separation = inputs["radii"]["separation"]
		radius_alignment = inputs["radii"]["alignment"]
		radius_cohesion = inputs["radii"]["cohesion"]
		radius_predator = inputs["radii"]["predator"]
		radius_prey = inputs["radii"]["prey"]
		weightSeparation = inputs["weights"]["separation"]
		weightAlignment = inputs["weights"]["alignment"]
		weightCohesion = inputs["weights"]["cohesion"]
		weightPredator = inputs["weights"]["predator"]
		weightPredatorBoost = inputs["weights"]["predatorBoost"]
		weightPrey = inputs["weights"]["prey"]
		weightPreyBoost = inputs["weights"]["preyBoost"]
		maximumSpeed = inputs["maximumSpeed"]
	except (FileNotFoundError, json.decoder.JSONDecodeError, UnboundLocalError, KeyError, IndexError, TypeError) as e:
		if verbosity > 0:
			print("EXCEPTION: %s" % (str(e)))
		
		boidSize = random.random() * 10
		radius_separation = random.random() + 1
		radius_alignment = random.random() * 100
		radius_cohesion = random.random() * 100
		radius_predator = random.random() * 100
		radius_prey = random.random() * 100
		weightSeparation = random.random()
		weightAlignment = random.random()
		weightCohesion = random.random()
		weightPredator = random.random()
		weightPredatorBoost = random.random() + 1
		weightPrey = random.random()
		weightPreyBoost = random.random() + 1
		maximumSpeed = random.random() * 100
	radiusSeparationSquared = (radius_separation + (radius_separation * boidSize) + boidSize) ** 2
	radiusAlignmentSquared = (radius_alignment + (radius_alignment * boidSize) + boidSize) ** 2
	radiusCohesionSquared = (radius_cohesion + (radius_cohesion * boidSize) + boidSize) ** 2
	radiusPredatorSquared = (radius_predator + (radius_predator * boidSize) + boidSize) ** 2
	radiusPreySquared = (radius_prey + (radius_prey * boidSize) + boidSize) ** 2
	maximumSpeedSquared = maximumSpeed ** 2
	
	try:
		with open(sceneFilename, "rt") as scene_file:
			scene = json.load(scene_file)
		
		ticks = scene["ticks"]
		positions = numpy.array(
			scene["boids"]["positions"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		positionsPredator = numpy.array(
			scene["predators"]["positions"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		positionsPrey = numpy.array(
			scene["preys"]["positions"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		rotations = numpy.array(
			scene["boids"]["rotations"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		rotationsPredator = numpy.array(
			scene["predators"]["rotations"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		rotationsPrey = numpy.array(
			scene["preys"]["rotations"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		velocities = numpy.array(
			scene["boids"]["velocities"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		velocitiesPredator = numpy.array(
			scene["predators"]["velocities"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		velocitiesPrey = numpy.array(
			scene["preys"]["velocities"], dtype=float, copy=False, order=None, subok=False, ndmin=0
		)
		n = positions.shape[0]
		nPredators = positionsPredator.shape[0]
		nPreys = positionsPrey.shape[0]
		dimension = positions.shape[1]
		width = scene["window"]["width"]
		height = scene["window"]["height"]
		if dimension > 2:
			depth = scene["window"]["depth"]
		typePredator = scene["predators"]["type"]
		typePrey = scene["preys"]["type"]
		if typePredator == 2:
			waypointsPredator = numpy.array(
				scene["predators"]["waypoints"], dtype=float, copy=False, order=None, subok=False, ndmin=0
			)
		
		if nPredators < 1:
			positionsPredator.shape = (0, dimension)
			rotationsPredator.shape = (0, dimension)
			velocitiesPredator.shape = (0, dimension)
		if nPreys < 1:
			positionsPrey.shape = (0, dimension)
			rotationsPrey.shape = (0, dimension)
			velocitiesPrey.shape = (0, dimension)
		
		if positions.shape[0] != n or positions.shape[1] != dimension \
			or positionsPredator.shape[0] != nPredators or positionsPredator.shape[1] != dimension \
			or positionsPrey.shape[0] != nPreys or positionsPrey.shape[1] != dimension:
				raise IndexError
	except (FileNotFoundError, json.decoder.JSONDecodeError, UnboundLocalError, KeyError, IndexError, TypeError) as e:
		if verbosity > 0:
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
		nPredators = random.randint(0, round(n * 0.1)) \
			if nPredators is None else nPredators
		nPreys = random.randint(0, round(n * 0.1)) \
			if nPreys is None else nPreys
		positions = (min(width, height) / (40 / 9)) * numpy.random.randn(n, dimension) \
			if positions is None else positions
		positionsPredator = (min(width, height) / (40 / 9)) * numpy.random.randn(nPredators, dimension) \
			if positionsPredator is None else positionsPredator
		positionsPrey = (min(width, height) / (40 / 9)) * numpy.random.randn(nPreys, dimension) \
			if positionsPrey is None else positionsPrey
		rotations = numpy.random.rand(n, dimension) \
			if rotations is None else rotations
		rotationsPredator = numpy.random.rand(nPredators, dimension) \
			if rotationsPredator is None else rotationsPredator
		rotationsPrey = numpy.random.rand(nPreys, dimension) \
			if rotationsPrey is None else rotationsPrey
		velocities = numpy.random.randn(n, dimension) \
			if velocities is None else velocities
		velocitiesPredator = numpy.random.randn(nPredators, dimension) \
			if velocitiesPredator is None else velocitiesPredator
		velocitiesPrey = numpy.random.randn(nPreys, dimension) \
			if velocitiesPrey is None else velocitiesPrey
		typePredator = random.randint(1, 2) \
			if typePredator is None else typePredator
		typePrey = random.randint(1, 2) \
			if typePrey is None else typePrey
		waypointsPredator = (
			numpy.random.rand(nPredators, random.randint(2, 24), dimension) * (2 * min(width, height))
		) - min(width, height) \
			if typePredator == 2 and waypointsPredator is None else waypointsPredator
		
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
	if __name__ == "__main__":
		# verbosity = 2
		dimension = 2
		# n = 400
		# nPredators = 4
		# nPreys = 4
		# ticks = 1024
	if positions.shape[0] != n or positions.shape[1] != dimension \
		or positionsPredator.shape[0] != nPredators or positionsPredator.shape[1] != dimension \
		or positionsPrey.shape[0] != nPreys or positionsPrey.shape[1] != dimension:
			if verbosity > 0:
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
			nPredators = random.randint(0, round(n * 0.1)) \
				if nPredators is None else nPredators
			nPreys = random.randint(0, round(n * 0.1)) \
				if nPreys is None else nPreys
			positions = (min(width, height) / (40 / 9)) * numpy.random.randn(n, dimension) \
				if positions is None or positions.shape[0] != n or positions.shape[1] != dimension else positions
			positionsPredator = (min(width, height) / (40 / 9)) * numpy.random.randn(nPredators, dimension) \
				if positionsPredator is None or positionsPredator.shape[0] != nPredators \
				or positionsPredator.shape[1] != dimension else positionsPredator
			positionsPrey = (min(width, height) / (40 / 9)) * numpy.random.randn(nPreys, dimension) \
				if positionsPrey is None or positionsPrey.shape[0] != nPreys \
				or positionsPrey.shape[1] != dimension else positionsPrey
			rotations = numpy.random.rand(n, dimension) \
				if rotations is None or rotations.shape[0] != n or rotations.shape[1] != dimension else rotations
			rotationsPredator = numpy.random.rand(nPredators, dimension) \
				if rotationsPredator is None or rotationsPredator.shape[0] != nPredators \
				or rotationsPredator.shape[1] != dimension else rotationsPredator
			rotationsPrey = numpy.random.rand(nPreys, dimension) \
				if rotationsPrey is None or rotationsPrey.shape[0] != nPreys \
				or rotationsPrey.shape[1] != dimension else rotationsPrey
			velocities = numpy.random.randn(n, dimension) \
				if velocities is None or velocities.shape[0] != n or velocities.shape[1] != dimension else velocities
			velocitiesPredator = numpy.random.randn(nPredators, dimension) \
				if velocitiesPredator is None or velocitiesPredator.shape[0] != nPredators \
				or velocitiesPredator.shape[1] != dimension else velocitiesPredator
			velocitiesPrey = numpy.random.randn(nPreys, dimension) \
				if velocitiesPrey is None or velocitiesPrey.shape[0] != nPreys \
				or velocitiesPrey.shape[1] != dimension else velocitiesPrey
			typePredator = random.randint(1, 2) \
				if typePredator is None else typePredator
			typePrey = random.randint(1, 2) \
				if typePrey is None else typePrey
			if typePredator == 2:
				waypointsPredator = (
					numpy.random.rand(nPredators, random.randint(2, 24), dimension) * (2 * min(width, height))
				) - min(width, height) \
					if waypointsPredator is None or waypointsPredator.shape[0] != nPredators \
					or waypointsPredator.shape[2] != dimension else waypointsPredator
			
			positions[0] = numpy.array(
				numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, 
				ndmin=0
			)
			rotations[0] = numpy.array(
				numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, 
				ndmin=0
			)
			velocities[0] = numpy.array(
				numpy.ones(dimension, dtype=float, order=None), dtype=float, copy=False, order=None, subok=False, 
				ndmin=0
			)
	
	if verbosity > 0:
		print(
			"------------------------------------------------------------"
			"------------------------------------------------------------"
		)
		print("{")
		print(
			"\t\"n\": %d, "
			"\"nPredators\": %d, "
			"\"nPreys\": %d, "
			"\"ticks\": %d, " % (n, nPredators, nPreys, ticks)
		)
		print(
			"\t\"boids\": {\n"
			"\t\t\"positions\": %s, \n"
			"\t\t\"velocities\": %s \n"
			"\t}, "
			% (str(positions.tolist()), str(velocities.tolist()))
		)
		if nPredators > 0:
			print(
				"\t\"predators\": {\n"
				"\t\t\"type\": %s, \n"
				"\t\t\"positions\": %s, \n"
				"\t\t\"velocities\": %s \n"
				"\t}, "
				% (typePredator, str(positionsPredator.tolist()), str(velocitiesPredator.tolist()))
			)
		if nPreys > 0:
			print(
				"\t\"preys\": {\n"
				"\t\t\"type\": %s, \n"
				"\t\t\"positions\": %s, \n"
				"\t\t\"velocities\": %s \n"
				"\t}, "
				% (typePrey, str(positionsPrey.tolist()), str(velocitiesPrey.tolist()))
			)
		print(
			"\t\"parameters\": {\n"
			"\t\t\"boidSize\": %f,\n"
			"\t\t\"radiuses\": {\n"
			"\t\t\t\"separation\": %f, \n"
			"\t\t\t\"alignment\": %f, \n"
			"\t\t\t\"cohesion\": %f, \n"
			"\t\t\t\"predator\": %f, \n"
			"\t\t\t\"prey\": %f\n"
			"\t\t}, \n"
			"\t\t\"weights\": {\n"
			"\t\t\t\"separation\": %f, \n"
			"\t\t\t\"alignment\": %f, \n"
			"\t\t\t\"cohesion\": %f, \n"
			"\t\t\t\"predator\": %f, \n"
			"\t\t\t\"predatorBoost\": %f, \n"
			"\t\t\t\"prey\": %f\n"
			"\t\t\t\"preyBoost\": %f\n"
			"\t\t}, \n"
			"\t\t\"maximumSpeed\": %f\n"
			"\t}" % (
				boidSize, 
				radius_separation, radius_alignment, radius_cohesion, radius_predator, radius_prey, 
				weightSeparation, weightAlignment, weightCohesion, weightPredator, weightPredatorBoost, weightPrey, 
				weightPreyBoost, 
				maximumSpeed
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
			output["nPredators"] = nPredators
			output["nPreys"] = nPreys
			output["ticks"] = ticks
			output["boids"] = {
				"positions": str(positions).replace('\n', ""), 
				"rotations": str(rotations).replace('\n', ""), 
				"velocities": str(velocities).replace('\n', "")
			}
			output["predators"] = {
				"type": typePredator, 
				"positions": str(positionsPredator).replace('\n', ""), 
				"rotations": str(rotationsPredator).replace('\n', ""), 
				"velocities": str(velocitiesPredator).replace('\n', "")
			}
			output["preys"] = {
				"type": typePrey, 
				"positions": str(positionsPrey).replace('\n', ""), 
				"rotations": str(rotationsPrey).replace('\n', ""), 
				"velocities": str(velocitiesPrey).replace('\n', "")
			}
			output["measurementFitness"] = []
			output["measurement"] = []
			output["fitness"] = []
	
	'''# Normalise
	total_weights = \
		weightSeparation + weightAlignment + weightCohesion \
		+ weightPredator + weightPrey
	weightSeparation /= total_weights
	weightAlignment /= total_weights
	weightCohesion /= total_weights
	weightPredator /= total_weights
	weightPrey /= total_weights'''
	
	separations = numpy.zeros((n, dimension), dtype=float, order=None)
	alignments = numpy.zeros((n, dimension), dtype=float, order=None)
	cohesions = numpy.zeros((n, dimension), dtype=float, order=None)
	predator = numpy.zeros((n, dimension), dtype=float, order=None)
	prey = numpy.zeros((n, dimension), dtype=float, order=None)
	
	predatorsWaypoint = [0]
	
	measurement = [] if measure else None
	
	return


def __update__(tick: int) -> None:
	global scatter
	global scatterPredators
	global scatterPreys
	global dimension
	global width
	global height
	global depth
	global n
	global nPredators
	global nPreys
	global positions
	global positionsPredator
	global positionsPrey
	global velocities
	global velocitiesPredator
	global velocitiesPrey
	global matrixSeparations
	global matrixAlignments
	global matrixCohesions
	global matrixPredators
	global matrixPreys
	global separations
	global alignments
	global cohesions
	global predator
	global prey
	global differences
	global differencesPredator
	global differencesPrey
	global distances
	global distancesPredator
	global distancesPrey
	global predatorsWaypoint
	
	differences = positions - positions.reshape(n, 1, dimension)
	differencesPredator = positionsPredator - positions.reshape(n, 1, dimension)
	differencesPrey = positionsPrey - positions.reshape(n, 1, dimension)
	distances = numpy.einsum("...i,...i", differences, differences)
	distancesPredator = numpy.einsum("...i,...i", differencesPredator, differencesPredator)
	distancesPrey = numpy.einsum("...i,...i", differencesPrey, differencesPrey)
	
	# Separation
	matrixSeparations = (distances < radiusSeparationSquared) * (distances != 0)
	separations = numpy.nan_to_num(
		numpy.sum(differences * matrixSeparations.reshape(n, n, 1), axis=1) * -1, 
		copy=False
	)
	# Alignment
	matrixAlignments = (distances < radiusAlignmentSquared) * (distances != 0)
	alignments = numpy.nan_to_num(
		numpy.sum(
			matrixAlignments.reshape(n, n, 1) * numpy.repeat(velocities.reshape(1, n, dimension), n, axis=0), axis=1
		) / numpy.sum(matrixAlignments, axis=0).reshape(n, 1), 
		copy=False
	)
	# Cohesion
	matrixCohesions = (distances < radiusCohesionSquared) * (distances != 0)
	cohesions = numpy.nan_to_num(
		(positions + numpy.sum(
			matrixCohesions.reshape(n, n, 1) * numpy.repeat(positions.reshape(1, n, dimension), n, axis=0), 
			axis=1
		)) / (numpy.ones(n) + numpy.sum(matrixCohesions, axis=0)).reshape(n, 1), 
		copy=False
	) - positions
	# Predator
	matrixPredators = (distancesPredator < radiusPredatorSquared)
	predator = numpy.nan_to_num(
		numpy.sum(
			differencesPredator * numpy.repeat(matrixPredators.reshape(n, nPredators, 1), dimension, axis=2), 
			axis=1
		) * -1, 
		copy=False
	)
	# Prey
	matrixPreys = (distancesPrey < radiusPreySquared)
	if nPreys > 0:
		prey = numpy.nan_to_num(
			differencesPrey[numpy.arange(n), numpy.argmin(distancesPrey, axis=1)]
			* (numpy.sum(matrixPreys, axis=1) > 0).reshape(n, 1), 
			copy=False
		)
	
	# Others
	# Predators
	if typePredator == 1:
		'''differencesPredator = positions - positionsPredator.reshape(nPredators, 1, dimension)
		distancesPredator = numpy.einsum("...i,...i", differencesPredator, differencesPredator)
		predators = numpy.nan_to_num(positions[numpy.argmin(distancesPredator, axis=1)] - positionsPredator, copy=False)'''
		predators = numpy.nan_to_num(positions[numpy.argmin(distancesPredator, axis=0)] - positionsPredator, copy=False)
	elif typePredator == 2:
		predators = waypointsPredator[0][predatorsWaypoint[0]] - positionsPredator
		if round(waypointsPredator[0][predatorsWaypoint[0]][0]) == round(positionsPredator[0][0]) \
			and round(waypointsPredator[0][predatorsWaypoint[0]][1]) == round(positionsPredator[0][1]):
				predatorsWaypoint[0] = (predatorsWaypoint[0] + 1) % len(waypointsPredator[0])
	# Preys
	if tick % 200 == 0:
		if typePrey == 1:
			if dimension == 1:
				positionsPrey = numpy.random.uniform(-width, width, (nPreys, 1))
			elif dimension == 2:
				positionsPrey = numpy.concatenate(
					(
						numpy.random.uniform(-width, width, (nPreys, 1)), 
						numpy.random.uniform(-height, height, (nPreys, 1))
					), 
					axis=1
				)
			elif dimension == 3:
				positionsPrey = numpy.concatenate(
					(
						numpy.random.uniform(-width, width, (nPreys, 1)), 
						numpy.random.uniform(-width, width, (nPreys, 1)), 
						numpy.random.uniform(-depth, depth, (nPreys, 1))
					), 
					axis=1
				)
		elif typePrey == 2:
			preyY = None
			preyM = None
			preyX = None
			preyC = None
			preyR = None
			if dimension == 1:
				positionsPrey = numpy.random.uniform(-width, width, (nPreys, 1))
			elif dimension == 2:
				preyY = None
				preyM = (
					waypointsPredator[0][(predatorsWaypoint[0] + 1) % len(waypointsPredator[0])][1] 
					- waypointsPredator[0][predatorsWaypoint[0]][1]
				) / (
					waypointsPredator[0][(predatorsWaypoint[0] + 1) % len(waypointsPredator[0])][0] 
					- waypointsPredator[0][predatorsWaypoint[0]][0]
				)
				preyX = random.uniform(-width, width)
				preyC = 0
				preyY = preyM * preyX + preyC
				preyR = random.randint(0, int(min(width, height) / 4))
				
				positionsPrey = numpy.concatenate(
					(
						numpy.random.uniform(preyX - preyR, preyX + preyR, (nPreys, 1)), 
						numpy.random.uniform(preyY - preyR, preyY + preyR, (nPreys, 1))
					), 
					axis=1
				)
			elif dimension == 3:
				positionsPrey = numpy.concatenate(
					(
						numpy.random.uniform(-width, width, (nPreys, 1)), 
						numpy.random.uniform(-width, width, (nPreys, 1)), 
						numpy.random.uniform(-depth, depth, (nPreys, 1))
					), 
					axis=1
				)
	
	# ------------------------------------------------------------------------------------------------------------------------
	'''position = None
	position_other = None
	neighbours = None
	c = None'''
	# ------------------------------------------------------------------------------------------------------------------------
	'''
	# Separation
	for boid in range(n):
		position = positions[boid, :]
		c = numpy.zeros(dimension, dtype=float, order=None)
		for other in range(n):
			if other != boid:
				position_other = positions[other]
				difference = position_other - position
				if numpy.einsum("...i,...i", difference, difference) < radiusSeparationSquared:
					c -= difference
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
				if numpy.einsum(
					"...i,...i", position_other - position, position_other - position
				) < radiusAlignmentSquared:
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
				if numpy.einsum(
					"...i,...i", position_other - position, position_other - position
				) < radiusCohesionSquared:
					c += position_other
					neighbours += 1.0
		cohesions[boid] = c if neighbours < 1 else (c / neighbours) - position
	'''
	# ------------------------------------------------------------------------------------------------------------------------
	
	# Normalise
	separations = numpy.nan_to_num(
		separations / numpy.sqrt(numpy.einsum("...i,...i", separations, separations).reshape(n, 1)), copy=False
	)
	alignments = numpy.nan_to_num(
		alignments / numpy.sqrt(numpy.einsum("...i,...i", alignments, alignments).reshape(n, 1)), copy=False
	)
	cohesions = numpy.nan_to_num(
		cohesions / numpy.sqrt(numpy.einsum("...i,...i", cohesions, cohesions).reshape(n, 1)), copy=False
	)
	predator = numpy.nan_to_num(
		predator / numpy.sqrt(numpy.einsum("...i,...i", predator, predator).reshape(n, 1)), copy=False
	)
	prey = numpy.nan_to_num(
		prey / numpy.sqrt(numpy.einsum("...i,...i", prey, prey).reshape(n, 1)), copy=False
	)
	# Normalise Others
	predators = numpy.nan_to_num(
		predators / numpy.sqrt(numpy.einsum("...i,...i", predators, predators).reshape(nPredators, 1)), copy=False
	)
	'''preys = numpy.nan_to_num(
		preys / numpy.sqrt(numpy.einsum("...i,...i", preys, preys).reshape(nPreys, 1)), copy=False
	)'''
	
	target = \
		(
			(weightSeparation * separations) + (weightAlignment * alignments) + (weightCohesion * cohesions)
			+ (weightPredator * weightPredatorBoost * predator) + (weightPrey * weightPreyBoost * prey)
		)
	
	velocities += target + numpy.random.randn(n, dimension)
	velocitiesPredator += predators  # + numpy.random.randn(n, dimension)
	# velocitiesPrey += preys + numpy.random.randn(n, dimension)
	
	boundaries = positions + (dT * velocities)
	out_of_bounds = numpy.array(
		(boundaries[:, 0] < -width) + (boundaries[:, 0] > width), 
		dtype=float, copy=False, order=None, subok=False, ndmin=0
	).reshape(n, 1)
	if dimension > 1:
		out_of_bounds += numpy.array(
			(boundaries[:, 1] > height) + (boundaries[:, 1] < -height), 
			dtype=float, copy=False, order=None, subok=False, ndmin=0
		).reshape(n, 1)
		if dimension > 2:
			out_of_bounds += numpy.array(
				(boundaries[:, 2] > depth) + (boundaries[:, 2] > depth), 
				dtype=float, copy=False, order=None, subok=False, ndmin=0
			).reshape(n, 1)
	if boundary_type == 1:
		velocities += out_of_bounds * (velocities * -2)
	elif boundary_type == 2:
		positions += out_of_bounds * (positions * -2)
	
	'''velocities = numpy.clip(velocities, -maximumSpeed, maximumSpeed)
	velocitiesPredator = numpy.clip(velocitiesPredator, -maximumSpeed, maximumSpeed)
	velocitiesPrey = numpy.clip(velocitiesPrey, -maximumSpeed, maximumSpeed)'''
	velocities_squared = numpy.einsum("...i,...i", velocities, velocities).reshape(n, 1)
	velocities_predator_squared = numpy.einsum(
		"...i,...i", velocitiesPredator, velocitiesPredator
	).reshape(nPredators, 1)
	'''velocities_prey_squared = numpy.einsum(
		"...i,...i", velocitiesPrey, velocitiesPrey
	).reshape(nPreys, 1)'''
	velocities = (
		velocities * (velocities_squared < maximumSpeedSquared) 
		+ numpy.nan_to_num(
			velocities / numpy.sqrt(velocities_squared)
		) * (velocities_squared > maximumSpeedSquared) 
		* maximumSpeed
	)
	velocitiesPredator = (
		velocitiesPredator * (velocities_predator_squared < maximumSpeedSquared) 
		+ numpy.nan_to_num(
			velocitiesPredator / numpy.sqrt(velocities_predator_squared)
		) * (velocities_predator_squared > maximumSpeedSquared) 
		* maximumSpeed
	)
	
	positions += dT * velocities
	positionsPredator += dT * velocitiesPredator
	# positionsPrey += dT * velocitiesPrey
	
	velocities *= 0.9
	velocitiesPredator *= 0.9
	# velocitiesPrey *= 0.9
	
	if animation:
		matplotlib.pyplot.title("Tick %d" % tick)
		if animation_type == 1:
			if dimension == 1:
				scatter.set_offsets(numpy.insert(positions, [1], [0], axis=1))
				scatterPredators.set_offsets(numpy.insert(positionsPredator, [1], [0], axis=1))
				scatterPreys.set_offsets(numpy.insert(positionsPrey, [1], [0], axis=1))
			elif dimension == 2:
				scatter.set_offsets(positions)
				scatterPredators.set_offsets(positionsPredator)
				scatterPreys.set_offsets(positionsPrey)
			elif dimension == 3:
				scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
				scatterPredators._offsets3d = (
					positionsPredator[:, 0], positionsPredator[:, 1], positionsPredator[:, 2]
				)
				scatterPreys._offsets3d = (positionsPrey[:, 0], positionsPrey[:, 1], positionsPrey[:, 2])
		elif animation_type == 2:
			if dimension == 1:
				scatter = matplotlib.pyplot.scatter(
					positions[:, 0], numpy.zeros((n, dimension), dtype=float, order=None), s=8, marker="o"
				)
				scatterPredators = matplotlib.pyplot.scatter(
					positionsPredator[:, 0], numpy.zeros((nPredators, dimension), dtype=float, order=None), 
					s=8, c="Red", marker="s"
				)
				scatterPreys = matplotlib.pyplot.scatter(
					positionsPrey[:, 0], numpy.zeros((nPreys, dimension), dtype=float, order=None), 
					s=8, c="Green", marker="^"
				)
			elif dimension == 2:
				scatter = matplotlib.pyplot.scatter(
					positions[:, 0], positions[:, 1], s=8, marker="o"
				)
				scatterPredators = matplotlib.pyplot.scatter(
					positionsPredator[:, 0], positionsPredator[:, 1], s=8, c="Red", marker="s"
				)
				scatterPreys = matplotlib.pyplot.scatter(
					positionsPrey[:, 0], positionsPrey[:, 1], s=8, c="Green", marker="^"
				)
			elif dimension == 3:
				scatter = ax.scatter(
					positions[:, 0], positions[:, 1], positions[:, 2], s=8, marker="o"
				)
				scatterPredators = ax.scatter(
					positionsPredator[:, 0], positionsPredator[:, 1], positionsPredator[:, 2], s=8, c="Red", marker="o"
				)
				scatterPreys = ax.scatter(
					positionsPrey[:, 0], positionsPrey[:, 1], positionsPrey[:, 2], s=8, c="Green", marker="^"
				)
	
	if measure:
		__measure__(tick)
	
	return


def __measure__(tick: int) -> None:
	global output
	global ticks
	global n
	global nPredators
	global nPreys
	global velocities
	'''global velocitiesPredator
	global velocitiesPrey'''
	global matrixSeparations
	global matrixAlignments
	global matrixCohesions
	global matrixPredators
	global matrixPreys
	global distances
	global distancesPredator
	global distancesPrey
	global measurement
	
	measurement_velocities = numpy.sum(numpy.einsum("...i,...i", velocities, velocities)) / n  # velocities_squared
	measurement_predators = 1 / ((numpy.sum(matrixPredators) + 1) / (n * nPredators))
	measurement_preys = numpy.sum(matrixPreys) / (n * nPreys)
	measurement_distances = 1 / (numpy.sum(distances) / (n * n))
	measurement_distances_predator = numpy.sum(distancesPredator) / (nPredators * n)
	measurement_distances_prey = 1 / (numpy.sum(distancesPrey) / (nPreys * n))
	
	measurement.append([
		measurement_velocities, 
		measurement_predators, 
		measurement_preys, 
		measurement_distances, 
		measurement_distances_predator, 
		measurement_distances_prey
	])
	
	if tick == ticks - 1:
		output["measurementFitness"] = [
			"velocities", 
			"predators", 
			"preys", 
			"distances", 
			"distancesPredator", 
			"distancesPrey"
		]
		
		output["measurement"] = measurement
		
		measurement_array = numpy.array(measurement, dtype=float, copy=False, order=None, subok=False, ndmin=0)
		
		output["fitness"] = [
			numpy.nan_to_num(numpy.mean(measurement_array[:, 0]), copy=False), 
			numpy.nan_to_num(numpy.mean(measurement_array[:, 1]), copy=False), 
			numpy.nan_to_num(numpy.mean(measurement_array[:, 2]), copy=False), 
			numpy.nan_to_num(numpy.mean(measurement_array[:, 3]), copy=False), 
			numpy.nan_to_num(numpy.mean(measurement_array[:, 4]), copy=False), 
			numpy.nan_to_num(numpy.mean(measurement_array[:, 5]), copy=False)
		]
	
	return


def __run__(parameters: dict, scene_file: str) -> (tuple, [float]):
	global dT
	global boundary_type
	global animation
	global measure
	global sceneFilename
	# global outputFilename
	global verbosity
	'''global outputFile
	global output'''
	global boidSize
	global radiusSeparationSquared
	global radiusAlignmentSquared
	global radiusCohesionSquared
	global radiusPredatorSquared
	global radiusPreySquared
	global weightSeparation
	global weightAlignment
	global weightCohesion
	global weightPredator
	global weightPredatorBoost
	global weightPrey
	global weightPreyBoost
	global maximumSpeed
	global maximumSpeedSquared
	global ticks
	
	random.seed(24)
	numpy.random.seed(24)
	numpy.warnings.filterwarnings("ignore")
	dT = 0.1
	boundary_type = 1  # [1 = Bound (Velocity), 2 = Wrap (Position)]
	animation = False
	measure = True
	
	# inputFilename
	sceneFilename = scene_file
	# outputFilename = output_file
	verbosity = 0
	
	__global__()
	
	boidSize = parameters["boidSize"]
	radius_separation = parameters["radii"]["separation"]
	radius_alignment = parameters["radii"]["alignment"]
	radius_cohesion = parameters["radii"]["cohesion"]
	radius_predator = parameters["radii"]["predator"]
	radius_prey = parameters["radii"]["prey"]
	weightSeparation = parameters["weights"]["separation"]
	weightAlignment = parameters["weights"]["alignment"]
	weightCohesion = parameters["weights"]["cohesion"]
	weightPredator = parameters["weights"]["predator"]
	weightPredatorBoost = parameters["weights"]["predatorBoost"]
	weightPrey = parameters["weights"]["prey"]
	weightPreyBoost = parameters["weights"]["preyBoost"]
	maximumSpeed = parameters["maximumSpeed"]
	radiusSeparationSquared = (radius_separation + (radius_separation * boidSize) + boidSize) ** 2
	radiusAlignmentSquared = (radius_alignment + (radius_alignment * boidSize) + boidSize) ** 2
	radiusCohesionSquared = (radius_cohesion + (radius_cohesion * boidSize) + boidSize) ** 2
	radiusPredatorSquared = (radius_predator + (radius_predator * boidSize) + boidSize) ** 2
	radiusPreySquared = (radius_prey + (radius_prey * boidSize) + boidSize) ** 2
	maximumSpeedSquared = maximumSpeed ** 2
	
	for tick in range(ticks):
		__update__(tick)
	
	return (
		(
			"velocities", 
			"predators", 
			"preys", 
			"distances", 
			"distancesPredator", 
			"distancesPrey"
		), 
		output["fitness"]
	)


if __name__ == "__main__":
	__main__()
