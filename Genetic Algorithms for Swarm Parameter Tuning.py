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
sceneFile = None
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
	__argparse__()
	
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


if __name__ == "__main__":
	__main__()
