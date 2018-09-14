import pandas as pd
import numpy as np

import matplotlib as m_plt
from matplotlib import pyplot as plt

from copy import copy
from pprint import pprint

import sys
import os

import personal_plotter as p_plt

# Data Generation

dummy_shape = (10)
dummy_data = np.identity(dummy_shape)
print(dummy_data)

# Data Sampling

input_data = dummy_data.copy()
objetive_data = dummy_data.copy()

# Activation Function

def sigmoid(*args):
	return 1 / (1 + np.exp(-x_val))

# Forward Engine

def forward_engine(input_data, objetive_data, layer_container, theta_container=None,kernel=sigmoid):
	if theta_container is None:
		theta_container = [ np.zeros(layer_container[ii].shape) for ii in len(layer_container)]
	else:
		pass
	
	output = [ None for _ in range(len(layer_container)) ]
	
	relative_layer = input_data.copy()
	for index in range(len(layer_container)):
			carry =  kernel((relative_layer *  layer_container[index]) + theta_container)
			output[index] = carry.copy()
			
			relative_laye = carry.copy()
	
	return output

# Gradient Derivate

def d_sigmoid(*args):
	pass

# Backward Engine

def backward_engine(input_data, objetive_data, layer_container, theta_container=None,
					forward_layer,
					learning_rate=0.01, epochs=int(1e6), threashold=0.05):
	if theta_container is None:
		theta_container = [ np.zeros(layer_container[ii].shape) for ii in len(layer_container)]
	else:
		pass
	
	output = [ None for _ in range(len(layer_container)) ]
	
	relative_layer = input_data.copy()
	for index in range(len(layer_container)):
			carry =  kernel((relative_layer *  layer_container[index]) + theta_container)
			output[index] = carry.copy()
			
			relative_laye = carry.copy()
	
	return output