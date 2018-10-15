#! /usr/bin/python3.6 -s
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### Help Functions

def sigmoid(x_val):
	return 1 / (1 + np.exp(-x_val))

def d_sigmoid( x_val, mode="layer", kernel=sigmoid):
	"""
	in the layer mode, the input data was already procesed using some kernel
	in the input mode, the input data wasnt pass throw some kernel.
	"""
	if mode == "layer":
		return np.multiply(x_val , (x_val - 1))
	elif mode == "input":
		temp = kernel(x_val)
		return np.multiply(temp, (temp - 1))

###### Fordward Function

def forward_engine(input_data, objective_data, model, kernel=sigmoid, 
                   verbose=False, *args):
	
	

    output = [ None for _ in range(len(weigth_data_container)) ]
    
    relative_layer = input_data.copy()
    for index in range(len(weigth_data_container)):
            
            if verbose:
                print(relative_layer.shape, weigth_data_container[index].shape, theta_container[index].shape)
            
            carry =  kernel((relative_layer *  weigth_data_container[index]) + theta_container[index])
            output[index] = carry.copy()
            
            relative_layer = carry.copy()
    
    return output