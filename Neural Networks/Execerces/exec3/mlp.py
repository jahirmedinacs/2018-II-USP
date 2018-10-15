#! /usr/bin/python3.6 -s
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

##### Help Functions

def sigmoid(x_val):
	return 1 / (1 + np.exp(-x_val))

def d_sigmoid( x_val, mode="layer", kernel=sigmoid):
	"""
	in the Layer mode, the input data was already procesed using some kernel
	in the Input mode, the input data wasnt pass throw some kernel.
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


def main():
	
	targets_names = ["Kama", "Rosa", "Canadian"]
	
	seeds_df = pd.DataFrame()
	seeds_df = pd.DataFrame.from_csv(path="seeds_df.csv", header=0 ,sep=";", encoding="utf-8")
	
	target = np.zeros(seeds_df.shape[0], dtype=int)
	ii = 1
	for names in targets_names:
		target += seeds_df[names] * ii
		ii += 1
	
	to_plot = seeds_df.iloc[:, :-3]
	to_plot["target"] = target
	
	sns.pairplot(data=to_plot, vars=to_plot.columns.tolist()[:-1], hue='target')
	plt.show()


	
if __name__ == "__main__":
	main()