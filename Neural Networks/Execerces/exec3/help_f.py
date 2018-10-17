# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sys
import os

##### Help Functions

def normalizacion(input_data):
	output = np.zeros(input_data.shape)

	for i in  np.arange(input_data.shape[1]):
		temp_array = input_data[:, i]
		output[:, i] = ( temp_array - temp_array.min() ) / ( temp_array.max() - temp_array.min() )
	return output


def sub_sampler(data_samples, ratio):
	top_id = int(data_samples.shape[0] * ratio)
	rows_id = np.arange(data_samples.shape[0])

	sub_sample_ids_train = np.random.choice(rows_id, top_id, replace=False)

	sub_sample_ids_test = np.setdiff1d(rows_id, sub_sample_ids_train)

	return [sub_sample_ids_train,sub_sample_ids_test]

##### RBF

def gaussiana(x_val, std_deviation):
	return np.power(np.e, -np.power(x_val,2)/(np.power(std_deviation,2)))


def euclidean_dist(x_val, y_val):
	return np.sqrt ( np.sum ( np.power ( x_val - y_val , 2)))

##### MLP

def sigmoid(x_val):
	return  (1 / ( 1 + np.exp( -x_val ) ))


def d_sigmoid(x_val):
	return ( x_val * ( 1 - x_val ))
