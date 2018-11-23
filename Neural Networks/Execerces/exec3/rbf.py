#! /usr/bin/python3.6 -s
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sys
import os

import help_f as hf

##### RBF

def k_means(data, n_of_clusters=3, threshold=1e-5 , max_iter=1000):

	rows_id = np.arange(data.shape[0])
	center_ids = np.random.choice(rows_id, n_of_clusters, replace=False)

	centers = data[center_ids, :]

	error = 1e3
	for _  in range(max_iter):
		if error > threshold:
			pass
		else:
			break

		distances =	[[hf.euclidean_dist(column_x, centers[ii, :]).tolist() for column_x in data] for ii in range(n_of_clusters)]

		index_container = []
		for i in range(data.shape[0]):
			min_dist = distances[0][i]
			index = 0
			for j in range(n_of_clusters):
				if min_dist > distances[j][i]:
					min_dist = distances[j][i]
					index = j
			index_container += [index];

		error = 0
		index_container = np.array(index_container)
		for i in range(n_of_clusters):
			rows = np.where(index_container == i)
			centers[i, :] = data[rows, :].mean(1)

			error += hf.euclidean_dist(centers[i, :], data[rows, :].mean(1))

	return [n_of_clusters, centers, index_container, 1.0]


def phi(model,data):
	[n_of_clusters, centers, _, spread] = model

	output = np.zeros((data.shape[0], n_of_clusters))

	for i in range(n_of_clusters):
		for j in range(data.shape[0]):
			output[j,i] = hf.gaussiana( hf.euclidean_dist(centers[i,],data[j,]), spread)

	return output


def adaline(X,Y,eta=0.1, threshold=1e-5 ,maxiter=100):
	W = np.random.random_sample((Y.shape[1],X.shape[1])) - 0.5 # randomização dos pesos sinapticos
	c = 0

	while c < maxiter: # criterio de parada
		tam = X.shape[0]

		for i in np.arange(tam):
			Yi = np.dot(W,X[i,])
			A = (Y[i,] - Yi)
			B = X[i,]
			if(np.argmax(Yi)!=np.argmax(Y[i,])):
				W = W + eta*np.dot(A.reshape((A.shape[0],1)),B.reshape((1,B.shape[0])))
		c = c + 1
	return W

def clasificacion(W,X,Y, verbose=False):
	acierto = 0;
	tam = Y.shape[0]

	for i in np.arange(tam):
		Yesperado = np.dot(W,X[i,]);
		Yi = np.argmax(Yesperado)

		if verbose:
			temp_arry = np.zeros(3)
			temp_arry[Yi] += 1
			print("\n\t", i,"\n\tPredicted :\t", temp_arry, " -\t- ", Yesperado, "\n\tExpected :\t", Y[i,])

		if np.sum(Yi - np.argmax(Y[i,])) == 0:
			 acierto += 1
	return (acierto * 100.0 )/tam

##### Exercice Functions


def seed_test(Data, cluster=12, siz=0.75, maxiterK=1000, maxiterA=1000, eta = 0.2):

	X = hf.normalizacion(Data.iloc[:, :-3].values)

	Y = Data.iloc[:, -3:].values.astype(float)

	model  = k_means(X.copy(), n_of_clusters=cluster, max_iter=maxiterK)
	X = phi(model,X.copy())

	sizes = hf.sub_sampler(X, siz)


	X1 = X[sizes[0],:]
	Y1 = Y[sizes[0],:]

	X2 = X[sizes[1],:]
	Y2 = Y[sizes[1],:]

	X1 = hf.normalizacion(X1)
	X2 = hf.normalizacion(X2)

	W = adaline(X1,Y1,maxiter = maxiterA)

	A = clasificacion(W,X1,Y1)
	B = clasificacion(W,X2,Y2, verbose=True)

	print(
	"""
	\t\t|-----------|
	% Train: {:f} Max.Iter: {:d} Eta: {:f}
	% Precition (Train) :>> \t{:f}
	% Precition (Test) :>> \t{:f}
	\t\t|-----------|
	""".format(siz, maxiterA, eta, A, B)
	)

	return [A,B]


def main():

	targets_names = ["Kama", "Rosa", "Canadian"]

	seeds_df = pd.DataFrame()
	seeds_df = pd.DataFrame.from_csv(path="seeds_df.csv", header=0 ,sep=";", encoding="utf-8")

	seed_test(seeds_df)


if __name__ == "__main__":
	main()
