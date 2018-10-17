#! /usr/bin/python3.6 -s
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sys
import os

import help_f as hf

##### RBF

def kmeans(X, clusters = 3,threshold = 1e-5 , maxiter = 1000):

	rows_id = np.arange(X.shape[0])
	ids = np.random.choice(rows_id, clusters, replace=False)

	centers = X[ids,:]
	error = 2*threshold
	iteration = 0


	while error > threshold and iteration < maxiter:

		distances = []
		for i in np.arange(clusters):
			distances.append(list(map( hf.euclidean_dist, [a for a in X],[centers[i,] for b in np.arange(X.shape[0])]) ))

		ids = []
		for i in np.arange(X.shape[0]):
			mini = distances[0][i]
			indice = 0
			for j in np.arange(clusters):
				if mini > distances[j][i]:
					mini = distances[j][i]
					indice = j
			ids.append(indice);

		error = 0
		ids = np.asarray(ids.copy())
		for i in np.arange(clusters):
			rowIds = np.where( ids == i)
			error = error + hf.euclidean_dist(centers[i,],np.mean(X[rowIds,],axis = 1))
			centers[i,] = np.mean(X[rowIds,],axis = 1)
		iteration = iteration+1

	spread = 1.0
	return [clusters,centers,ids,spread]

def phi(model,X):
	clusters = model[0]
	centers = model[1]
	spread = model[3]

	X1 = np.zeros((X.shape[0],clusters))

	for i in np.arange(clusters):
		for j in np.arange(X.shape[0]):
			X1[j,i] = hf.gaussiana(hf.euclidean_dist(centers[i,],X[j,]),spread)
	return X1

def adaline(X,Y,eta =0.1,threshold = 1e-5 ,maxiter = 100):
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

	parameters = Data.iloc[:, :-3].values
	X = hf.normalizacion(parameters)

	objetive = Data.iloc[:, -3:].values
	Y = objetive.astype(float)

	model  = kmeans(X.copy(),clusters= cluster,maxiter= maxiterK)
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
