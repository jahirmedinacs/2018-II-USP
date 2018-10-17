#! /usr/bin/python3.6 -s
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

##### Help Functions

def normalizacion(Data):
	
	for i in  np.arange(Data.shape[1]):
		Data[:,i] = ( Data[:,i] - np.min(Data[:,i]) )/( np.max(Data[:,i]) - np.min(Data[:,i]) )
	return Data

##### RBF

def gaussiana(v,spread):
	return np.power(np.e, -np.power(v,2)/(np.power(spread,2)))    


def dist_euclidian(A,B):
	return np.sqrt(np.sum(np.power(A-B,2)))

def kmeans(X, clusters = 3,threshold = 1e-5 , maxiter = 1000):
	
	rows_id = np.arange(X.shape[0])
	ids = np.random.choice(rows_id, clusters, replace=False)
	
	centers = X[ids,:]
	error = 2*threshold
	iteration = 0
	
	
	while error > threshold and iteration < maxiter:
	
		distances = []
		for i in np.arange(clusters):
			distances.append(list(map( dist_euclidian,[a for a in X],[centers[i,] for b in np.arange(X.shape[0])]) ))

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
			error = error + dist_euclidian(centers[i,],np.mean(X[rowIds,],axis = 1))
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
			X1[j,i] = gaussiana(dist_euclidian(centers[i,],X[j,]),spread)
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


def normalizacion(Data):
	for i in  np.arange(Data.shape[1]): 
		Data[:,i] = ( Data[:,i] - np.min(Data[:,i]) )/( np.max(Data[:,i]) - np.min(Data[:,i]) )
	return Data

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


def sub_sampler(data_samples, ratio, verbose=False):
	top_id = int(data_samples.shape[0] * ratio)
		
	rows_id = np.arange(data_samples.shape[0])
	sub_sample_ids_train = np.random.choice(rows_id, top_id, replace=False)
		
	sub_sample_ids_test = np.setdiff1d(rows_id, sub_sample_ids_train)
	
	return [sub_sample_ids_train,sub_sample_ids_test]


##### Exercice Functions


def seed_test(Data, cluster=12, siz=0.75, maxiterK=1000, maxiterA=1000, eta = 0.2):
	
	parameters = Data.iloc[:, :-3].values
	X = normalizacion(parameters)
		
	objetive = Data.iloc[:, -3:].values
	Y = objetive.astype(float)
	
	model  = kmeans(X.copy(),clusters= cluster,maxiter= maxiterK)
	X = phi(model,X.copy())
		
	sizes = sub_sampler(X, siz)
	
	
	X1 = X[sizes[0],:]
	Y1 = Y[sizes[0],:]
	
	X2 = X[sizes[1],:]
	Y2 = Y[sizes[1],:]

	X1 = normalizacion(X1)
	X2 = normalizacion(X2)
	
	W = adaline(X1,Y1,maxiter = maxiterA)
	
	A = clasificacion(W,X1,Y1)
	B = clasificacion(W,X2,Y2, verbose=True)
	
	print("Particionamiento: "+ str(siz) +" Max.Iter: " + str(maxiterA) + " Eta: " + str(eta))    
	print("Acuracia na data de treinamento",A)
	print("Acuracia na data de test",B)
	print("\n")
			
	return [A,B]


def main():

	targets_names = ["Kama", "Rosa", "Canadian"]

	seeds_df = pd.DataFrame()
	seeds_df = pd.DataFrame.from_csv(path="seeds_df.csv", header=0 ,sep=";", encoding="utf-8")
	
	for _  in range(5):
		print("\n")
	
	# Making the objetive data
	
	"""
	objetive = seeds_df.iloc[:, -3:].values

	parameters = seeds_df.iloc[:, :-3].values
	"""
	
	print(seed_test(seeds_df))
	

if __name__ == "__main__":
	main()


# def XOR(Data):
	
# 	X = Data[:,0:Data.shape[1]-1]
# 	Y = binarizar(Data[:,-1],siz = 2)

# 	model  = kmeans(X.copy(),clusters= 2,maxiter= 10)
# 	print(model[1])
# 	X = phi(model,X.copy())
# 	print(X)
	
# 	X = normalizacion(X)
	
# 	W = adaline(X,Y,maxiter = 50000)
	
	
# 	A = clasificacion(W,X,Y)
# 	print("Acuracia na data de treinamento",A)
	
# 	return W

# def main():
	
#    #X = np.matrix([[0,0,1],[0,1,0],[1,0,0],[1,1,1]])
#    #XOR(X)
   
#    datos = pd.read_csv("seeds_dataset.data",sep = ",") # leitura dos dados
#    seedData = datos.values
#    seed_test(seedData)
# if __name__ == "__main__":
# 	main()