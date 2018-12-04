import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns


def wine(data):
    return data[:,1:]

def normalizacion(Data):
    for i in  np.arange(Data.shape[1]): 
        Data[:,i] = ( Data[:,i] - np.min(Data[:,i]) )/( np.max(Data[:,i]) - np.min(Data[:,i]) )
    return Data

def dist_euclidian(A,B):
    return np.sqrt(np.sum(np.power(A-B,2),axis = 1))

def arquitectura(Isize = 13 , Osize = 10):
    
    W = np.random.random_sample((Osize*Osize,Isize))
    return [Isize,Osize,W]

def kohonen(model,X,eta = 0.1,epochs = 100):
    
    vizinhança = model[1]//3
    
    tam = X.shape[0]
    T = model[1]*model[1]
    for t in np.arange(epochs):
        E = eta /(t+1)
        V = (int)(vizinhança - (vizinhança*((t+1)/100))) + 1
        for i in np.arange(tam):
            Xi = X[i,]
            BMU = np.argmin(dist_euclidian(Xi,model[2]))
            Ib = BMU//model[1]
            Jb = BMU%model[1]
            for j in np.arange(T):
                I = j//model[1]
                J = j%model[1]
                if np.abs(Ib-I)+np.abs(Jb - J)<= V:
                    model[2][j,] = model[2][j,] + E*(X[i,] - model[2][j,])
    return model

def main():   
   datos = pd.read_csv("wine.data",sep = ",") # leitura dos dados
   X = datos.values
   Y = X [ : ,0]
   X = wine(X)
   X = normalizacion(X)
   model = arquitectura()
   trained = kohonen(model,X)
   print(trained[2])
   
   I = np.zeros((10,10))
   #print(I)
   for i in np.arange(X.shape[0]):
       BMU = np.argmin(dist_euclidian(X[i,],model[2]))
       I[BMU//10][BMU%10] = Y[i]    
   #print(I)
   
   sns.heatmap(I)
if __name__ == "__main__":
   main()