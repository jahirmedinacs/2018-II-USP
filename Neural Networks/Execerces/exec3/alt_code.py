import pandas as pd
import numpy as np

import matplotlib as m_plt
from matplotlib import pyplot as plt

from copy import copy
from pprint import pprint

import sys
import os

import personal_plotter as p_plt

#____________

dummy_shape = (10)
dummy_data = np.identity(dummy_shape)
print(dummy_data)

#____________

input_data = dummy_data.copy()
temp_shape = copy(input_data.shape)
input_data = list(map(lambda x : np.matrix(x), np.split(input_data.flatten(), temp_shape[1])))

objetive_data = dummy_data.copy()
temp_shape = copy(objetive_data.shape)
objetive_data = list(map(lambda x : np.matrix(x), np.split(objetive_data.flatten(), temp_shape[1])))

#____________

def sigmoid(x_val):
    return 1 / (1 + np.exp(-x_val))

#____________

def d_sigmoid( x_val, mode="layer"):
    if mode == "layer":
        return np.multiply(x_val , (x_val - 1))
    elif mode == "input":
        temp = sigmoid(x_val)
        return np.multiply(temp, (temp - 1))

#____________

def forward_engine(input_data, objective_data, weigth_data_container, 
                   theta_container,kernel=sigmoid, 
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

#____________

def backward_engine(input_data, objective_data, weigth_data_container, theta_container,
                    forward_data_container, derivate=d_sigmoid,
                    learning_rate=0.05, verbose=False, *args):
    
    output = copy(weigth_data_container)
    t_output = copy(theta_container)
    
    last_layer = True
    for index in range( len(weigth_data_container) - 1, -1, -1 ):
        
            if last_layer:
                
                e = np.multiply((forward_data_container[index] - objective_data) , derivate(forward_data_container[index]))
                
                output[index] += learning_rate * ( e.T * forward_data_container[index - 1]).T
                
                t_output[index] += learning_rate * e
                
                last_layer = False
                
                if verbose:
                    print("output layer")
            else:
                
                if index > 0:
                    e = np.multiply( (e * weigth_data_container[index + 1].T) , derivate(forward_data_container[index]))
                                        
                    output[index] += learning_rate * np.multiply( e.T , forward_data_container[index - 1]).T
                    
                    t_output[index] += learning_rate * e
                    
                    if verbose:
                        print("in between layers ", index)
                    
                else:
                    e = np.multiply( (e * weigth_data_container[index + 1].T) , derivate(forward_data_container[index]))
                    
                    output[index] += learning_rate * np.multiply( e.T , input_data).T
                    
                    t_output[index] += learning_rate * e
                    
                    if  verbose:
                        print("input layer")
                        
            if verbose:
                print("e \t")
                print(e)
                print(e.T.shape)
                
                print("Forwared Shape \t")
                print(forward_data_container[index - 1].shape)
                
                print("Weigth \t")
                print (output[index].shape)
                print(output[index])
                
                print("Theta \t")
                print (t_output[index].shape)
                print(t_output[index])
    
    return output, t_output

#____________

def MLP_engine(input_data, objetive_data,
               layer_set_up = [10,4,10],
               theta = False,
               function_set=[sigmoid, d_sigmoid],
               learning_rate=0.05, epochs=int(1e3), threshold=0.05,
               verbose=False):
    
    shape_container = layer_set_up
    
    weigth_data_container = [ np.random.sample((shape_container[ii], shape_container[ii + 1])) + 1
                             for ii in range(len(shape_container[:-1])) ]

    if theta:
        theta_container = list(map(lambda x : np.matrix(x), [ np.random.sample(ii)  + 1 for ii in shape_container[1:]]))
    else:
        theta_container = list(map(lambda x : np.matrix(x), [ np.zeros(ii) for ii in shape_container[1:] ]))
    
    if verbose:
        print("weigth", weigth_data_container)
        print("theta", theta_container)
    
    iters = 0
    while epochs != 0 :
        epoch_cum = 0
        
        for id_data in range(len(input_data)):
            
            forward_data_container = forward_engine(input_data[id_data], objetive_data[id_data],
                                                    weigth_data_container, theta_container,
                                                    kernel=function_set[0],
                                                    verbose=verbose)
            
            if  verbose:
                print(forward_data_container)
            
            weigth_data_container, theta_container = backward_engine(input_data[id_data], objetive_data[id_data],
                            weigth_data_container, theta_container,
                            forward_data_container, derivate=function_set[1],
                            learning_rate=learning_rate,
                                                    verbose=verbose)
            
            epoch_cum += np.sqrt(np.sum(np.power(objetive_data[id_data] - forward_data_container[-1], 2))/ len(forward_data_container[-1]))
            #epoch_cum += objetive_data[id_data] - forward_data_container[-1]
        
        if iters < 1000:
            if epochs % 50 == 0:
                print("Epoch :> {:d} \n\tError :> {:.4f} %".format(iters, (epoch_cum / len(input_data)) * 100))
        
        if epochs%10000 == 0:
            print("Epoch :> {:d} \n\tError :> {:.4f} %".format(iters, (epoch_cum / len(input_data)) * 100))
            
        if epoch_cum / len(input_data) < threshold:
            break
        else:
            pass
        
        iters += 1
        epochs -= 1
    
    print("Total of Epochs {:d} \n\tError :> {:.4f} %".format(iters, (epoch_cum / len(input_data)) * 100))
    return weigth_data_container, theta_container, shape_container

#____________

a = [[1,1], [1,0], [0,1], [0,0]]
a = list(map(lambda x: np.matrix(x), a))
print(a)
b = [1, 0, 0, 0]
b = list(map(lambda x: np.matrix(x), b))
print(b)

w_d, t_d, s_d = MLP_engine(a, b,
                           layer_set_up=[2, 1], theta=True,
                           epochs=int(1e3), learning_rate=1,
                           verbose=False)
    
#____________

for i in range(4):
    print("\n\n____________***____________\n\n")
    print("\t\tCASE", i)
    print("\tA\t::>>\n", a[i].T)
    print("\tB\t::>>\t", b[i])
    
    temp = forward_engine(a[i], b[i], w_d, t_d)
    
    print("\n\tRAW Result\n")
    print(temp)
    print("\t\nBinary Result\n \t\t", end="")
    print( round( float(temp[0]) , 0), end="\n")

#____________

w_d, t_d, s_d = MLP_engine(input_data, objetive_data,
                           layer_set_up=[10, 4, 10], theta=True,
                           epochs=int(1e4), learning_rate=0.5,
                           verbose=False)

#____________

for jj in range(10):
    
    print("\t_____***_____\n")
    print("\t\t\033[1mCASE\033[0m", jj)
    
    print("Input\n")
    print(input_data[jj])

    temp = forward_engine(input_data[jj], objetive_data[jj], w_d, t_d)
    
    print("\n\t\033[1mRAW Result\033[0m\t", temp, "\n")
    
    print("\nResult\t", np.round(temp[-1], 0))
    print("Objective\t", objetive_data[jj])
    print("\033[1mMatch? :>>\t", np.prod(np.round(temp[-1], 0) == objetive_data[jj]).astype(np.bool))