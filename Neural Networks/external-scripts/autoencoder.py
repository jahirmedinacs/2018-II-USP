import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def function(net): # função sigmoide
    return (1/(1+np.exp(-net)))

def dnet(net):
    return (net*(1-net))

def feedforward(X,model):
    
    Whidden = model[3] 
    Woutput = model[4]
    
    X = np.concatenate((X,np.ones(1)),axis = 0) 
    netH = np.dot(Whidden,X)
    fnetH = function(netH)
    
    Rd = np.concatenate((fnetH,np.ones(1)),axis = 0) # redução de dimensionalidade
    netO = np.dot(Woutput,Rd)
    fnetO = function(netO)
    
    return [fnetO,fnetH]

def backpropagation(model,dataset,eta = 0.1,threshold = 1e-3):
    sError = 2*threshold
    c = 0
    while sError > threshold: # criterio de parada
        sError = 0
        X = dataset
        Y = dataset
        
        results = feedforward(X,model);
        
        O = results[0]
        #Error
        Error = np.sum(Y-O)
        
        sError = sError + np.sum(np.power(Error,2))
        #Treinamento capa de salida
        
        deltaO = Error * dnet(results[0])
        #Treinamento capa de entrada
        
        Wk = model[4][:,0:model[1]]
       
        deltaH = results[1] * (np.dot(deltaO,Wk))
        #atualização dos pesos
        Rp = np.concatenate((results[1],np.ones(1)),axis=0)
        Rp = np.reshape(Rp,(model[1]+1,1))
        
        Xp = np.concatenate((X,np.ones(1)),axis=0)
        Xp = np.reshape(Xp,(model[2]+1,1))
        
        model[4] = model[4] +  eta*(np.dot(np.reshape(deltaO,(model[2],1)),np.transpose(Rp)))
        model[3] = model[3] +  eta*(np.dot(np.reshape(deltaH,(model[1],1)),np.transpose(Xp)))
        
        #contador
        print("iteração ",c)
        print("Error:",sError)
        print("\n");
        c = c+1
    return model

def mlp(Isize = 100,Hsize = int(np.log2(100)),Osize = 100):
    
    # Isize tamano da camada de entrada
    # Osize tamano da camada de salida
    # Hsize tamano de camada oculta
    
    Whidden = np.random.random_sample((Hsize,Isize +1)) - 0.5 # randomização dos pesos sinapticos
    Woutput = np.random.random_sample((Osize,Hsize +1)) - 0.5 
    
    model = [Isize,Hsize,Osize,Whidden,Woutput]
    return model

def main():
    
    Siz = int(input("Tamanho da matriz\n"))
    I = np.identity(Siz) # matriz identidade 
    I = np.reshape(I,-1) # matriz to array
    
    S2 = np.power(Siz,2) 
    
    M = mlp(Isize = S2, Hsize = int(np.log2(S2)), Osize = S2) #inicialização do modelo
    
    trained = backpropagation(M,I) # Treinamento do modelo
    
    R = feedforward(I,trained); # prueba do modelo
    
    R = R[0]
    R = np.reshape(R,(Siz,Siz))
    
    Rd = R[1]
    print("Redução")
    print(Rd)
    print("\nMapeamento")
    print(R)
    
if __name__ == "__main__":
    main()