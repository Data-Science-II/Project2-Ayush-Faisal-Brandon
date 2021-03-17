import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold


class Perceptron():
    
    def __init__(self, data, response, eta = 0.05): 
        """
        Builds a perceptron from numpy arrays of x variables and a response. 
        
        Keyword Arguments: 
        data -- the X variable matrix 
        response -- the y response vector 
        """
        self.x = data
        self.y = response
        self.k = self.x.shape[1] - 1
        self.n = self.x.shape[0]
        self.model = Perceptron.build_model(self.x, self.y, eta)
        
    def build_model(x, y, suppress = False, eta = 0.05):
        """Builds a perceptron given the input matrix and response variable."""
        model = keras.Sequential()
        model.add(layers.Dense(1, input_dim = x.shape[1], kernel_initializer = "normal", activation = "sigmoid", use_bias=False))

        optimizer = keras.optimizers.Adam(learning_rate=eta)      

        model.compile(loss = "mean_squared_error", optimizer = optimizer)
        if not suppress: 
            model.summary()
        return model

    def forward_selection(self, iter = 1000): 
        return VariableSelection.forward_selection(self.x, self.y, Perceptron.build_model, iter)

    def backward_elimination(self, iter = 1000): 
        return VariableSelection.backward_elimination(self.x, self.y, Perceptron.build_model, iter)

    def stepwise_regression(self, iter = 1000): 
        return VariableSelection.stepwise_regression(self.x, self.y, Perceptron.build_model, iter)
    
    def train(self, epochs = 20): 
        """Fits the model over a predetermined number of epochs."""
        self.model.fit(self.x, self.y, 
                    epochs = epochs, 
                    batch_size = 20, 
                    verbose = 0, 
                    shuffle = False)
        
    def rsq(self): 
        """Uses metrics.rsq using perceptron parameters."""
        return metrics.rsq(self.model, self.x, self.y)
    
    def rsq_adj(self):
        """Uses metrics.rsq_adj using perceptron parameters."""
        return metrics.rsq_adj(self.model, self.x, self.y)
    
    def aic(self):
        """Uses metrics.aic using perceptron measures."""
        return metrics.aic(self.model, self.x, self.y)
    
    def rsq_cv(self, folds = 10):
        """
        Peforms k-fold cross-validation and returns the average Rsq on the testing samples. 
        
        Keyword Arguments: 
        folds -- the number of folds to use for cross validation
        """
        return metrics.rsq_cv(self.model, self.x, self.y, folds)   
        

class metrics(): 
    def rsq(model, x, y):
        """Computes and returns the rsq for a trained model."""
        yhat = model.predict(x)
        mse = keras.metrics.MeanSquaredError()
        mse(y, np.mean(y))
        sst = mse.result().numpy()
        mse = keras.metrics.MeanSquaredError()
        mse(y, yhat)
        sse = mse.result().numpy()
        return 1 - sse/sst
    
    def rsq_adj(model, x, y):
        """Computes and returns the rsq_adj for a trained model."""
        rsq = metrics.rsq(model, x, y)
        n = x.shape[0] 
        k = x.shape[1] - 1
        top = (n-1)
        bottom = (n-k-1)
        return 1-(1-rsq)*top/bottom        
    
    def aic(model, x, y): 
        """Computes and returns the aic for a trained model."""
        yhat = model.predict(x)
        n = x.shape[0]
        k = x.shape[1] - 1
        m = keras.metrics.MeanSquaredError()
        m(y, yhat)
        mse = m.result().numpy()
        aic = n*np.log(mse) + 2*k
        return aic
    
    def rsq_cv(model, x, y, folds = 10):
        """
        Peforms k-fold cross-validation and returns the average Rsq on the testing samples. 
        
        Keyword Arguments: 
        folds -- the number of folds to use for cross validation
        """
        kfold = KFold(n_splits = folds, shuffle = True)
        rsq_cv = 0
        for train, test in kfold.split(x, y): 
            model.fit(x[train], y[train], epochs = 500, batch_size = 20, verbose = 0)
            rsq_cv = rsq_cv + metrics.rsq(model, x[test], y[test])
            
        return rsq_cv/folds 

class NeuralNet3L (Perceptron):
    def __init__(self, data, response): 
        """
        Builds a 3 Layer Nueral Network based on the input data. 
        
        Keyword Arguments: 
        
        data -- the x variable matrix 
        response -- the y response vector 
        """
        self.x = data
        self.y = response
        self.k = self.x.shape[1] - 1
        self.n = self.x.shape[0]
        self.model = NeuralNet3L.build_model(self.x, self.y)
    
    def build_model(x, y, suppress = False, width = 0): 
        k = x.shape[1] - 1
        #Default Width Value
        if (width == 0): 
            width = k + 1
        
        model = keras.Sequential()
        model.add(layers.Dense(width, input_dim = x.shape[1], 
                            kernel_initializer = "normal", 
                            use_bias = False,
                            activation = "relu", ))
        model.add(layers.Dense(1, kernel_initializer = "normal"))
        model.compile(loss = "mean_squared_error", optimizer = "adam")
        if not suppress: 
            model.summary()
        return model
    
    def forward_selection(self, iter = 200): 
        return VariableSelection.forward_selection(self.x, self.y, 
                                                NeuralNet3L.build_model,
                                                iter)
    
    def backward_elimination(self, iter = 200):
        return VariableSelection.backward_elimination(self.x, self.y, 
                                                NeuralNet3L.build_model,
                                                iter)
    
    def stepwise_regression(self, iter = 200): 
        return VariableSelection.stepwise_regression(self.x, self.y, 
                                                NeuralNet3L.build_model,
                                                iter)

class NeuralNet4L (Perceptron):
    def __init__(self, data, response): 
        """
        Builds a 3 Layer Nueral Network based on the input data. 
        
        Keyword Arguments: 
        
        data -- the x variable matrix 
        response -- the y response vector 
        """
        self.x = data
        self.y = response
        self.k = self.x.shape[1] - 1
        self.n = self.x.shape[0]
        self.model = NeuralNet4L.build_model(self.x, self.y)
    
    def build_model(x, y, suppress = False, width1 = 0, width2 = 0): 
        k = x.shape[1] - 1
        #Default Width Value
        if (width1 == 0): 
            width1 = k + 1
            
        if (width2 == 0): 
            width2 = width1//2
        
        model = keras.Sequential()
        model.add(layers.Dense(width1, input_dim = x.shape[1], 
                            kernel_initializer = "normal", 
                            use_bias = False,
                            activation = "relu", ))
        model.add(layers.Dense(width2,
                            kernel_initializer = "normal", 
                            activation = "relu"))
        model.add(layers.Dense(1, kernel_initializer = "normal"))
        model.compile(loss = "mean_squared_error", optimizer = "adam")
        if not suppress: 
            model.summary()
        return model
    
    def forward_selection(self, iter = 200): 
        return VariableSelection.forward_selection(self.x, self.y, 
                                                NeuralNet4L.build_model,
                                                iter)
    
    def backward_elimination(self, iter = 200):
        return VariableSelection.backward_elimination(self.x, self.y, 
                                                NeuralNet4L.build_model,
                                                iter)
    
    def stepwise_regression(self, iter = 200): 
        return VariableSelection.stepwise_regression(self.x, self.y, 
                                                    NeuralNet4L.build_model,
                                                    iter)

class VariableSelection(): 

    def forward_selection(x, y, build_fn, iter = 1000): 
        """Does forward selection all the way through adding one variable using Rsq_adj."""
        k = x.shape[1] - 1
        toBeIncluded = [0]
        toBeTested = [_ for _ in range(1, k + 1)]
        rsq = []
        rsq_adj = []
        rsq_cv = []
        aic = []
        for i in range(k):
            bestVar, bestModel, _ = VariableSelection.forward_sel_one(build_fn, toBeIncluded, toBeTested, x, y, iter = iter)
            
            toBeIncluded = toBeIncluded + [bestVar]
            toBeTested.remove(bestVar)



            subsetX = x[:, toBeIncluded]

            rsq.append(metrics.rsq(bestModel, subsetX, y))
            rsq_adj.append(metrics.rsq_adj(bestModel, subsetX, y))
            aic.append(metrics.aic(bestModel, subsetX, y))
            rsq_cv.append(metrics.rsq_cv(bestModel, subsetX, y))
            
            print(toBeIncluded)
            print(metrics.rsq_adj(bestModel, subsetX, y))


        return rsq, rsq_adj, rsq_cv, aic

    def forward_sel_one(build_fn, toBeIncluded, toBeTested, x, y, iter = 1000):
        bestAdjRsq = 0
        bestVar = 0
        bestModel = None
        for j in toBeTested:
            subsetList = toBeIncluded + [j]
            subsetX = x[:, subsetList]
            model = build_fn(subsetX, y, True)
            model.fit(subsetX, y, epochs = iter, batch_size = 20, verbose = 0)
            adjRsq = metrics.rsq_adj(model, subsetX, y)
            if (bestAdjRsq == 0): 
                bestAdjRsq = adjRsq
            if (adjRsq >= bestAdjRsq): 
                bestAdjRsq = adjRsq
                bestVar = j
                bestModel = model

        return bestVar, bestModel, bestAdjRsq
    
    def backward_elimination(x, y, build_fn, iter = 1000):
        k = x.shape[1] - 1
        toBeIncluded = [_ for _ in range(k + 1)]
        
        initial_model = build_fn(x, y, True)
        initial_model.fit(x , y, epochs = iter, batch_size = 20, verbose = 0)
        rsq = [metrics.rsq(initial_model, x, y)]
        rsq_adj = [metrics.rsq_adj(initial_model, x, y)]
        rsq_cv = [metrics.rsq_cv(initial_model, x, y)]
        aic = [metrics.aic(initial_model, x, y)]
        

        
        for i in range(k - 1): 
            worstVar, bestModel, _ = VariableSelection.backward_elim_one(build_fn, toBeIncluded, x, y, iter)
            
            toBeIncluded.remove(worstVar)
            subsetX = x[:, toBeIncluded]
                        
            rsq.append(metrics.rsq(bestModel, subsetX, y))
            rsq_adj.append(metrics.rsq_adj(bestModel, subsetX, y))
            aic.append(metrics.aic(bestModel, subsetX, y))
            rsq_cv.append(metrics.rsq_cv(bestModel, subsetX, y))
            
            print(toBeIncluded)
            print(metrics.rsq_adj(bestModel, subsetX, y))
            
        return rsq, rsq_adj, rsq_cv, aic
            
            
    def backward_elim_one(build_fn, toBeIncluded, x, y, iter = 1000): 
        worstVar = 0 
        worstRsqAdj = 0
        bestRsqAdj = 0
        bestModel = None
        for j in toBeIncluded: 
            subsetList = toBeIncluded.copy()
            subsetList.remove(j)
            if (j != 0): 
                subsetX = x[:, subsetList]
                model = build_fn(subsetX, y, True)
                model.fit(subsetX, y, epochs = iter, batch_size = 20, verbose = 0)
                
                adjRsq = metrics.rsq_adj(model, subsetX, y)
                if (worstRsqAdj == 0): 
                    worstRsqAdj = adjRsq
                    bestRsqAdj = adjRsq
                    
                if (adjRsq <= worstRsqAdj): 
                    worstRsqAdj = adjRsq
                    worstVar = j
                    
                if (adjRsq >= bestRsqAdj): 
                    bestRsqAdj = adjRsq
                    bestModel = model
        
        return worstVar, bestModel, bestRsqAdj
    
    def stepwise_regression(x, y, build_fn, iter = 1000):
        k = x.shape[1] - 1
        toBeIncluded = [0]
        toBeTested = [_ for _ in range(1, k + 1)]
        
        rsq = []
        rsq_adj = []
        rsq_cv = []
        aic = []
        
        improving = True
        toBeat = 0
        counter = 0
        while improving: 
            if (counter == k): 
                return rsq, rsq_adj, rsq_cv, aic

            print(toBeIncluded)

            bestVar, bestAddModel, bestAddAdjRsq = VariableSelection.forward_sel_one(build_fn,
                                                                                        toBeIncluded, 
                                                                                        toBeTested,
                                                                                        x, y, 
                                                                                        iter = iter)
            worstVar, bestElimModel, bestElimAdjRsq = VariableSelection.backward_elim_one(build_fn, 
                                                                                        toBeIncluded, 
                                                                                        x, y, 
                                                                                        iter)

            if (toBeat == 0): 
                toBeat = bestAddAdjRsq
                
            if (bestElimAdjRsq == 0): 
                bestElimAdjRsq = bestAddAdjRsq - 10
                
            if (toBeat > bestAddAdjRsq and toBeat > bestElimAdjRsq): 
                improving = False
                
            if (bestElimAdjRsq > bestAddAdjRsq and bestElimAdjRsq >= toBeat and bestElimAdjRsq != 0): 
                toBeIncluded.remove(worstVar)
                toBeTested.append(worstVar)
                
                subsetX = x[:, toBeIncluded]
                rsq.append(metrics.rsq(bestElimModel, subsetX, y))
                rsq_adj.append(metrics.rsq_adj(bestElimModel, subsetX, y))
                aic.append(metrics.aic(bestElimModel, subsetX, y))
                rsq_cv.append(metrics.rsq_cv(bestElimModel, subsetX, y))
                
                toBeat = bestAdjRsq
                
            if (bestAddAdjRsq > bestElimAdjRsq and bestAddAdjRsq >= toBeat): 
                toBeIncluded.append(bestVar)
                if (bestVar in toBeTested):
                    toBeTested.remove(bestVar)
                subsetX = x[:, toBeIncluded]
                rsq.append(metrics.rsq(bestAddModel, subsetX, y))
                rsq_adj.append(metrics.rsq_adj(bestAddModel, subsetX, y))
                aic.append(metrics.aic(bestAddModel, subsetX, y))
                rsq_cv.append(metrics.rsq_cv(bestAddModel, subsetX, y))
                
                toBeat = bestAddAdjRsq

            counter = counter + 1
            
        return rsq, rsq_adj, rsq_cv, aic