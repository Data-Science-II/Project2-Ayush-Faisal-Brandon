#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import numpy.linalg as lalg
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.regularizers as rg
from sklearn.model_selection import KFold
from Models import *


# In[2]:


emissions = pd.read_csv("../data/emissions.csv")
emissions.fillna(emissions.mean())
x = emissions[emissions.columns[1:11]].to_numpy()
ox = np.insert(x, 10, 1.0, axis = 1)
y = emissions[emissions.columns[0]].to_numpy()


# In[ ]:


auto_perceptron = Perceptron(ox, y, 0.1, build_fn = Perceptron.build_model)
forward_pcp = auto_perceptron.forward_selection(5000)
backward_pcp = auto_perceptron.backward_elimination(5000)
step_pcp = auto_perceptron.stepwise_regression(5000)


# In[ ]:


auto_3L = NeuralNet3L(ox, y, build_fn = NeuralNet3L.build_model)
forward_3L = auto_3L.forward_selection()
backward_3L = auto_3L.backward_elimination()
step_3L = auto_3L.stepwise_regression()


# In[ ]:


auto_4L = NeuralNet4L(ox, y, build_fn = NeuralNet4L.build_model)
forward_4L = auto_4L.forward_selection()
backward_4L = auto_4L.backward_elimination()
step_4L = auto_4L.stepwise_regression()


# In[ ]:


ridge_perceptron = keras.Sequential()
ridge_perceptron.add(layers.Dense(1, input_dim = 11, 
                                 kernel_initializer = "uniform", 
                                 activation = "relu", 
                                 use_bias = False,
                                 kernel_regularizer = rg.l2(0.01)))

optimizer = keras.optimizers.Adam(learning_rate = 0.0005)
ridge_perceptron.compile(loss = "mean_squared_error", optimizer = optimizer)


# In[ ]:


ridge_perceptron.fit(ox, y, epochs = 50, batch_size = 10, verbose = 0)
rsq = metrics.rsq(ridge_perceptron, ox, y)
rsq_cv = metrics.rsq_cv(ridge_perceptron, ox, y, epochs = 50)
print(f"Rsq = {rsq} Rsq_cv = {rsq_cv}")


# In[ ]:


def plot_and_save(arrays, name, basepath = "../plots/python/"): 
    rsq, rsq_a, rsq_cv, aic = arrays
    x = [_ for _ in range(len(rsq))]
    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = [10,10]
    plt.plot(x, np.array([rsq, rsq_a, rsq_cv]).transpose())
    plt.xlabel("Number of variables")
    plt.ylabel("Rsq Value")
    plt.legend(["Rsq", "RsqAdj", "RsqCV"])
    plt.savefig(basepath+name)
    plt.show()
    
    plt.style.use("fivethirtyeight")
    plt.plot(x, aic)
    plt.xlabel("Number of Variables")
    plt.ylabel("AIC")
    plt.savefig(basepath+"AIC"+name)
    plt.show()
    
plot_and_save(forward_pcp, "AutoForwardPCP.png")
plot_and_save(backward_pcp, "BackWardPCP.png")
plot_and_save(step_pcp, "StepwisePCP.png")

plot_and_save(forward_3L, "AutoForward3L.png")
plot_and_save(backward_3L, "BackWard3L.png")
plot_and_save(step_3L, "Stepwise3L.png")

plot_and_save(forward_4L, "AutoForward4L.png")
plot_and_save(backward_4L, "BackWard4L.png")
plot_and_save(step_4L, "Stepwise4L.png")


# In[ ]:




