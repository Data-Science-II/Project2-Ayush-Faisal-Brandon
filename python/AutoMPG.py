import pandas as pd 
import numpy as np
import numpy.linalg as lalg
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from Models import *

#Prepare variables 
auto = pd.read_csv("../data/auto-mpg.csv")
auto.fillna(auto.mean())
x = auto[auto.columns[1:7]].to_numpy()
ox = np.insert(x, 0, 1.0, axis = 1)
ox = ox/lalg.norm(ox)
y = auto[auto.columns[0]].to_numpy()
y = y/lalg.norm(y)

###Perceptron 
auto_perceptron = Perceptron(ox, y, 0.1, build_fn = Perceptron.build_model)
print(auto_perceptron.forward_selection(5000))
print(auto_perceptron.backward_elimination(5000))
print(auto_perceptron.stepwise_regression(5000))

###3LayerNetwork 
auto_3L = NeuralNet3L(ox, y, build_fn = NeuralNet3L.build_model)
print(auto_3L.forward_selection())
print(auto_3L.backward_elimination())
print(auto_3L.stepwise_regression())

###4LayerNetwork 
auto_4L = NeuralNet4L(ox, y, build_fn = NeuralNet4L.build_model)
print(auto_4L.forward_selection())
print(auto_4L.backward_elimination())
print(auto_4L.stepwise_regression())
