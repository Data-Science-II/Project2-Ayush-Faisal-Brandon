# Import Libraries
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


# Helper Functions
def concrete_pcp_build(x, y, suppress = False): 
    model = keras.Sequential()
    model.add(layers.Dense(1, input_dim = x.shape[1], 
                           kernel_initializer = "uniform", 
                           activation = "relu", 
                           use_bias = False))
    optimizer = keras.optimizers.Adam(learning_rate = 0.0005)
    model.compile(loss = "mean_squared_error", optimizer = optimizer)
    return model

def print_fit(model): 
    print(f"""
            Rsq = {model.rsq()}
            Rsq_adj = {model.rsq_adj()}
            AIC = {model.aic()}
          """)

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


# Prepare variables 
concrete = pd.read_csv("../data/concrete.csv")
concrete.fillna(concrete.mean())
x = concrete[concrete.columns[0:7]].to_numpy()
ox = np.insert(x, 0, 1.0, axis = 1)
ox = ox/lalg.norm(ox)
y = concrete[concrete.columns[8]].to_numpy()
y = y/lalg.norm(y)


###Perceptron 
concrete_perceptron = Perceptron(ox, y, 0.1, build_fn = concrete_pcp_build)
# Perceptron Forward
forward_pcp = concrete_perceptron.forward_selection(20)
#print("Perceptron Forward: ", forward_pcp)
# Perceptron Backward
backward_pcp = concrete_perceptron.backward_elimination(20)
#print("Perceptron Backward: ", backward_pcp)
# Perceptron Stepwise
stepwise_pcp = concrete_perceptron.stepwise_regression(20)
#print("Perceptron Stepwise: ", stepwise_pcp)


###3LayerNetwork 
concrete_3L = NeuralNet3L(ox, y, build_fn = NeuralNet3L.build_model)
# 3LayerNN Forward
forward_3L = concrete_3L.forward_selection()
#print("3LayerNN Forward: ", forward_3L)
# 3LayerNN Backward
backward_3L = concrete_3L.backward_elimination()
#print("3LayerNN Backward: ", backward_3L)
# 3LayerNN Stepwise
stepwise_3L = concrete_3L.stepwise_regression()
#print("3LayerNN Stepwise: ", stepwise_3L)


###4LayerNetwork 
concrete_4L = NeuralNet4L(ox, y, build_fn = NeuralNet4L.build_model)
# 4LayerNN Forward
forward_4L = concrete_4L.forward_selection()
#print("4LayerNN Forward: ", forward_4L)
# 4LayerNN Backward
backward_4L = concrete_4L.backward_elimination()
#print("4LayerNN Backward: ", backward_4L)
# 4LayerNN Stepwise
stepwise_4L = concrete_4L.stepwise_regression()
#print("4LayerNN Stepwise: ", stepwise_4L)

concrete_perceptron = Perceptron(ox, y, 0.1, build_fn = concrete_pcp_build)
concrete_3L = NeuralNet3L(ox, y, build_fn = NeuralNet3L.build_model)
concrete_4L = NeuralNet4L(ox, y, build_fn = NeuralNet4L.build_model)

concrete_perceptron.train(500)
concrete_3L.train(200)
concrete_4L.train(300)

print_fit(concrete_perceptron)
print_fit(concrete_3L)
print_fit(concrete_4L)
"""
###Ridge Perceptron 
ridge_perceptron = keras.Sequential()
ridge_perceptron.add(layers.Dense(1, input_dim = 9, 
                                 kernel_initializer = "uniform", 
                                 activation = "relu", 
                                 use_bias = False,
                                 kernel_regularizer = rg.l2(0.01)))

optimizer = keras.optimizers.Adam(learning_rate = 0.0005)
ridge_perceptron.compile(loss = "mean_squared_error", optimizer = optimizer)

ridge_perceptron.fit(ox, y, epochs = 50, batch_size = 10, verbose = 0)
rsq_cv = metrics.rsq_cv(ridge_perceptron, ox, y, epochs = 500)
print(f"Rsq = {rsq} Rsq_cv = {rsq_cv}")
"""

# Plots    
plot_and_save(forward_pcp, "ConcreteForwardPCP.png")
plot_and_save(backward_pcp, "ConcreteBackWardPCP.png")
plot_and_save(stepwise_pcp, "ConcreteStepwisePCP.png")

plot_and_save(forward_3L, "ConcreteForward3L.png")
plot_and_save(backward_3L, "ConcreteBackWard3L.png")
plot_and_save(stepwise_3L, "ConcreteStepwise3L.png")

plot_and_save(forward_4L, "ConcreteForward4L.png")
plot_and_save(backward_4L, "ConcreteBackWard4L.png")
plot_and_save(stepwise_4L, "ConcreteStepwise4L.png")
