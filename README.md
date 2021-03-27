# Project Two 

## How to use the Models module 

There are 3 classes for the Models, these are the Perceptron class, 
the NeuralNet_3L class and he NeuralNet_4L class. Each of these classes 
function much the same, and have default models with them that work 
with the AutoMPG dataset and  may or may not work with other datasets as well. 

If you want to pass your own model into these classes you need to create a build 
function. The build function must have the following signature: 

```python 
    def build_fn(x, y, suppress = False, learning_rate = 0.05, **kwargs): 
```

Any builld function with this signature will function, but if you don't want to 
go throught the trouble of making all the arguments, this minimial signature 
will work as well. 

```python 
    def build_fn(x, y, suppress = False):
```    

The build function must return a compiled keras model, and can be passed as an argument 
into the constructor of any of the 3 model modules. Here is an example build function 
(NeuralNet_4L Default build_function): 

```python
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
```

Here you can notice that model widths are dependent on the size of input vector x. 
The additional two width arguments are not needed for an external build function. 
Happy Coding! 

## ToDo List 

The ToDo list can be found in the excel file, and does not take into account plotting and 
reporting. It is only a list of the models that must be coded. Scalation Code is coming soon. 

## Known Issues 

### Models.metrics 

 - The `rsq_adj` function uses an incorrect k value to calculate the rsq_cv. It uses the number of 
input paramters as the model complexity, but obviously this breaks down for larger models like the 
3 and 4 Layer Nets. 
- The `rsq_cv` function suffers from data leakage, must be fixed to reset model before training otherwise
it returns higher values than the `rsq` function. Currently unreliable, but can be fixed by resetting 
model parameters before fitting. Other alternative is to pass in a build_fn. Fix is coming soon. 
