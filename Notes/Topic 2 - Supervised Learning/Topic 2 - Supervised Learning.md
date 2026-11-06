## Definition

As mentioned in the previous [[Topic 1 - Notes|topic]], **Supervised Learning (SL)** is a type of machine learning where input and output pairs are associated in order to map a pattern. In this context, **training** involves finding parameters that predict outputs "well" from inputs for a **training dataset** of input and output pairs.

## Basic Model Notation

We begin with a model that takes $x$ and returns $y$. This model can be defined as $$y = f[x]$$
In this instance, when we compute the prediction $y$ from the input $x$, we call it **inference**. The model is, therefore, just a function with a fixed form. We could also include in this model the use of $\phi$ parameters, such that $$y = f[x, \phi]$$
The presence of parameters is relevant to **training**. When we train a model, we determine the set of parameters that allows it to make sensible predictions from the same input, making inferences more precise. 

That being said, to train a model, another important parameter is the **Loss Function**.
## Loss Function

Given a training dataset of $I$ pairs of input/output$$\{x_i,y_i\}^I_{i=1}$$
the Loss function measures how inadequate the model is.$$L[\phi, f[x,\phi], \{x_i,y_i\}^I_{i=1}]$$
this function returns a scalar that is smaller when models map inputs to outputs better. The idea is to find the parameters that minimize this loss function or, in mathematical terms $$\hat{\phi} = argmin[L[\phi]]$$ 

## Training a Supervised Learning Model

Let us take as an example a linear $1D$ regression model:

![[Pasted image 20261006180607.png]]

based on training data, we can calculate the loss function like so:

![[Pasted image 20261006180746.png]]

This visual example shows that the used parameters have a higher loss function mathematically. Ideally we can change these parameters to minimize the loss:

![[Pasted image 20261006181011.png]]

![[Pasted image 20261006181025.png]]

![[Pasted image 20261006181041.png]]

Once we reach the optimized parameters for a minimal loss function, we can consider that the model was effectively **trained**. The next step is to pair this model with another dataset, which is a DIFFERENT dataset than the training dataset, called an **inference dataset**, to measure performance. When this happens, the degree to which this performance is the same as training is called the degree of **generalization**.  