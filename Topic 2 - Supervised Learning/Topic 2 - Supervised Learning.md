## Definition

As mentioned in the previous [[Topic 1 - Notes|topic]], **Supervised Learning (SL)** is a type of machine learning where input and output pairs are associated in order to map a pattern. In this context, **training** involves finding parameters that predict outputs "well" from inputs for a **training dataset** of input and output pairs.

## Notation

An SL model can be defined as $$y = f[x]\space\space\space or \space\space\space y=f[x,\phi]$$
where $y$ is the output, $x$ is the input, and $\phi$ are the parameters.

## Loss Function

Given a dataset of $I$ pairs of input/output$$\{x_i,y_i\}^I_{i=1}$$
the Loss function measures how inadequate the model is.$$L[\phi, f[x,\phi], \{x_i,y_i\}^I_{i=1}]$$
this function returns a scalar that is smaller when models map inputs to outputs better. The idea is to find the parameters that minimize this loss function. 

## Training a Supervised Learning Model

Let us take as an example a linear $1D$ regression model:

![[Pasted image 20261006180607.png]]

based on training data, we can calculate the loss function like so:

![[Pasted image 20261006180746.png]]

This visual example shows that the used parameters have a higher loss function mathematically. Ideally we can change these parameters to minimize this loss function:

![[Pasted image 20261006181011.png]]

![[Pasted image 20261006181025.png]]

![[Pasted image 20261006181041.png]]

Once we reach the optimized parameters for a minimal loss function, we can consider that the model was effectively **trained**. The next step is to pair this model with another dataset, which is necessarily a different dataset than the training dataset, called an **inference dataset**, to measure performance. When this happens, the degree to which this performance is the same as training is called the degree of **generalization**. 