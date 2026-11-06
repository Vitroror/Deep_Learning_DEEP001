## Definition

On the last [[Topic 2 - Supervised Learning#Training a Supervised Learning Model|topic]], we used a linear 1D model to describe the training process of a model. The issue is this model can only describe behavior of an input/output relationship as a line. Therefore we need more complex learning models to better describe more complex relationships. For this, we use **Neural Networks**. In this topic, we discuss **Shallow Neural Networks**.

Shallow networks are also functions $y = f[x, \phi]$ as seen [[Topic 2 - Supervised Learning#Basic Model Notation|previously]]. In this context, we can provide a more simplistic definition of a neural network as an example, considering ten parameters $\phi = \{\phi_1,\phi_2,\phi_3,\theta_{10},\theta_{11},\theta_{20},\theta_{21},\theta_{30},\theta_{31} \}$ $$y = f[x, \phi]\\\space=\phi_0+ \phi_1a[\theta_{10}+\theta_{11}x] + \phi_2a[\theta_{20}+\theta_{21}x]+\phi_3a[\theta_{30}+\theta_{31}x]$$
First, we compute the three linear functions of input data, then we pass the results into an **activation function** and finally we weigh the three resulting activations, sum them and add an offset. 

## The Activation Function 

There are many possibilities for an activation function, but for this concept we will use the **rectified linear unit** or **ReLU**. $$\text{a}[z] = \text{ReLU}[z] = \begin{cases} 0 & z < 0 \\ z & z \ge 0 \end{cases}.$$
## Hidden Units and the depiction of a neural network

We can also write each activation as intermediate quantities $$h_1 = a[\theta_{10}+\theta_{11}x]\\h_2 = a[\theta_{20}+\theta_{21}x]\\h_3 = a[\theta_{30}+\theta_{31}x]$$
we refer to each intermediate activation equation as a **hidden unit**. this way, we can rewrite the model as $$y = \phi_0 + \phi_1h_1 + \phi_2h_2 + \phi_3h_3$$
This way, the figure below describes each slope and activation unit, and its effects visually:

![[Pasted image 20261013174320.png]]

with this final definition, the visual depiction of a neural network can be seen in the image below. the leftmost depiction is not the usual one, but serves to represent the intercept parameters. The leftmost bubbles represent the input, the rightmost are the output and the middle bubbles represent the hidden layer, where each bubble is a hidden unit, or **neuron**.

![[Pasted image 20261013175513.png]]

## Multivariate inputs

In the case of a multi variable input, the hidden units might be expressed by$$h_1 = a[\theta_{10}+\theta_{11}x_1 + \theta_{12}x_2]\\h_2 = a[\theta_{20}+\theta_{21}x+\theta_{22}x_2]\\h_3 = a[\theta_{30}+\theta_{31}x+\theta_{32}x_2]$$
and the output equation remains the same. This way, the graphical representation of the network can be shown in the 3D plane as below

![[Pasted image 20261013180745.png]]

## Generalization of a Shallow Neural Network

After illustrating the multivariate inputs required, a neural network with a dimension of $D$ hidden units can be described as$$y_j = \phi_{j0} + \sum^D_{d=1}\phi_{jd}h_d$$ where $$h_d = a[\theta_{d0}+\sum^{Di}_{i=1}\theta_{di}x_i]$$
