
# Deep Neural Networks


## Definition

[[Topic 3.1 - Shallow Networks|Previously]], we discussed Shallow networks, and how they work and function. This topic surrounds Deep Networks. These neural networks are named **Deep**, because they oppose shallow networks in the sense that they comprise more than one **hidden layer**. But why would i use a deep network instead of just a shallow one? Well the figure below allows us to figure out a benefit.

![[Pasted image 20261026160335.png]]

Notice how, when using a deep network, sometimes we can achieve a higher number of separation regions with mostly the same ammount of parameters, and more separation regions define how complex the network can become.

To better define this, we must start from the principle of combining shallow networks.

## Composing and Combining Two Networks

Let us define two shallow neural networks, the first being $$h_1 = a[\theta_{10}+\theta_{11}x]\\h_2 = a[\theta_{20}+\theta_{21}x]\\h_3 = a[\theta_{30}+\theta_{31}x]$$ with $y = \phi_0 + \phi_1h_1 + \phi_2h_2 + \phi_3h_3$, and the second being $$h'_1 = a[\theta'_{10}+\theta'_{11}y]\\ h'_2 = a[\theta'_{20}+\theta'_{21}y]\\ h'_3 = a[\theta'_{30}+\theta'_{31}y]$$ with $y' = \phi'_0 + \phi'_1h'_1 + \phi'_2h'_2 + \phi'_3h'_3$. This definition gives us two networks, where one network's output becomes the input of another network.

![[Pasted image 20261026155003.png]]


We can expand the previous equation by substituting $y$, like so $$h'_1 = a[\theta'_{10}+\theta'_{11}\phi_0+\theta'_{11}\phi_1h_1+\theta'_{11}\phi_2h_2+\theta'_{11}\phi_3h_3]\\ h'_2 = a[\theta'_{20}+\theta'_{21}\phi_0+\theta'_{21}\phi_1h_1+\theta'_{21}\phi_2h_2+\theta'_{21}\phi_3h_3]\\ h'_3 = a[\theta'_{30}+\theta'_{31}\phi_0+\theta'_{31}\phi_1h_1+\theta'_{31}\phi_2h_2+\theta'_{31}\phi_3h_3]$$
we can also reduce the effective size of this equation by introducing a new variable, in a way that the equation becomes $$h'_1 = a[\psi_10+\psi_{11}h_1+\psi_{12}h_2+\psi_{13}h_3]\\ h'_2 = a[\psi_20+\psi_{21}h_1+\psi_{22}h_2+\psi_{23}h_3]\\ h'_3 = a[\psi_30+\psi_{31}h_1+\psi_{32}h_2+\psi_{33}h_3]$$
This equation generates the following network

![[Pasted image 20261026162511.png]]

![[Pasted image 20261026162753.png]]


## Notation

For deep networks, a different notation is used to better generalize the topic. Let $\beta_0$ be the bias vector, and $\Omega_0$ be the weight matrix, we can generalize all previous equations in a matrix\vector notation as being$$h_1 = a[\beta_0+\Omega_0x]\\ h_2 = a[\beta_1 + \Omega_1h_1]\\ h_3 = a[\beta_2+\Omega_2h_2]\\ .\\ .\\ .\\ h_K=a[\beta_{K-1}+\Omega_{K-1}h_{K-1}]$$
where $y = \beta_K+\Omega_Kh_K$. If we consider the **hyperparameters** $D$ being network width and $K$ being network depth, we have.

![[Pasted image 20261026165813.png]]


Recovering the previous idea of the capacity of deep networks, the below figure determines the complexity boost deep networks can achieve when compared to shallow networks.

![[Pasted image 20261026170140.png]]


