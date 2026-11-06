## Definition

[[Topic 4.2 - Fitting Models|Previously]] we used SGD and Adam to update neural-network parameters. These updates require the gradient of the loss with respect to every weight and bias. **Backpropagation** efficiently computes these derivatives.

It consists of a **forward pass**, which calculates and stores intermediate activations, and a **backward pass**, which applies the chain rule from the loss back through the network while reusing computations.

## Problem Definition

Consider a network with three hidden layers:

$$
\mathbf{h}_1=a[\boldsymbol{\beta}_0+\mathbf{\Omega}_0\mathbf{x}],
\quad
\mathbf{h}_2=a[\boldsymbol{\beta}_1+\mathbf{\Omega}_1\mathbf{h}_1],
$$

$$
\mathbf{h}_3=a[\boldsymbol{\beta}_2+\mathbf{\Omega}_2\mathbf{h}_2],
\quad
f[\mathbf{x},\boldsymbol{\phi}]
=\boldsymbol{\beta}_3+\mathbf{\Omega}_3\mathbf{h}_3.
$$

The parameters are bias vectors $\boldsymbol{\beta}_k$ and weight matrices $\mathbf{\Omega}_k$. For training pair $(\mathbf{x}_i,y_i)$, let $\ell_i$ be its individual loss; then:

$$
L[\boldsymbol{\phi}]=\sum_{i=1}^{I}\ell_i.
$$

SGD uses a batch $B_t$ to update:

$$
\boldsymbol{\phi}_{t+1}
\leftarrow
\boldsymbol{\phi}_t
-\alpha
\sum_{i\in B_t}
\frac{\partial\ell_i[\boldsymbol{\phi}_t]}
{\partial\boldsymbol{\phi}}.
$$

Therefore, each layer requires:

$$
\frac{\partial\ell_i}{\partial\boldsymbol{\beta}_k},
\qquad
\frac{\partial\ell_i}{\partial\mathbf{\Omega}_k}.
$$

## Computing Derivatives

Each weight multiplies the activation of its source hidden unit. Consequently, the effect of changing that weight is proportional to the stored source activation. This motivates the forward pass: run each example through the network and cache all hidden activations.

A parameter change also propagates forward through every subsequent layer before affecting the loss. The downstream derivatives needed for an early parameter have already been calculated when differentiating a later parameter. The backward pass exploits this redundancy by working from output to input.

## Toy Example

Consider a scalar model:

$$
f[x,\boldsymbol{\phi}]
=\beta_3+\omega_3\cos\left[
\beta_2+\omega_2\exp\left[
\beta_1+\omega_1\sin[\beta_0+\omega_0x]
\right]\right].
$$

with least-squares loss:

$$
\ell_i=(f[x_i,\boldsymbol{\phi}]-y_i)^2.
$$

Rather than differentiate the entire nested expression separately for every parameter, define intermediate values:

$$
\begin{aligned}
f_0&=\beta_0+\omega_0x_i, & h_1&=\sin[f_0],\\
f_1&=\beta_1+\omega_1h_1, & h_2&=\exp[f_1],\\
f_2&=\beta_2+\omega_2h_2, & h_3&=\cos[f_2],\\
f_3&=\beta_3+\omega_3h_3, & \ell_i&=(f_3-y_i)^2.
\end{aligned}
$$

### Backward pass

Start at the output:

$$
\frac{\partial\ell_i}{\partial f_3}=2(f_3-y_i).
$$

Then use the chain rule repeatedly:

$$
\frac{\partial\ell_i}{\partial h_3}
=
\frac{\partial f_3}{\partial h_3}
\frac{\partial\ell_i}{\partial f_3}.
$$

Each preceding derivative is its local derivative multiplied by the derivative already obtained from the following computation. This reuse is the core efficiency of backpropagation.

Once $\partial\ell_i/\partial f_k$ is known, the parameter derivatives follow:

$$
\frac{\partial\ell_i}{\partial\beta_k}
=
\frac{\partial f_k}{\partial\beta_k}
\frac{\partial\ell_i}{\partial f_k},
\qquad
\frac{\partial\ell_i}{\partial\omega_k}
=
\frac{\partial f_k}{\partial\omega_k}
\frac{\partial\ell_i}{\partial f_k}.
$$

For $f_k=\beta_k+\omega_kh_k$:

$$
\frac{\partial f_k}{\partial\beta_k}=1,
\qquad
\frac{\partial f_k}{\partial\omega_k}=h_k.
$$

Thus, a weight gradient is proportional to the activation the weight multiplied in the forward pass.

## Backpropagation Algorithm

For a deep ReLU network, define:

$$
\mathbf{f}_0=\boldsymbol{\beta}_0+\mathbf{\Omega}_0\mathbf{x}_i,
$$

$$
\mathbf{h}_k=a[\mathbf{f}_{k-1}],
\qquad
\mathbf{f}_k=\boldsymbol{\beta}_k+\mathbf{\Omega}_k\mathbf{h}_k,
\qquad
k\in\{1,\ldots,K\},
$$

$$
\ell_i=l[\mathbf{f}_K,y_i].
$$

The $\mathbf{f}_k$ values are pre-activations; the $\mathbf{h}_k$ values are post-ReLU activations.

### Backward pass through ReLU

ReLU has derivative zero for negative inputs and one for positive inputs. During backpropagation, each layer:

1. multiplies the arriving derivative by the transposed weight matrix; and
2. masks it with the stored ReLU inputs.

$$
\frac{\partial\ell_i}{\partial\mathbf{f}_{k-1}}
=
\mathbf{I}[\mathbf{f}_{k-1}>0]
\odot
\left(
\mathbf{\Omega}_k^T
\frac{\partial\ell_i}{\partial\mathbf{f}_k}
\right).
$$

Here, $\odot$ is elementwise multiplication.

### Bias and weight derivatives

The bias derivative is:

$$
\frac{\partial\ell_i}{\partial\boldsymbol{\beta}_k}
=
\frac{\partial\ell_i}{\partial\mathbf{f}_k}.
$$

The weight derivative is an outer product:

$$
\frac{\partial\ell_i}{\partial\mathbf{\Omega}_k}
=
\frac{\partial\ell_i}{\partial\mathbf{f}_k}
\mathbf{h}_k^T.
$$

For the first layer, use the input vector instead:

$$
\frac{\partial\ell_i}{\partial\mathbf{\Omega}_0}
=
\frac{\partial\ell_i}{\partial\mathbf{f}_0}
\mathbf{x}_i^T.
$$

Compute these gradients for every example in the batch, then sum them for the optimizer update.

### Algorithmic differentiation

Frameworks such as PyTorch and TensorFlow automate the forward and backward passes through **algorithmic differentiation**. Each layer knows its local derivative, and the framework records the computational graph needed to apply the chain rule in reverse.

Backpropagation is computationally efficient because its costly operations are matrix multiplications. Its main cost is memory: activations from the forward pass must remain available until the backward pass.