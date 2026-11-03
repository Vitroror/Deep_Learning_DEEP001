## Definition

Following the concept of [[Topic 5.1 - Backpropagation|backpropagation]], **parameter initialization** means choosing weights and biases before training so that forward activations and backward gradients retain sensible magnitudes across network depth.

If values shrink rapidly, training suffers from vanishing activations or gradients. If they grow rapidly, it suffers from exploding activations or gradients.

## Parameter Initialization

At a layer:

$$
\mathbf{f}_k
=
\boldsymbol{\beta}_k+\mathbf{\Omega}_k\mathbf{h}_k
=
\boldsymbol{\beta}_k+\mathbf{\Omega}_ka[\mathbf{f}_{k-1}].
$$

Suppose biases begin at zero and weights are sampled with mean zero and variance $\sigma_{\Omega}^2$.

- If $\sigma_{\Omega}^2$ is too small, activations decrease at each layer; ReLU removes roughly half of them as well.
- If it is too large, activation magnitudes grow with depth.

The same multiplicative behavior occurs in the backward pass, where gradients are multiplied by $\mathbf{\Omega}^T$. Small values produce vanishing gradients and negligible updates; large values produce exploding gradients and unstable updates.

## Initialization for the Forward Pass

For adjacent layers:

$$
\mathbf{h}=a[\mathbf{f}],
\qquad
\mathbf{f}'=\boldsymbol{\beta}+\mathbf{\Omega}\mathbf{h}.
$$

If the input layer has $D_h$ units, its pre-activations have variance $\sigma_f^2$, and ReLU is used, then:

$$
\sigma_{f'}^2
=
\frac{1}{2}D_h\sigma_{\Omega}^2\sigma_f^2.
$$

The factor $1/2$ reflects that a symmetric distribution has about half its values clipped to zero by ReLU. To preserve activation variance, set:

$$
\sigma_{\Omega}^2=\frac{2}{D_h}.
$$

This is **He initialization**, where $D_h$ is the number of inputs to the layer.

## Initialization for the Backward Pass

To preserve the variance of backward gradients, the corresponding condition is:

$$
\sigma_{\Omega}^2=\frac{2}{D_{h'}},
$$

where $D_{h'}$ is the number of units the weights feed into.

When neighboring layers differ in width, no one variance satisfies both conditions exactly. A compromise is:

$$
\sigma_{\Omega}^2
=
\frac{4}{D_h+D_{h'}}.
$$

Appropriate initialization keeps activations and gradients stable and avoids exploding or vanishing values.

## Example Training Code

The chapter's PyTorch example shows the complete workflow:

1. define a network with linear layers and ReLU activations;
2. apply He initialization to linear weights and set biases to zero;
3. choose mean-squared error and SGD with momentum;
4. load shuffled minibatches;
5. clear gradients, run the forward pass, compute loss, run the backward pass, and update parameters; and
6. reduce the learning rate on a schedule.

Although backpropagation is mathematically detailed, frameworks hide it behind the loss backward operation.
