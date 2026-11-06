## Definition

[[Topic 4.1 - Loss Functions|In the previous topic]], we defined a loss function $L[\boldsymbol{\phi}]$ as a single number that measures how poorly a neural network with parameters $\boldsymbol{\phi}$ describes the training data. **Fitting a model** (or training it) means finding the parameter values that minimize this loss:

$$
\hat{\boldsymbol{\phi}}
= \arg\min_{\boldsymbol{\phi}} L[\boldsymbol{\phi}].
$$

Training is iterative. We first choose initial parameter values, then repeatedly:

1. compute the derivatives of the loss with respect to the parameters; and
2. adjust the parameters in a direction that decreases the loss.

This chapter concerns the second step: the algorithms used to update parameters. The next chapter explains how to calculate gradients for neural networks and how to initialize their parameters.

## Gradient Descent

**Gradient descent** starts from an initial parameter vector $\boldsymbol{\phi}=[\phi_0,\phi_1,\ldots,\phi_N]^T$. At each iteration, it computes the gradient:

$$
\frac{\partial L}{\partial\boldsymbol{\phi}}
=
\begin{bmatrix}
\frac{\partial L}{\partial\phi_0}\\
\frac{\partial L}{\partial\phi_1}\\
\vdots\\
\frac{\partial L}{\partial\phi_N}
\end{bmatrix},
$$

then updates the parameters:

$$
\boldsymbol{\phi}
\leftarrow
\boldsymbol{\phi}
-\alpha\frac{\partial L}{\partial\boldsymbol{\phi}}.
$$

The gradient points in the direction in which the loss increases fastest, so the negative gradient points downhill. The positive scalar $\alpha$ determines the size of the step. When $\alpha$ is fixed, it is called the **learning rate**. Alternatively, a **line search** tests several candidate step sizes and selects the one that reduces the loss most.

At a minimum, the loss surface is flat, so its gradient is zero. In practice, optimization is stopped when the gradient becomes sufficiently small.

### Linear regression example

For a one-dimensional linear model,

$$
y=f[x,\boldsymbol{\phi}]
=\phi_0+\phi_1x,
$$

the least-squares loss over $I$ training examples is:

$$
L[\boldsymbol{\phi}]
=\sum_{i=1}^{I}\ell_i
=\sum_{i=1}^{I}(\phi_0+\phi_1x_i-y_i)^2.
$$

The total gradient is the sum of the per-example gradients:

$$
\frac{\partial L}{\partial\boldsymbol{\phi}}
=\sum_{i=1}^{I}\frac{\partial\ell_i}{\partial\boldsymbol{\phi}},
$$

where

$$
\frac{\partial\ell_i}{\partial\boldsymbol{\phi}}
=
\begin{bmatrix}
2(\phi_0+\phi_1x_i-y_i)\\
2x_i(\phi_0+\phi_1x_i-y_i)
\end{bmatrix}.
$$

For linear regression, the loss is **convex**: it has one well-defined global minimum. Therefore, if we continually move downhill, gradient descent reaches the best solution regardless of where it starts.

### Non-convex loss functions

Neural networks are nonlinear models, so their loss functions are usually **non-convex**. The chapter illustrates this with a two-parameter Gabor model:

$$
f[x,\boldsymbol{\phi}]
=\sin[\phi_0+0.06\phi_1x]
\cdot
\exp\left[
-\frac{(\phi_0+0.06\phi_1x)^2}{32.0}
\right].
$$

This function oscillates while its amplitude falls away from its center. Here, $\phi_0$ controls the center position, and $\phi_1>0$ stretches or compresses the function along the $x$-axis. Its least-squares loss can contain many valleys and flat regions, unlike the single bowl-shaped loss of linear regression.

### Local minima and saddle points

A **local minimum** is a point where small changes in every direction increase the loss, but another point elsewhere may have lower loss. The best minimum across the whole parameter space is the **global minimum**.

Gradient descent is entirely determined by its starting point. On a non-convex surface, it may descend into a local minimum and stop even though a better solution exists elsewhere.

A **saddle point** has zero gradient but is neither a minimum nor a maximum: the loss rises in some directions and falls in others. Gradient descent can escape if it is not exactly at the saddle point, but the surrounding surface may be very flat. A small gradient can therefore make training appear to have converged when it has only stalled near a saddle point.

## Stochastic Gradient Descent

For neural networks with millions of parameters, it is impractical to exhaustively search the parameter space or repeatedly restart gradient descent. **Stochastic gradient descent** (SGD) introduces noise into the update process. It still moves downhill on average, but an individual step may not follow the steepest downhill direction and may even move temporarily uphill.

This randomness can move the model from one loss valley to another, helping it escape poor local minima.

### Batches and epochs

At iteration $t$, SGD chooses a random subset $B_t$ of the training data, called a **minibatch** (or simply a batch), and computes the gradient using only those examples:

$$
\boldsymbol{\phi}_{t+1}
\leftarrow
\boldsymbol{\phi}_t
-\alpha
\sum_{i\in B_t}
\frac{\partial\ell_i[\boldsymbol{\phi}_t]}
{\partial\boldsymbol{\phi}}.
$$

The algorithm normally draws batches without replacement until each training example has been used once. One complete pass through the dataset is an **epoch**.

- A batch can contain one training example.
- A batch can contain the full dataset; this is **full-batch gradient descent**, identical to ordinary gradient descent.
- Smaller batches make each update cheaper but introduce more randomness.

Another useful interpretation is that SGD performs gradient descent on a different batch-specific loss function at every iteration. Although the update direction varies from batch to batch, the expected loss and expected gradient remain the same as in full-batch gradient descent.

### Properties of SGD

SGD has several practical benefits:

- Each update improves the model fit to its current batch, so updates tend to be useful even when noisy.
- Sampling without replacement gives all examples an equal contribution over an epoch.
- Computing a gradient from a batch costs less than computing it from the entire dataset.
- Noise can help the optimizer escape local minima and reduces the chance of stalling near saddle points.
- In practice, SGD often finds parameters that generalize well to unseen data.

SGD does not necessarily converge to one exact parameter vector. Near a good minimum, however, all batches should have small gradients, so updates become small.

### Learning-rate schedules

SGD is often paired with a **learning-rate schedule**. The learning rate starts relatively high and is reduced by a constant factor every $N$ epochs.

Early in training, larger steps encourage exploration of the parameter space and make it easier to cross from one valley to another. Later, smaller steps allow fine-tuning near a promising solution.

## Momentum

Momentum smooths SGD by combining the current batch gradient with the previous update direction:

$$
\mathbf{m}_{t+1}
\leftarrow
\beta\mathbf{m}_t
+(1-\beta)
\sum_{i\in B_t}
\frac{\partial\ell_i[\boldsymbol{\phi}_t]}
{\partial\boldsymbol{\phi}},
$$

$$
\boldsymbol{\phi}_{t+1}
\leftarrow
\boldsymbol{\phi}_t
-\alpha\mathbf{m}_{t+1}.
$$

Here, $\mathbf{m}_t$ is the momentum term and $\beta\in[0,1)$ controls how strongly earlier gradients affect the current update.

The recursive equation makes momentum an infinite weighted sum of previous gradients, with older gradients receiving smaller weights. When gradients keep pointing in a similar direction, the effective step grows. When their direction changes repeatedly, they partly cancel. The result is a smoother trajectory with less oscillation in narrow valleys and typically faster convergence.

### Nesterov accelerated momentum

Momentum can be viewed as a rough prediction of the next location. **Nesterov accelerated momentum** evaluates the gradient at this predicted position rather than at the current one:

$$
\mathbf{m}_{t+1}
\leftarrow
\beta\mathbf{m}_t
+(1-\beta)
\sum_{i\in B_t}
\frac{\partial\ell_i[\boldsymbol{\phi}_t-\alpha\beta\mathbf{m}_t]}
{\partial\boldsymbol{\phi}},
$$

$$
\boldsymbol{\phi}_{t+1}
\leftarrow
\boldsymbol{\phi}_t
-\alpha\mathbf{m}_{t+1}.
$$

The gradient therefore corrects the path that momentum alone would have taken.

## Adam

With a fixed learning rate, ordinary gradient descent takes large steps in directions with large gradients and small steps in directions with small gradients. If the loss surface is steep in one direction and shallow in another, one learning rate is difficult to choose: a rate large enough to make progress in the shallow direction may overshoot in the steep direction.

A simple normalization idea is:

$$
\mathbf{m}_{t+1}
\leftarrow
\frac{\partial L[\boldsymbol{\phi}_t]}{\partial\boldsymbol{\phi}},
\qquad
\mathbf{v}_{t+1}
\leftarrow
\left(
\frac{\partial L[\boldsymbol{\phi}_t]}{\partial\boldsymbol{\phi}}
\right)^2,
$$

$$
\boldsymbol{\phi}_{t+1}
\leftarrow
\boldsymbol{\phi}_t
-\alpha
\frac{\mathbf{m}_{t+1}}
{\sqrt{\mathbf{v}_{t+1}}+\epsilon}.
$$

All squaring, square roots, and divisions are elementwise. This normalizes away the gradient magnitude, leaving its direction in each coordinate. It makes progress in both steep and shallow directions, but it tends to oscillate around a minimum rather than converge exactly.

**Adaptive moment estimation (Adam)** adds momentum to both the gradient and its squared value:

$$
\mathbf{m}_{t+1}
\leftarrow
\beta\mathbf{m}_t
+(1-\beta)
\frac{\partial L[\boldsymbol{\phi}_t]}{\partial\boldsymbol{\phi}},
$$

$$
\mathbf{v}_{t+1}
\leftarrow
\gamma\mathbf{v}_t
+(1-\gamma)
\left(
\frac{\partial L[\boldsymbol{\phi}_t]}{\partial\boldsymbol{\phi}}
\right)^2.
$$

At the beginning of training, these moving averages are biased toward zero. Adam corrects this with:

$$
\tilde{\mathbf{m}}_{t+1}
=\frac{\mathbf{m}_{t+1}}{1-\beta^{t+1}},
\qquad
\tilde{\mathbf{v}}_{t+1}
=\frac{\mathbf{v}_{t+1}}{1-\gamma^{t+1}}.
$$

It then updates parameters using the bias-corrected estimates:

$$
\boldsymbol{\phi}_{t+1}
\leftarrow
\boldsymbol{\phi}_t
-\alpha
\frac{\tilde{\mathbf{m}}_{t+1}}
{\sqrt{\tilde{\mathbf{v}}_{t+1}}+\epsilon}.
$$

In practice, Adam usually computes these quantities from minibatches, so its trajectory remains noisy. It balances updates across parameters and network layers, can converge smoothly, and is less sensitive to the initial learning rate than ordinary SGD. Consequently, it usually needs less elaborate learning-rate scheduling.

## Training Algorithm Hyperparameters

The training **hyperparameters** are choices that affect final model performance but are not learned model parameters. They include:

- the learning algorithm;
- batch size;
- learning rate and learning-rate schedule; and
- momentum coefficients such as $\beta$ and $\gamma$.

Choosing them is often empirical. A common approach is **hyperparameter search**: train several models using different hyperparameter settings, then select the best-performing model.