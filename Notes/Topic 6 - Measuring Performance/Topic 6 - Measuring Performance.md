## Definition

[[Topic 2 - Supervised Learning|Previously]], we learned how to train a network. **Generalization** is the extent to which its performance on unseen test data matches its performance on training data.

Training performance alone is insufficient: a network can memorize the training set and still make poor predictions on new examples.

## Training a Simple Model

The chapter uses MNIST-1D, a ten-class dataset. A neural network achieves zero training error, but its test error remains around 40%. This gap shows that fitting the training data is not equivalent to learning the underlying task.

Use a separate **test set** of input/output pairs produced by the same process to estimate true performance. Test loss and test error can differ: a model may make the same mistakes while becoming more confident, which increases negative log-likelihood without changing error rate.

## Sources of Error

For least-squares regression, test error has three sources.

### Noise

**Noise** is irreducible variation in outputs for the same input. It can arise from stochastic processes, measurement error, incorrect labels, or unobserved explanatory variables. Noise limits test performance but does not necessarily prevent memorization of the training set.

### Bias

**Bias** is systematic error caused by insufficient model flexibility. A model with too few linear regions cannot exactly describe a curved target function, even with optimal parameters.

### Variance

**Variance** results from finite, noisy training data. Different samples of the same population give different fitted parameters and predictions. Stochastic training can add further variance.

### Bias-variance decomposition

Let $\mu[x]$ be the true mean output at $x$ and $\sigma^2$ its noise variance. Let $f_\mu[x]$ be the average prediction over all possible training datasets. Then:

$$
\mathbb{E}_{\mathcal{D}}
\left[
\mathbb{E}_y[L[x]]
\right]
=
\underbrace{
\mathbb{E}_{\mathcal{D}}
\left[
(f[x,\boldsymbol{\phi}[\mathcal{D}]]-f_\mu[x])^2
\right]
}_{\text{variance}}
+
\underbrace{
(f_\mu[x]-\mu[x])^2
}_{\text{bias}}
+
\underbrace{\sigma^2}_{\text{noise}}.
$$

For least-squares regression these components add together. Their interaction is more complicated for other losses.

## Reducing Error

Noise is a fundamental limit, but bias and variance can often be reduced.

### Reducing variance

More relevant training data reduce variance. With few examples, different sampled datasets yield very different fitted models; with more examples, random noise averages out and models become more consistent.

### Reducing bias

Increasing capacity gives the model a richer family of functions and usually reduces bias. In neural networks, capacity grows with additional hidden units and layers.

### Bias-variance trade-off

For a fixed dataset, increasing capacity can reduce bias but increase variance. A model may then fit random training fluctuations rather than the underlying function: **overfitting**.

In the classical, under-parameterized regime, this produces an optimal intermediate capacity.

## Double Descent

Modern networks may show **double descent** as capacity increases:

1. test error initially falls as bias decreases;
2. it rises near the capacity needed to memorize the training set; and
3. it falls again in the strongly over-parameterized regime.

The initial pattern matches the bias-variance trade-off. The second decline is less fully understood. In high-dimensional spaces, training data are extremely sparse, so predictions between data points matter. Larger models can represent smoother interpolation, and initialization or optimization may implicitly favor such smooth solutions.

The three regions are called the classical or under-parameterized regime, the critical regime, and the modern or over-parameterized regime.

## Choosing Hyperparameters

Hyperparameters include network architecture, learning rate, batch size, and training algorithm settings. Do not choose them by repeatedly inspecting test performance: that would tune the procedure to accidental properties of the test set.

Use three datasets:

- **training set:** learn parameters;
- **validation set:** compare hyperparameter settings;
- **test set:** evaluate the chosen model once.

Hyperparameter search is expensive because every candidate needs a training run. It is also difficult because settings can be discrete, conditional, and non-differentiable. Random search and more targeted optimization methods use validation results to explore promising configurations.

