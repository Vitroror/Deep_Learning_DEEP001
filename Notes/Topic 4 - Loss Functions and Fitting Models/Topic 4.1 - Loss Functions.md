## Definition

[[Topic 1 - Notes|Previously]], we defined that a neural network is a family of functions $f[\mathbf{x}, \boldsymbol{\phi}]$, where $\boldsymbol{\phi}$ denotes the learnable parameters. Training needs a way to decide which member of this family best matches a dataset of input/output pairs $\{\mathbf{x}_i, y_i\}_{i=1}^{I}$. A **loss function** (or cost function) $L[\boldsymbol{\phi}]$ returns one number measuring the mismatch between predictions and ground truth. We train by choosing parameters that minimize it:

$$
\hat{\boldsymbol{\phi}} = \arg\min_{\boldsymbol{\phi}} L[\boldsymbol{\phi}].
$$

The central idea of this is that the network need not directly predict $y$. Instead, it predicts the parameters of a probability distribution over possible outputs. A principled loss then rewards the model for assigning high probability to what was actually observed.

## Maximum likelihood

For each input $\mathbf{x}$, interpret the model as defining a conditional distribution $\Pr(y\mid\mathbf{x})$, rather than only a single output. The appropriate distribution depends on the output domain: a normal distribution for an unbounded real-valued target, a Bernoulli distribution for a binary label, or a categorical distribution for one of $K$ classes.

### Computing a distribution over outputs

Choose a parametric distribution $\Pr(y\mid\boldsymbol{\theta})$ on the required output domain. Then make the network compute one or more of its parameters:

$$
\boldsymbol{\theta} = f[\mathbf{x}, \boldsymbol{\phi}].
$$

For example, for a real-valued target, we can use a normal distribution with mean $\mu$ and variance $\sigma^2$. The network may predict $\mu$, while $\sigma^2$ is initially treated as an unknown constant.

### The maximum-likelihood criterion

For training pairs $\{\mathbf{x}_i,y_i\}$, choose parameters that make all observed outputs probable:

$$
\hat{\boldsymbol{\phi}}
= \arg\max_{\boldsymbol{\phi}}
\prod_{i=1}^{I}\Pr\!\left(y_i\mid f[\mathbf{x}_i,\boldsymbol{\phi}]\right).
$$

This product is the **likelihood**. It assumes that the output distributions have the same form for each example and that outputs are conditionally independent given their inputs:

$$
\Pr(y_1,\ldots,y_I\mid\mathbf{x}_1,\ldots,\mathbf{x}_I)
=\prod_{i=1}^{I}\Pr(y_i\mid\mathbf{x}_i).
$$

### Log-likelihood and negative log-likelihood

Products of many small probabilities are numerically awkward. Because $\log$ is monotonically increasing, maximizing the likelihood has the same solution as maximizing its logarithm:

$$
\arg\max_{\boldsymbol{\phi}}\sum_{i=1}^{I}
\log\Pr\!\left(y_i\mid f[\mathbf{x}_i,\boldsymbol{\phi}]\right).
$$

By convention, training minimizes a loss, so negate this quantity:

$$
L[\boldsymbol{\phi}]
=-\sum_{i=1}^{I}\log\Pr\!\left(y_i\mid f[\mathbf{x}_i,\boldsymbol{\phi}]\right).
$$


### Inference

After training, the model supplies a distribution $\Pr(y\mid f[\mathbf{x},\hat{\boldsymbol{\phi}}])$. We may report the full distribution, including uncertainty, or a point estimate at its most probable value:

$$
\hat y=\arg\max_y\Pr\!\left(y\mid f[\mathbf{x},\hat{\boldsymbol{\phi}}]\right).
$$

## Constructing Loss Functions

For a new supervised prediction task:

1. Choose a probability distribution $\Pr(y\mid\boldsymbol{\theta})$ whose domain matches the target $y$.
2. Configure the network to predict the needed distribution parameters: $\boldsymbol{\theta}=f[\mathbf{x},\boldsymbol{\phi}]$.
3. Train by minimizing negative log-likelihood:

   $$
   \hat{\boldsymbol{\phi}}
   =\arg\min_{\boldsymbol{\phi}}
   \left[-\sum_{i=1}^{I}\log\Pr\!\left(y_i\mid f[\mathbf{x}_i,\boldsymbol{\phi}]\right)\right].
   $$

4. At test time, return either the predicted distribution or its most probable output.

This makes the choice of loss a modeling decision: select a distribution that expresses the kind of target and uncertainty the task has.

## Univariate Regression

In univariate regression, $y\in\mathbb{R}$. Choose a normal distribution:

$$
\Pr(y\mid\mu,\sigma^2)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left[-\frac{(y-\mu)^2}{2\sigma^2}\right].
$$

Let the network predict its mean, $\mu=f[\mathbf{x},\boldsymbol{\phi}]$, while the variance is fixed. The negative log-likelihood is then equivalent, for optimization, to:

$$
L[\boldsymbol{\phi}]
=\sum_{i=1}^{I}\left(y_i-f[\mathbf{x}_i,\boldsymbol{\phi}]\right)^2.
$$

### Least squares loss

Thus **least squares** is not an arbitrary choice: it follows from assuming independent targets drawn from normal distributions whose means are predicted by the network. Terms independent of $\boldsymbol{\phi}$, and positive constant scaling factors, can be removed without moving the minimum.

For a normal distribution, the most probable value is its mean. Therefore, the usual regression prediction is simply:

$$
\hat y=f[\mathbf{x},\hat{\boldsymbol{\phi}}].
$$

### Estimating variance and heteroscedastic regression

The variance can be learned as a global parameter, $\hat\sigma^2$, alongside network parameters. It describes the overall uncertainty remaining around predictions.

The constant-variance assumption is **homoscedastic**. When uncertainty varies with the input, use a network with two outputs:

$$
\mu=f_1[\mathbf{x},\boldsymbol{\phi}],
\qquad
\sigma^2=f_2[\mathbf{x},\boldsymbol{\phi}]^2.
$$

Squaring ensures that the variance is non-negative. This is **heteroscedastic regression**: the model predicts both its mean response and an input-dependent uncertainty.

![[Pasted image 20261026172219.png]]

## Binary Classification

For binary labels $y\in\{0,1\}$, choose a Bernoulli distribution with parameter $\lambda$, the probability that $y=1$:

$$
\Pr(y\mid\lambda)=(1-\lambda)^{1-y}\lambda^y.
$$

The parameter must lie in $[0,1]$, so transform the unconstrained network output with the logistic sigmoid:

$$
\lambda=\operatorname{sig}(f[\mathbf{x},\boldsymbol{\phi}]),
\qquad
\operatorname{sig}(z)=\frac{1}{1+\exp(-z)}.
$$

The resulting negative log-likelihood is the **binary cross-entropy loss**:

$$
L[\boldsymbol{\phi}]
=\sum_{i=1}^{I}\left[
-(1-y_i)\log\!\left(1-\operatorname{sig}(f[\mathbf{x}_i,\boldsymbol{\phi}])\right)
-y_i\log\!\left(\operatorname{sig}(f[\mathbf{x}_i,\boldsymbol{\phi}])\right)
\right].
$$

The sigmoid output is $\Pr(y=1\mid\mathbf{x})$; $1-\lambda$ is $\Pr(y=0\mid\mathbf{x})$. For a hard decision, predict class 1 when $\lambda>0.5$, otherwise class 0.

![[Pasted image 20261026172330.png]]

## Multiclass Classification

For $K>2$ classes, $y\in\{1,2,\ldots,K\}$, choose the categorical distribution with one probability $\lambda_k$ per class:

$$
\Pr(y=k)=\lambda_k,
\qquad
0\leq\lambda_k\leq1,
\qquad
\sum_{k=1}^{K}\lambda_k=1.
$$

The network produces $K$ unconstrained outputs (logits). Apply softmax to convert them into valid probabilities:

$$
\operatorname{softmax}_k(\mathbf{z})
=\frac{\exp(z_k)}{\sum_{k'=1}^{K}\exp(z_{k'})}.
$$

The probability assigned to training label $y_i$ is $\operatorname{softmax}_{y_i}(f[\mathbf{x}_i,\boldsymbol{\phi}])$. The **multiclass cross-entropy loss** is:

$$
L[\boldsymbol{\phi}]
=-\sum_{i=1}^{I}
\log\!\left[\operatorname{softmax}_{y_i}(f[\mathbf{x}_i,\boldsymbol{\phi}])\right].
$$

Equivalently, for logits $f_k[\mathbf{x}_i,\boldsymbol{\phi}]$:

$$
L[\boldsymbol{\phi}]
=-\sum_{i=1}^{I}\left(
f_{y_i}[\mathbf{x}_i,\boldsymbol{\phi}]
-\log\sum_{k'=1}^{K}\exp(f_{k'}[\mathbf{x}_i,\boldsymbol{\phi}])
\right).
$$

At inference, choose the class with the greatest predicted probability. The same recipe extends beyond these examples: select a distribution appropriate to the target, such as Poisson for non-negative counts, von Mises for directions, beta for proportions in $[0,1]$, or mixtures of Gaussians for multimodal real-valued targets.

![[Pasted image 20261026172351.png]]

## Multiple outputs

When $\mathbf{y}$ contains several predictions, a common simplification is to assume its components are independent. Then the joint probability factorizes:

$$
\Pr(\mathbf{y}\mid f[\mathbf{x},\boldsymbol{\phi}])
=\prod_d \Pr\!\left(y_d\mid f_d[\mathbf{x},\boldsymbol{\phi}]\right).
$$

Taking negative logs turns the product into a sum:

$$
L[\boldsymbol{\phi}]
=-\sum_{i=1}^{I}\sum_d
\log\Pr\!\left(y_{id}\mid f_d[\mathbf{x}_i,\boldsymbol{\phi}]\right).
$$

For example, use one normal distribution per continuous output or one categorical distribution per pixel in a segmentation task. Different output types can also be combined: predicting wind direction and strength could pair a circular distribution with a positive-valued distribution. The additive loss comes directly from the independence assumption.

## Cross-entropy loss

Cross-entropy provides another route to the same objective. Let $q(y)$ be the empirical distribution of the observed targets and $p(y)=\Pr(y\mid\boldsymbol{\theta})$ be the model distribution. Their KL divergence is:

$$
D_{\mathrm{KL}}(q\Vert p)
=\int q(z)\log q(z)\,dz
-\int q(z)\log p(z)\,dz.
$$

The first term does not depend on the model parameters. Minimizing KL divergence therefore minimizes the second term, the **cross-entropy**. For an empirical distribution represented by point masses at the observations,

$$
q(y)=\frac{1}{I}\sum_{i=1}^{I}\delta(y-y_i),
$$

the cross-entropy reduces, up to the constant factor $1/I$, to:

$$
-\sum_{i=1}^{I}\log\Pr(y_i\mid\boldsymbol{\theta}).
$$

Once $\boldsymbol{\theta}=f[\mathbf{x}_i,\boldsymbol{\phi}]$, this is exactly the negative log-likelihood loss. Hence, maximizing likelihood and minimizing cross-entropy are equivalent views of fitting the model distribution to the data.

![[Pasted image 20261026172441.png]]