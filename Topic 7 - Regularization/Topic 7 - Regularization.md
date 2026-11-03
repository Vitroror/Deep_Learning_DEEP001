## Definition

[[Topic 6 - Measuring Performance|Previously]], we saw that models can overfit training data or make poor predictions in unconstrained regions. **Regularization** is a family of methods that improves generalization and narrows the training-test performance gap.

Strictly, regularization adds a preference to the loss function. In common machine-learning use, it also includes optimization effects and heuristics that improve test performance.

## Explicit Regularization

Ordinary fitting minimizes:

$$
\hat{\boldsymbol{\phi}}
=
\arg\min_{\boldsymbol{\phi}}
\left[
\sum_{i=1}^{I}\ell_i[\mathbf{x}_i,y_i]
\right].
$$

Explicit regularization adds a penalty:

$$
\hat{\boldsymbol{\phi}}
=
\arg\min_{\boldsymbol{\phi}}
\left[
\sum_{i=1}^{I}\ell_i[\mathbf{x}_i,y_i]
+\lambda g[\boldsymbol{\phi}]
\right].
$$

Here, $g[\boldsymbol{\phi}]$ is larger for less-preferred parameters and $\lambda$ controls the trade-off between data fit and that preference.

### Probabilistic interpretation

Regularization can be interpreted as a prior $\Pr(\boldsymbol{\phi})$. Instead of maximum likelihood, optimize the maximum a posteriori criterion:

$$
\hat{\boldsymbol{\phi}}
=
\arg\max_{\boldsymbol{\phi}}
\left[
\Pr(\boldsymbol{\phi})
\prod_{i=1}^{I}\Pr(y_i\mid\mathbf{x}_i,\boldsymbol{\phi})
\right].
$$

The penalty is the negative log-prior:

$$
\lambda g[\boldsymbol{\phi}]
=-\log\Pr(\boldsymbol{\phi}).
$$

### L2 regularization

The common L2 penalty is:

$$
\hat{\boldsymbol{\phi}}
=
\arg\min_{\boldsymbol{\phi}}
\left[
\sum_{i=1}^{I}\ell_i[\mathbf{x}_i,y_i]
+\lambda\sum_j\phi_j^2
\right].
$$

Applied to weights, this is often called **weight decay**. Smaller weights encourage smoother output functions. This can reduce overfitting by accepting slightly greater bias for lower variance, and can favor smooth interpolation where data are absent. Excessively large $\lambda$ makes the model too smooth and causes underfitting.

## Implicit Regularization

Optimization itself has preferences. Finite-step gradient descent behaves like optimization of:

$$
\widetilde{L}_{\mathrm{GD}}[\boldsymbol{\phi}]
=
L[\boldsymbol{\phi}]
+\frac{\alpha}{4}
\left\|
\frac{\partial L}{\partial\boldsymbol{\phi}}
\right\|^2.
$$

It is thus repelled from steep regions with large gradient norms, changing its trajectory and potentially the solution it reaches.

SGD adds a term that penalizes disagreement between batch gradients:

$$
\widetilde{L}_{\mathrm{SGD}}[\boldsymbol{\phi}]
=
L[\boldsymbol{\phi}]
+\frac{\alpha}{4}
\left\|
\frac{\partial L}{\partial\boldsymbol{\phi}}
\right\|^2
+\frac{\alpha}{4B}
\sum_{b=1}^{B}
\left\|
\frac{\partial L_b}{\partial\boldsymbol{\phi}}
-\frac{\partial L}{\partial\boldsymbol{\phi}}
\right\|^2.
$$

SGD therefore favors points where batches agree about the slope. This may help explain why smaller batches can generalize better than full-batch optimization.

## Heuristics to Improve Performance

### Early stopping

Stop before full convergence. A model often learns coarse structure before it learns noise, so early stopping can reduce overfitting. Select the stopping point by monitoring validation performance and retaining the best checkpoint.

### Ensembling

An **ensemble** averages several models' predictions, allowing partly independent errors to cancel. Use mean outputs for regression and mean pre-softmax activations for classification. Models may differ by random initialization, resampled training data (**bagging**), hyperparameters, or architecture.

### Dropout

**Dropout** randomly sets a subset of hidden units to zero at every training step. It prevents excessive reliance on particular units and discourages unnecessary changes in the function away from data. At standard inference, use all units and scale weights to compensate; Monte Carlo dropout instead makes multiple stochastic predictions and combines them.

### Applying noise

- Input noise smooths the learned function; adversarial training uses worst-case small input perturbations.
- Weight noise favors broad, flat minima where small parameter changes have little effect.
- **Label smoothing** replaces an absolute one-hot classification target with a softened target distribution, discouraging overconfident predictions.

### Bayesian inference

Bayesian inference treats parameters as uncertain:

$$
\Pr(\boldsymbol{\phi}\mid\{(\mathbf{x}_i,y_i)\})
=
\frac{
\left[\prod_{i=1}^{I}\Pr(y_i\mid\mathbf{x}_i,\boldsymbol{\phi})\right]
\Pr(\boldsymbol{\phi})
}{
\int
\left[\prod_{i=1}^{I}\Pr(y_i\mid\mathbf{x}_i,\boldsymbol{\phi})\right]
\Pr(\boldsymbol{\phi})
d\boldsymbol{\phi}
}.
$$

Predictions average over parameter settings by their posterior probability:

$$
\Pr(y\mid\mathbf{x},\mathcal{D})
=
\int
\Pr(y\mid\mathbf{x},\boldsymbol{\phi})
\Pr(\boldsymbol{\phi}\mid\mathcal{D})
d\boldsymbol{\phi}.
$$

This is an infinite weighted ensemble and can provide uncertainty estimates, but practical neural-network methods require approximations.

### Transfer, multi-task, and self-supervised learning

**Transfer learning** pre-trains on a related data-rich task, then replaces or adds output layers and adapts the network to the target task. **Multi-task learning** trains several related tasks together so they can share useful internal representations.

**Self-supervised learning** creates labels from unlabeled data. Generative methods predict masked content; contrastive methods distinguish related example pairs from unrelated ones. The resulting representation is then fine-tuned for a labeled task.

### Data augmentation

**Data augmentation** creates valid extra examples by transforming inputs without changing their labels: image rotation, flipping, blur, and color changes; synonym replacement or back-translation for text; or frequency changes for audio. It increases effective data and teaches invariance to irrelevant transformations.
