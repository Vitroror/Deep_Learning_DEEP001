[[Topic 8 - Convolutional Neural Networks|Previously]], we saw that CNNs can learn spatial features by repeatedly applying convolutional layers. However, simply adding more layers does not always improve performance. This lecture covers **Residual Networks**, which allow much deeper neural networks to be trained effectively.

## Definition

A standard neural network processes its data sequentially: each layer receives the output of the previous layer and transforms it before passing it forward.

$$
\mathbf{h}_1 = f_1[\mathbf{x}], \quad
\mathbf{h}_2 = f_2[\mathbf{h}_1], \quad
\mathbf{y} = f_3[\mathbf{h}_2].
$$

In theory, deeper networks should have more capacity and therefore perform better. In practice, after a certain depth, both training and test performance can become worse. This is not an overfitting problem, as the deeper model also struggles to fit the training data.

One explanation is the **shattered gradients** problem. In a very deep sequential network, small changes in early layers can cause the gradient to change unpredictably. Gradient descent then has difficulty making stable progress.

## Residual Connections

A **residual connection**, also called a skip connection, adds a layer's input directly back to its output:

$$
\mathbf{h}_{k+1} = \mathbf{h}_k + f_k[\mathbf{h}_k].
$$

Instead of learning a completely new representation, the function $f_k[\cdot]$ learns a change, or residual, to the current representation. This combination is called a **residual block**.

The shortcut path allows information to move directly through the network. It also gives gradients a shorter path during backpropagation, which reduces the shattered gradients problem. A residual network can be interpreted as many smaller networks of different depths working together.

For the addition to be valid, the input and output of the residual function must have the same dimensions. If the number of channels or spatial size changes, a $1\times1$ convolution can be used to match them.

![[Pasted image 20261110184925.png]]

## Order of Operations

A residual block contains nonlinear activation functions, otherwise the complete network would still be linear. However, it is common to apply BatchNorm and ReLU before the convolutional layer:

$$
\text{input} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Convolution}.
$$

This means the final operation in the residual branch is linear, so it can add either positive or negative corrections to the input. In practice, a residual block can contain several convolutional layers before its output is added back to the original representation.

## Exploding Gradients and Batch Normalization

Residual connections make vanishing gradients less problematic, but they can cause the variance of activations to grow. At every residual block, the input and the processed output are added together. If both have similar variance, the combined variance increases repeatedly as the network gets deeper. This may lead to **exploding gradients**.

**Batch normalization**, or **BatchNorm**, addresses this problem. It computes the mean and standard deviation of the activations in a mini-batch and normalizes them:

$$
h_i \leftarrow \frac{h_i-m_h}{s_h+\epsilon}.
$$

The normalized output is then scaled and shifted using learned parameters:

$$
h_i \leftarrow \gamma h_i + \delta.
$$

BatchNorm helps keep the forward pass stable, makes the loss surface smoother, and allows higher learning rates. It also acts as a small source of regularization, since the normalization of each example changes depending on the other examples in the mini-batch.

## Common Residual Architectures

### ResNet

**ResNet** is the standard residual CNN architecture for image classification. Its residual blocks usually contain BatchNorm, ReLU, and convolutional layers. The network periodically downsamples the spatial representation while increasing the number of channels.

Very deep ResNets use **bottleneck blocks**. These first apply a $1\times1$ convolution to reduce the number of channels, then a $3\times3$ convolution to process local spatial information, and finally another $1\times1$ convolution to restore the original number of channels. This reduces the number of parameters while preserving the receptive field.

![[Pasted image 20261110185013.png]]

### DenseNet

**DenseNet** uses a different kind of skip connection. Instead of adding the previous representation to the new one, it concatenates them along the channel dimension. Each layer therefore receives the output of all earlier layers.

This allows later layers to reuse features learned by previous ones. However, the number of channels grows quickly, so $1\times1$ convolutions are often used to control the representation size.

![[Pasted image 20261110185029.png]]

### U-Net and Hourglass Networks

A **U-Net** is an encoder-decoder architecture used mainly for semantic segmentation. The encoder repeatedly downsamples the input, increasing the receptive field and collecting global information. The decoder upsamples the representation back to the original image resolution.

The U-Net uses skip connections between corresponding encoder and decoder layers. These pass high-resolution spatial details directly to the decoder, so the central low-resolution representation does not need to remember every detail of the original image.

![[Pasted image 20261110185056.png]]

**Hourglass networks** use a similar encoder-decoder structure, but normally add the processed skip connections instead of concatenating them. They are commonly used for pose estimation, where the network predicts a heatmap for each body joint.


![[Pasted image 20261110185113.png]]