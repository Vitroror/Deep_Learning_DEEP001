[[Topic 7 - Regularization| Previous lectures]] described the supervised learning pipeline as a whole. This lecture covers **Convolutional Neural Networks (CNNs)**, and their most effective applications.

## Definition

Some inputs, like images for example, have characteristics that require specific modelling definitions:

* Firstly, images are high-dimensional in nature, which means that a fully conected, even a shallow network, would require billions of parameters even for simple images.
* Also, image pixels are very close together, so they are statistically related. Fully connected networks have no notion of "nearby", and treat the relationship of every input equally
* Lastly, images have no geometrical interpretation variation. For example, a picture of a tree still is a tree, even if you take two pictures at different angles. This, however, changes the input of the network completely, so to the machine, both pictures are different things altogether. This means that fully connected networks are inefficient in this context.

Therefore, the processing of images requires a different type of network, that being the **Convolutional Neural Network**. A CNN is mainly characterized by the usage of **convolutional layers**, which process each local image region independently, using parameters that are shared accross the image.

## The principle of Invariance and Equivariance

Before diving into the topic, we must first devine precisely what is invariance and equivariance. A function $\mathbf{f[x]}$ of an image $\mathbf{x}$ is deemed *invariant*  to a transformation $\mathbf{t[x]}$ if $$\mathbf{f[t[x]] = f[x]}$$in other words, the network should be able to identify an image as containing the same object, even if it was transformed in any way. By the same logic, a function is also *equivariant* to a transformation if $$\mathbf{f[t[x]] = t[f[x]]}$$
this means that the function's output changes in the same way under the transformation as the input.

![[Pasted image 20261110165103.png]]

## One-Dimensional Convolution

In one dimension, a convolution takes an input $\mathbf{x}$ and returns an output $\mathbf{z}$ so that each output is a weighted sum of nearby inputs. These weights are used at every position and are called the **convolution kernel**. For a kernel of 3, we have $$z_i = \omega_1x_{i-1}+\omega_2x_i+\omega_3x_{i+1}$$
where $\omega=[\omega_1, \omega_2, \omega_3]^T$ is the kernel itself. Notice how the operation is **equivariant** with respect to translation, so if we translate the input, the corresponding output is translated in the same way.

![[Pasted image 20261110171310.png]]

For the image above, some new terminology is required, namely the network's **stride**, which represents the shift to which we evaluate the kernel, the network's **size** which represents how many weights there are in the convolutional layer. Higher sizes imply the requirement of more weight, which can me compensated with **dilation**, where kernel weights are interlaced with zeros.

However, the equation poses the question: How do we treat the input? For this we use **padding**, more specifically **zero padding**, where we assume the input is zero outside of the valid range.

## Convolutional Layers

A convolutional layer combines the convolution itself with a bias and an activation function. For the previous kernel, each hidden unit is computed as$$h_i = a[\beta + \omega_1x_{i-1}+\omega_2x_i+\omega_3x_{i+1}],$$

where $\beta$ and $\omega$ are learned parameters, and $a[\cdot]$ is the activation function. The same three weights and the same bias are used at every position. This is the key difference to a fully connected layer, which would need a separate weight for every connection between input and hidden units.

For an input with $D$ positions, a fully connected layer with $D$ hidden units has $D^2$ weights and $D$ biases. A convolutional layer with kernel size three has only three weights and one bias, regardless of the input size. It can be seen as a very constrained fully connected layer: most connections are zero, and the nonzero connections are forced to share their values. This restriction is useful because it encodes the prior knowledge that the same local pattern can occur anywhere in the input.

## Channels

One convolution alone would discard a considerable amount of information, since it combines nearby values into one output and then passes them through a nonlinearity. Therefore, convolutional layers normally apply several different kernels in parallel. Each kernel produces a **feature map**, also called a **channel**.

For example, a first kernel may respond to a rising edge in a signal while a second one responds to a falling edge. Both feature maps are preserved, so the next layer can use both kinds of information. In deeper layers, each output channel combines values from all the input channels in a local neighborhood.

If an incoming layer has $C_i$ channels, kernel size $K$, and the next layer has $C_o$ channels, then the layer contains $C_iC_oK$ weights and $C_o$ biases in 1D. Although the number of channels usually grows as the network becomes deeper, the parameters are still shared across spatial positions.

![[Pasted image 20261110183735.png]]
## Convolutional Networks and Receptive Fields

A complete CNN is a sequence of convolutional layers. The **receptive field** of a hidden unit is the region of the original input that can affect its value. With a kernel size of three, units in the first layer have receptive fields of size three. A unit in the second layer combines three of those first-layer units, so its receptive field has size five. Each successive layer integrates information from a larger part of the original input.

This allows a CNN to learn local patterns in its earliest layers and gradually combine them into more global patterns. Stride and dilation can make the receptive field grow faster: stride reduces the spatial resolution, whereas dilation increases the distance between the kernel weights without introducing more learned parameters.

## Example: MNIST-1D

The chapter applies these ideas to the MNIST-1D classification problem. The input is a vector with 40 values, and the output is a vector with ten class activations followed by a softmax layer. The CNN uses three convolutional hidden layers, each with fifteen channels, a kernel size of three, a stride of two, and valid padding. The spatial dimension gradually falls from 40 positions to 19, then 9, and finally 4 positions. The last $4\times15=60$ hidden units are flattened and passed to a fully connected output layer.

This model has 2,050 parameters, while a comparable fully connected network has 59,065. Both models fit the training data, but the CNN has much lower test error. This is not merely due to having fewer parameters: the convolutional network has a better **inductive bias** for this data. MNIST-1D examples are translated versions of templates, so applying the same feature detector at every position is a sensible architectural assumption. In this way, convolution can also be interpreted as a form of regularization, since it rules out many implausible input-output mappings.

![[Pasted image 20261110183819.png]]

![[Pasted image 20261110183833.png]]
## Convolutional Networks for 2D Inputs

The same principle extends naturally to images. A 2D convolutional kernel moves horizontally and vertically over the input. For a $3\times3$ kernel, each hidden unit is computed as$$h_{ij} = a\left[\beta + \sum_{m=1}^{3}\sum_{n=1}^{3}\omega_{mn}x_{i+m-2,j+n-2}\right].$$

Thus, each output is a learned weighted sum over a local square region of the image. With zero padding, the resulting feature map can have the same height and width as the input.

Color images have three input channels: red, green, and blue. A $3\times3$ kernel applied to an RGB image therefore has $3\times3\times3$ weights, since it looks at all three colors over a $3\times3$ area. To make several output channels, we use several such kernels. More generally, a layer with kernel size $K\times K$, $C_i$ input channels, and $C_o$ output channels has $C_iC_oK^2$ weights and $C_o$ biases.

![[Pasted image 20261110183910.png]]

## Downsampling and Upsampling

CNNs often progressively reduce the height and width of their internal representations. This is called **downsampling**. A strided convolution with stride two retains outputs at every other position, effectively reducing both spatial dimensions by a factor of two. The receptive field grows at the same time, allowing later layers to reason about larger image regions.

Two other common downsampling methods are **max pooling** and **mean pooling**. Max pooling retains the largest value from each $2\times2$ region, while mean pooling takes their average. Max pooling gives a limited degree of translation invariance: a small shift in the input may leave the maximum unchanged. These operations are applied independently to each channel, so they alter spatial dimensions but not the number of channels.

When the output must also be an image, we need **upsampling** to restore the resolution. The simplest method duplicates each input value into a larger grid. Other methods are max unpooling, which restores values to the recorded positions of previous maxima, and bilinear interpolation, which estimates intermediate values smoothly.

**Transposed convolution** is a learned upsampling operation. It is named because its weight matrix is the transpose of the matrix representing the corresponding strided convolution; it is not the inverse of convolution in general. Each input can contribute to several output positions, which lets the network learn how the higher-resolution representation should be formed.

## Changing the Number of Channels

Sometimes we want to combine or change channels without changing spatial resolution. A **$1\times1$ convolution** does exactly this. At every spatial position, it takes a weighted sum over all input channels, adds a bias, and passes it through an activation function. Repeating the operation with different weights creates the desired number of output channels.

Despite the name, a $1\times1$ convolution does not inspect neighboring pixels. It is equivalent to applying the same fully connected layer to the channel vector at every location in the image. This is especially useful when two branches of a network need compatible numbers of channels before they are combined.

## Applications

### Image Classification

In image classification, the input image is assigned to one of a fixed set of categories. A standard benchmark is **ImageNet**, which contains over one million training images belonging to 1,000 classes. A typical network receives a $224\times224$ RGB image and outputs a probability distribution over these classes.

**AlexNet** was the first CNN to perform extremely well on ImageNet. It uses five convolutional layers followed by three fully connected layers. The representation is gradually downsampled with convolutions and max pooling, while the number of channels increases. Data augmentation, dropout, weight decay, and SGD with momentum were also essential to its performance.

![[Pasted image 20261110183940.png]]

**VGG** follows the same general pattern, but uses a much deeper sequence of smaller $3\times3$ convolutions. Its success showed that increasing network depth could significantly improve image classification performance. However, deeper networks eventually become difficult to train, which motivates the residual connections studied in the next chapter.

![[Pasted image 20261110183958.png]]

### Object Detection

Image classification predicts what is present somewhere in the image. **Object detection** must additionally locate each object, possibly when several objects are present. The early **YOLO** system divides a $448\times448$ image into a $7\times7$ grid. For each grid location, it predicts object classes and a fixed number of bounding boxes.

Each bounding box is described by its center position, width, height, and a confidence value. After prediction, low-confidence boxes are removed and overlapping boxes for the same object are suppressed, keeping the most confident one. YOLO is typically pretrained on image classification and then fine-tuned for detection, which is an example of transfer learning.

![[Pasted image 20261110184035.png]]

### Semantic Segmentation

**Semantic segmentation** assigns a class label to every pixel. Its output is therefore spatial, rather than a single label for the whole image. For example, for 21 possible classes, a network can output a $224\times224\times21$ tensor, where the values at each pixel are passed through a softmax function to obtain the class probabilities.

An early segmentation network first uses convolution and pooling layers to create a low-resolution representation containing information from the entire image. It then uses an upsampling path with max unpooling and transposed convolutions to recover the original resolution. The downsampling half is called an **encoder**, and the upsampling half is called a **decoder**. Because of this shape, these models are also known as **encoder-decoder** or **hourglass** networks.
