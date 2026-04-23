# CNNs and ResNet

Convolutional Neural Networks (CNNs) are the foundational architecture for
processing spatial data. They exploit two structural priors — local connectivity
(nearby pixels are correlated) and translation equivariance (a cat is a cat
regardless of where it appears in the image) — to achieve dramatic parameter
efficiency over fully connected networks. ResNet (He et al., 2016) added
residual connections that solved the degradation problem and made very deep
networks trainable, enabling the ImageNet breakthrough that catalysed modern
deep learning.

## 1. The Core Intuition (The "Why")

A fully connected layer applied to a $224 \times 224 \times 3$ image has
$224 \times 224 \times 3 = 150{,}528$ inputs. With 1000 hidden units, that is
150 million parameters in a single layer, most of which learn redundant local
patterns. An MLP has no prior knowledge that adjacent pixels are related.

LeCun et al. (1989, 1998) showed that constraining weights to be shared across
spatial locations (convolution) allows a single set of filters to detect the
same edge or texture anywhere in the image. This **weight sharing** reduces
parameters by orders of magnitude and introduces a powerful inductive bias.

He et al. (2016) showed that simply making networks deeper did not improve
performance — very deep networks had higher training error than shallow ones
(the degradation problem). Residual connections, which add the input directly
to the output of each block, solved this by providing a gradient highway to
early layers.

## 2. The Theoretical Underpinning

### Discrete Convolution

A 2D convolution of input $X$ with kernel $K$ produces output:

$$(X \star K)[i, j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X[i+m, j+n] \cdot K[m, n]$$

where $k$ is the kernel size. In a convolutional layer, there are $C_{\text{out}}$
kernels, each of depth $C_{\text{in}}$, producing $C_{\text{out}}$ feature maps.

The number of parameters in a convolutional layer is:

$$P = C_{\text{out}} \times C_{\text{in}} \times k \times k$$

regardless of the spatial dimensions $H$ and $W$ of the input. This scale-
invariance is the key efficiency gain over fully connected layers.

### Receptive Field

The receptive field of a neuron is the region of the input that influences
it. With kernel size $k$ and $L$ convolutional layers (no pooling), the
receptive field grows linearly:

$$\text{RF}(L) = 1 + L(k - 1)$$

With stride-2 convolutions or max pooling, it grows faster. Deep networks
build hierarchical representations: early layers detect edges, middle layers
detect textures, deep layers detect objects.

### Batch Normalisation

Batch Norm (Ioffe & Szegedy, 2015) normalises the pre-activations within
each mini-batch:

$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{z}_i + \beta$$

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, and
$\gamma, \beta$ are learned scale and shift parameters. This stabilises the
input distribution to each layer, enabling higher learning rates and reducing
sensitivity to initialisation.

### Residual Connections

A standard convolutional block computes $h = \sigma(W_2 \sigma(W_1 x))$.
A residual block adds a skip connection:

$$h = \sigma\!\left(W_2 \sigma(W_1 x) + x\right) = \mathcal{F}(x) + x$$

The gradient of the loss with respect to $x$ is:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h} \cdot \left(1 + \frac{\partial \mathcal{F}(x)}{\partial x}\right)$$

The $+1$ term provides a gradient highway: even if $\partial \mathcal{F} / \partial x$
is small (vanishing gradients), the gradient still flows through with
magnitude $\partial L / \partial h$. This is the mathematical reason why
residual connections solve the degradation problem.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np


def conv2d_naive(
    x:      np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    pad:    int = 0,
) -> np.ndarray:
    """Naive 2D convolution for a single channel.

    Args:
        x:      Input of shape (H, W).
        kernel: Filter of shape (kH, kW).
        stride: Stride along both spatial dimensions.
        pad:    Zero-padding size.

    Returns:
        Output feature map of shape (H_out, W_out).
    """
    H, W   = x.shape
    kH, kW = kernel.shape
    # Pad input with zeros on all sides
    x_pad  = np.pad(x, pad, mode="constant")
    H_out  = (H + 2 * pad - kH) // stride + 1
    W_out  = (W + 2 * pad - kW) // stride + 1
    out    = np.zeros((H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            patch     = x_pad[i*stride:i*stride+kH, j*stride:j*stride+kW]
            out[i, j] = float(np.sum(patch * kernel))
    return out


def maxpool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """2D max pooling with stride = pool_size."""
    H, W  = x.shape
    H_out = H // pool_size
    W_out = W // pool_size
    out   = np.zeros((H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            patch     = x[i*pool_size:(i+1)*pool_size,
                          j*pool_size:(j+1)*pool_size]
            out[i, j] = float(np.max(patch))
    return out
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard ResNet residual block (He et al., 2016).

    For a network with many blocks, the residual connection ensures
    gradients flow directly from the loss to every block, solving
    the vanishing gradient problem for very deep networks.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                          # save identity
        out = F.relu(self.bn1(self.conv1(x))) # first conv + BN + ReLU
        out = self.bn2(self.conv2(out))        # second conv + BN
        out = F.relu(out + residual)           # add skip, then activate
        return out


class SmallResNet(nn.Module):
    """Minimal ResNet for illustration; suitable for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem    = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1  = ResidualBlock(64)
        self.layer2  = ResidualBlock(64)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using `bias=True` in convolutional layers that are immediately
  followed by BatchNorm. BatchNorm subtracts the mean, which cancels any
  bias added by the previous layer.

  **Fix:** Always set `bias=False` in conv layers followed by BatchNorm.

- **Mistake:** Using BatchNorm with very small batch sizes (< 8). The
  per-batch statistics are highly noisy with small batches.

  **Fix:** Use GroupNorm or LayerNorm for small batches. GroupNorm divides
  channels into groups and normalises within each group.

- **Mistake:** Choosing a kernel size of 5 or 7 when building deep networks.
  Two stacked 3×3 convolutions have the same receptive field as one 5×5 but
  with fewer parameters and two nonlinearities.

  **Fix:** Use 3×3 kernels throughout. Use 1×1 convolutions for channel
  projection (pointwise operations).

- **Mistake:** Applying data augmentation during evaluation. Augmentation
  is a training-time regularisation, not an inference transform.

  **Fix:** Use a separate transform pipeline for training and validation.
  Common pitfall with `torchvision.transforms`.

- **Mistake:** Freezing all BatchNorm layers when fine-tuning a pretrained
  ResNet on a small dataset. The BN statistics from ImageNet may not match
  the new domain.

  **Fix:** Set `track_running_stats=False` or use `model.train()` with a
  very small learning rate to allow BN statistics to adapt.

## 6. Knowledge Check

1. **Conceptual:** Show that two stacked $3 \times 3$ convolutional layers
   have the same receptive field as one $5 \times 5$ layer. How many
   parameters does each require for $C$ input and output channels? Which
   has more nonlinearities?

2. **Conceptual:** Explain the degradation problem that motivated residual
   connections. If you add more layers to a network and the deeper network
   does worse on the training set, what does this imply about the optimiser's
   ability to learn an identity mapping?

3. **Coding challenge:** Implement a simple CNN in PyTorch and plot the
   activation statistics (mean and std) at each layer for a forward pass
   on a random input, with and without BatchNorm. Explain what you observe.

## References

1. LeCun, Y., Boser, B., Denker, J. S., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4), 541–551.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS 2012*.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. arXiv:1512.03385.
5. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML 2015*.
