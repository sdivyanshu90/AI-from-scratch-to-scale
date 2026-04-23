# Backpropagation

Backpropagation is the algorithm that makes training deep neural networks
computationally feasible. It computes the gradient of the loss with respect
to every parameter in the network in a single backward pass — at a cost
proportional to the forward pass, regardless of the number of parameters.
Without it, each parameter would require a separate finite-difference
approximation, making training infeasibly slow.

## 1. The Core Intuition (The "Why")

The gradient of the loss $L$ with respect to a parameter $w_{ij}^{(l)}$ in
layer $l$ requires applying the chain rule through every subsequent layer.
The naive approach recomputes intermediate values many times. The key insight
of backpropagation is to **reuse** intermediate results: compute
$\partial L / \partial h^{(l)}$ once and use it to compute all parameter
gradients at layer $l$.

Paul Werbos derived this algorithm in his 1974 PhD thesis for training
multi-layer networks. Rumelhart, Hinton, and Williams (1986) independently
rediscovered and popularised it with practical demonstrations, launching
the modern deep learning era. LeCun applied it to convolutional networks
for handwriting recognition (1989).

The algorithm is simply reverse-mode automatic differentiation applied to
a neural network computation graph. Modern frameworks (PyTorch, JAX) implement
a generalised version of this idea.

## 2. The Theoretical Underpinning

### The Chain Rule for Computational Graphs

For a scalar output $L$ and scalar input $x$ connected through intermediates
$z_1, z_2, \ldots$:

$$\frac{\partial L}{\partial x} = \sum_{\text{paths } x \to L} \prod_{\text{edges}} \frac{\partial z_{i+1}}{\partial z_i}$$

For a network with $L = \ell(y_{\text{pred}}, y)$ and layers
$a^{(1)}, a^{(2)}, \ldots, a^{(L)}$, the chain rule gives:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

### The Delta/Error Signal

Define the **error signal** at layer $l$ as:

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

where $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$ is the pre-activation.

The backpropagation recurrence is:

$$\delta^{(l)} = \left(W^{(l+1)}\right)^{\top} \delta^{(l+1)} \odot \sigma'\!\left(z^{(l)}\right)$$

where $\odot$ is element-wise multiplication and $\sigma'(z^{(l)})$ is the
derivative of the activation function.

Starting from $\delta^{(L)} = \partial L / \partial z^{(L)}$ (computed at
the output layer), we propagate backwards to compute all $\delta^{(l)}$.

### Parameter Gradients

Given the error signals, parameter gradients are:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^{\top}$$
$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

For a mini-batch of $n$ examples, the gradient is the mean across examples:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{1}{n} \Delta^{(l)} \left(A^{(l-1)}\right)^{\top}$$

where $\Delta^{(l)} \in \mathbb{R}^{n_l \times n}$ stacks the error signals.

### Computational Complexity

The forward pass computes $a^{(1)}, \ldots, a^{(L)}$ and costs $O(P)$ where
$P$ is the total number of parameters. The backward pass walks the same graph
in reverse, also costing $O(P)$. Total cost: $O(P)$, the same order as the
forward pass. This is why training is feasible: the cost of computing all
$P$ gradients is $2 \times$ the cost of one forward pass.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_hat - y) ** 2))


def mse_grad(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * (y_hat - y) / y.size


class BackpropNet:
    """Two-hidden-layer network with full backprop implementation.

    All four equations of backpropagation:
      BP1: delta at output layer
      BP2: delta at hidden layers (recurrence)
      BP3: gradient of W
      BP4: gradient of b
    """

    def __init__(self, sizes: List[int], lr: float = 0.01) -> None:
        self.lr = lr
        rng = np.random.default_rng(0)
        self.W = [rng.normal(0, np.sqrt(2/s), (s2, s)).astype(np.float32)
                  for s, s2 in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((s, 1), dtype=np.float32)
                  for s in sizes[1:]]

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass storing pre-activations (z) and activations (a).

        Args:
            x: Input of shape (n_in, batch_size).

        Returns:
            (output, list_of_z, list_of_a)
        """
        zs, acts = [], [x]
        a = x
        for W, b in zip(self.W, self.b):
            z = W @ a + b
            zs.append(z)
            a = sigmoid(z)
            acts.append(a)
        return a, zs, acts

    def backward(
        self,
        y:    np.ndarray,
        zs:   List[np.ndarray],
        acts: List[np.ndarray],
    ) -> float:
        """Backpropagation; updates weights in-place; returns loss."""
        batch  = y.shape[1]
        y_hat  = acts[-1]
        loss   = mse_loss(y_hat, y)

        # BP1: error at output layer
        delta = mse_grad(y_hat, y) * sigmoid_prime(zs[-1])

        for l in reversed(range(len(self.W))):
            # BP3 and BP4: parameter gradients
            dW = delta @ acts[l].T / batch
            db = delta.sum(axis=1, keepdims=True) / batch
            # BP2: propagate error to previous layer
            if l > 0:
                delta = self.W[l].T @ delta * sigmoid_prime(zs[l - 1])
            # SGD update
            self.W[l] -= self.lr * dW
            self.b[l]  -= self.lr * db

        return loss
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def check_gradient_flow(model: nn.Module) -> dict[str, float]:
    """Return the mean absolute gradient for each layer.

    Useful for diagnosing vanishing or exploding gradients.
    A layer with mean_grad near 0 is not learning.
    A layer with mean_grad >> 1 may cause instability.
    """
    grad_stats: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats[name] = float(param.grad.abs().mean().item())
    return grad_stats


def train_with_gradient_clipping(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    loss_fn:    nn.Module,
    X:          torch.Tensor,
    y:          torch.Tensor,
    clip_norm:  float = 1.0,
) -> float:
    """Single training step with gradient norm clipping.

    Gradient clipping prevents gradient explosion in deep networks.
    The global gradient norm is computed across all parameters and
    clipped to `clip_norm` if exceeded.

    Args:
        clip_norm: Maximum allowed L2 norm of the concatenated gradient vector.

    Returns:
        Scalar loss value.
    """
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    # Clip global gradient norm (sum over all parameter gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    optimizer.step()
    return loss.item()
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Not checking for exploding or vanishing gradients in deep
  networks. Training loss may plateau or diverge with no clear error message.

  **Fix:** Monitor gradient norms per layer during the first few epochs.
  Use `check_gradient_flow()` or TensorBoard histograms.

- **Mistake:** Using ReLU throughout without residual connections in very
  deep networks (> 20 layers). Dead ReLU neurons (always outputting 0)
  propagate zero gradients backward, blocking learning in early layers.

  **Fix:** Use residual (skip) connections as in ResNet, or use GELU
  which has non-zero gradient everywhere.

- **Mistake:** Forgetting to average gradients over the batch before the
  update. If you sum rather than mean, the effective learning rate scales
  with batch size.

  **Fix:** Use `reduction="mean"` in loss functions (PyTorch default) and
  divide batch gradient sums by batch size.

- **Mistake:** Recomputing the forward pass inside the backward function
  for custom `torch.autograd.Function` implementations.

  **Fix:** Save tensors needed for the backward pass using `ctx.save_for_backward`
  during the forward pass; retrieve them in backward.

- **Mistake:** Using float16 throughout without loss scaling in mixed-precision
  training. Gradients of small activations underflow to zero in float16.

  **Fix:** Use `torch.cuda.amp.autocast` with `GradScaler` for numerically
  stable mixed-precision training.

## 6. Knowledge Check

1. **Conceptual:** Write out all four backpropagation equations (BP1–BP4)
   from memory. Explain the role of each equation and identify which ones
   involve the activation derivative $\sigma'(z)$.

2. **Conceptual:** The backward pass costs roughly the same as two forward
   passes. Where does this factor of 2 come from? What additional memory
   does the backward pass require compared to the forward pass?

3. **Coding challenge:** Implement gradient checking: for each parameter
   $w_i$, numerically estimate $\partial L / \partial w_i$ using the
   finite-difference formula $[L(w_i + \epsilon) - L(w_i - \epsilon)] / 2\epsilon$.
   Compare these numerical gradients to those computed by backpropagation
   and report the relative error.

## References

1. Werbos, P. J. (1974). *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences*. PhD Thesis, Harvard University. *(First derivation of backpropagation.)*
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
3. LeCun, Y., Boser, B., Denker, J. S., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4), 541–551.
4. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: A survey. *JMLR*, 18, 1–43.
