# Multilayer Perceptrons (MLPs)

A Multilayer Perceptron (MLP) is a sequence of affine transformations
alternating with element-wise nonlinear activations. It is the simplest deep
architecture and the building block inside every transformer, CNN, and
diffusion model. The Universal Approximation Theorem gives MLPs their
theoretical justification; understanding the theorem's conditions and
limitations is essential for knowing when depth matters.

## 1. The Core Intuition (The "Why")

Frank Rosenblatt's perceptron (1958) could only learn linearly separable
functions — famously, it could not learn XOR. Minsky and Papert (1969) proved
this limitation formally, which contributed to the first AI winter.

The key insight that restored interest was the addition of hidden layers and
nonlinear activations. A single hidden layer with sigmoid activations can
approximate any continuous function (Cybenko, 1989; Hornik, 1991). However,
the number of neurons required may be exponential in the input dimension.
Deep networks with multiple hidden layers can represent functions that require
exponentially many neurons in a shallow network — this is the theoretical
motivation for depth.

Practically: a two-layer MLP with enough hidden units can fit any training
dataset. The question is generalisation. Regularisation (dropout, weight
decay, batch normalisation) is what makes deep MLPs useful in practice.

## 2. The Theoretical Underpinning

### The Affine + Activation Block

A single MLP layer computes:

$$h^{(l)} = \sigma\!\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$$

where $h^{(l-1)} \in \mathbb{R}^{n_{l-1}}$ is the input activations from
the previous layer, $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the
weight matrix, $b^{(l)} \in \mathbb{R}^{n_l}$ is the bias, and $\sigma$
is an element-wise nonlinearity.

Without $\sigma$, stacking $L$ layers collapses to a single affine map:

$$W^{(L)}\cdots W^{(1)} x + \text{bias}$$

which is no more expressive than one layer. Nonlinearity is what makes
composition create new representations.

### Activation Functions

| Activation | Formula | Gradient at 0 | Saturates? |
|------------|---------|---------------|------------|
| Sigmoid | $1/(1+e^{-x})$ | 0.25 | Yes (both sides) |
| Tanh | $(e^x - e^{-x})/(e^x + e^{-x})$ | 1.0 | Yes (both sides) |
| ReLU | $\max(0, x)$ | 0 | Yes (negative side) |
| GELU | $x \Phi(x)$ | 0.5 | No |
| SiLU/Swish | $x/(1+e^{-x})$ | 0.5 | No |

ReLU (Glorot et al., 2011) solved the vanishing gradient problem for
positive activations: its gradient is 1 for positive inputs, so gradients
do not decay through many layers. GELU and SiLU are now preferred in
transformers because they are smooth and have non-zero gradient for all
inputs.

### The Universal Approximation Theorem

**Cybenko (1989):** Any continuous function $f : [0,1]^d \to \mathbb{R}$
can be approximated to arbitrary accuracy $\epsilon > 0$ by a network of
the form

$$F(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^{\top} x + b_j)$$

provided $\sigma$ is a bounded, continuous, non-constant activation
(e.g., sigmoid). The key limitation: $N$ may need to be exponentially large.

**Hornik (1991)** extended this to arbitrary bounded measurable functions
and any non-polynomial activation, emphasising that the universal
approximation property comes from the multilayer structure, not the specific
choice of activation.

### Weight Initialisation

A critical practical point: if weights are initialised too large, activations
explode; too small, they vanish. He et al. (2015) derived the initialisation
variance for ReLU networks:

$$\text{Var}[W_{ij}] = \frac{2}{n_{\text{in}}}$$

where $n_{\text{in}}$ is the number of input connections to the layer.
This "He initialisation" keeps activation variance stable across depth.
For tanh activations, Glorot (Xavier) initialisation is preferred:

$$\text{Var}[W_{ij}] = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 for x > 0, 0 otherwise."""
    return (x > 0).astype(np.float32)


def he_init(n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    """He initialisation for ReLU networks: Var = 2 / n_in."""
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0, std, size=(n_out, n_in)).astype(np.float32)


class MLPNumpy:
    """Two-hidden-layer MLP for binary classification.

    Demonstrates forward pass, MSE loss, and manual backpropagation.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        rng      = np.random.default_rng(seed)
        self.lr  = lr
        self.weights = []
        self.biases  = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(he_init(n_in, n_out, rng))
            self.biases.append(np.zeros(n_out, dtype=np.float32))

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass; returns (output, pre-activations, activations)."""
        pre_acts: List[np.ndarray] = []
        acts:     List[np.ndarray] = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b          # pre-activation
            pre_acts.append(z)
            # last layer: no activation (logits for MSE); others: ReLU
            h = z if i == len(self.weights) - 1 else relu(z)
            acts.append(h)
        return h, pre_acts, acts

    def backward(
        self,
        x:        np.ndarray,
        y:        np.ndarray,
        pre_acts: List[np.ndarray],
        acts:     List[np.ndarray],
    ) -> float:
        """Backpropagation with MSE loss. Returns loss."""
        y_hat  = acts[-1]
        loss   = float(np.mean((y_hat - y) ** 2))
        delta  = 2 * (y_hat - y) / len(y)   # dL/d(logit)

        for i in reversed(range(len(self.weights))):
            dW    = delta.T @ acts[i]
            db    = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.weights[i] * relu_grad(pre_acts[i - 1])
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db

        return loss
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductionMLP(nn.Module):
    """Configurable MLP with GELU activations, dropout, and layer norm.

    This is the feedforward block found inside every transformer.
    """

    def __init__(
        self,
        layer_sizes:   List[int],
        dropout_rate:  float = 0.1,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i, (n_in, n_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layers.append(nn.Linear(n_in, n_out))
            if i < len(layer_sizes) - 2:
                # Hidden layers get activation and optionally dropout/norm
                if use_layer_norm:
                    layers.append(nn.LayerNorm(n_out))
                layers.append(nn.GELU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply He initialisation to all Linear layers."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using sigmoid or tanh activations in deep networks without
  residual connections. Both saturate: gradients in the saturation region
  are near zero, stalling training in deep networks.

  **Fix:** Use ReLU, GELU, or SiLU for hidden layers. Reserve sigmoid for
  output layers in binary classification.

- **Mistake:** Initialising weights from a standard normal distribution
  without scaling by layer width. Activations grow exponentially with depth.

  **Fix:** Use He initialisation for ReLU layers, Glorot/Xavier for tanh
  layers. PyTorch applies Kaiming uniform by default in `nn.Linear`.

- **Mistake:** Not using dropout regularisation on large MLPs.
  MLPs with more parameters than training examples memorise training data.

  **Fix:** Add `nn.Dropout(p=0.1–0.5)` after each hidden activation for
  large models. Tune dropout rate via cross-validation.

- **Mistake:** Using a learning rate that is too large for deep MLPs.
  Large gradients compound multiplicatively across layers, causing NaN losses.

  **Fix:** Use gradient clipping (`torch.nn.utils.clip_grad_norm_`) and
  warm up the learning rate for the first few epochs.

- **Mistake:** Forgetting to call `model.eval()` during inference. This
  leaves dropout and batch normalisation in training mode, causing
  non-deterministic predictions at test time.

  **Fix:** Always use `model.eval()` and `torch.no_grad()` together during
  evaluation and inference.

## 6. Knowledge Check

1. **Conceptual:** State the Universal Approximation Theorem precisely.
   What does it say about the number of neurons needed? What does it NOT
   say about learning (optimisation) or generalisation?

2. **Conceptual:** Why do ReLU networks not suffer from the vanishing
   gradient problem for positive activations? What is the "dying ReLU"
   problem and how do Leaky ReLU and GELU address it?

3. **Coding challenge:** Train a 4-layer MLP on the MNIST dataset using
   PyTorch. Experiment with ReLU, GELU, and sigmoid activations. Plot the
   training loss curves. Explain any differences you observe in terms of
   the activation function properties discussed in this lesson.

## References

1. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organisation in the brain. *Psychological Review*, 65(6), 386–408.
2. Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
3. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
4. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *ICCV 2015*.
6. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*.
