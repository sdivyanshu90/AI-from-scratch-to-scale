# Calculus for Machine Learning

Calculus is the engine that turns a loss function into a trained model.
Without it, the only way to improve a model's parameters would be exhaustive
search — computationally infeasible even for a network with a few hundred
weights. Derivatives give us a principled, scalable, local rule for
improvement, and the chain rule makes that rule composable across arbitrarily
deep computation graphs.

## 1. The Core Intuition (The "Why")

Imagine you are blindfolded on a hilly landscape and your only goal is to find
the lowest point. You cannot see the whole terrain, but you *can* feel which
direction is downhill beneath your feet. That local slope is what a derivative
gives you: the instantaneous rate of change of the loss with respect to a
parameter.

Before gradient-based optimisation became standard, practitioners used grid
search, coordinate ascent, and evolutionary strategies. These work acceptably
with fewer than a dozen parameters. They are completely infeasible for modern
neural networks with millions or billions of weights.

The breakthrough insight of Rumelhart, Hinton, and Williams (1986) was that
the chain rule of calculus lets you compute the gradient of the loss with
respect to *every* parameter in a network in a single backward pass —
computational cost proportional to the forward pass, not to the number of
parameters. Neural networks are compositions of differentiable functions, and
the chain rule tells us exactly how to compose those derivatives.

## 2. The Theoretical Underpinning

### Derivatives and the Gradient

Let $L(w)$ be a scalar loss as a function of parameter $w \in \mathbb{R}$.
The derivative at $w$ is defined as

$$
\frac{dL}{dw} = \lim_{\epsilon \to 0}
\frac{L(w + \epsilon) - L(w)}{\epsilon}
$$

If $\frac{dL}{dw} > 0$, increasing $w$ increases the loss; we should decrease
$w$. For a parameter vector $\theta \in \mathbb{R}^{d}$, the gradient stacks
all partial derivatives into a vector pointing in the direction of steepest
ascent:

$$
\nabla_{\theta} L =
\left[\,
\frac{\partial L}{\partial \theta_{1}},\;
\cdots,\;
\frac{\partial L}{\partial \theta_{d}}
\,\right]^{\top}
$$

### Gradient Descent Update Rule

$$
\theta^{(t + 1)} = \theta^{(t)} - \eta \, \nabla_{\theta}L\!\left(\theta^{(t)}\right)
$$

$\eta > 0$ is the **learning rate** — the step size in parameter space. Too
large: the update overshoots minima and the loss diverges. Too small:
convergence is prohibitively slow. Gradient descent was first described for
convex functions by Cauchy (1847). Robbins & Monro (1951) extended it to the
stochastic setting (SGD), which is the basis for every modern deep-learning
optimiser.

### The Chain Rule — Why Deep Learning Scales

If the output of one function feeds into another, $y = f(g(x))$, the chain
rule gives

$$
\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}, \quad u = g(x)
$$

For a depth-$L$ network with parameters $\theta_k$ at layer $k$, the gradient
with respect to layer $k$ is

$$
\frac{\partial \mathcal{L}}{\partial \theta_k} =
\frac{\partial \mathcal{L}}{\partial f_L} \cdot
\frac{\partial f_L}{\partial f_{L-1}} \cdots
\frac{\partial f_k}{\partial \theta_k}
$$

Each term is a local Jacobian. Backpropagation computes these products in
reverse, reusing activations cached during the forward pass. This is why
backpropagation costs $O(\text{forward pass})$, not $O(\text{parameters}^2)$.

### Worked Example: Linear Regression Gradients

For $\hat{y}_i = w x_i + b$ with MSE loss
$L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^{2}$:

$$
\frac{\partial L}{\partial w} =
\frac{2}{n}\sum_{i=1}^{n} x_i(\hat{y}_i - y_i)
\qquad
\frac{\partial L}{\partial b} =
\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

The weight update is large when the input $x_i$ is large and the error is
large. The model "learns faster" from informative, poorly-predicted examples.

### Jacobians and Vector-Valued Functions

When a layer maps $x \in \mathbb{R}^{d}$ to $y \in \mathbb{R}^{m}$, the
derivative is the **Jacobian** $J \in \mathbb{R}^{m \times d}$ with
$J_{ij} = \partial y_i / \partial x_j$. Backpropagation propagates a
row-vector of upstream gradients through $J$ via a vector-Jacobian product
(VJP), which is why frameworks expose a `backward` hook rather than
materialising the full Jacobian.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def predict_linear(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    if y_hat.shape != y.shape:
        raise ValueError("Shape mismatch between predictions and targets.")
    return float(np.mean((y_hat - y) ** 2))


def linear_gradients(
    x: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float,
) -> Tuple[float, float]:
    """Compute analytic gradients of MSE w.r.t. weight and bias.

    dL/dw = (2/n) * sum(x * (y_hat - y))  -- input scales the error signal
    dL/db = (2/n) * sum(y_hat - y)         -- bias sees every residual equally
    """
    y_hat  = predict_linear(x, w, b)
    errors = y_hat - y
    n      = float(x.shape[0])
    dw = float((2.0 / n) * np.sum(x * errors))
    db = float((2.0 / n) * np.sum(errors))
    return dw, db


def train_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float,
    steps: int,
) -> Dict[str, float]:
    """Train a scalar linear model via gradient descent."""
    w, b = 0.0, 0.0
    for _ in range(steps):
        dw, db = linear_gradients(x, y, w, b)
        w -= lr * dw   # theta -= eta * grad
        b -= lr * db
    preds = predict_linear(x, w, b)
    return {"weight": w, "bias": b, "loss": mse_loss(preds, y)}
```

## 4. Implementation: Production-Grade (PyTorch / SOTA Library)

PyTorch's `autograd` builds a computational graph during the forward pass and
traverses it in reverse during `.backward()`. The user writes the forward
computation; autograd handles the rest.

```python
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class ScalarRegressor(nn.Module):
    """Single-feature linear regressor backed by autograd."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_with_autograd(
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    steps: int,
) -> Dict[str, float]:
    """Train with SGD; autograd computes gradients automatically."""
    model     = ScalarRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    for _ in range(steps):
        optimizer.zero_grad()          # clear accumulated gradients
        loss = loss_fn(model(x), y)
        loss.backward()                # chain rule applied automatically
        optimizer.step()               # theta -= lr * grad

    return {
        "weight": float(model.linear.weight.detach().item()),
        "bias":   float(model.linear.bias.detach().item()),
        "loss":   float(loss.detach().item()),
    }
```

## 5. Production Pitfalls & Pro-Tips

- ❌ **The mistake:** Choosing a learning rate by intuition and blaming the
  model when loss explodes or plateaus.
  ✅ **The fix:** Use a learning rate range test (Smith, 2017). Sweep $\eta$
  over several orders of magnitude for 100 steps and plot loss vs. $\eta$.
  The optimal value is just below the divergence point.

- ❌ **The mistake:** Shipping a custom autograd function without verifying
  its gradient.
  ✅ **The fix:** Use `torch.autograd.gradcheck`. It computes finite-difference
  numerical gradients and compares them to your analytic ones. Discrepancies
  larger than $10^{-5}$ indicate bugs.

- ❌ **The mistake:** Forgetting to call `optimizer.zero_grad()` before each
  forward pass, causing unintentional gradient accumulation across batches.
  ✅ **The fix:** Make `zero_grad()` the very first line of your training step.
  Intentional gradient accumulation requires explicit tracking.

- ❌ **The mistake:** Using the same learning rate for all layers and not
  knowing that early layers in fine-tuned models often need much smaller
  updates.
  ✅ **The fix:** Use per-layer learning rate multipliers or layer-wise
  adaptive rate scaling (LARS/LAMB) for large-batch training.

- ❌ **The mistake:** Interpreting a low training loss as proof of finding the
  global minimum. Gradient descent is a local method.
  ✅ **The fix:** Monitor validation loss and use proper initialisation (He or
  Xavier), batch normalisation, and gradient clipping to keep optimisation
  stable.

## 6. Knowledge Check

1. **Conceptual:** In deep networks, the chain rule produces a product of
   Jacobians from the loss back to the inputs. Why does this product tend to
   vanish or explode in very deep networks, and what architectural choices
   (residual connections, normalisation, activation functions) mitigate it?

2. **Conceptual:** SGD computes the gradient on a random mini-batch rather than
   the full dataset. Why does this noisy estimate still converge, and what
   unexpected benefit does the noise provide in non-convex loss landscapes?

3. **Coding challenge:** Implement gradient descent for $L(w) = (w - 3)^{2}$
   starting from $w = 0$. Run 50 steps with $\eta = 0.1$. What is the minimum
   number of steps required to reach $L < 10^{-4}$?

## References

1. Cauchy, A. L. (1847). Méthode générale pour la résolution des systèmes d'équations simultanées. *Compte Rendu*, 25, 536–538. *(First gradient descent.)*
2. Robbins, H., & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407. *(Origin of SGD.)*
3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
4. Werbos, P. (1974). *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences*. PhD thesis, Harvard University.
5. Bengio, Y. (2012). Practical recommendations for gradient-based training of deep architectures. In *Neural Networks: Tricks of the Trade*, Springer.
6. Smith, L. N. (2017). Cyclical learning rates for training neural networks. *WACV 2017*.
