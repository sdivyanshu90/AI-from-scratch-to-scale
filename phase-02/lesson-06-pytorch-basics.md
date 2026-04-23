# PyTorch Basics

PyTorch is the de facto research and production framework for deep learning.
Its dynamic computation graph and imperative execution model make debugging
natural, while its autograd engine handles derivative computation automatically.
Understanding how PyTorch works internally — especially the autograd tape and
tensor memory model — is the foundation for writing efficient, correct deep
learning code.

## 1. The Core Intuition (The "Why")

The preceding frameworks (Theano, early TensorFlow) required you to define
a static computation graph, compile it, and then feed data through it.
Debugging required special sessions and graph inspection tools.

PyTorch (Paszke et al., 2017) introduced **define-by-run**: the computation
graph is built dynamically as Python code executes. Each operation records
its inputs in a gradient tape; backpropagation walks this tape in reverse to
compute gradients. This means you can use ordinary Python control flow (loops,
conditionals, print statements) in your model code.

The result: PyTorch made deep learning research dramatically faster to iterate.
You can inspect intermediate tensors with the same tools you use for NumPy
arrays, print shapes at any point, and step through your model in a debugger.

## 2. The Theoretical Underpinning

### Tensors as Memory Buffers

A PyTorch tensor is a typed, multi-dimensional array backed by a contiguous
memory buffer. The buffer itself is managed by a `Storage` object that may
live on CPU or GPU memory. A tensor is a *view* into this storage defined by
a pointer, shape, and strides.

Strides define how many elements to skip to advance one step along each
dimension. For a row-major $3 \times 4$ tensor, strides are $(4, 1)$:
advancing one row requires skipping 4 elements; advancing one column skips 1.
A transposed view has strides $(1, 4)$ and uses the same underlying storage
— no data is copied.

### The Autograd Tape

When you create a tensor with `requires_grad=True`, every operation on it
creates an internal `Function` node that records:
1. The operation type (e.g., `MmBackward` for matrix multiply).
2. References to the input tensors.
3. The partial derivatives (as closures) for the backward pass.

Together, these nodes form a directed acyclic graph (DAG). Calling `.backward()`
on a scalar loss triggers a reverse-mode traversal: the gradient of the output
is multiplied by each function's Jacobian and propagated to the inputs.

For a computation $y = f(x)$, the gradient rule is:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

where $\partial L / \partial y$ is the incoming gradient and $\partial y / \partial x$
is the local Jacobian of $f$.

### Computational Graph Example

For `z = x @ W + b`:

```
x ──┐
    ├─ MatMul ──┐
W ──┘           ├─ Add ── z
b ──────────────┘
```

The backward pass computes:

$$\frac{\partial L}{\partial W} = x^{\top} \cdot \frac{\partial L}{\partial z}, \quad \frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot W^{\top}, \quad \frac{\partial L}{\partial b} = \sum_i \frac{\partial L}{\partial z_i}$$

These are the rules for dense linear layers that every deep learning framework
implements.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

from typing import Callable, Optional
import numpy as np


class Tensor:
    """A minimal autograd tensor supporting add, matmul, and relu.

    Each Tensor stores its value and, if it is a leaf with requires_grad,
    will accumulate .grad during backward(). Non-leaf tensors store a
    _backward function that propagates gradients to inputs.
    """

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
    ) -> None:
        self.data          = np.asarray(data, dtype=np.float32)
        self.grad:         Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward:    Callable[[], None] = lambda: None
        self._prev:        list["Tensor"] = []

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data)
        out._prev = [self, other]

        def _backward() -> None:
            # dL/dA = dL/dC @ B^T
            # dL/dB = A^T @ dL/dC
            if self.requires_grad:
                g = out.grad @ other.data.T
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = self.data.T @ out.grad
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data))
        out._prev = [self]

        def _backward() -> None:
            if self.requires_grad:
                g = out.grad * (self.data > 0).astype(np.float32)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def backward(self) -> None:
        """Topological sort and reverse-mode AD."""
        topo  = []
        visited: set[int] = set()

        def build(t: "Tensor") -> None:
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    build(p)
                topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Generator


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation.

    This demonstrates the standard PyTorch Module pattern:
    - Define parameters in __init__ via nn.Linear layers.
    - Implement the forward pass in forward().
    - PyTorch builds the autograd graph dynamically during forward().
    """

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def training_loop(
    model:    nn.Module,
    X:        torch.Tensor,
    y:        torch.Tensor,
    lr:       float = 1e-3,
    epochs:   int   = 100,
    batch_size: int = 256,
) -> list[float]:
    """Standard PyTorch training loop.

    Returns:
        List of per-epoch average loss values.
    """
    dataset    = TensorDataset(X, y)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn    = nn.CrossEntropyLoss()
    epoch_losses: list[float] = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()           # clear accumulated gradients
            logits = model(X_batch)
            loss   = loss_fn(logits, y_batch)
            loss.backward()                 # compute gradients via autograd
            optimizer.step()               # update parameters
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(loader))

    return epoch_losses


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Forgetting `optimizer.zero_grad()` at the start of each
  training step. PyTorch accumulates gradients by default; skipping this
  call adds gradients from the previous batch to the current one.

  **Fix:** Always call `optimizer.zero_grad()` before the forward pass.
  Alternatively, `optimizer.zero_grad(set_to_none=True)` is slightly faster.

- **Mistake:** Performing gradient computations inside `torch.no_grad()`
  during evaluation — or forgetting to use it. Without `no_grad()`,
  PyTorch builds an autograd graph for every forward pass, wasting memory.

  **Fix:** Wrap evaluation code in `with torch.no_grad():`. For inference
  pipelines, additionally call `model.eval()` to disable dropout and
  batch normalisation.

- **Mistake:** Mixing NumPy operations and PyTorch tensors mid-computation.
  NumPy operations detach tensors from the autograd graph silently.

  **Fix:** Keep all computations in PyTorch until the final step. Convert
  to NumPy only after `.detach().cpu().numpy()`.

- **Mistake:** Moving tensors between CPU and GPU inside the training loop.
  `.to(device)` calls within the batch loop add significant overhead.

  **Fix:** Move the entire dataset to the device before training, or use
  `pin_memory=True` and `non_blocking=True` in the DataLoader.

- **Mistake:** Using `.data` to access tensor values inside the training loop.
  `.data` bypasses the autograd graph and can cause silent gradient errors.

  **Fix:** Use `.detach()` or `with torch.no_grad():` for operations that
  should not contribute to gradients.

## 6. Knowledge Check

1. **Conceptual:** Explain what "strides" are for a PyTorch tensor. If you
   call `.T` (transpose) on a 2D tensor, does a new memory buffer get
   allocated? What does `.contiguous()` do and when is it needed?

2. **Conceptual:** Why does PyTorch accumulate gradients rather than
   resetting them each backward pass? Give a practical use case where
   gradient accumulation is intentional and useful.

3. **Coding challenge:** Build a small neural network in PyTorch and
   implement gradient checkpointing manually: split the forward pass into
   two segments and only keep the activations at the segment boundary,
   recomputing the first segment during the backward pass.

## References

1. Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*. arXiv:1912.01703.
2. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: A survey. *JMLR*, 18, 1–43.
3. Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys*, 23(1), 5–48.
