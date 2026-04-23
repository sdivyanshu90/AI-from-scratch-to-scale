# AdamW and Optimisers

The optimiser is the algorithm that updates model parameters given their
gradients. Gradient descent is the foundation; modern optimisers like Adam
and AdamW add adaptive learning rates, momentum, and decoupled weight decay.
Choosing the right optimiser and tuning its hyperparameters is one of the
highest-leverage decisions in training a deep learning model.

## 1. The Core Intuition (The "Why")

Vanilla SGD updates all parameters with the same learning rate:
$w \leftarrow w - \eta \nabla_w L$. This has a fundamental problem: features
with very different gradient scales (common in NLP, where some tokens are
much rarer than others) benefit from different learning rates. A parameter
that rarely receives gradient should have its update amplified; a parameter
that receives large gradients every step should be damped.

Adam (Kingma & Ba, 2014) solves this by maintaining a **per-parameter
adaptive learning rate**: it tracks the exponential moving average of
gradients (momentum) and the exponential moving average of squared gradients
(scale), and divides the step by the scale. This normalises the update to
approximately unit scale regardless of the gradient magnitude.

Loshchilov & Hutter (2019) discovered that Adam's L2 regularisation is
mathematically different from weight decay: adding $\lambda w$ to the
gradient (L2) does not produce the same result as multiplying the parameter
by $(1 - \lambda)$ after the update (weight decay). AdamW implements the
correct decoupled weight decay and is now the standard for training
transformers.

## 2. The Theoretical Underpinning

### Exponential Moving Averages

Adam maintains two moment estimates for each parameter $w$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment, momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment, gradient scale)}$$

where $g_t$ is the gradient at step $t$, $\beta_1 \approx 0.9$, and
$\beta_2 \approx 0.999$.

The intuition: $m_t$ smooths the gradient direction (like ball rolling with
momentum). $v_t$ estimates the variance of the gradient — high variance means
the gradient is noisy and steps should be small; low variance means we can
take larger steps.

### Bias Correction

At early steps, $m_0 = v_0 = 0$, so the estimates are biased toward zero.
Adam corrects this:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

At step $t=1$: $\beta_1^1 = 0.9$, so $\hat{m}_1 = m_1 / 0.1 = 10 m_1$.
This amplifies the first-step estimate to remove the zero-initialisation
bias. As $t \to \infty$, the correction factor $\to 1$.

### The Adam Update Rule

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\, \hat{m}_t$$

- $\eta$: global learning rate (default 1e-3).
- $\epsilon$: numerical stability constant (default 1e-8); prevents division by zero.
- $\sqrt{\hat{v}_t}$ normalises the step by the gradient scale.

### Decoupled Weight Decay (AdamW)

Adam + L2 regularisation adds $\lambda w$ to the gradient before computing
moments. This means the regularisation effect is scaled by $1/\sqrt{\hat{v}_t}$,
attenuating decay for infrequently-updated parameters. AdamW decouples:

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\,\hat{m}_t - \eta \lambda w_t$$

The weight decay term $-\eta \lambda w_t$ is applied independently of the
adaptive scaling. For the same $\lambda$, AdamW provides stronger
regularisation on rarely-updated parameters than Adam + L2.

### Learning Rate Schedules

A fixed learning rate is rarely optimal. Common schedules:

- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$
- **Linear warmup + cosine decay**: used in nearly all large language models.
  The first $T_w$ steps linearly ramp up from 0 to $\eta_{\max}$.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np
from typing import Dict, List


class AdamW:
    """AdamW optimiser with decoupled weight decay.

    Implements Algorithm 2 from Loshchilov & Hutter (2019).
    """

    def __init__(
        self,
        params:       List[np.ndarray],
        lr:           float = 1e-3,
        betas:        tuple = (0.9, 0.999),
        eps:          float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        self.params       = params
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        # Initialise moment estimates to zero
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        """Update all parameters given their gradients.

        Args:
            grads: List of gradient arrays, one per parameter.
        """
        self.t += 1
        # Bias correction factors
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, (p, g) in enumerate(zip(self.params, grads)):
            # Update biased moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            # Bias-corrected estimates
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            # Adaptive step (Adam part)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            # Decoupled weight decay (AdamW part)
            decay = self.lr * self.weight_decay * p
            p -= step + decay
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_optimizer_and_scheduler(
    model:        nn.Module,
    lr:           float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int   = 1000,
    total_steps:  int   = 10_000,
) -> tuple:
    """Build AdamW with linear warmup + cosine decay schedule.

    This is the standard recipe for training transformers.

    Args:
        warmup_steps: Number of steps to linearly increase lr from 0 to lr.
        total_steps:  Total number of training steps.

    Returns:
        (optimizer, scheduler) — call scheduler.step() after each batch.
    """
    # Separate weight-decayed and non-decayed parameters.
    # Bias and LayerNorm parameters should NOT be weight-decayed.
    no_decay    = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    decay_params   = [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)]

    optimizer = AdamW([
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # Linear warmup
    warmup = LinearLR(optimizer,
                      start_factor=1e-8,
                      end_factor=1.0,
                      total_iters=warmup_steps)
    # Cosine decay from lr to lr/10 over remaining steps
    cosine = CosineAnnealingLR(optimizer,
                               T_max=total_steps - warmup_steps,
                               eta_min=lr / 10)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup, cosine],
                             milestones=[warmup_steps])
    return optimizer, scheduler
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using the same learning rate for all parameter groups. Bias
  and LayerNorm parameters behave differently from weight matrices.

  **Fix:** Set `weight_decay=0` for biases and norm parameters. PyTorch
  param groups make this easy.

- **Mistake:** Forgetting learning rate warmup when training large models.
  Starting with the full learning rate causes large initial gradients and
  unstable training.

  **Fix:** Use linear warmup for the first 1–5% of training steps.

- **Mistake:** Using `beta2=0.999` (Adam default) for training with batch
  sizes above 1024. High beta2 causes the second moment to change slowly,
  making the optimiser less responsive.

  **Fix:** For large-batch training, use `beta2=0.95` (used by LLaMA, GPT).

- **Mistake:** Treating AdamW weight decay as equivalent to L2
  regularisation. They are mathematically different for adaptive optimisers.

  **Fix:** Always use AdamW for weight decay with adaptive optimisers. Use
  L2 regularisation only with SGD.

- **Mistake:** Not checkpointing the optimiser state when saving models.
  Adam's moment estimates encode training history; restarting from a saved
  model without them effectively restarts training.

  **Fix:** Save `{"model": model.state_dict(), "optimizer": optimizer.state_dict()}`
  together.

## 6. Knowledge Check

1. **Conceptual:** Explain the bias correction in Adam. Why does the
   initialisation of moment estimates to zero cause a problem, and how does
   the bias correction factor $1/(1 - \beta^t)$ remove it? What happens
   to the correction factor as $t \to \infty$?

2. **Conceptual:** Describe the difference between L2 regularisation and
   decoupled weight decay. Show with equations why they produce different
   effective regularisation strengths for parameters with small vs large
   gradient variance.

3. **Coding challenge:** Implement a learning rate finder (Smith, 2017):
   train for one epoch while exponentially increasing the learning rate
   from 1e-7 to 1e0. Plot the loss vs learning rate. The optimal learning
   rate is just before the loss starts to increase sharply.

## References

1. Robbins, H., & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407. *(Original SGD.)*
2. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*. arXiv:1412.6980.
3. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*. arXiv:1711.05101.
4. Smith, L. N. (2017). Cyclical learning rates for training neural networks. *WACV 2017*. arXiv:1506.01186.
