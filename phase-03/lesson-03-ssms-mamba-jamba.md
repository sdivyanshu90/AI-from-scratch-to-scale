# State Space Models, Mamba, and Jamba

State Space Models (SSMs) are a family of sequence models grounded in
classical control theory that offer linear-time inference and constant memory
— properties transformers fundamentally cannot achieve due to the $O(n^2)$
attention matrix. Mamba (Gu & Dao, 2023) made SSMs practical for language
modelling by introducing input-dependent (selective) state transitions.
Jamba (Lieber et al., 2024) demonstrated that combining attention and SSM
layers in a hybrid architecture achieves the best of both worlds.

## 1. The Core Intuition (The "Why")

Transformers have two fundamental scaling problems:
1. **Quadratic attention**: the $n \times n$ attention matrix costs $O(n^2)$
   in memory and time with sequence length $n$.
2. **Linear KV cache growth**: during autoregressive generation, the key-value
   cache grows as $O(n \cdot d)$, making long-context generation expensive.

State Space Models compress the entire sequence history into a fixed-size
**state vector** $h_t \in \mathbb{R}^N$ updated at each step. This gives
$O(1)$ memory for inference and $O(n)$ computation for training (using the
parallel convolutional form).

The tradeoff: SSMs lose the ability to recall a specific token from far back
with the precision of attention. Hybrid models (Jamba) address this by using
attention layers for precise retrieval and SSM layers for efficient local
processing.

## 2. The Theoretical Underpinning

### The Continuous SSM

A linear time-invariant (LTI) state space model is described by:

$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$

where $h(t) \in \mathbb{R}^N$ is the latent state, $x(t) \in \mathbb{R}$
is the input, and $y(t) \in \mathbb{R}$ is the output. $A, B, C, D$ are
parameter matrices.

### Discretisation

For digital sequences with step size $\Delta$, the continuous system is
discretised using the Zero-Order Hold (ZOH) rule:

$$\bar{A} = e^{\Delta A}, \quad \bar{B} = (e^{\Delta A} - I) A^{-1} B \approx \Delta B$$

The discrete recurrence is:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t + D x_t$$

This is a linear RNN. For inference, it runs recurrently in $O(N)$ per step.
For training, it can be computed as a 1D convolution: the kernel
$\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$ and
$y = \bar{K} \star x$, computable in $O(n \log n)$ via FFT.

### HiPPO: Principled A Initialisation

The choice of $A$ matrix determines how well the model retains history. Gu
et al. (2020) derived the **HiPPO matrix** that keeps the state $h(t)$ as
the optimal polynomial approximation of the history $x(\tau), \tau \leq t$:

$$A_{nk} = -\begin{cases} \sqrt{(2n+1)(2k+1)} & n > k \\ n + 1 & n = k \\ 0 & n < k \end{cases}$$

This structured matrix is the key to S4's ability to model long-range
dependencies, and is used to initialise $A$ in SSM layers.

### Selective SSM: Mamba's Key Innovation

In S4, $A, B, C$ are fixed (input-independent), limiting selectivity.
Mamba makes $B$, $C$, and $\Delta$ **input-dependent**:

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \text{softplus}(s_\Delta(x_t))$$

where $s_B, s_C, s_\Delta$ are linear projections of the input. This allows
the model to decide *what to forget* and *what to remember* based on content,
which is impossible in LTI systems but essential for language (proper nouns,
referents, and context switches all require selective memory updates).

The cost: input-dependent parameters break the ability to use the FFT
convolution for training. Mamba solves this with a custom **parallel scan**
kernel that achieves $O(n \log n)$ training time while remaining memory-
efficient via hardware-aware implementation.

### Hybrid Architecture: Jamba

Jamba interleaves transformer attention layers with Mamba SSM layers in a
ratio of roughly 1 attention layer per 7 SSM layers. This provides:
- Attention layers: precise retrieval, handling of rare tokens, global context.
- SSM layers: efficient local processing, linear inference cost.

The resulting model achieves higher throughput and lower memory usage than
a pure transformer at the same effective capacity.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def build_hippo_matrix(N: int) -> np.ndarray:
    """Construct the HiPPO-LegS A matrix for state size N.

    The HiPPO matrix provides a mathematically principled initialisation
    that encourages the SSM to retain a polynomial approximation of
    its input history.

    Args:
        N: State dimension.

    Returns:
        A matrix of shape (N, N).
    """
    A = np.zeros((N, N))
    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt((2*n + 1) * (2*k + 1))
            elif n == k:
                A[n, k] = -(n + 1)
    return A


def discretize_zoh(
    A: np.ndarray,
    B: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-Order Hold discretisation of continuous SSM matrices.

    Args:
        A:     Continuous A matrix of shape (N, N).
        B:     Continuous B matrix of shape (N, 1).
        delta: Discretisation step size.

    Returns:
        (A_bar, B_bar): Discrete A and B matrices.
    """
    A_bar = expm(delta * A)
    B_bar = np.linalg.solve(A, (A_bar - np.eye(len(A))) @ B)
    return A_bar, B_bar


def ssm_recurrence(
    A_bar: np.ndarray,
    B_bar: np.ndarray,
    C:     np.ndarray,
    xs:    np.ndarray,
) -> np.ndarray:
    """Run the SSM recurrence over a sequence.

    Args:
        A_bar: Discrete A of shape (N, N).
        B_bar: Discrete B of shape (N, 1).
        C:     Output matrix of shape (1, N).
        xs:    Input sequence of shape (T,).

    Returns:
        Output sequence of shape (T,).
    """
    N = A_bar.shape[0]
    h = np.zeros(N)
    ys = []
    for x in xs:
        h = A_bar @ h + B_bar.ravel() * x
        y = float(C @ h)
        ys.append(y)
    return np.array(ys)
```

## 4. Implementation: Production-Grade (Mamba)

```python
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Mamba SSM block from Gu & Dao (2023).

    This wraps the Mamba layer from the official mamba_ssm package,
    which includes the CUDA-optimised parallel scan kernel.

    The Mamba block processes sequences in linear time during training
    and with constant memory (state size) during inference.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model = d_model,
            d_state = d_state,   # state dimension N
            d_conv  = d_conv,    # local convolution width before SSM
            expand  = expand,    # expand factor for inner dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm residual Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of the same shape.
        """
        return x + self.mamba(self.norm(x))


class MambaLM(nn.Module):
    """Minimal Mamba language model stack."""

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_layers:   int,
    ) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: embedding and lm_head share weights
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using the naive SSM recurrence for training. The sequential
  loop is $O(nN)$ and cannot be parallelised on GPU.

  **Fix:** Use the parallel associative scan for training. The Mamba package
  provides a CUDA kernel that achieves near-theoretical efficiency.

- **Mistake:** Initialising $A$ randomly. Random $A$ matrices do not have the
  structured eigenspectrum needed for long-range memory.

  **Fix:** Initialise $A$ from the HiPPO matrix for best long-range memory
  properties.

- **Mistake:** Choosing a very large state dimension $N$ without profiling.
  Large $N$ increases both memory and compute for the recurrence.

  **Fix:** Start with $N = 16$ (Mamba's default) and increase only if
  benchmarks show under-fitting.

- **Mistake:** Expecting SSMs to match transformers on tasks requiring
  precise long-range recall (e.g., retrieving a specific fact mentioned
  earlier in a long document).

  **Fix:** Use a hybrid architecture (Mamba + attention layers) or use
  attention-based retrieval (RAG) for such tasks.

- **Mistake:** Not using `torch.bfloat16` when running Mamba. The
  hardware-aware CUDA kernel is optimised for BF16.

  **Fix:** Load the model in BF16 and ensure inputs are cast accordingly.

## 6. Knowledge Check

1. **Conceptual:** Compare the inference complexity of a transformer
   (with KV cache) and an SSM for generating one new token from a context
   of $n$ tokens. Why does the SSM scale better for very long contexts?

2. **Conceptual:** The selective SSM (Mamba) makes $B$, $C$, and $\Delta$
   input-dependent. Why does this break the FFT convolution trick used
   in S4? How does the parallel scan recover $O(n)$ time complexity?

3. **Coding challenge:** Implement the parallel prefix scan (associative
   scan) for a linear SSM recurrence $h_t = a_t h_{t-1} + b_t$ using
   PyTorch. Verify that your parallel implementation produces the same
   output as the sequential loop for a small example.

## References

1. Gu, A., Goel, K., & Re, C. (2021). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*. arXiv:2111.00396. *(S4.)*
2. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
3. Lieber, O., Lenz, B., Bata, H., et al. (2024). Jamba: A hybrid Transformer-Mamba language model. arXiv:2403.19887.
4. Gu, A., Dao, T., Ermon, S., Rudra, A., & Re, C. (2020). HiPPO: Recurrent memory with optimal polynomial projections. *NeurIPS 2020*.
