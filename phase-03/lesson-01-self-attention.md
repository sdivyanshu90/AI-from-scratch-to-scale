# Self-Attention and the Transformer

The Transformer (Vaswani et al., 2017) replaced recurrent networks for
sequence modelling by introducing self-attention: a mechanism that computes
pairwise interactions between all positions in a sequence in parallel. Every
large language model, vision transformer, and multimodal model built since
2018 is fundamentally a stack of self-attention layers. Understanding the
attention mechanism at the mathematical level — and its computational
constraints — is the single most important technical skill for modern ML
engineering.

## 1. The Core Intuition (The "Why")

RNNs process sequences one step at a time. This means:
1. The computation is inherently sequential — you cannot parallelise across
   positions on a GPU.
2. Information must travel through many RNN steps to connect distant positions,
   compressing it into a fixed-size hidden state. Long-range dependencies are
   hard to learn.

Self-attention solves both problems simultaneously: it computes relationships
between all pairs of positions in a single matrix operation, is fully
parallelisable, and provides constant-length dependency paths between any
two positions.

Bahdanau et al. (2014) introduced attention as an add-on to RNNs: the decoder
learns to query the encoder states selectively. Vaswani et al. (2017) made
attention the entire architecture, removing recurrence entirely.

## 2. The Theoretical Underpinning

### Queries, Keys, and Values

Given an input sequence $X \in \mathbb{R}^{n \times d}$ (n tokens, d
dimensions), self-attention projects it into three matrices via learned
linear projections:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$. The **scaled dot-product
attention** is then:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V$$

- $QK^{\top} \in \mathbb{R}^{n \times n}$: the attention logit matrix.
  Entry $(i, j)$ is the dot product between query $i$ and key $j$, measuring
  how much position $i$ should attend to position $j$.
- $\sqrt{d_k}$ scaling: without this, large $d_k$ causes dot products to grow
  large in magnitude, pushing softmax into its saturation region where gradients
  vanish. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products at
  approximately 1.
- $V$: each row of the output is a weighted average of value vectors, with
  weights given by the attention probabilities.

### Multi-Head Attention

A single attention head can only attend to one "type" of relationship at a
time. Multi-head attention runs $H$ attention heads in parallel, each with
its own $W^Q_h, W^K_h, W^V_h$, then concatenates the outputs:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

$$\text{head}_h = \text{Attention}(QW^Q_h, KW^K_h, VW^V_h)$$

With $H$ heads and $d_k = d / H$, the total computational cost is the same
as single-head attention. Different heads learn to attend to different aspects
(syntactic roles, coreference, positional relationships).

### Causal (Masked) Self-Attention

For autoregressive language models (GPT), position $i$ must not attend to
positions $j > i$ (the future). A causal mask sets the attention logits to
$-\infty$ for $j > i$ before the softmax, making those probabilities zero:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V$$

where $M_{ij} = 0$ if $j \leq i$ and $M_{ij} = -\infty$ if $j > i$.

### Positional Encoding

Self-attention is permutation-equivariant: it treats the sequence as a set.
To encode order, the original Transformer adds sinusoidal positional encodings:

$$PE_{(pos, 2i)}   = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Modern LLMs use Rotary Position Embedding (RoPE, Su et al., 2021), which
applies a rotation matrix to Q and K that encodes relative position, providing
better extrapolation to longer sequences.

### Complexity

Standard self-attention is $O(n^2 d)$ in time and $O(n^2)$ in memory due to
the $n \times n$ attention matrix. For long sequences (> 4096 tokens), this
is prohibitive. FlashAttention (Dao et al., 2022) reorders the computation
using tiling to avoid materialising the full $n \times n$ matrix, achieving
$O(n^2 d / M)$ time (where $M$ is SRAM size) and $O(n)$ memory.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last dimension."""
    x_shift = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x_shift)
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scaled dot-product attention (Vaswani et al., 2017).

    Args:
        Q:    Queries of shape (..., n, d_k).
        K:    Keys    of shape (..., n, d_k).
        V:    Values  of shape (..., n, d_v).
        mask: Optional additive mask of shape (..., n, n).
              Use 0 for allowed positions and -1e9 for masked positions.

    Returns:
        Output of shape (..., n, d_v) and attention weights (..., n, n).
    """
    d_k    = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)  # (..., n, n)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)                        # (..., n, n)
    return weights @ V, weights


def causal_mask(n: int) -> np.ndarray:
    """Lower-triangular causal mask; -1e9 for future positions."""
    mask = np.triu(np.full((n, n), -1e9), k=1)
    return mask.astype(np.float32)


class MultiHeadAttentionNumpy:
    """Multi-head self-attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int, seed: int = 0) -> None:
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        rng          = np.random.default_rng(seed)
        std          = 0.02
        self.Wq = rng.normal(0, std, (d_model, d_model)).astype(np.float32)
        self.Wk = rng.normal(0, std, (d_model, d_model)).astype(np.float32)
        self.Wv = rng.normal(0, std, (d_model, d_model)).astype(np.float32)
        self.Wo = rng.normal(0, std, (d_model, d_model)).astype(np.float32)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, n, d_model) -> (batch, heads, n, d_k)."""
        b, n, d = x.shape
        x = x.reshape(b, n, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(
        self, x: np.ndarray, causal: bool = False
    ) -> np.ndarray:
        b, n, _ = x.shape
        Q = self._split_heads(x @ self.Wq)
        K = self._split_heads(x @ self.Wk)
        V = self._split_heads(x @ self.Wv)
        mask = causal_mask(n) if causal else None
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # Merge heads: (batch, heads, n, d_k) -> (batch, n, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return out @ self.Wo
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Single Transformer block: multi-head attention + FFN.

    Uses PyTorch's built-in scaled_dot_product_attention which
    dispatches to FlashAttention when available (PyTorch >= 2.0).
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        ffn_dim:     int,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.attn    = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        x:           torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-norm transformer block (GPT-2 style).

        Pre-norm (applying LayerNorm before attention) is more stable
        during training of very deep models than post-norm.

        Args:
            x:           Input tensor of shape (batch, seq_len, d_model).
            causal_mask: Boolean mask of shape (seq_len, seq_len).
        """
        # Multi-head self-attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=causal_mask,
                                need_weights=False)
        x = x + self.drop(attn_out)
        # Feed-forward with residual connection
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Forgetting the $\sqrt{d_k}$ scaling in attention. Without it,
  large $d_k$ leads to large logit magnitudes, saturated softmax, and near-
  zero gradients. Training stalls.

  **Fix:** Always divide by $\sqrt{d_k}$. This is the single most common
  bug in from-scratch attention implementations.

- **Mistake:** Using post-norm (LayerNorm after residual) for training deep
  transformers from scratch. Post-norm training is less stable and requires
  learning rate warmup.

  **Fix:** Use pre-norm (LayerNorm before the sub-layer) as in GPT-2 and
  LLaMA. Pre-norm is more stable and generally outperforms post-norm.

- **Mistake:** Materialising the full $n \times n$ attention matrix for
  sequences longer than a few thousand tokens. Memory grows quadratically.

  **Fix:** Use `F.scaled_dot_product_attention` (PyTorch >= 2.0) which
  dispatches to FlashAttention. For even longer sequences, use sliding
  window attention or xFormers.

- **Mistake:** Applying dropout inside attention during inference.

  **Fix:** Always call `model.eval()` before inference. This disables
  all dropout layers.

- **Mistake:** Attending over padding tokens. Attention over padding
  adds noise and wastes computation.

  **Fix:** Pass a padding mask to `nn.MultiheadAttention` via the
  `key_padding_mask` argument.

## 6. Knowledge Check

1. **Conceptual:** Derive the $\sqrt{d_k}$ scaling factor from first
   principles. Assume the query and key vectors have entries drawn from
   $\mathcal{N}(0, 1)$. What is the variance of their dot product? Why
   does this push softmax into its saturation region?

2. **Conceptual:** Multi-head attention runs $H$ attention heads in parallel.
   If you used a single head with $d_k = d_{\text{model}}$ (same total
   parameters as multi-head), would performance be the same? Why or why not?

3. **Coding challenge:** Implement scaled dot-product attention from scratch
   in PyTorch (without using `F.scaled_dot_product_attention`). Test that
   your implementation matches PyTorch's for small $n$ and $d_k$. Then
   benchmark both against each other for $n = 2048$.

## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *ICLR 2015*. arXiv:1409.0473.
2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS 2017*. arXiv:1706.03762.
3. Su, J., Lu, Y., Pan, S., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
4. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS 2022*. arXiv:2205.14135.
