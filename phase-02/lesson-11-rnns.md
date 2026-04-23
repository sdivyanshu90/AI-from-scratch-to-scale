# Recurrent Neural Networks, LSTMs, and GRUs

Recurrent Neural Networks (RNNs) process sequential data by maintaining a
hidden state that is updated at each time step. They were the dominant
architecture for language modelling, machine translation, and speech
recognition before the transformer. Understanding their core failure mode —
the vanishing gradient problem — and how LSTMs and GRUs solve it explains
both the design principles behind these architectures and why attention-based
models eventually superseded them.

## 1. The Core Intuition (The "Why")

Processing a sentence word by word requires *memory*: the meaning of "bank"
in "I went to the river bank" depends on the earlier word "river." An MLP
has no memory — it maps a fixed-size input to output with no concept of
order or history. An RNN solves this by passing a hidden state from one
time step to the next, creating a form of recurrent memory.

The fundamental problem: for sequences of length $T$, backpropagating through
$T$ time steps multiplies the same weight matrix $T$ times. If the spectral
radius of this matrix is $< 1$, gradients shrink exponentially and early
time steps receive no gradient (vanishing gradients). If it is $> 1$,
gradients grow exponentially (exploding gradients).

Hochreiter and Schmidhuber (1997) designed the LSTM with explicit gates to
control information flow, creating a gradient highway through time that
prevents vanishing. Cho et al. (2014) introduced the GRU as a simpler
alternative with fewer gates.

## 2. The Theoretical Underpinning

### Vanilla RNN

At each time step $t$, the RNN computes:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$$

$$\hat{y}_t = W_y h_t + b_y$$

where $h_t \in \mathbb{R}^{d_h}$ is the hidden state, $x_t \in \mathbb{R}^{d_x}$
is the input, and $W_h, W_x, W_y$ are weight matrices.

### Backpropagation Through Time (BPTT)

The gradient of the loss $L = \sum_t \ell_t$ at step $T$ with respect to
the hidden state at step $t$ is:

$$\frac{\partial L}{\partial h_t} = \frac{\partial \ell_T}{\partial h_T} \cdot \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Each factor is:

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(\tanh'(z_k)) \cdot W_h$$

The product of $T - t$ such Jacobians determines whether gradients vanish or
explode. For large $T$, if $\lVert W_h \rVert < 1$, these products tend to zero.

### LSTM: The Gating Solution

The LSTM (Hochreiter & Schmidhuber, 1997) introduces a **cell state** $c_t$
that runs through time with only element-wise operations, avoiding the
repeated matrix multiplication that causes vanishing gradients.

Four gates at time step $t$:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(cell candidate)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$

Cell state update and hidden state:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

The key insight: the gradient of $c_t$ with respect to $c_{t-1}$ is just
$f_t$ (element-wise). Since the forget gate $f_t \in (0,1)$ is learnable,
the LSTM can maintain gradients with magnitude close to 1 when $f_t \approx 1$,
preventing vanishing.

### GRU: Simplified Gating

The GRU (Cho et al., 2014) merges cell and hidden state, using only two gates:

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t]) \quad \text{(candidate)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

The update gate $z_t$ interpolates between the old state and the candidate:
when $z_t \approx 0$, the state is copied forward unchanged. This achieves
similar gradient flow to LSTM with fewer parameters.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np


class VanillaRNN:
    """Single-layer vanilla RNN for sequence classification."""

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        output_size: int,
        seed:        int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        std = 0.01
        self.Wh = rng.normal(0, std, (hidden_size, hidden_size)).astype(np.float32)
        self.Wx = rng.normal(0, std, (hidden_size, input_size)).astype(np.float32)
        self.bh = np.zeros(hidden_size, dtype=np.float32)
        self.Wy = rng.normal(0, std, (output_size, hidden_size)).astype(np.float32)
        self.by = np.zeros(output_size, dtype=np.float32)

    def forward(self, xs: list[np.ndarray]) -> tuple[list, list]:
        """Forward pass over a sequence.

        Args:
            xs: List of T input vectors of shape (input_size,).

        Returns:
            (ys, hs): Output logits and hidden states.
        """
        h = np.zeros(self.Wh.shape[0], dtype=np.float32)
        hs, ys = [h], []
        for x in xs:
            h = np.tanh(self.Wh @ h + self.Wx @ x + self.bh)
            y = self.Wy @ h + self.by
            hs.append(h)
            ys.append(y)
        return ys, hs


class LSTMCell:
    """Single LSTM cell step."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        rng  = np.random.default_rng(0)
        concat = input_size + hidden_size
        # Pack all four gate weights into one matrix for efficiency
        self.W = rng.normal(0, 0.01, (4 * hidden_size, concat)).astype(np.float32)
        self.b = np.zeros(4 * hidden_size, dtype=np.float32)

    def step(
        self,
        x:  np.ndarray,
        hc: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """One LSTM step.

        Args:
            x:  Input of shape (input_size,).
            hc: (h_{t-1}, c_{t-1}).

        Returns:
            (h_t, c_t).
        """
        h_prev, c_prev = hc
        xh    = np.concatenate([x, h_prev])
        gates = self.W @ xh + self.b
        d     = len(h_prev)
        f = 1 / (1 + np.exp(-gates[0*d : 1*d]))   # forget gate
        i = 1 / (1 + np.exp(-gates[1*d : 2*d]))   # input gate
        g = np.tanh(gates[2*d : 3*d])              # cell candidate
        o = 1 / (1 + np.exp(-gates[3*d : 4*d]))   # output gate
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        return h, c
```

## 4. Implementation: Production-Grade (PyTorch)

```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier with variable-length input support.

    Handles variable-length sequences correctly using PackedSequence.
    """

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int,
        hidden_size:  int,
        num_layers:   int,
        num_classes:  int,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm   = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths:   torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: Padded token indices of shape (batch, seq_len).
            lengths:   True sequence lengths of shape (batch,).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x       = self.embed(token_ids)
        # Pack to skip computation on padding tokens
        packed  = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                       enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers, batch, hidden_size)
        h_last  = h_n[-1]                    # last layer's hidden state
        return self.fc(self.dropout(h_last))
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Not handling variable-length sequences with padding.
  Computing LSTM on padding tokens wastes computation and pollutes the
  hidden state with meaningless updates.

  **Fix:** Use `pack_padded_sequence` / `pad_packed_sequence` in PyTorch.
  Sort sequences by length or use `enforce_sorted=False`.

- **Mistake:** Initialising the LSTM forget gate bias to zero. At the start
  of training, $f_t \approx 0.5$, meaning the network forgets half its state
  at each step. Long-range dependencies never get a chance to form.

  **Fix:** Initialise the forget gate bias to a positive value (e.g., 1.0).
  This makes the network remember by default initially.

- **Mistake:** Using a unidirectional LSTM for tasks where the full context
  is available at inference time (e.g., text classification).

  **Fix:** Use a bidirectional LSTM (`bidirectional=True`) to leverage
  context from both directions.

- **Mistake:** Applying dropout incorrectly to LSTM outputs. Applying dropout
  to the recurrent connections directly destroys the LSTM's memory.

  **Fix:** Use `dropout=p` in `nn.LSTM` (applies only to inter-layer outputs,
  not recurrent connections) or use LockedDropout (Merity et al., 2017).

- **Mistake:** Using LSTM for very long sequences (> 1000 tokens). BPTT
  through 1000 steps is slow and gradients still vanish despite LSTM gates.

  **Fix:** Use Transformers with self-attention for long sequences.
  Alternatively, truncate BPTT to 100-200 steps.

## 6. Knowledge Check

1. **Conceptual:** Show mathematically why the gradient of the LSTM cell
   state $c_t$ with respect to $c_{t-1}$ is the forget gate $f_t$ rather
   than a product of weight matrices. Why does this prevent vanishing gradients
   when $f_t \approx 1$?

2. **Conceptual:** Compare LSTM and GRU architecturally. How many gate
   matrices does each have? What role does the GRU's reset gate play, and
   how does it differ from the LSTM's forget gate?

3. **Coding challenge:** Train a character-level language model using a
   vanilla RNN and an LSTM on a short text corpus. Plot the training loss
   for both models over 100 epochs. Explain the difference in convergence
   in terms of the gradient flow analysis.

## References

1. Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179–211.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
3. Cho, K., van Merriënboer, B., Gulcehre, C., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP 2014*. arXiv:1406.1078.
4. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML 2013*. arXiv:1211.5063.
