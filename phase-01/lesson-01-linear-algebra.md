# Linear Algebra

Linear algebra is the mathematical backbone of every modern AI system. From the
pixel matrix fed into a vision model to the weight tensors inside a trillion-
parameter language model, every operation that a GPU accelerates is ultimately
a matrix multiplication. Understanding it deeply — not just mechanically — is
what separates engineers who can debug training failures from those who simply
run example notebooks and hope.

## 1. The Core Intuition (The "Why")

Before linear algebra became central to ML, the dominant paradigm was
feature engineering: a human expert would hand-craft scalar features (house
square footage, age, zip code) and feed them one at a time into statistical
estimators. This worked for dozens of features. It collapsed completely for
high-dimensional inputs.

An image at 224 × 224 × 3 resolution is 150,528 scalar values. A sentence
tokenised to 512 subword units, each embedded in 768 dimensions, is 393,216
numbers. Handling these one scalar at a time in Python loops is not just slow
— it is architecturally wrong. The structure of the problem lives in the
*relationships between* those numbers, not in the numbers individually.

Linear algebra gives you the language to express those relationships:
- A **vector** is a point in a high-dimensional space. The direction encodes
  semantic meaning; two embedding vectors pointing in the same direction mean
  similar things.
- A **matrix** is a function. Multiplying a vector by a matrix *transforms* it —
  rotating, scaling, or projecting it into a new space where a linear
  classifier can more easily separate classes.
- A **batch** of inputs as a matrix lets you apply that same function to
  thousands of examples simultaneously in one hardware instruction.

The key hardware insight is that GPUs are designed around matrix multiplication
(GEMM kernels). Every extra loop you write in Python instead of a matmul is
compute you are leaving on the table.

## 2. The Theoretical Underpinning

### Vectors and Linear Maps

Let a feature vector be $x \in \mathbb{R}^{d}$, where $d$ is the number of
features. A **linear layer** is a function parameterised by a weight matrix
$W \in \mathbb{R}^{m \times d}$ and a bias $b \in \mathbb{R}^{m}$:

$$y = Wx + b$$

Each row $W_{i,:}$ is a learned direction. The $i$-th output $y_i = W_{i,:}
\cdot x + b_i$ measures how much $x$ aligns with that direction. Stacking $m$
such rows lets the layer ask $m$ independent questions about $x$ simultaneously.

### The Dot Product as a Compatibility Score

The inner product of two vectors $u, v \in \mathbb{R}^{d}$ is

$$s = u^{\top}v = \sum_{j=1}^{d} u_{j}v_{j}$$

Geometrically, $s = \lVert u \rVert_2 \lVert v \rVert_2 \cos\theta$, where
$\theta$ is the angle between them. Large positive $s$ means the vectors point
in similar directions. This single formula is the computational primitive
behind attention scores, retrieval ranking, recommendation systems, and dense
passage ranking — essentially every "how relevant is $A$ to $B$?" question.

### Matrix Multiplication as Function Composition

For $A \in \mathbb{R}^{n \times p}$ and $B \in \mathbb{R}^{p \times m}$:

$$
(AB)_{ij} = \sum_{k=1}^{p} A_{ik}B_{kj}
$$

The shared index $k$ is the **latent dimension** — the space being projected
through. Composing two linear maps $A$ and $B$ is the same as asking: "first
project into a $p$-dimensional intermediate space, then project onward to
$m$ dimensions." This is exactly what stacking two layers in a network does.

### Cosine Similarity

In retrieval we care about *angle*, not magnitude, because a vector can be
scaled without changing what it represents. For $u, v \in \mathbb{R}^{d}$:

$$
\operatorname{cos}(u, v) =
\frac{u^{\top}v}{\lVert u \rVert_{2}\lVert v \rVert_{2}}
$$

where $\lVert u \rVert_{2} = \sqrt{\sum_{j=1}^{d} u_j^2}$ is the Euclidean
norm. After L2-normalisation, cosine similarity reduces to a plain dot product,
which is why modern vector databases (FAISS, Annoy, ScaNN) can exploit BLAS
kernels even when the user asks for "cosine search."

### Batched Linear Maps

In practice, inputs arrive as a batch of $n$ examples:
$X \in \mathbb{R}^{n \times d}$. Applying the same linear transformation to
every row simultaneously gives

$$
Y = XW^{\top} + \mathbf{1}b^{\top}
$$

where $Y \in \mathbb{R}^{n \times m}$ and $\mathbf{1} \in \mathbb{R}^{n}$
is the all-ones vector for broadcasting. A single GEMM call on a GPU now
handles all $n$ examples in parallel, which is why batch size is the most
important throughput lever in deep learning.

### Singular Value Decomposition (SVD)

Every matrix $A \in \mathbb{R}^{m \times n}$ can be factored as
$A = U \Sigma V^{\top}$, where $U$ and $V$ are orthogonal and $\Sigma$ is
diagonal with non-negative entries (singular values). The singular values
measure the "energy" in each direction. Keeping only the top $r$ singular
values gives the best rank-$r$ approximation — the theoretical basis for PCA,
embedding compression, and LoRA weight decomposition.

## 3. Implementation: From Scratch (Python + NumPy)

The goal is to make the mechanics tangible: a batched linear layer and a
cosine similarity search pipeline that you can trace line by line.

```python
from __future__ import annotations

from typing import Tuple

import numpy as np


def linear_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Apply a batched affine transformation Y = X W^T + 1 b^T.

    Args:
        inputs: Input matrix $X \in \mathbb{R}^{n \times d}$.
        weights: Weight matrix $W \in \mathbb{R}^{m \times d}$.
        bias: Bias vector $b \in \mathbb{R}^{m}$.

    Returns:
        Output matrix $Y \in \mathbb{R}^{n \times m}$.

    Raises:
        ValueError: If the array shapes are incompatible.
    """
    if inputs.ndim != 2 or weights.ndim != 2 or bias.ndim != 1:
        raise ValueError("Expected 2D inputs, 2D weights, and 1D bias.")
    if inputs.shape[1] != weights.shape[1]:
        raise ValueError("Input feature dim must match weight feature dim.")
    if weights.shape[0] != bias.shape[0]:
        raise ValueError("Weight rows must match the bias length.")

    # W stores m row-vectors; X @ W.T computes dot products between every
    # input row and every weight row — exactly what the equation says.
    projected = inputs @ weights.T
    # Broadcasting adds the same bias to every one of the n input rows.
    return projected + bias


def l2_normalize(vectors: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize each row to unit L2 norm.

    Args:
        vectors: Matrix $X \in \mathbb{R}^{n \times d}$.
        epsilon: Small constant that prevents division by zero.

    Returns:
        Row-normalized matrix where every row satisfies ||row||_2 = 1.

    Raises:
        ValueError: If the input is not a 2D array.
    """
    if vectors.ndim != 2:
        raise ValueError("Expected a 2D matrix of row vectors.")

    # keepdims=True lets us broadcast division across d columns.
    norms = np.sqrt(np.sum(vectors ** 2, axis=1, keepdims=True))
    return vectors / np.clip(norms, a_min=epsilon, a_max=None)


def cosine_similarity_matrix(
    queries: np.ndarray,
    documents: np.ndarray,
) -> np.ndarray:
    """Compute the full (q, r) cosine similarity matrix.

    After L2-normalisation, cosine similarity is a dot product, so the entire
    pairwise matrix is a single matmul — O(q * r * d) not O(q * r * d) loops.

    Args:
        queries: Query matrix with shape $(q, d)$.
        documents: Document matrix with shape $(r, d)$.

    Returns:
        Similarity matrix with shape $(q, r)$ in [-1, 1].

    Raises:
        ValueError: If the feature dimensions do not match.
    """
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("Query and document embedding dimensions must match.")

    q_unit = l2_normalize(queries)
    d_unit = l2_normalize(documents)
    # Each entry (i, j) is the cosine similarity between query i and doc j.
    return q_unit @ d_unit.T


def top_match(
    queries: np.ndarray,
    documents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the best document index and cosine score for each query.

    Args:
        queries: Query matrix with shape $(q, d)$.
        documents: Document matrix with shape $(r, d)$.

    Returns:
        Tuple of $(indices, scores)$ arrays, each with shape $(q,)$.
    """
    scores = cosine_similarity_matrix(queries, documents)
    indices = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(scores.shape[0]), indices]
    return indices, best_scores
```

## 4. Implementation: Production-Grade (PyTorch / SOTA Library)

At scale, `nn.Linear` fuses the matmul and bias-add, autograd tracks gradients,
and device placement lets the same code run on CPU, GPU, or TPU without change.

```python
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingProjector(nn.Module):
    """Learned affine projection for embedding spaces.

    A thin wrapper around nn.Linear that makes the intent explicit: we project
    embeddings from one space into another, optionally changing dimensionality.

    Args:
        input_dim: Dimensionality of the input embedding $d$.
        output_dim: Dimensionality of the output embedding $m$.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        # nn.Linear stores W in shape (output_dim, input_dim) and computes
        # x @ W^T + b — identical to our from-scratch version, but fused.
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the projection Y = X W^T + 1 b^T.

        Args:
            inputs: Tensor $X \in \mathbb{R}^{n \times d}$.

        Returns:
            Tensor $Y \in \mathbb{R}^{n \times m}$.
        """
        return self.linear(inputs)


def retrieve_top_match(
    query_vectors: torch.Tensor,
    document_vectors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieve the highest-cosine-similarity document for each query.

    Args:
        query_vectors: Query tensor of shape $(q, d)$.
        document_vectors: Document tensor of shape $(r, d)$.

    Returns:
        Tuple of $(indices, scores)$ tensors, each with shape $(q,)$.
    """
    # F.normalize computes L2 norm across dim=-1 and divides — GPU-accelerated.
    q_unit = F.normalize(query_vectors, p=2.0, dim=-1)
    d_unit = F.normalize(document_vectors, p=2.0, dim=-1)
    # One matmul produces all q × r cosine scores.
    similarity = q_unit @ d_unit.T
    scores, indices = torch.max(similarity, dim=1)
    return indices, scores


def demo_inference() -> Tuple[torch.Tensor, torch.Tensor]:
    """Demonstrate a tiny embedding projection + retrieval pipeline."""
    projector = EmbeddingProjector(input_dim=4, output_dim=3)
    queries = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    documents = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.1, 1.0, 0.0, 1.0]])

    with torch.inference_mode():
        pq = projector(queries)
        pd = projector(documents)
        return retrieve_top_match(pq, pd)
```

## 5. Production Pitfalls & Pro-Tips

- ❌ **The mistake:** Mixing row-vector and column-vector conventions in
  different parts of the codebase until a matmul silently broadcasts the wrong
  axis and produces wrong-shaped output with no error.
  ✅ **The fix:** Adopt a single convention — "batch first, features last" — and
  assert shapes at every function boundary. Use `einops` for readable
  rearrangements instead of ambiguous `reshape` calls.

- ❌ **The mistake:** Using raw dot products for retrieval when embedding norms
  vary across the corpus, causing high-magnitude vectors to dominate results
  purely due to scale.
  ✅ **The fix:** L2-normalise all embeddings before indexing. If magnitude is
  semantically meaningful (e.g., token frequency), use explicit inner-product
  indices with documented semantics.

- ❌ **The mistake:** Operating on Python lists or row-by-row loops when the
  same computation can be expressed as a batched matrix operation.
  ✅ **The fix:** Profile with `torch.profiler` or `py-spy`. Any hot loop over a
  first axis is almost always replaceable with a single matmul.

- ❌ **The mistake:** Ignoring dtype: mixing `float32` and `float16` tensors
  produces silent upcasting, unexpected memory use, or NaN under overflow.
  ✅ **The fix:** Set dtype explicitly at the model boundary. Use `autocast`
  for mixed-precision training with bfloat16 on Ampere+ hardware.

- ❌ **The mistake:** Forgetting that SVD is the correct lens for understanding
  why LoRA works. Low-rank updates are not an arbitrary trick — they exploit
  the empirical observation that weight updates have low intrinsic rank.
  ✅ **The fix:** When debugging LoRA or other low-rank methods, plot the
  singular value spectrum of the weight delta to verify the assumption holds.

## 6. Knowledge Check

1. **Conceptual:** Why does L2-normalisation allow cosine similarity to be
   computed as a plain dot product, and what information is discarded in that
   normalisation? When would you want to keep magnitude information?

2. **Conceptual:** A linear layer $y = Wx + b$ maps $\mathbb{R}^{d}$ to
   $\mathbb{R}^{m}$. If $m < d$, the layer is a dimensionality reduction. If
   $m > d$, it is an expansion. What does the SVD of $W$ tell you about what
   the layer "chooses" to preserve?

3. **Coding challenge:** In at most 25 lines of NumPy, compute the top-$k$
   cosine-similarity matches for a batch of queries against a corpus, returning
   indices and scores — without using any explicit Python loops over the query
   or corpus dimension.

## References

1. Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
2. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
3. Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM*, 18(11), 613–620. *(Origin of cosine similarity for information retrieval.)*
4. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. *(SVD perspective on weight updates.)*
