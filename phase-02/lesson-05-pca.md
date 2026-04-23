# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) finds the low-dimensional linear subspace
that best preserves the variance of a dataset. It is simultaneously a
dimensionality reduction technique, a noise-filtering method, a visualisation
tool, and a compression algorithm. Understanding PCA deeply — through the lens
of eigendecomposition and SVD — equips you to debug any method that operates
on high-dimensional representations.

## 1. The Core Intuition (The "Why")

Karl Pearson (1901) and Harold Hotelling (1933) independently developed PCA
to handle the following problem: you have $d$ correlated measurements, but
the _actual_ degrees of freedom in the system are much fewer. A camera pointed
at a human arm records 640×480 = 307,200 pixel values, but the arm has
roughly 7 degrees of freedom (joint angles). PCA recovers those underlying
degrees of freedom as a linear combination of the original measurements.

The geometric intuition: PCA finds the axis of greatest variance in the data
(first principal component), then the axis of greatest remaining variance
orthogonal to the first (second principal component), and so on. Each axis is
a direction in the original $d$-dimensional feature space that captures
independent variation.

## 2. The Theoretical Underpinning

### Covariance Matrix and Eigendecomposition

Given centred data $X \in \mathbb{R}^{n \times d}$ (zero mean), the
**empirical covariance matrix** is

$$\Sigma = \frac{1}{n-1} X^{\top}X \in \mathbb{R}^{d \times d}$$

$\Sigma_{ij}$ measures how much features $i$ and $j$ vary together.

Since $\Sigma$ is symmetric positive semi-definite, it has a real eigendecomposition:

$$\Sigma = V \Lambda V^{\top}$$

where the columns of $V \in \mathbb{R}^{d \times d}$ are the eigenvectors
(principal components), and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$
contains the eigenvalues in descending order. Each eigenvalue $\lambda_i$ is
the variance explained by the $i$-th principal component.

### SVD: The Numerically Stable Route

Rather than forming $\Sigma = X^{\top}X$ explicitly (which squares the
condition number), we use the **Singular Value Decomposition** directly on $X$:

$$X = U \Sigma_s V^{\top}$$

where $U \in \mathbb{R}^{n \times n}$, $\Sigma_s \in \mathbb{R}^{n \times d}$
is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$,
and $V \in \mathbb{R}^{d \times d}$ contains the right singular vectors.

The connection to PCA: the right singular vectors of $X$ are exactly the
eigenvectors of $X^{\top}X$ (up to a constant). The singular values satisfy

$$\lambda_i = \frac{\sigma_i^2}{n-1}$$

Numeric libraries (NumPy, LAPACK) always compute PCA via SVD, never by
explicitly forming the covariance matrix.

### Dimensionality Reduction

Project data onto the top $r < d$ principal components:

$$\tilde{X} = X V_r$$

where $V_r$ contains the first $r$ columns of $V$ (the $r$ eigenvectors with
largest eigenvalues). The reconstruction is

$$\hat{X} = \tilde{X} V_r^{\top}$$

The **explained variance ratio** of the $r$-dimensional approximation is

$$\text{EVR}(r) = \frac{\sum_{i=1}^{r} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

Choose $r$ to achieve a target EVR (e.g., 95%).

### The Eckart-Young Theorem

The PCA projection $\hat{X} = U_r \Sigma_{s,r} V_r^{\top}$ is the _best
rank-$r$ approximation_ of $X$ in the Frobenius norm:

$$\hat{X} = \arg\min_{\text{rank}(M) \leq r} \|X - M\|_F$$

This gives PCA a rigorous optimality guarantee: no other linear projection
to $r$ dimensions preserves more of the original data in a Frobenius sense.

## 3. Implementation: From Scratch (Python + NumPy)

```python

import numpy as np


class PCA:
    """PCA via SVD with explained variance diagnostics."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_:          np.ndarray | None = None   # (n_components, d)
        self.explained_variance_:  np.ndarray | None = None   # (n_components,)
        self.explained_variance_ratio_: np.ndarray | None = None
        self.mean_:                np.ndarray | None = None   # (d,)

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA on X.

        Args:
            X: Data matrix of shape (n, d).
        """
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_           # centre data

        # Full thin SVD: X_c = U S Vt
        # Only need Vt (right singular vectors = principal components)
        _, s, Vt = np.linalg.svd(X_c, full_matrices=False)

        # Eigenvalues of covariance matrix
        variances     = s ** 2 / (n - 1)
        total_var     = variances.sum()

        self.components_                = Vt[:self.n_components]
        self.explained_variance_        = variances[:self.n_components]
        self.explained_variance_ratio_  = (variances[:self.n_components]
                                           / total_var)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto the principal components.

        Args:
            X: Data matrix of shape (n, d).

        Returns:
            Projected data of shape (n, n_components).
        """
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct approximate data from projections."""
        return Z @ self.components_ + self.mean_
```

## 4. Implementation: Production-Grade (scikit-learn)

```python

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_pca_pipeline(n_components: int) -> Pipeline:
    """Build a scaling + PCA pipeline.

    StandardScaler centres data AND scales to unit variance.
    If features have different units (e.g., height in cm and weight in kg),
    scaling is essential before PCA; otherwise high-variance features
    dominate the principal components.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=42)),
    ])


def choose_n_components(X: np.ndarray, target_evr: float = 0.95) -> int:
    """Find the minimum number of components to explain target_evr of variance.

    Args:
        X: Feature matrix of shape (n, d).
        target_evr: Target explained variance ratio (e.g., 0.95 for 95%).

    Returns:
        Minimum number of components.
    """
    scaler = StandardScaler()
    pca    = PCA()
    pca.fit(scaler.fit_transform(X))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n          = int(np.searchsorted(cumulative, target_evr) + 1)
    return n


def large_scale_pca(
    X: np.ndarray,
    n_components: int,
) -> TruncatedSVD:
    """Use TruncatedSVD for large or sparse matrices.

    TruncatedSVD uses randomised SVD (Halko et al., 2011) which is O(n*k*log(k))
    rather than O(n*d^2). Note: TruncatedSVD does not centre the data
    (important for sparse matrices where centering destroys sparsity).
    For dense data, centre manually before calling this.

    Args:
        X: Data matrix of shape (n, d). Should be centred for dense data.
        n_components: Number of components.

    Returns:
        Fitted TruncatedSVD.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)
    return svd
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Applying PCA without scaling when features have different
  units or variance ranges. A feature with variance 1000 will dominate the
  first principal component regardless of its semantic importance.

  **Fix:** Always apply `StandardScaler` before PCA unless you have a
  specific reason to preserve scale differences.

- **Mistake:** Computing PCA by forming the covariance matrix $X^{\top}X$
  explicitly, then calling `np.linalg.eig`. This squares the condition
  number and loses numerical precision.

  **Fix:** Use `np.linalg.svd` directly on the centred data, or
  `sklearn.decomposition.PCA` which does this automatically.

- **Mistake:** Fitting PCA on the entire dataset including the test set.
  This leaks test distribution information into the principal components.

  **Fix:** Fit PCA only on training data. Transform test data using the
  training PCA (use sklearn's `Pipeline` to enforce this).

- **Mistake:** Choosing $n\_components$ by eyeballing the scree plot (eigenvalue
  vs component number). This is subjective and unreliable.

  **Fix:** Choose $r$ to achieve a target cumulative explained variance ratio
  (90-99% depending on application).

- **Mistake:** Using full PCA on a sparse matrix (e.g., TF-IDF features).
  Centering a sparse matrix makes it dense, causing memory explosion.

  **Fix:** Use `TruncatedSVD` (randomised SVD) which skips centering and
  operates on sparse matrices directly.

## 6. Knowledge Check

1. **Conceptual:** Why does PCA via SVD of the data matrix $X$ give the same
   principal components as eigendecomposition of the covariance matrix
   $\Sigma = X^{\top}X / (n-1)$? Show the algebraic relationship.

2. **Conceptual:** PCA finds the directions of maximum variance. Is this
   always the same as the directions most useful for a downstream classifier?
   Give an example where PCA might discard information that is important for
   classification.

3. **Coding challenge:** Implement PCA from scratch using SVD. Then implement
   the Kernel PCA kernel (using an RBF kernel matrix) to handle non-linear
   structure. Apply both to a toy dataset where standard PCA fails (e.g.,
   two concentric circles) and compare the projections.

## References

1. Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. _Philosophical Magazine_, 2(11), 559–572.
2. Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. _Journal of Educational Psychology_, 24(6), 417–441.
3. Jolliffe, I. T. (2002). _Principal Component Analysis_ (2nd ed.). Springer.
4. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. _SIAM Review_, 53(2), 217–288.
