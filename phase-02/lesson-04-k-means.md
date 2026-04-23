# K-Means Clustering

K-means is the most widely deployed unsupervised learning algorithm in
production. It segments customers, compresses images, initialises neural
network weights, and builds quantisation codebooks for vector databases.
Its simplicity conceals a surprisingly rich mathematical structure that
explains both when it excels and when it silently fails.

## 1. The Core Intuition (The "Why")

Before k-means, clustering required either hierarchical methods (slow, $O(n^2)$
memory) or manual rule-based segmentation. Stuart Lloyd (1957, published 1982)
at Bell Labs developed what is now called Lloyd's algorithm while working on
pulse code modulation for audio compression: find the $k$ representative code
vectors that minimise the total distortion of quantised audio samples.

The key insight is a **fixed-point iteration**: given fixed cluster assignments,
the optimal centroid is the mean of the assigned points. Given fixed centroids,
the optimal assignment is the nearest centroid. These two steps reduce the
objective monotonically, guaranteeing convergence — but to a local minimum, not
a global one. This is the central limitation: k-means finds a locally optimal
partition, and the solution depends heavily on initialisation.

## 2. The Theoretical Underpinning

### The Objective Function

K-means minimises the **within-cluster sum of squares** (WCSS):

$$J = \sum_{k=1}^{K}\sum_{i \in C_k} \|x_i - \mu_k\|^2_2$$

where $C_k$ is the set of points assigned to cluster $k$ and $\mu_k$ is the
centroid of that cluster. This objective is non-convex; finding the global
minimum is NP-hard in general dimensions (Dasgupta, 2008).

### Lloyd's Algorithm

Lloyd's algorithm performs coordinate descent on $J$:

1. **Assignment step**: $c_i \leftarrow \arg\min_{k} \|x_i - \mu_k\|^2$
2. **Update step**: $\mu_k \leftarrow \frac{1}{|C_k|}\sum_{i \in C_k} x_i$

Each step can only decrease $J$:

- Assignment: for each point, we replace its cluster assignment with the
  nearest centroid, which can only decrease $\|x_i - \mu_{c_i}\|^2$.
- Update: the mean minimises squared Euclidean distance within a set.

This guarantees convergence to a local minimum, but the number of steps
can be exponential in the worst case (though is linear in practice).

### The Voronoi Diagram

The assignment step partitions space into **Voronoi cells**: region $V_k$ is
the set of all points closer to $\mu_k$ than to any other centroid.

$$V_k = \{x : \|x - \mu_k\|_2 \leq \|x - \mu_j\|_2 \text{ for all } j \neq k\}$$

The boundaries between Voronoi cells are hyperplanes equidistant between
pairs of centroids. This means k-means always produces **convex** clusters,
and cannot represent non-convex cluster shapes.

### K-Means++ Initialisation

Random initialisation of centroids can place multiple centroids in the same
cluster, leading to poor solutions. Arthur and Vassilvitskii (2007) proved
that the **k-means++** initialisation guarantees:

$$E[J(\text{solution})] \leq 8(\ln k + 2) \cdot J(\text{OPT})$$

The initialisation is:

1. Choose the first centroid uniformly at random.
2. For each subsequent centroid $k$, choose point $x$ with probability
   proportional to $D(x)^2$, where $D(x)$ is the distance from $x$ to the
   nearest already-chosen centroid.

This seeding spreads the initial centroids, making it much less likely that
two centroids start in the same true cluster.

### Choosing K: The Elbow Method and Silhouette Score

The **elbow method** plots WCSS vs $k$. As $k$ increases, WCSS decreases; the
"elbow" in the curve suggests the point of diminishing returns.

The **silhouette score** for point $i$ is

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

where $a_i$ is the mean distance to all points in the same cluster, and
$b_i$ is the mean distance to all points in the nearest other cluster.
$s_i \in [-1, 1]$; values near 1 indicate well-separated clusters.

## 3. Implementation: From Scratch (Python + NumPy)

```python

import numpy as np


def kmeans_plusplus_init(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """K-Means++ initialisation.

    Args:
        X:   Data matrix of shape (n, d).
        k:   Number of clusters.
        rng: NumPy random generator for reproducibility.

    Returns:
        Initial centroids of shape (k, d).
    """
    n = X.shape[0]
    # Choose first centroid uniformly at random
    first_idx   = rng.integers(0, n)
    centroids   = [X[first_idx]]

    for _ in range(k - 1):
        # Squared distance from each point to nearest centroid
        dists = np.min(
            [np.sum((X - c) ** 2, axis=1) for c in centroids],
            axis=0,
        )
        # Sample proportional to D(x)^2 — k-means++ rule
        probs = dists / dists.sum()
        idx   = rng.choice(n, p=probs)
        centroids.append(X[idx])

    return np.array(centroids)


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """K-Means clustering with k-means++ initialisation.

    Args:
        X:            Data matrix of shape (n, d).
        k:            Number of clusters.
        max_iter:     Maximum number of Lloyd iterations.
        tol:          Stop when centroid shift < tol.
        random_state: Seed for reproducibility.

    Returns:
        (labels, centroids, wcss) where labels has shape (n,),
        centroids has shape (k, d), and wcss is the objective value.
    """
    rng       = np.random.default_rng(random_state)
    centroids = kmeans_plusplus_init(X, k, rng)

    for iteration in range(max_iter):
        # Assignment step: assign each point to the nearest centroid
        dists  = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)  # shape (n,)

        # Update step: recompute centroids as cluster means
        new_centroids = np.array([
            X[labels == c].mean(axis=0) if (labels == c).any() else centroids[c]
            for c in range(k)
        ])

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    # Compute final WCSS
    wcss = float(sum(
        np.sum((X[labels == c] - centroids[c]) ** 2)
        for c in range(k)
    ))
    return labels, centroids, wcss
```

## 4. Implementation: Production-Grade (scikit-learn)

```python

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    random_state: int = 42,
) -> KMeans:
    """Fit K-Means with multiple restarts.

    Args:
        n_init: Number of times to run k-means with different initialisations.
                The run with the best WCSS is returned.

    Returns:
        Fitted KMeans instance.
    """
    km = KMeans(
        n_clusters   = k,
        init         = "k-means++",
        n_init       = n_init,
        max_iter     = 300,
        random_state = random_state,
    )
    km.fit(X)
    return km


def select_k_by_silhouette(
    X: np.ndarray,
    k_range: range,
) -> dict[int, float]:
    """Compute silhouette score for each k in k_range.

    Returns a dict mapping k to silhouette score. Choose the k with
    the highest score.
    """
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)
    scores    = {}
    for k in k_range:
        km     = KMeans(n_clusters=k, init="k-means++", n_init=5,
                        random_state=42)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            scores[k] = -1.0
        else:
            scores[k] = float(silhouette_score(X_scaled, labels,
                                               sample_size=10_000))
    return scores


def mini_batch_kmeans(
    X: np.ndarray,
    k: int,
    batch_size: int = 1024,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans for large-scale data.

    Processes mini-batches of size `batch_size` rather than the full
    dataset at each iteration, enabling clustering of datasets that
    do not fit in memory.
    """
    mb = MiniBatchKMeans(
        n_clusters   = k,
        batch_size   = batch_size,
        n_init       = 5,
        random_state = 42,
    )
    mb.fit(X)
    return mb
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Running k-means on raw, un-scaled features. Euclidean distance
  is dominated by features with large variance, making other features
  irrelevant.

  **Fix:** Always apply `StandardScaler` before k-means. For very high-
  dimensional data (text, images), first apply PCA to reduce dimensionality.

- **Mistake:** Running k-means once and using the result. K-means converges
  to a local minimum that depends on initialisation. A single run can produce
  a poor partition.

  **Fix:** Run with `n_init=10` (the scikit-learn default) and take the
  run with the lowest WCSS.

- **Mistake:** Treating the elbow method as definitive. The "elbow" is often
  not clearly visible, leading to subjective choices.

  **Fix:** Combine elbow plot with silhouette scores and domain knowledge.

- **Mistake:** Assuming spherical, equally-sized clusters. K-means enforces
  Voronoi partitions: it cannot handle elongated, non-convex, or very
  differently-sized clusters.

  **Fix:** For non-convex clusters, use DBSCAN or Gaussian Mixture Models.

- **Mistake:** Using standard k-means on 100M+ rows in production.
  Each iteration is $O(nkd)$; this is prohibitively slow at scale.

  **Fix:** Use `MiniBatchKMeans` for large datasets. It converges to near-
  optimal solutions in a fraction of the time.

## 6. Knowledge Check

1. **Conceptual:** Prove that each step of Lloyd's algorithm (assignment and
   update) cannot increase the WCSS objective. Use this to argue convergence.
   Why does convergence not guarantee a global minimum?

2. **Conceptual:** K-means uses Euclidean distance, which makes it equivalent
   to assuming clusters are spherical Gaussians with equal covariance. What
   would you use instead if you wanted to model clusters with different shapes
   and sizes?

3. **Coding challenge:** Implement the silhouette score from scratch using
   NumPy. Verify your implementation matches `sklearn.metrics.silhouette_score`
   on a small dataset. Then use it to compare k=2, 3, 4, 5 on the Iris dataset.

## References

1. Lloyd, S. P. (1982). Least squares quantization in PCM. _IEEE Transactions on Information Theory_, 28(2), 129–137. _(Originally a 1957 Bell Labs technical report.)_
2. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. _Proceedings of 5th Berkeley Symposium_, 1, 281–297.
3. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. _Proceedings of SODA 2007_, 1027–1035.
4. Dasgupta, S. (2008). The hardness of k-means clustering. _Technical Report CS2008-0916_, UCSD.
