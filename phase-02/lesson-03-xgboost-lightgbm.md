# XGBoost and LightGBM

Gradient boosted trees are the dominant algorithm for structured/tabular data
in production. They win the majority of Kaggle competitions on tabular
problems and power ranking, fraud detection, and pricing systems at every
major tech company. XGBoost and LightGBM are the two most widely used
implementations, both of which extend Friedman's (2001) gradient boosting
framework with algorithmic and systems-level innovations.

## 1. The Core Intuition (The "Why")

The breakthrough insight of boosting is that you do not need a single powerful
model — you can combine many _weak_ models (models that are only slightly
better than random) into a strong model, as long as each new model corrects
the mistakes of the ensemble so far.

Gradient boosting (Friedman 2001) reframed this as **functional gradient
descent**: the ensemble is a function in function space, and each new tree
is a step in the direction of steepest descent of the loss function. This
reframing is crucial because it allows any differentiable loss function
(not just squared error) and provides a clean theoretical grounding.

XGBoost (Chen & Guestrin, 2016) added a second-order Taylor expansion of the
loss, which improves convergence and allows deeper regularisation. LightGBM
(Ke et al., 2017) introduced histogram-based splitting and leaf-wise growth,
which dramatically speeds up training on large datasets.

## 2. The Theoretical Underpinning

### Functional Gradient Descent

Let $F_m(x)$ be the ensemble after $m$ trees. The $(m+1)$-th tree $h_{m+1}$
is fit to the **pseudo-residuals**: the negative gradient of the loss with
respect to the current prediction:

$$r_i = -\left[\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_m}$$

The new ensemble is $F_{m+1}(x) = F_m(x) + \eta \cdot h_{m+1}(x)$, where
$\eta$ is the learning rate. This is exactly gradient descent in the space
of functions.

### Second-Order Taylor Expansion (XGBoost)

XGBoost computes a second-order Taylor expansion of the loss around the
current prediction:

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^{n}\left[g_i f_m(x_i) + \tfrac{1}{2}h_i f_m(x_i)^2\right] + \Omega(f_m)$$

where $g_i = \partial \ell / \partial \hat{y}_i$ is the first-order gradient
(gradient statistic), $h_i = \partial^2 \ell / \partial \hat{y}_i^2$ is the
second-order gradient (Hessian statistic), and

$$\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

is a regularisation term penalising the number of leaves $T$ and the L2
norm of leaf weights $w_j$.

For each candidate split of a leaf into left/right children, the optimal
gain formula is:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

where $G_L, H_L$ are the sums of gradients and Hessians in the left child,
and analogously for the right child. This is the gain formula you will see
in every XGBoost implementation and paper.

### Histogram-Based Splitting (LightGBM)

Rather than evaluating every distinct feature value as a split threshold
(XGBoost's exact greedy algorithm), LightGBM bins each feature into at most
$b$ discrete buckets (default 255). Finding the best split then requires
only $O(b)$ comparisons per feature instead of $O(n)$, reducing the total
split search from $O(nd)$ to $O(bd)$.

The memory savings are also significant: storing feature values as `uint8`
indices instead of `float32` values reduces memory by 4×.

### Leaf-Wise vs Level-Wise Growth

LightGBM grows trees **leaf-wise**: at each step, split the leaf with the
highest gain, regardless of tree level. XGBoost grows **level-wise**: split
all leaves at the current depth. Leaf-wise growth achieves lower training
loss with fewer leaves, but can produce unbalanced trees that overfit on
small datasets. Limit `max_depth` or `num_leaves` to control this.

## 3. Implementation: From Scratch (Python + NumPy)

```python

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StumpNode:
    """A decision stump (1-split regression tree)."""
    feature: int
    threshold: float
    left_value:  float
    right_value: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.where(X[:, self.feature] <= self.threshold,
                       self.left_value, self.right_value)
        return out


def fit_stump(
    X: np.ndarray,
    residuals: np.ndarray,
    hessians: np.ndarray,
    l2_lambda: float = 1.0,
) -> StumpNode:
    """Fit a single-split stump using the XGBoost gain formula.

    Args:
        X:         Feature matrix (n, d).
        residuals: First-order gradients g_i.
        hessians:  Second-order gradients h_i.
        l2_lambda: L2 regularisation on leaf weights.

    Returns:
        Best stump found by exhaustive split search.
    """
    n, d = X.shape
    best_gain  = -np.inf
    best_stump = None

    G_total = residuals.sum()
    H_total = hessians.sum()

    for feat in range(d):
        order = np.argsort(X[:, feat])
        G_L, H_L = 0.0, 0.0
        for idx in order[:-1]:
            G_L += residuals[idx]
            H_L += hessians[idx]
            G_R = G_total - G_L
            H_R = H_total - H_L
            # XGBoost gain formula
            gain = 0.5 * (
                G_L**2 / (H_L + l2_lambda) +
                G_R**2 / (H_R + l2_lambda) -
                G_total**2 / (H_total + l2_lambda)
            )
            if gain > best_gain:
                best_gain  = gain
                threshold  = float(X[idx, feat])
                w_L = -G_L / (H_L + l2_lambda)
                w_R = -G_R / (H_R + l2_lambda)
                best_stump = StumpNode(feat, threshold, w_L, w_R)

    return best_stump


class GBMFromScratch:
    """Minimal gradient boosting for binary cross-entropy loss."""

    def __init__(self, n_rounds: int = 50, lr: float = 0.1) -> None:
        self.n_rounds = n_rounds
        self.lr       = lr
        self.stumps:   List[StumpNode] = []
        self.base_score: float = 0.0

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit gradient boosted stumps for binary classification."""
        # Base prediction: log-odds of mean label
        mean_y = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(mean_y / (1 - mean_y)))
        F = np.full(len(y), self.base_score)

        for _ in range(self.n_rounds):
            p   = self._sigmoid(F)
            g   = p - y              # first-order gradient
            h   = p * (1 - p)        # Hessian for log-loss
            stump = fit_stump(X, g, h)
            self.stumps.append(stump)
            F += self.lr * stump.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.base_score)
        for stump in self.stumps:
            F += self.lr * stump.predict(X)
        return self._sigmoid(F)
```

## 4. Implementation: Production-Grade (XGBoost / LightGBM)

```python

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    l2_lambda: float = 1.0,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with standard regularisation settings.

    Args:
        subsample: Fraction of rows sampled per tree (stochastic GBM).
        colsample_bytree: Fraction of columns sampled per tree.
        l2_lambda: L2 regularisation on leaf weights (reg_lambda in XGBoost).

    Returns:
        Fitted XGBClassifier.
    """
    model = xgb.XGBClassifier(
        n_estimators      = n_estimators,
        learning_rate     = learning_rate,
        max_depth         = max_depth,
        subsample         = subsample,
        colsample_bytree  = colsample_bytree,
        reg_lambda        = l2_lambda,
        eval_metric       = "logloss",
        early_stopping_rounds = 50,
        random_state      = 42,
        n_jobs            = -1,
    )
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with leaf-wise growth.

    Args:
        num_leaves: Maximum number of leaves per tree. Controls model
            complexity. Rule of thumb: num_leaves < 2^max_depth.

    Returns:
        Fitted LGBMClassifier.
    """
    model = lgb.LGBMClassifier(
        n_estimators    = n_estimators,
        learning_rate   = learning_rate,
        num_leaves      = num_leaves,
        subsample       = 0.8,
        colsample_bytree = 0.8,
        reg_lambda      = 1.0,
        n_jobs          = -1,
        random_state    = 42,
    )
    return model
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Training to convergence on training data without early stopping.
  Boosting never stops improving training loss. Without early stopping, you
  will always overfit.

  **Fix:** Hold out a validation set and use `early_stopping_rounds=50`.
  Stop when validation loss has not improved for 50 rounds.

- **Mistake:** Using too large a learning rate with too few trees. A high
  learning rate makes each tree's correction too aggressive; the ensemble
  oscillates rather than converges.

  **Fix:** Use `learning_rate=0.05` with `n_estimators=1000` rather than
  `lr=0.3` with `n_estimators=100`. More trees with lower lr almost always
  wins.

- **Mistake:** Forgetting that gradient boosting is sensitive to outliers in
  the target variable. Squared-error loss gives extreme targets high weight.

  **Fix:** Use `objective="reg:quantileerror"` (XGBoost) or Huber loss for
  robust regression.

- **Mistake:** Treating feature importances as the ground truth for feature
  selection. Split-count importance is biased toward high-cardinality features.

  **Fix:** Use SHAP values (`shap` library) for reliable, consistent feature
  attribution that is theoretically grounded in cooperative game theory.

- **Mistake:** Not accounting for class imbalance. Boosting in the default
  log-loss objective will over-weight the majority class.

  **Fix:** Set `scale_pos_weight = neg_count / pos_count` in XGBoost or
  `is_unbalance=True` in LightGBM.

## 6. Knowledge Check

1. **Conceptual:** In gradient boosting, the pseudo-residuals are the negative
   gradient of the loss. For squared-error loss, what are the pseudo-residuals?
   For log-loss (binary cross-entropy), what are they?

2. **Conceptual:** Explain why the XGBoost gain formula includes the Hessian
   $h_i$ in the denominator. What goes wrong if you only use the first-order
   gradient (as in vanilla gradient boosting)?

3. **Coding challenge:** Implement a simple gradient boosting regressor
   (squared-error loss) from scratch. Use depth-1 stumps as base learners.
   Track training and validation MSE versus the number of rounds and plot
   the result. At what round does overfitting begin?

## References

1. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. _Annals of Statistics_, 29(5), 1189–1232.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings of KDD 2016_. arXiv:1603.02754.
3. Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. _NeurIPS 2017_.
4. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. _NeurIPS 2017_. _(SHAP values.)_
