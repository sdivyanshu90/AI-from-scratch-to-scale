# Decision Trees

A decision tree partitions feature space into rectangular regions by
recursively splitting on individual feature thresholds, and assigns a
constant prediction to each region. Despite their simplicity, trees are the
workhorse inside ensembles (random forests, gradient boosting) that win most
structured-data competitions. Understanding how they are built — and why they
overfit — is essential before studying those ensembles.

## 1. The Core Intuition (The "Why")

Human decision-making naturally resembles a tree: "Is it raining? Yes → take
an umbrella. No → Is it cold? Yes → take a jacket." A decision tree formalises
this process as a greedy search over feature-threshold pairs.

Before decision trees, classification required hand-designed rules or linear
models. The breakthrough of Hunt et al. (1966) and later Quinlan (1986) was
to automate rule discovery through information theory: find the split that
maximally reduces _uncertainty_ about the label distribution.

The key insight is that information gain quantifies how much knowing the value
of a feature reduces your uncertainty about the class label. A pure split —
one where all examples fall into the same class — has zero remaining
uncertainty. Impure splits leave you still uncertain, which is wasteful.

## 2. The Theoretical Underpinning

### Shannon Entropy

For a discrete distribution $P = (p_1, p_2, \ldots, p_k)$ over $k$ classes,
the **entropy** is

$$H(P) = -\sum_{c=1}^{k} p_c \log_2 p_c$$

- $H = 0$ when one class probability is 1 (perfectly pure node — certain).
- $H = \log_2 k$ when all classes are equally likely (maximally impure).

Entropy is measured in bits. A single coin flip has $H = 1$ bit.

### Information Gain

A split on feature $j$ at threshold $t$ divides the training set $S$ into
two subsets $S_L$ (where $x_j \leq t$) and $S_R$ (where $x_j > t$). The
information gain of this split is

$$\text{IG}(S, j, t) = H(S) - \frac{|S_L|}{|S|}\,H(S_L) - \frac{|S_R|}{|S|}\,H(S_R)$$

We greedily choose the split $(j^*, t^*)$ that maximises $\text{IG}$.

### Gini Impurity

An alternative to entropy is the **Gini impurity**:

$$G(P) = 1 - \sum_{c=1}^{k} p_c^{2}$$

Gini has a probabilistic interpretation: it equals the probability that two
randomly drawn examples from the node have different classes. It is slightly
faster to compute than entropy (no logarithms) and is the default in
scikit-learn's `DecisionTreeClassifier`.

### CART vs ID3 vs C4.5

| Algorithm | Split criterion  | Handles continuous features | Pruning         |
| --------- | ---------------- | --------------------------- | --------------- |
| ID3       | Information gain | No (requires binning)       | None            |
| C4.5      | Gain ratio       | Yes                         | Error-based     |
| CART      | Gini impurity    | Yes                         | Cost-complexity |

CART (Breiman et al., 1984) always builds binary trees, while ID3/C4.5 can
create multi-way splits.

### Overfitting and Pruning

An unpruned tree can perfectly memorise training data by creating one leaf
per training example ($H(\text{leaf}) = 0$). This achieves zero training
error but generalises poorly.

Cost-complexity pruning adds a penalty proportional to the number of leaves:

$$R_\alpha(T) = R(T) + \alpha \cdot |T_{\text{leaves}}|$$

where $R(T)$ is the misclassification rate. Increasing $\alpha$ prunes
branches whose split provides less benefit than the complexity cost.

## 3. Implementation: From Scratch (Python + NumPy)

```python

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Node:
    """A decision tree node."""
    feature_index: Optional[int]   = None   # feature to split on
    threshold:     Optional[float] = None   # split threshold
    left:          Optional["Node"] = None
    right:         Optional["Node"] = None
    value:         Optional[int]   = None   # leaf class prediction


def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity: 1 - sum(p_c^2)."""
    if len(y) == 0:
        return 0.0
    probs = np.bincount(y) / len(y)
    return float(1.0 - np.sum(probs ** 2))


def best_split(X: np.ndarray, y: np.ndarray):
    """Find the feature and threshold that minimise weighted Gini impurity.

    Returns (best_feature, best_threshold) or (None, None) if no split
    reduces impurity.
    """
    n, d        = X.shape
    best_gain   = -np.inf
    best_feat   = None
    best_thresh = None
    parent_gini = gini_impurity(y)

    for feat in range(d):
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            mask_l = X[:, feat] <= t
            mask_r = ~mask_l
            if mask_l.sum() == 0 or mask_r.sum() == 0:
                continue
            # Weighted Gini of children
            gini_children = (
                (mask_l.sum() / n) * gini_impurity(y[mask_l]) +
                (mask_r.sum() / n) * gini_impurity(y[mask_r])
            )
            gain = parent_gini - gini_children
            if gain > best_gain:
                best_gain   = gain
                best_feat   = feat
                best_thresh = t

    return best_feat, best_thresh


def build_tree(
    X: np.ndarray, y: np.ndarray,
    max_depth: int, depth: int = 0,
) -> Node:
    """Recursively build a CART decision tree.

    Args:
        X: Features, shape (n, d).
        y: Integer class labels, shape (n,).
        max_depth: Maximum tree depth to prevent overfitting.
        depth: Current recursion depth.
    """
    # Leaf conditions: pure node, depth limit, or no valid split
    if depth >= max_depth or len(np.unique(y)) == 1:
        return Node(value=int(np.bincount(y).argmax()))

    feat, thresh = best_split(X, y)
    if feat is None:
        return Node(value=int(np.bincount(y).argmax()))

    mask = X[:, feat] <= thresh
    return Node(
        feature_index = feat,
        threshold     = thresh,
        left  = build_tree(X[mask],  y[mask],  max_depth, depth + 1),
        right = build_tree(X[~mask], y[~mask], max_depth, depth + 1),
    )


def predict_one(node: Node, x: np.ndarray) -> int:
    if node.value is not None:
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict_one(node.left, x)
    return predict_one(node.right, x)
```

## 4. Implementation: Production-Grade (scikit-learn)

```python

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score


def fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    ccp_alpha: float = 0.0,
) -> DecisionTreeClassifier:
    """Fit a CART decision tree with optional cost-complexity pruning.

    Args:
        X: Feature matrix of shape (n, d).
        y: Class labels.
        max_depth: Hard depth limit.
        min_samples_leaf: Minimum samples required at a leaf node.
            Prevents very small, potentially overfitted leaves.
        ccp_alpha: Cost-complexity pruning parameter alpha.
            Larger values prune more aggressively.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    tree = DecisionTreeClassifier(
        criterion        = "gini",
        max_depth        = max_depth,
        min_samples_leaf = min_samples_leaf,
        ccp_alpha        = ccp_alpha,
    )
    tree.fit(X, y)
    return tree


def cross_val_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = 5,
    cv: int = 5,
) -> tuple[float, float]:
    """Cross-validate a decision tree; return (mean_accuracy, std)."""
    tree   = DecisionTreeClassifier(max_depth=max_depth, criterion="gini")
    scores = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def show_rules(tree: DecisionTreeClassifier, feature_names: list[str]) -> str:
    """Return a human-readable text representation of the tree rules."""
    return export_text(tree, feature_names=feature_names)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Allowing a tree to grow to full depth without regularisation.
  A deep tree memorises training noise: features that are pure coincidences
  in training data become hard rules.

  **Fix:** Set `max_depth`, `min_samples_leaf`, and `ccp_alpha`. Use the
  cross-validated pruning path (`.cost_complexity_pruning_path()`).

- **Mistake:** Treating feature importance from a single tree as reliable.
  Feature importances are high-variance estimates; they change dramatically
  between bootstrapped samples.

  **Fix:** Use a random forest and average importances across hundreds of
  trees for stable estimates.

- **Mistake:** Using decision trees on high-cardinality categorical features
  without encoding. scikit-learn requires numeric input.

  **Fix:** Use `OrdinalEncoder` or target encoding for high-cardinality
  categories. Consider `CatBoost` which handles categoricals natively.

- **Mistake:** Assuming axis-aligned splits are always appropriate.
  Trees cannot represent diagonal decision boundaries without many splits,
  leading to very deep trees for linearly separable data.

  **Fix:** Recognise when the problem has linear structure; use logistic
  regression or kernel SVM there.

- **Mistake:** Not checking class imbalance. Information gain and Gini are
  biased toward the majority class in imbalanced datasets.

  **Fix:** Use `class_weight="balanced"` or oversample the minority class
  with SMOTE before fitting.

## 6. Knowledge Check

1. **Conceptual:** Entropy and Gini impurity both measure node impurity, but
   which is more sensitive to changes in class probability near 0 and 1?
   Sketch both functions for binary classification and explain the implication
   for split selection.

2. **Conceptual:** Why does greedy top-down tree building not guarantee a
   globally optimal tree? What computational property of the optimal search
   makes it intractable?

3. **Coding challenge:** Implement cost-complexity pruning from scratch:
   compute the weakest link for a trained tree (the internal node whose
   removal increases training error least per unit reduction in tree size)
   and remove it. Repeat until a single root node remains.

## References

1. Shannon, C. E. (1948). A mathematical theory of communication. _Bell System Technical Journal_, 27(3), 379–423.
2. Quinlan, J. R. (1986). Induction of decision trees. _Machine Learning_, 1(1), 81–106. _(ID3 algorithm.)_
3. Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). _Classification and Regression Trees_. Wadsworth. _(CART.)_
4. Quinlan, J. R. (1993). _C4.5: Programs for Machine Learning_. Morgan Kaufmann.
