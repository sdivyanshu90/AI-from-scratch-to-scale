# Regression

Regression is the oldest formal machine learning technique and the foundation
upon which almost every supervised learning method rests. Understanding it
deeply — its assumptions, its failure modes, its geometric interpretation —
gives you a lens for debugging every subsequent model you will ever build.

## 1. The Core Intuition (The "Why")

Carl Friedrich Gauss developed least-squares regression in 1809 to fit
orbital paths of celestial bodies from noisy measurements. Francis Galton
coined the term "regression to the mean" in 1886 studying heights of parents
and children. Both were solving the same problem: given noisy observations
of a system, find the best-fit relationship between inputs and outputs.

The core insight is that instead of trying to fit every data point exactly
(which would simply memorise noise), we look for the _simplest_ function that
explains the systematic pattern in the data. This tension between fitting the
training data and generalising to new data — the bias-variance tradeoff — is
the central problem in all of machine learning.

Linear regression assumes the relationship is linear. This is a strong
assumption. But it is also often approximately true locally, and the resulting
model is interpretable: you can read off exactly how much each feature
contributes to the prediction.

## 2. The Theoretical Underpinning

### Ordinary Least Squares (OLS)

Given training data $\{(x_i, y_i)\}_{i=1}^{n}$ with
$x_i \in \mathbb{R}^{d}$ and $y_i \in \mathbb{R}$, the linear model is

$$\hat{y}_i = w^{\top}x_i + b$$

The OLS objective minimises the sum of squared residuals:

$$L(w, b) = \sum_{i=1}^{n}(\hat{y}_i - y_i)^{2} = \|Xw - y\|^{2}_{2}$$

where $X \in \mathbb{R}^{n \times d}$ is the design matrix with rows $x_i$,
and $y \in \mathbb{R}^n$ is the target vector.

### The Normal Equations

Setting $\nabla_w L = 0$ and solving gives the closed-form solution:

$$w^* = (X^{\top}X)^{-1}X^{\top}y$$

Each term has a clear role: $X^{\top}X \in \mathbb{R}^{d \times d}$ is the
feature covariance matrix; $X^{\top}y \in \mathbb{R}^{d}$ is the
feature-target covariance vector. The inverse $(X^{\top}X)^{-1}$ is only
defined when $X^{\top}X$ is full rank — when the columns of $X$ are linearly
independent (no perfectly collinear features).

The Gauss-Markov theorem states that OLS gives the **Best Linear Unbiased
Estimator** (BLUE): among all linear unbiased estimators, OLS has minimum
variance. This is only true under specific assumptions: zero-mean errors,
homoscedasticity (constant error variance), and no correlation between errors.

### Ridge Regression (L2 Regularisation)

When features are correlated or $d > n$, $(X^{\top}X)$ is singular or
near-singular. Ridge regression adds an L2 penalty that makes the problem
well-conditioned:

$$L_{\text{ridge}}(w) = \|Xw - y\|^{2}_{2} + \lambda\|w\|^{2}_{2}$$

The closed-form solution becomes

$$w^*_{\text{ridge}} = (X^{\top}X + \lambda I)^{-1}X^{\top}y$$

The $\lambda I$ term shifts the eigenvalues of $X^{\top}X$ away from zero,
stabilising the inverse. Geometrically, ridge shrinks all weights toward
zero proportionally. Tikhonov (1963) generalised this regularisation idea,
which is why ridge regression is also called Tikhonov regularisation.

### LASSO (L1 Regularisation)

$$L_{\text{lasso}}(w) = \|Xw - y\|^{2}_{2} + \lambda\|w\|_{1}$$

L1 regularisation induces **sparsity**: many weights are driven exactly to
zero, performing implicit feature selection. The geometry is key: the L1
ball has corners at the coordinate axes; the optimum lands at a corner
(sparse solution) with much higher probability than L2, which has a smooth
ball with no corners.

## 3. Implementation: From Scratch (Python + NumPy)

```python

import numpy as np
from typing import Optional


class LinearRegression:
    """OLS and Ridge regression with closed-form and gradient descent solvers."""

    def __init__(self, ridge_lambda: float = 0.0) -> None:
        self.ridge_lambda = ridge_lambda
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit_normal_equations(self, X: np.ndarray, y: np.ndarray) -> None:
        """Solve w* = (X^T X + lambda I)^-1 X^T y directly.

        Args:
            X: Design matrix of shape (n, d).
            y: Target vector of shape (n,).
        """
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be (n, d) and y must be (n,).")
        n, d = X.shape
        # Augment X with a bias column of ones
        X_aug = np.column_stack([X, np.ones(n)])
        # Ridge penalty applied to weights but NOT bias (last column)
        reg = self.ridge_lambda * np.eye(d + 1)
        reg[-1, -1] = 0.0  # do not regularise bias
        # Normal equations: w* = (X_aug^T X_aug + reg)^{-1} X_aug^T y
        solution = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
        self.weights = solution[:-1]
        self.bias    = float(solution[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions y_hat = X w + b."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        return X @ self.weights + self.bias

    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE on a held-out dataset."""
        return float(np.mean((self.predict(X) - y) ** 2))
```

## 4. Implementation: Production-Grade (PyTorch / scikit-learn)

```python

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


def build_sklearn_ridge(alpha: float = 1.0) -> Pipeline:
    """Build a Ridge regression pipeline with standard scaling.

    Args:
        alpha: Regularisation strength (equivalent to lambda in the lesson).

    Returns:
        sklearn Pipeline: scale -> ridge.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha)),
    ])


class TorchLinearRegressor(nn.Module):
    """Linear regression model backed by PyTorch autograd.

    Using PyTorch allows gradient-based optimisation on large datasets
    where the normal equations are too slow (O(d^3) cost).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_torch_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_lambda: float = 0.01,
    lr: float = 1e-3,
    epochs: int = 500,
) -> TorchLinearRegressor:
    """Train a linear regressor with L2 regularisation using Adam."""
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    model     = TorchLinearRegressor(X.shape[1])
    # weight_decay implements L2 regularisation on the weight matrix
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=l2_lambda)
    loss_fn   = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    return model
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Fitting a regression model on raw features without scaling.
  Ridge and LASSO penalties are scale-sensitive: a feature measured in
  thousands has a very small coefficient that receives almost no regularisation.

  **Fix:** Always apply `StandardScaler` or `MinMaxScaler` before regularised
  regression.

- **Mistake:** Interpreting OLS coefficients causally when features are
  correlated. Multicollinearity inflates coefficient variance and makes
  individual coefficients uninterpretable.

  **Fix:** Check the condition number of $X^{\top}X$. Values above 1000
  indicate severe multicollinearity. Use Ridge or VIF analysis.

- **Mistake:** Using the normal equations for $d > 10{,}000$ features. The
  cost is $O(nd^2 + d^3)$: both terms are prohibitive at scale.

  **Fix:** Use gradient descent (SGD with L2 penalty) or sparse solvers
  (scipy's `lsqr` for large sparse systems).

- **Mistake:** Selecting $\lambda$ by training set performance rather than
  cross-validation. Training loss always decreases as $\lambda \to 0$.

  **Fix:** Use 5-fold or 10-fold cross-validation. `RidgeCV` does this
  efficiently with generalised cross-validation.

- **Mistake:** Forgetting to check residual diagnostics after fitting.
  Heteroscedasticity (non-constant variance) or non-linear patterns in
  residuals invalidate the Gauss-Markov assumptions.

  **Fix:** Plot residuals vs fitted values. A random horizontal band is good.
  Any pattern (funnel shape, curve) suggests model misspecification.

## 6. Knowledge Check

1. **Conceptual:** The Gauss-Markov theorem says OLS is BLUE. What are the
   specific assumptions required, and which are most commonly violated in
   practice? How does each violation affect predictions?

2. **Conceptual:** Explain geometrically why L1 regularisation produces sparse
   solutions but L2 does not. How does the shape of the constraint region
   differ between the two?

3. **Coding challenge:** Implement k-fold cross-validation for Ridge regression
   from scratch using NumPy. For $\lambda \in \{0.001, 0.01, 0.1, 1, 10\}$,
   report the mean and standard deviation of MSE across 5 folds.

## References

1. Gauss, C. F. (1809). _Theoria Motus Corporum Coelestium_. Perthes & Besser. _(First least-squares derivation.)_
2. Galton, F. (1886). Regression towards mediocrity in hereditary stature. _Journal of the Anthropological Institute_, 15, 246–263. _(Origin of "regression".)_
3. Tikhonov, A. N. (1963). Solution of incorrectly formulated problems and the regularization method. _Soviet Mathematics Doklady_, 4, 1035–1038.
4. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. _Technometrics_, 12(1), 55–67.
5. Tibshirani, R. (1996). Regression shrinkage and selection via the LASSO. _JRSS-B_, 58(1), 267–288.
