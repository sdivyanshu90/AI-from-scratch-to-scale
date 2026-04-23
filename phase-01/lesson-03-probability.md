# Probability and Statistics

Probability is the formal language of uncertainty, and uncertainty is
everywhere in machine learning: the targets we train on are noisy, the
features we observe are incomplete, and the generalisations we make to unseen
data are inherently uncertain. Without probability theory we cannot reason
rigorously about any of these — and we build models that are confidently wrong.

## 1. The Core Intuition (The "Why")

Early AI used hard rules: "if this symptom, then this disease." Rule-based
expert systems worked in narrow, well-specified domains but collapsed in the
real world because evidence is rarely clean. A symptom is *suggestive*, not
conclusive. A word in a spam email might appear in legitimate email too.

The predecessor approach had no mechanism for updating beliefs as new evidence
arrived. You either had the rule or you did not.

Probability gives us three capabilities rule-based systems lack:
1. **Degrees of belief** — a model can say "70 % likely spam" rather than
   just "spam" or "not spam."
2. **Update rules** — Bayes' theorem is a formal mechanism for revising
   beliefs when new evidence arrives.
3. **Calibration** — a well-calibrated model's confidence scores match
   empirical frequencies, which is essential for any system that feeds its
   outputs into downstream decisions.

## 2. The Theoretical Underpinning

### Bayes' Theorem

Let $A$ and $B$ be events with $P(B) > 0$. Conditional probability is defined as

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

Rearranging the joint probability symmetrically gives **Bayes' theorem**:

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
$$

- $P(A)$: the **prior** — our belief about $A$ before observing $B$.
- $P(B \mid A)$: the **likelihood** — how probable the evidence is if $A$ is true.
- $P(B)$: the **marginal likelihood** — a normalisation constant.
- $P(A \mid B)$: the **posterior** — our updated belief after observing $B$.

This update rule underlies Bayesian inference, spam filtering, medical
diagnosis, and the fundamental framing of supervised learning: given data
$x$, what is the probability of class $c$?

### Gaussian Naive Bayes

For class $c$ and feature vector $x = (x_1, \ldots, x_d)$, Naive Bayes
assumes **conditional independence** of features:

$$
P(x \mid c) = \prod_{j=1}^{d} P(x_j \mid c)
$$

This assumption is almost never exactly true, but it is a surprisingly
effective approximation. Under a Gaussian model, each feature in class $c$ has
mean $\mu_{cj}$ and variance $\sigma^2_{cj}$:

$$
P(x_j \mid c) =
\frac{1}{\sqrt{2\pi\sigma_{cj}^{2}}}
\exp\!\left(-\frac{(x_j - \mu_{cj})^{2}}{2\sigma_{cj}^{2}}\right)
$$

### The Log-Sum-Exp Trick

Multiplying many probabilities underflows IEEE float64 at ~700 independent
features. Work in log space:

$$
\hat{c} = \arg\max_{c \in \mathcal{C}}
\left[
\log P(c) + \sum_{j=1}^{d} \log P(x_j \mid c)
\right]
$$

This is numerically safe because we only add numbers, never multiply them.

### KL Divergence — Why It Appears in Every Loss Function

The Kullback-Leibler divergence between distributions $P$ and $Q$ is

$$
D_{\mathrm{KL}}(P \,\Vert\, Q) =
\sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

It measures how many extra bits are needed to encode samples from $P$ using a
code designed for $Q$. Cross-entropy loss is $H(P, Q) = H(P) + D_{\mathrm{KL}}(P \Vert Q)$.
Minimising cross-entropy is equivalent to minimising KL divergence from the
true label distribution to the model's predicted distribution. This connection
explains why cross-entropy is the *right* loss function for classification
from an information-theoretic standpoint.

### Calibration

A model is **calibrated** if, among all inputs assigned probability $p$ to
class $c$, exactly fraction $p$ actually belong to class $c$. Modern deep
networks are systematically overconfident (Guo et al., 2017). This matters
because miscalibrated probabilities mislead downstream decisions.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianNaiveBayes:
    class_priors: np.ndarray    # shape (K,)
    class_means: np.ndarray     # shape (K, d)
    class_variances: np.ndarray # shape (K, d)
    classes: np.ndarray         # shape (K,)


def fit_gaussian_nb(
    features: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 1e-6,
) -> GaussianNaiveBayes:
    """Estimate class priors and per-class Gaussian parameters.

    The prior P(c) is estimated by relative frequency.
    epsilon is variance smoothing that prevents log(0).
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of rows.")

    classes = np.unique(labels)
    K, d = len(classes), features.shape[1]

    priors    = np.zeros(K)
    means     = np.zeros((K, d))
    variances = np.zeros((K, d))

    for k, c in enumerate(classes):
        mask           = labels == c
        data           = features[mask]
        priors[k]      = data.shape[0] / features.shape[0]
        means[k]       = np.mean(data, axis=0)
        variances[k]   = np.var(data, axis=0) + epsilon

    return GaussianNaiveBayes(priors, means, variances, classes)


def predict_gaussian_nb(
    model: GaussianNaiveBayes,
    features: np.ndarray,
) -> np.ndarray:
    """Predict class labels using log-posterior scoring."""
    log_posts = []

    for k in range(len(model.classes)):
        # log normalisation constant: -0.5 * log(2 pi sigma^2)
        log_norm = -0.5 * np.log(2.0 * np.pi * model.class_variances[k])
        # log exponent: -0.5 * (x - mu)^2 / sigma^2
        log_exp  = -0.5 * ((features - model.class_means[k]) ** 2) \
                   / model.class_variances[k]
        log_post = np.log(model.class_priors[k]) + \
                   np.sum(log_norm + log_exp, axis=1)
        log_posts.append(log_post)

    return model.classes[np.argmax(np.stack(log_posts, axis=1), axis=1)]
```

## 4. Implementation: Production-Grade (PyTorch / SOTA Library)

```python
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class TorchGaussianNB:
    class_priors: torch.Tensor     # (K,)
    class_means: torch.Tensor      # (K, d)
    class_variances: torch.Tensor  # (K, d)
    classes: torch.Tensor          # (K,)


def fit_torch_gnb(
    features: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 1e-6,
) -> TorchGaussianNB:
    """Fit Gaussian Naive Bayes with fully vectorised tensor ops."""
    classes = torch.unique(labels, sorted=True)
    priors, means, variances = [], [], []

    for c in classes:
        data = features[labels == c]
        priors.append(data.shape[0] / features.shape[0])
        means.append(data.mean(dim=0))
        variances.append(data.var(dim=0, unbiased=False) + epsilon)

    return TorchGaussianNB(
        torch.tensor(priors),
        torch.stack(means),
        torch.stack(variances),
        classes,
    )


def predict_torch_gnb(
    model: TorchGaussianNB,
    features: torch.Tensor,
) -> torch.Tensor:
    """Vectorised log-posterior prediction for all classes at once."""
    x   = features[:, None, :]                  # (n, 1, d)
    mu  = model.class_means[None, :, :]         # (1, K, d)
    var = model.class_variances[None, :, :]     # (1, K, d)

    # Sum log Gaussian likelihoods over features, add log prior
    log_post = (
        -0.5 * torch.log(2.0 * torch.pi * var)
        - 0.5 * ((x - mu) ** 2) / var
    ).sum(dim=-1) + torch.log(model.class_priors)[None, :]  # (n, K)

    return model.classes[torch.argmax(log_post, dim=1)]
```

## 5. Production Pitfalls & Pro-Tips

- ❌ **The mistake:** Multiplying probabilities in floating-point space for
  sequences longer than ~100 features, causing numerical underflow to zero.
  ✅ **The fix:** Work entirely in log space. Replace products with sums and
  use the log-sum-exp trick for the final normalisation.

- ❌ **The mistake:** Treating the conditional independence assumption in Naive
  Bayes as if it were actually true when presenting model probabilities.
  ✅ **The fix:** Use NB as a baseline classifier, not as a calibrated
  probability estimator. Apply Platt scaling or isotonic regression post-hoc
  if you need calibrated probabilities.

- ❌ **The mistake:** Shipping raw softmax scores as probabilities without
  checking calibration. Modern neural networks are systematically overconfident.
  ✅ **The fix:** Evaluate with Expected Calibration Error (ECE) and reliability
  diagrams. Apply temperature scaling if ECE is high.

- ❌ **The mistake:** Using accuracy as the only metric when class distributions
  are skewed. A model predicting the majority class always achieves high accuracy
  without learning anything useful.
  ✅ **The fix:** Report precision, recall, F1, and area under the
  precision-recall curve for imbalanced problems.

- ❌ **The mistake:** Confusing cross-entropy minimisation with calibration.
  Optimising cross-entropy produces a good classifier, but not a calibrated one.
  ✅ **The fix:** Evaluate these as separate objectives and apply post-hoc
  calibration when calibrated probabilities are required.

## 6. Knowledge Check

1. **Conceptual:** Naive Bayes often works better than its independence
   assumption deserves. Explain why a model with wrong assumptions can still
   make correct predictions, and describe conditions under which the
   independence assumption is most harmful.

2. **Conceptual:** Show algebraically that minimising cross-entropy loss
   $H(y, \hat{p}) = -\sum_c y_c \log \hat{p}_c$ is equivalent to minimising
   KL divergence from the true label distribution to the model's predicted
   distribution.

3. **Coding challenge:** Implement a function that takes a numpy array of
   predicted probabilities and ground-truth binary labels, and returns the
   Expected Calibration Error (ECE) using 10 equally-spaced bins.

## References

1. Bayes, T. (1763). An essay towards solving a problem in the doctrine of chances. *Philosophical Transactions of the Royal Society*, 53, 370–418.
2. Laplace, P.-S. (1812). *Théorie analytique des probabilités*. Courcier.
3. Mitchell, T. (1997). *Machine Learning*. McGraw-Hill. *(Chapter 6: Bayesian Learning.)*
4. Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. *Annals of Mathematical Statistics*, 22(1), 79–86.
5. Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*.
