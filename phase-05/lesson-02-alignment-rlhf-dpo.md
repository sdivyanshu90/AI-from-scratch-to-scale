# Alignment: RLHF and Direct Preference Optimisation (DPO)

Pre-trained language models are trained to predict the next token, not to
be helpful, harmless, or honest. Alignment training shapes model behaviour
to match human values and task requirements. Reinforcement Learning from
Human Feedback (RLHF) was the technique behind ChatGPT and InstructGPT.
Direct Preference Optimisation (DPO) simplifies RLHF by eliminating the
explicit reward model and solving the alignment problem analytically.

## 1. The Core Intuition (The "Why")

A pre-trained LLM is a probability distribution over tokens that reflects
the distribution of internet text. Internet text contains harmful content,
factual errors, and responses that prioritise engagement over accuracy.
Simply asking the model to be helpful in a system prompt is insufficient —
the weights encode the full training distribution.

Christiano et al. (2017) introduced the RLHF framework: collect human
preferences over model outputs, train a reward model on these preferences,
then use RL to optimise the LM's policy against the reward model. Ouyang
et al. (2022) applied this to InstructGPT (GPT-3), showing that a 1.3B
RLHF model was preferred over the 175B GPT-3 base model by human raters.

Rafailov et al. (2023) observed that the RLHF objective has a closed-form
solution: the optimal policy can be expressed directly in terms of the data
without ever training an explicit reward model. This gave us DPO.

## 2. The Theoretical Underpinning

### The RLHF Objective

RLHF optimises the LM policy $\pi_\theta$ to maximise the expected reward
while staying close to the reference (SFT) model via a KL penalty:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[R_\phi(x, y)\right] - \beta \, \text{KL}\!\left[\pi_\theta(y|x) \,\|\, \pi_\text{ref}(y|x)\right]$$

- $R_\phi$: reward model trained on human preferences
- $\beta$: KL penalty coefficient (larger = stay closer to reference model)
- $\pi_\text{ref}$: the supervised fine-tuned (SFT) model, frozen

The KL term prevents the policy from "reward hacking" — finding outputs
that score high on $R_\phi$ but are incoherent or degenerate.

### Bradley-Terry Preference Model

The reward model is trained using the **Bradley-Terry model** for pairwise
preferences. Given a prompt $x$ and two responses $y_w$ (preferred) and
$y_l$ (rejected):

$$P(y_w \succ y_l \mid x) = \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)$$

The reward model is trained to maximise:

$$\mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\!\left(R_\phi(x, y_w) - R_\phi(x, y_l)\right)\right]$$

This is a binary cross-entropy loss: the model should score the preferred
response higher than the rejected response.

### PPO: Policy Optimisation

With $R_\phi$ trained, RLHF uses **Proximal Policy Optimisation (PPO)**
to optimise $\pi_\theta$. The full reward signal combines the reward model
score with the KL penalty:

$$r(x, y) = R_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$$

PPO clips the policy update to prevent large steps:

$$\mathcal{L}_\text{PPO}(\theta) = \mathbb{E}\!\left[\min\!\left(\rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ is
the probability ratio and $A_t$ is the advantage estimate.

### DPO: Closed-Form Solution

Rafailov et al. (2023) showed that the optimal policy for the RLHF
objective is:

$$\pi^*(y|x) = \frac{\pi_\text{ref}(y|x) \exp\!\left(\frac{1}{\beta} R^*(x, y)\right)}{Z(x)}$$

where $Z(x)$ is a normalisation constant. Rearranging, the implicit reward
of any policy $\pi_\theta$ relative to $\pi_\text{ref}$ is:

$$R(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

Substituting this into the Bradley-Terry objective and noting that $Z(x)$
cancels in the difference, the DPO loss is:

$$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

DPO directly optimises the policy using preference pairs — no explicit
reward model, no RL loop, no advantage estimation.

## 3. Implementation: From Scratch (Python + PyTorch)

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_logps_w:    torch.Tensor,
    policy_logps_l:    torch.Tensor,
    reference_logps_w: torch.Tensor,
    reference_logps_l: torch.Tensor,
    beta:              float = 0.1,
) -> torch.Tensor:
    """Compute the DPO loss for a batch of preference pairs.

    Args:
        policy_logps_w:    Log-probabilities of preferred responses under pi_theta.
                           Shape: (batch_size,)
        policy_logps_l:    Log-probabilities of rejected responses under pi_theta.
        reference_logps_w: Log-probabilities of preferred responses under pi_ref.
        reference_logps_l: Log-probabilities of rejected responses under pi_ref.
        beta:              KL penalty coefficient.

    Returns:
        Scalar DPO loss (mean over batch).
    """
    # Log-ratio: log(pi_theta / pi_ref) for each response
    log_ratio_w = policy_logps_w - reference_logps_w   # Preferred
    log_ratio_l = policy_logps_l - reference_logps_l   # Rejected

    # DPO objective: prefer responses where policy improves more over reference
    # than for rejected responses
    logits = beta * (log_ratio_w - log_ratio_l)

    # Binary cross-entropy: preferred should score higher than rejected
    loss = -F.logsigmoid(logits).mean()
    return loss


def compute_sequence_logprobs(
    model,
    input_ids:      torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Compute the per-token log-probs of a response sequence.

    Only computes log-probs for the response tokens (not the prompt),
    then averages over response tokens.

    Args:
        model:          Causal LM model.
        input_ids:      (batch, seq_len) token IDs of prompt + response.
        response_start: Index of the first response token.

    Returns:
        (batch_size,) mean log-prob of the response.
    """
    with torch.no_grad():
        logits = model(input_ids).logits   # (batch, seq_len, vocab)

    # Shift: predict token t+1 from hidden state t
    shift_logits = logits[:, response_start-1:-1, :]   # (batch, resp_len, vocab)
    shift_labels = input_ids[:, response_start:]        # (batch, resp_len)

    # Per-token log-probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.mean(dim=-1)   # Mean over response tokens
```

## 4. Implementation: Production-Grade (TRL DPOTrainer)

```python
from __future__ import annotations

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


def train_dpo(
    model_name:    str   = "mistralai/Mistral-7B-Instruct-v0.2",
    dataset_name:  str   = "trl-lib/ultrafeedback_binarized",
    output_dir:    str   = "./dpo-output",
    beta:          float = 0.1,
    learning_rate: float = 5e-7,
    num_epochs:    int   = 1,
) -> None:
    """Fine-tune a model using DPO with LoRA adapters.

    The dataset must have columns: prompt, chosen, rejected.
    TRL's DPOTrainer handles log-prob computation for both the policy
    and frozen reference model, and computes the DPO loss.

    Args:
        model_name:    Base model (should be SFT-fine-tuned first).
        dataset_name:  HuggingFace preference dataset.
        output_dir:    Where to save the DPO-trained adapter.
        beta:          KL penalty coefficient (0.01 = permissive, 0.5 = conservative).
        learning_rate: Learning rate (DPO needs very small LR, typically 1e-7 to 1e-6).
        num_epochs:    Number of training epochs.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load preference dataset
    dataset = load_dataset(dataset_name, split="train")

    # LoRA config for memory-efficient DPO
    peft_config = LoraConfig(
        r              = 16,
        lora_alpha     = 32,
        target_modules = "all-linear",
        lora_dropout   = 0.05,
    )

    training_args = DPOConfig(
        output_dir          = output_dir,
        beta                = beta,
        learning_rate       = learning_rate,
        num_train_epochs    = num_epochs,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        bf16                = True,
        logging_steps       = 50,
        save_strategy       = "epoch",
        remove_unused_columns = False,
    )

    trainer = DPOTrainer(
        model      = model,
        args       = training_args,
        train_dataset = dataset,
        tokenizer  = tokenizer,
        peft_config = peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Skipping the SFT stage and applying DPO/RLHF directly to
  a base model. Preference learning requires the model to already produce
  coherent responses; the base model's distribution is too wide.

  **Fix:** Always SFT on a high-quality instruction-following dataset
  first, then align with DPO or RLHF.

- **Mistake:** Setting $\beta$ too low (e.g., 0.01) in DPO. With a very
  small KL penalty, the model diverges from the reference distribution and
  produces repetitive or degenerate outputs.

  **Fix:** Start with $\beta=0.1$. Measure KL divergence from reference
  during training and ensure it stays below 5–10 nats.

- **Mistake:** Using a very high learning rate for DPO. DPO makes small
  but precise adjustments to the log-ratio; large updates destroy the
  calibration.

  **Fix:** Use learning rates in the range $10^{-7}$ to $5 \times 10^{-6}$.
  Monitor the reward margin (mean $\log(\pi_\theta(y_w)/\pi_\theta(y_l))$)
  rather than just loss.

- **Mistake:** Treating RLHF and DPO as interchangeable. DPO is simpler
  and more stable but requires offline preference data. RLHF with PPO can
  use online feedback (model generates, human rates, policy updates).

  **Fix:** Use DPO for offline preference datasets. Consider online DPO
  or PPO for tasks where you can generate new preference pairs during training.

- **Mistake:** Evaluating alignment by loss alone. DPO loss can decrease
  while helpfulness and harmlessness diverge.

  **Fix:** Evaluate with MT-Bench, AlpacaEval, or a held-out human preference
  test set. Track both reward margin and win rate against a reference model.

## 6. Knowledge Check

1. **Conceptual:** Walk through the DPO derivation. What mathematical
   property allows the normalisation constant $Z(x)$ to cancel out?
   How does this elimination of $Z(x)$ make DPO tractable compared to RLHF?

2. **Conceptual:** The KL penalty in RLHF prevents "reward hacking."
   What is reward hacking in this context? Give a concrete example of
   what a reward-hacking policy might do on a writing task.

3. **Coding challenge:** Implement the DPO loss from scratch and verify it
   by constructing a simple case: a vocabulary of 2 tokens, a policy that
   assigns [0.8, 0.2] to preferred/rejected, a reference that assigns
   [0.5, 0.5]. Compute the loss manually and verify your implementation
   matches.

## References

1. Christiano, P., Leike, J., Brown, T. B., et al. (2017). Deep reinforcement learning from human preferences. *NeurIPS 2017*. arXiv:1706.03741.
2. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*. arXiv:2203.02155. *(InstructGPT / RLHF.)*
3. Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS 2023*. arXiv:2305.18290.
4. Schulman, J., Wolski, F., Dhariwal, P., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
5. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39(3/4), 324–345.
