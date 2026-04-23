# Parameter-Efficient Fine-Tuning: LoRA, QLoRA, and DoRA

Full fine-tuning of large language models updates every parameter in the
model — often billions of weights — requiring enormous GPU memory and storage.
Parameter-Efficient Fine-Tuning (PEFT) methods adapt a pre-trained model to
a new task by updating only a tiny fraction of parameters, achieving
competitive performance at a fraction of the cost. LoRA, QLoRA, and DoRA
are the most widely adopted PEFT techniques in modern LLM workflows.

## 1. The Core Intuition (The "Why")

Aghajanyan et al. (2021) showed that pre-trained language models have a low
**intrinsic dimensionality**: there exists a random low-dimensional subspace
of the parameter space such that fine-tuning within this subspace achieves
near-full fine-tuning performance. For GPT-2, the intrinsic dimension of the
MRPC task is only ~200 — meaning you can effectively fine-tune a 117M-parameter
model by moving within a 200-dimensional subspace.

Hu et al. (2021) operationalised this insight with **LoRA** (Low-Rank Adaptation):
instead of updating a weight matrix $W \in \mathbb{R}^{d \times k}$ directly,
LoRA adds a low-rank perturbation $\Delta W = BA$ where
$B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and
$r \ll \min(d, k)$.

This reduces the number of trainable parameters from $d \times k$ to
$r(d + k)$, which for $r=8$, $d=k=4096$ is a $256\times$ reduction.

## 2. The Theoretical Underpinning

### LoRA: Low-Rank Adaptation

The full weight matrix $W_0$ is frozen. The adapted output is:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

- $W_0$: frozen pre-trained weight matrix
- $A$: randomly initialised (Kaiming uniform), trained
- $B$: zero-initialised, trained — ensures $\Delta W = 0$ at training start
- $\alpha$: scaling factor (typically set to $r$ so $\alpha/r = 1$ initially)
- $r$: rank hyperparameter (2–64 in practice)

At inference time, the adapter can be merged: $W = W_0 + \frac{\alpha}{r} BA$,
adding zero latency overhead.

**Which matrices to adapt?** Hu et al. applied LoRA to the query and value
projection matrices $W_Q$ and $W_V$ in each attention layer. Later work
(Dettmers et al., 2023) found that also adapting $W_K$, $W_O$, and the
feed-forward matrices improves performance.

### QLoRA: Quantised LoRA

Dettmers et al. (2023) showed that the frozen base model weights can be
quantised to **4-bit NormalFloat (NF4)** without losing adaptation quality,
enabling fine-tuning of 65B-parameter models on a single 48GB GPU:

**NF4 quantisation:** Map each block of $b$ weights to 4-bit values using
the quantile function of the standard normal distribution. This is optimal
for normally distributed weights (which most LLM weights are):

$$q_i = \Phi^{-1}\!\left(\frac{i + 0.5}{2^k}\right), \quad i = 0, 1, \ldots, 2^k - 1$$

where $k = 4$ gives 16 quantisation levels placed at the quantiles of
$\mathcal{N}(0, 1)$.

QLoRA additionally uses **double quantisation**: quantise the per-block
quantisation constants themselves, saving ~0.35 bits/parameter.

**Paged Optimiser**: QLoRA uses NVIDIA's unified memory to page optimiser
states from GPU to CPU RAM when the GPU is full, preventing OOM crashes
during gradient spikes.

### DoRA: Decompose Weight into Magnitude and Direction

Liu et al. (2024) observed that LoRA's linear interpolation adapts
magnitude and direction simultaneously, which may not reflect how humans
learn (we change direction more than magnitude). DoRA decomposes:

$$W = m \cdot \frac{V}{\lVert V \rVert_c}$$

where $m = \lVert W_0 \rVert_c$ is the column-wise magnitude vector and
$V / \lVert V \rVert_c$ is the unit-normalised direction matrix. DoRA
adapts the direction $V$ using LoRA while keeping the magnitude $m$ as
a separate trainable scalar:

$$W' = (m + \Delta m) \cdot \frac{(V_0 + BA)}{\lVert V_0 + BA \rVert_c}$$

DoRA matches or exceeds LoRA performance while using slightly more parameters
(one extra scalar per column).

## 3. Implementation: From Scratch (Python + PyTorch)

```python
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A linear layer with a LoRA low-rank adapter.

    Freezes the base weight and trains only the A and B matrices.

    Args:
        base_layer: Pre-trained nn.Linear module.
        r:          LoRA rank.
        alpha:      LoRA scaling factor (typically equals r).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r:          int   = 8,
        alpha:      float = 8.0,
    ) -> None:
        super().__init__()
        d_out, d_in = base_layer.weight.shape
        self.base   = base_layer
        self.r      = r
        self.scale  = alpha / r

        # Freeze base layer
        for p in self.base.parameters():
            p.requires_grad = False

        # A: Kaiming uniform init (same as standard nn.Linear weight init)
        self.lora_A = nn.Linear(d_in, r,    bias=False)
        # B: Zero init ensures delta=0 at training start
        self.lora_B = nn.Linear(r,    d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen) + scaled low-rank update
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base layer for zero-latency inference."""
        delta = self.scale * (self.lora_B.weight @ self.lora_A.weight)
        merged_weight = self.base.weight + delta
        out = nn.Linear(self.base.in_features, self.base.out_features,
                         bias=self.base.bias is not None)
        out.weight = nn.Parameter(merged_weight)
        if self.base.bias is not None:
            out.bias = nn.Parameter(self.base.bias.clone())
        return out


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 8.0) -> nn.Module:
    """Replace all nn.Linear layers with LoRALinear equivalents."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            inject_lora(module, r=r, alpha=alpha)
    return model
```

## 4. Implementation: Production-Grade (PEFT + bitsandbytes)

```python
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import datasets


def load_model_qlora(
    model_name: str   = "meta-llama/Meta-Llama-3-8B",
    r:          int   = 16,
    alpha:      float = 32.0,
    dropout:    float = 0.05,
) -> tuple:
    """Load a model in 4-bit (QLoRA) with LoRA adapters ready for fine-tuning.

    Uses NF4 quantisation for the base model weights (frozen),
    and trainable LoRA adapters on all linear layers.

    Args:
        model_name: HuggingFace model ID.
        r:          LoRA rank.
        alpha:      LoRA scaling (common: alpha = 2*r).
        dropout:    LoRA dropout for regularisation.

    Returns:
        (model, tokenizer) ready for SFTTrainer.
    """
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",           # NormalFloat4
        bnb_4bit_use_double_quant = True,             # Double quantisation
        bnb_4bit_compute_dtype    = torch.bfloat16,  # Compute in BF16, store in NF4
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
    )
    # Prepare for k-bit training: cast LayerNorm to FP32, enable gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r             = r,
        lora_alpha    = alpha,
        lora_dropout  = dropout,
        target_modules = "all-linear",   # Adapt all linear layers
        bias          = "none",
        task_type     = "CAUSAL_LM",
    )
    model     = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()
    return model, tokenizer


def train_qlora(
    model,
    tokenizer,
    dataset_name: str = "tatsu-lab/alpaca",
    output_dir:   str = "./lora-output",
) -> None:
    """Fine-tune a QLoRA model using SFTTrainer.

    SFTTrainer handles supervised fine-tuning with proper loss masking
    (only compute loss on assistant tokens, not prompt tokens).

    Args:
        model:        QLoRA-prepared model.
        tokenizer:    Corresponding tokenizer.
        dataset_name: HuggingFace dataset for instruction tuning.
        output_dir:   Directory to save LoRA adapter weights.
    """
    dataset = datasets.load_dataset(dataset_name, split="train")
    training_args = TrainingArguments(
        output_dir          = output_dir,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps        = 100,
        num_train_epochs    = 3,
        learning_rate       = 2e-4,
        fp16                = False,
        bf16                = True,
        logging_steps       = 50,
        save_strategy       = "epoch",
        optim               = "paged_adamw_8bit",   # Paged optimiser for QLoRA
    )
    trainer = SFTTrainer(
        model     = model,
        args      = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Setting rank $r$ too high (e.g., r=256) without a good
  reason. Higher rank increases parameters and can overfit on small datasets.

  **Fix:** Start with r=8 or r=16. Use r=64 only for tasks requiring
  major knowledge injection. Always validate on a held-out set.

- **Mistake:** Not merging LoRA weights before deployment. Running separate
  LoRA forward passes adds computation overhead and complicates serving.

  **Fix:** Call `model.merge_and_unload()` after training to fold the LoRA
  weights into the base model for zero-latency inference.

- **Mistake:** Using `load_in_4bit=True` without `prepare_model_for_kbit_training`.
  Without this call, LayerNorm layers remain in INT8 and gradients are wrong.

  **Fix:** Always call `prepare_model_for_kbit_training(model)` before
  adding LoRA adapters to a quantised model.

- **Mistake:** Fine-tuning only attention Q/V matrices when the task
  requires learning new factual associations. Attention layers learn routing;
  feed-forward layers store facts.

  **Fix:** For knowledge injection tasks, include `W_up`, `W_down`, and
  `W_gate` in `target_modules`. For style adaptation, Q/V may suffice.

- **Mistake:** Forgetting to set `tokenizer.pad_token = tokenizer.eos_token`
  for models like LLaMA that have no default pad token. This causes a
  runtime error during batching.

  **Fix:** Always set pad_token before creating the SFTTrainer. Also set
  `model.config.pad_token_id = tokenizer.pad_token_id`.

## 6. Knowledge Check

1. **Conceptual:** Explain why LoRA initialises $B$ to zero and $A$ with
   Kaiming uniform. What would happen if both were randomly initialised?
   What would happen if both were zero?

2. **Conceptual:** QLoRA quantises base model weights to NF4 but keeps
   the LoRA adapters in BF16. Why not quantise the adapters too? What is
   the role of `bnb_4bit_compute_dtype` in the forward pass?

3. **Coding challenge:** Implement LoRALinear from scratch with a `merge()`
   method. Verify correctness: instantiate a random `nn.Linear`, wrap it
   with LoRALinear, train for 100 steps, then merge and check that
   `LoRALinear.forward(x) ≈ merged_linear(x)` for all test inputs.

## References

1. Hu, E., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. arXiv:2106.09685.
2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*. arXiv:2305.14314.
3. Liu, S.-Y., Wang, C.-Y., Yin, H., et al. (2024). DoRA: Weight-decomposed low-rank adaptation. *ICML 2024*. arXiv:2402.09353.
4. Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). Intrinsic dimensionality explains the effectiveness of language model fine-tuning. *ACL 2021*. arXiv:2012.13255.
