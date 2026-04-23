# BERT vs GPT vs LLaMA

The transformer architecture supports three families of models that differ
in their training objectives and attention patterns: encoder-only (BERT),
decoder-only (GPT, LLaMA), and encoder-decoder (T5, BART). Understanding
the design differences — and the training objectives that produced each
capability — explains why you pick BERT for classification, GPT-4 for
generation, and why LLaMA-3 has largely displaced earlier open models.

## 1. The Core Intuition (The "Why")

Devlin et al. (2018) trained BERT with a Masked Language Model (MLM) objective:
randomly mask 15% of tokens and predict them from bidirectional context. This
produced representations that captured rich semantic relationships and dominated
NLP benchmarks from 2018–2020.

Radford et al. (GPT, 2018) trained a left-to-right language model: predict
the next token given all previous tokens. This autoregressive objective naturally
produces a generative model — you can sample from it to generate text.

The key insight: GPT-3 (Brown et al., 2020) showed that scaling autoregressive
LMs produces emergent capabilities (few-shot learning, instruction following)
not present in smaller models. This shifted the field toward decoder-only
models, which now dominate.

Touvron et al. (2023) showed with LLaMA that training a smaller model for
much longer on more data could match or exceed larger models trained for
less time — efficiency matters as much as model size.

## 2. The Theoretical Underpinning

### Encoder-Only: BERT

BERT uses bidirectional attention: every token attends to every other token.
This is implemented by removing the causal mask from self-attention.

**MLM objective**: For each token $x_t$ selected for masking:
- 80% of the time: replace with `[MASK]`.
- 10%: replace with a random token.
- 10%: keep the original token.

The model predicts the original token for all masked positions. This
corruption strategy forces the model to use context from both directions.

**NSP (Next Sentence Prediction)** was part of the original BERT but found
not to help much; RoBERTa (Liu et al., 2019) removed it.

For downstream tasks, BERT is fine-tuned by adding a classification head
to the `[CLS]` token representation:

$$p(\text{class}) = \text{softmax}(W \cdot h_{\text{CLS}} + b)$$

### Decoder-Only: GPT / LLaMA

GPT-style models use causal (left-to-right) attention: position $i$ can
only attend to positions $1, \ldots, i$. The training objective is
autoregressive:

$$\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log p(x_t \mid x_1, \ldots, x_{t-1})$$

For generation, you autoregressively sample $x_t \sim p(\cdot \mid x_{<t})$,
then append $x_t$ to the context and generate $x_{t+1}$.

### Architectural Differences: BERT vs LLaMA

| Feature | BERT-base | LLaMA-3 8B |
|---------|-----------|------------|
| Layers | 12 | 32 |
| Hidden dim | 768 | 4096 |
| Heads | 12 | 32 |
| Attention | Bidirectional | Causal |
| Positional encoding | Learned absolute | RoPE |
| Normalisation | Post-norm (LayerNorm) | Pre-norm (RMSNorm) |
| FFN activation | GELU | SwiGLU |
| Vocabulary | 30K (WordPiece) | 128K (BPE) |
| Parameters | 110M | 8B |

### RMSNorm

LLaMA replaces LayerNorm with Root Mean Square Layer Normalisation (RMSNorm),
which is cheaper (no mean subtraction):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

### SwiGLU FFN

LLaMA uses a gated FFN:

$$\text{FFN}(x) = \left(\text{Swish}(xW_1) \odot xW_2\right) W_3$$

where $\text{Swish}(x) = x \cdot \sigma(x)$. The gate $xW_2$ controls which
information passes through, empirically outperforming standard GELU FFN.

### Grouped Query Attention (GQA)

LLaMA-2 70B and LLaMA-3 use Grouped Query Attention (Ainslie et al., 2023):
queries have $H$ heads but keys and values share $G < H$ heads. This reduces
the KV cache size by a factor of $H / G$ while maintaining most of the
expressiveness of Multi-Head Attention (MHA).

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    Args:
        x:     Input of shape (..., d).
        gamma: Learned scale parameter of shape (d,).
        eps:   Numerical stability constant.

    Returns:
        Normalised tensor of same shape as x.
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def swiglu(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray) -> np.ndarray:
    """SwiGLU feed-forward block.

    Args:
        x:  Input of shape (n, d).
        W1, W2: Gating weight matrices of shape (d, ffn_dim).
        W3: Output weight matrix of shape (ffn_dim, d).

    Returns:
        Output of shape (n, d).
    """
    gate  = x @ W1
    value = x @ W2
    # Swish activation (SiLU): gate * sigmoid(gate)
    activated = gate / (1 + np.exp(-gate)) * value
    return activated @ W3
```

## 4. Implementation: Production-Grade (Hugging Face Transformers)

```python
from __future__ import annotations

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch


def load_bert_for_classification(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
) -> tuple:
    """Load a BERT model for sequence classification.

    Args:
        model_name: Hugging Face model identifier.
        num_labels: Number of output classes.

    Returns:
        (tokenizer, model) pair.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_labels,
    )
    return tokenizer, model


def load_llama_4bit(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
) -> tuple:
    """Load a LLaMA model in 4-bit NF4 quantisation using bitsandbytes.

    4-bit quantisation reduces memory from ~16 GB (BF16) to ~5 GB,
    enabling inference on a single consumer GPU.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        (tokenizer, model) pair.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,  # nested quantisation
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = quantization_config,
        device_map          = "auto",
    )
    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt:       str,
    max_new_tokens: int  = 256,
    temperature:  float = 0.7,
    top_p:        float = 0.9,
) -> str:
    """Generate text from a causal LM using nucleus sampling.

    Args:
        temperature: Scales logits before softmax. < 1 makes distribution
            sharper (less random), > 1 makes it flatter.
        top_p: Nucleus sampling: keep the smallest set of tokens whose
            cumulative probability exceeds top_p.

    Returns:
        Generated text string.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            do_sample      = True,
        )
    # Decode only the new tokens (exclude input prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using a BERT-style encoder for text generation tasks.
  BERT's bidirectional attention means it cannot autoregressively generate
  text — it needs to see the full sequence first.

  **Fix:** Use an encoder-decoder model (T5, BART) or a decoder-only
  model (GPT, LLaMA) for generation tasks.

- **Mistake:** Fine-tuning BERT with a learning rate of 1e-3. BERT's
  pre-trained weights are sensitive; a high LR destroys the representations.

  **Fix:** Fine-tune BERT with `lr = 2e-5` and linear warmup over 10% of
  steps. This is the standard recipe from the BERT paper.

- **Mistake:** Comparing model sizes by parameter count alone. LLaMA-3 8B
  outperforms GPT-3 175B on many tasks due to better training data and
  training duration.

  **Fix:** Evaluate on the benchmarks relevant to your task.

- **Mistake:** Not handling chat templates when using instruction-tuned models.
  Feeding raw text to an instruction-tuned LLaMA will produce poor outputs
  because the model expects a specific prompt format.

  **Fix:** Use `tokenizer.apply_chat_template(messages, tokenize=False)`
  to format prompts correctly.

- **Mistake:** Loading a large LLM in FP32 when BF16 suffices.
  LLaMA-3 8B requires 32 GB in FP32 vs 16 GB in BF16.

  **Fix:** Always load with `torch_dtype=torch.bfloat16` or use 4-bit
  quantisation for inference on consumer hardware.

## 6. Knowledge Check

1. **Conceptual:** Compare the MLM and autoregressive (CLM) training
   objectives. What capability does each objective specifically train for?
   Why does bidirectional context help for understanding but hurt for
   generation?

2. **Conceptual:** LLaMA replaces absolute positional encodings with RoPE.
   What are the limitations of absolute positional encodings that RoPE
   addresses? How does RoPE encode relative position?

3. **Coding challenge:** Using Hugging Face Transformers, fine-tune a
   BERT model on the SST-2 sentiment classification dataset. Then evaluate
   zero-shot sentiment classification using LLaMA-3 with a prompt. Compare
   accuracy and discuss when you would use each approach.

## References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*. arXiv:1810.04805.
2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. OpenAI Blog.
3. Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *NeurIPS 2020*. arXiv:2005.14165.
4. Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and efficient foundation language models. arXiv:2302.13971.
5. Su, J., Lu, Y., Pan, S., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
