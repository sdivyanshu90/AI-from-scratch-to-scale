# Quantisation: GGUF, AWQ, and GPTQ

Quantisation reduces the bit-width of model weights and activations, cutting
memory requirements and inference latency at the cost of some numerical
precision. A 70B-parameter model in FP16 requires ~140 GB of GPU memory —
more than 4 A100s. The same model in 4-bit requires ~35 GB, fitting on a
single A100. GGUF, AWQ, and GPTQ represent three different deployment
contexts: CPU/edge inference, GPU calibration-aware quantisation, and
GPU post-training quantisation respectively.

## 1. The Core Intuition (The "Why")

Neural network weights exhibit a striking empirical property: their values
follow approximately Gaussian distributions, and most of the information
in a weight matrix lies in a small fraction of salient channels. This means
that uniform quantisation (treating all values equally) wastes bits on
near-zero weights and loses precision on the large-magnitude weights that
matter most.

Dettmers et al. (2022) showed that 8-bit quantisation can be made nearly
lossless by quantising weights and activations **per-channel** (each row/column
gets its own scale factor) rather than per-tensor. Lin et al. (2023) with
AWQ showed that protecting only 1% of the "salient" weight channels (those
with the largest activation magnitudes) recovers most of the quality loss
from 4-bit quantisation.

## 2. The Theoretical Underpinning

### Uniform Affine Quantisation

For a real-valued tensor $x$ with range $[x_\min, x_\max]$, affine
quantisation maps to $b$-bit integers:

$$q = \operatorname{clip}\!\left(\left\lfloor \frac{x - z}{s} \right\rceil, 0, 2^b - 1\right)$$

where the **scale** $s = (x_\max - x_\min) / (2^b - 1)$ and the
**zero-point** $z = x_\min$ (asymmetric) or $z=0$ (symmetric).

Dequantisation recovers an approximation:

$$\hat{x} = s(q - z)$$

The **quantisation error** is bounded by $|x - \hat{x}| \leq s/2$, so
finer scale (smaller range per block) reduces error.

### Per-Channel and Per-Group Quantisation

Per-tensor quantisation uses one $(s, z)$ pair for the entire matrix.
Per-channel uses one pair per output channel (row). Per-group uses one
pair per contiguous block of $g$ elements within each channel.

Memory overhead of quantisation parameters: for a matrix of size $d_\text{out} \times d_\text{in}$
with group size $g$:

$$\text{params overhead} = \frac{d_\text{out} \times d_\text{in}}{g} \times \text{sizeof}(s, z)$$

Typical values: $g = 128$, scales stored in FP16 (2 bytes).

### GPTQ: One-Shot Weight Quantisation

Frantar et al. (2022) adapted the Optimal Brain Compression (OBC) framework
to quantise each weight row by row. After quantising weight $w_{ij}$ to
$\hat{w}_{ij}$, the error is compensated by adjusting the remaining
unquantised weights in the same row:

$$\delta W_{\text{remaining}} = -\frac{(w_{ij} - \hat{w}_{ij})}{[H^{-1}]_{jj}} \cdot [H^{-1}]_{j, \text{remaining}}$$

where $H = 2 X X^\top$ is the Hessian of the layer output error with
respect to the weights, computed from a calibration dataset.

### AWQ: Activation-Aware Weight Quantisation

Lin et al. (2023) observed that weight channels corresponding to large
activation magnitudes cause the most quantisation error. AWQ protects
these channels by scaling them up before quantisation (and scaling
activations down accordingly):

$$\tilde{w}_j = w_j \cdot s_j, \quad \tilde{x}_j = x_j / s_j$$

where $s_j \propto |x_j|^\alpha$ for a tuned $\alpha \in [0, 1]$.
The scale $s_j$ is absorbed into the preceding layer (e.g., the LayerNorm
scale), adding no runtime overhead.

### NormalFloat 4-bit (NF4)

Dettmers et al. (2023) (QLoRA) defined NF4, which places quantisation
bins at the quantiles of the standard normal distribution:

$$q_i = \Phi^{-1}\!\left(\frac{i + 0.5}{2^k}\right), \quad i = 0, \ldots, 2^k - 1$$

For $k=4$, the 16 bins are placed where they are most needed for normally
distributed weights, minimising expected quantisation error for LLM weights.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class QuantisedTensor:
    """Quantised tensor with per-group scale factors.

    Attributes:
        values:     Quantised integer values (int8 or int4 packed).
        scales:     Per-group scale factors.
        zero_points: Per-group zero points (0 for symmetric).
        group_size: Number of elements per quantisation group.
        orig_shape: Original tensor shape before quantisation.
    """
    values:      np.ndarray
    scales:      np.ndarray
    zero_points: np.ndarray
    group_size:  int
    orig_shape:  tuple


def quantise_per_group(
    tensor:     np.ndarray,
    group_size: int = 128,
    bits:       int = 8,
) -> QuantisedTensor:
    """Asymmetric per-group quantisation.

    Flattens the tensor, splits into groups of group_size elements,
    and computes (scale, zero_point) per group.

    Args:
        tensor:     Float32 input tensor.
        group_size: Elements per quantisation group.
        bits:       Quantisation bit-width.

    Returns:
        QuantisedTensor with integer values and per-group parameters.
    """
    flat    = tensor.flatten().astype(np.float32)
    n       = len(flat)
    # Pad to multiple of group_size
    pad     = (-n) % group_size
    flat    = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    groups  = flat.reshape(-1, group_size)  # (n_groups, group_size)

    qmin, qmax = 0, 2**bits - 1
    x_min = groups.min(axis=1, keepdims=True)
    x_max = groups.max(axis=1, keepdims=True)
    scales = (x_max - x_min) / (qmax - qmin)
    scales = np.where(scales == 0, 1.0, scales)   # Avoid division by zero
    zeros  = np.round(-x_min / scales).astype(np.int32)

    quantised = np.clip(np.round(groups / scales + zeros), qmin, qmax).astype(np.int8)
    return QuantisedTensor(
        values      = quantised[:len(flat)//group_size],
        scales      = scales.flatten(),
        zero_points = zeros.flatten(),
        group_size  = group_size,
        orig_shape  = tensor.shape,
    )


def dequantise_per_group(qt: QuantisedTensor) -> np.ndarray:
    """Reconstruct float32 tensor from per-group quantised representation.

    Args:
        qt: QuantisedTensor produced by quantise_per_group.

    Returns:
        Approximate float32 reconstruction.
    """
    n_groups   = qt.values.shape[0]
    groups     = qt.values.astype(np.float32)   # (n_groups, group_size)
    s          = qt.scales.reshape(-1, 1)
    z          = qt.zero_points.reshape(-1, 1)
    deq        = s * (groups - z)
    flat       = deq.flatten()
    total_orig = int(np.prod(qt.orig_shape))
    return flat[:total_orig].reshape(qt.orig_shape)


def quantisation_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean absolute error between original and quantised-dequantised tensor.

    Args:
        original:      Original float32 tensor.
        reconstructed: Reconstructed tensor after quantise + dequantise.

    Returns:
        Mean absolute error.
    """
    return float(np.mean(np.abs(original - reconstructed)))
```

## 4. Implementation: Production-Grade (AWQ + GGUF)

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def quantise_awq(
    model_name:    str,
    output_dir:    str,
    w_bit:         int = 4,
    group_size:    int = 128,
    calib_dataset: str = "pileval",
) -> Path:
    """Quantise a model with AWQ (Activation-Aware Weight Quantisation).

    AWQ protects salient weight channels (those with large activation
    magnitudes) during quantisation, recovering most quality lost by
    naive 4-bit quantisation.

    Args:
        model_name:    HuggingFace model identifier.
        output_dir:    Directory to write the quantised model.
        w_bit:         Target weight bit-width (4 or 8).
        group_size:    Quantisation group size (128 is standard).
        calib_dataset: Calibration dataset name (used to compute activations).

    Returns:
        Path to the saved quantised model directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoAWQForCausalLM.from_pretrained(
        model_name, safetensors=True, device_map="auto"
    )
    quant_config = {
        "zero_point": True,        # Asymmetric quantisation
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": "GEMM",         # GEMM kernel (faster) vs GEMV (smaller batch)
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_dataset=calib_dataset)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    return Path(output_dir)


def load_gguf_and_generate(
    gguf_path:  str,
    prompt:     str,
    max_tokens: int = 256,
    n_ctx:      int = 4096,
) -> str:
    """Load a GGUF model and generate a response.

    GGUF (GPT-Generated Unified Format) is the llama.cpp model format.
    It packages quantised weights, tokeniser, and metadata in one file,
    enabling CPU-only inference on consumer hardware.

    The quantisation type is embedded in the filename (e.g., Q4_K_M):
    Q4 = 4-bit, K = k-quants (mixed precision), M = medium accuracy.

    Args:
        gguf_path:  Path to the .gguf model file.
        prompt:     Input prompt.
        max_tokens: Maximum new tokens to generate.
        n_ctx:      Context window size.

    Returns:
        Generated text.
    """
    from llama_cpp import Llama
    llm    = Llama(model_path=gguf_path, n_ctx=n_ctx, verbose=False)
    output = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    return output["choices"][0]["text"]
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Assuming every layer tolerates 4-bit quantisation equally.
  The first and last embedding/unembedding layers, and attention output
  projection matrices, are typically most sensitive to quantisation error.

  **Fix:** Use mixed-precision quantisation: keep sensitive layers in 8-bit
  or FP16. GGUF's Q4_K_M and Q6_K formats do this automatically.

- **Mistake:** Comparing a quantised model's quality using perplexity only.
  Perplexity measures next-token prediction but does not capture task-specific
  degradation (e.g., instruction following, reasoning).

  **Fix:** Evaluate on your target task benchmark (MMLU, GSM8K, HumanEval)
  before and after quantisation. Report task accuracy alongside perplexity.

- **Mistake:** Treating GGUF as a quantisation method. GGUF is a packaging
  format; it can contain weights quantised by any method (Q4_0, Q8_0, F16).

  **Fix:** Distinguish between the quantisation algorithm (GPTQ, AWQ, GGML
  k-quants) and the file format (GGUF, safetensors, bin). Both choices affect
  the result independently.

- **Mistake:** Skipping calibration data for GPTQ or AWQ. Both methods
  compute Hessians or activation statistics from a small dataset. Using
  calibration data from the wrong domain degrades quality on your target domain.

  **Fix:** Use 128-512 samples from your target domain (or a close proxy)
  as calibration data. The AutoAWQ `calib_dataset` parameter accepts custom
  datasets.

- **Mistake:** Quantising the KV cache to 4-bit but not accounting for the
  quality impact on long-context tasks. KV cache quantisation reduces memory
  but is lossy for long sequences where early tokens are rarely re-attended.

  **Fix:** Evaluate long-context tasks (multi-hop QA, 128k-token summarisation)
  separately after enabling KV cache quantisation. Fall back to 8-bit KV
  cache for quality-critical deployments.

## 6. Knowledge Check

1. **Conceptual:** Derive the memory saving from moving a 7B-parameter model
   from FP16 (2 bytes/param) to 4-bit (0.5 bytes/param with group-size 128
   and FP16 scales). What is the effective bits-per-weight including scale
   overhead?

2. **Conceptual:** Explain why AWQ protects salient weight channels. What is
   the activation-weight relationship that motivates this choice? How does
   AWQ absorb the per-channel scale $s_j$ into the model without adding
   runtime overhead?

3. **Coding challenge:** Implement per-group 8-bit symmetric quantisation
   from scratch. Quantise a random (1024, 4096) weight matrix with group
   size 128. Compute the mean absolute error before and after quantisation.
   Compare to per-tensor quantisation and report the difference in error.

## References

1. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit matrix multiplication for transformers at scale. *NeurIPS 2022*. arXiv:2208.07339.
2. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate post-training quantization for generative pre-trained transformers. *ICLR 2023*. arXiv:2210.17323.
3. Lin, J., Tang, J., Tang, H., et al. (2023). AWQ: Activation-aware weight quantization for LLM compression and acceleration. arXiv:2306.00978.
4. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS 2023*. arXiv:2305.14314. *(NF4 quantisation.)*
5. Gerganov, G. (2023). llama.cpp and GGUF format. https://github.com/ggerganov/llama.cpp.
