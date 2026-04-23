# Ollama and LM Studio

Ollama and LM Studio are the two dominant tools for running large language
models locally — without GPU cloud costs, API rate limits, or data-privacy
concerns. Understanding how they work under the hood, and what their
limitations are, is essential for choosing the right tool for local
development, offline inference, and privacy-sensitive deployments.

## 1. The Core Intuition (The "Why")

Before tools like Ollama existed, running a large language model locally was
a multi-day project: find the model weights, convert them to the right format,
install the correct CUDA toolkit version, write inference scaffolding, and
debug incompatibilities. The result was that local LLM experimentation was
practically limited to researchers with ML engineering experience.

Two developments changed this:
1. **GGUF quantisation** (via llama.cpp, Gerganov 2023): A file format and
   quantisation scheme that compresses model weights from 16-bit floats to
   4-bit integers, reducing a 7B-parameter model from ~14 GB to ~4 GB —
   small enough to fit in the RAM of a consumer laptop.
2. **Metal and Vulkan GPU backends**: llama.cpp added backends for Apple
   Silicon's Metal API and AMD/Intel GPUs via Vulkan, making GPU-accelerated
   inference available without CUDA.

Ollama packages llama.cpp into a Docker-style CLI and API. LM Studio wraps
it in a desktop GUI. Both expose an OpenAI-compatible REST API, which means
any code written against the OpenAI SDK works against a local model with a
one-line endpoint change.

## 2. The Theoretical Underpinning

### Quantisation: Trading Precision for Memory

A full-precision (FP32) model weight uses 4 bytes. A BFloat16 weight uses 2
bytes. GGUF Q4 quantisation uses roughly 0.5 bytes per weight (4 bits). For
a 7B-parameter model:

| Format | Size per weight | Total size (7B params) |
|--------|-----------------|------------------------|
| FP32   | 4 bytes         | 28 GB                  |
| BF16   | 2 bytes         | 14 GB                  |
| Q8_0   | 1 byte          | 7 GB                   |
| Q4_K_M | ~0.5 bytes      | ~4 GB                  |
| Q2_K   | ~0.25 bytes     | ~2 GB                  |

The quality-size tradeoff: Q8_0 is nearly lossless. Q4_K_M (the most popular
format) loses ~2–5% on benchmarks but is imperceptible in most practical tasks.
Q2_K degrades noticeably. For production, Q4_K_M or Q5_K_M offer the best
quality-per-GB ratio.

### GGUF File Format

GGUF (GPT-Generated Unified Format) is a single-file binary format that stores:
- Model architecture metadata (JSON-serialised)
- Tokenizer vocabulary and merges
- All quantised weight tensors

The key design goal is that a model can be memory-mapped and loaded without
parsing or preprocessing. Inference begins immediately after `mmap()`.

### Prefill and Decode Phases

LLM inference has two distinct computational phases:
1. **Prefill**: The prompt tokens are processed in parallel (like a forward
   pass through a matrix). This is compute-bound and fast on GPU.
2. **Decode**: Tokens are generated one at a time, each requiring a full
   forward pass. This is memory-bandwidth-bound because the weight matrices
   must be read from VRAM/RAM on every step.

This is why generation speed (tokens/second) is limited primarily by memory
bandwidth, not FLOPS. An M3 Mac with 150 GB/s unified memory bandwidth often
generates tokens faster than a GPU with higher FLOPS but narrower memory bus.

### KV Cache

To avoid recomputing attention for all previous tokens on each decode step, a
**KV cache** stores the key and value tensors for each attention layer. Cache
size grows linearly with context length and number of layers:

$$
\text{KV cache size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L \times \text{dtype\_bytes}
$$

where $L$ is the current context length. For Llama-3-8B at 8K context with
BF16: $2 \times 32 \times 32 \times 128 \times 8192 \times 2 \approx 4.3$ GB.
Context length directly determines maximum memory required for inference.

## 3. Implementation: From Scratch (Python)

The key insight: once a model is running, the interaction protocol is just
HTTP. Here is a minimal client that communicates with any OpenAI-compatible
local server:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Generator, Iterator, List
import urllib.request
import urllib.error


@dataclass
class ChatMessage:
    role: str    # "system", "user", or "assistant"
    content: str


def build_chat_payload(
    messages: List[ChatMessage],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
) -> bytes:
    """Serialise a chat completion request to JSON bytes.

    Args:
        messages: Conversation history.
        model: Model identifier, e.g. 'llama3.2' or 'mistral'.
        temperature: Sampling temperature; 0 is greedy.
        max_tokens: Maximum tokens to generate.
        stream: Whether to request a streaming response.

    Returns:
        UTF-8 JSON payload.
    """
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    return json.dumps(payload).encode("utf-8")


def chat_completion(
    messages: List[ChatMessage],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Send a chat request to a local OpenAI-compatible server.

    Args:
        messages: Conversation history.
        model: Model name as configured in the local server.
        base_url: Base URL of the local server.
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens.

    Returns:
        Generated assistant message text.

    Raises:
        RuntimeError: If the server returns a non-200 status.
    """
    url     = f"{base_url}/v1/chat/completions"
    payload = build_chat_payload(messages, model, temperature, max_tokens)

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Server returned {exc.code}: {exc.read()}") from exc
```

## 4. Implementation: Production-Grade (Ollama + OpenAI SDK)

```python
from __future__ import annotations

from openai import OpenAI


def get_local_client(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",   # Ollama ignores the key; required by SDK
) -> OpenAI:
    """Return an OpenAI client configured to talk to a local Ollama server.

    The OpenAI SDK is used unchanged — only the base_url differs.
    This means any code written for the OpenAI API works locally with
    a one-line change.

    Args:
        base_url: URL of the local OpenAI-compatible server.
        api_key: Placeholder key (not used by Ollama).

    Returns:
        Configured OpenAI client.
    """
    return OpenAI(base_url=base_url, api_key=api_key)


def run_conversation(
    system_prompt: str,
    user_message: str,
    model: str = "llama3.2",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Run a two-turn conversation against a local model.

    Args:
        system_prompt: System instruction for the model.
        user_message: User's input message.
        model: Local model name.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Returns:
        Generated assistant response.
    """
    client = get_local_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

## 5. Production Pitfalls & Pro-Tips

- ❌ **The mistake:** Choosing model size based on parameter count alone.
  A 3B-parameter model with 16K context using Q8_0 might use more RAM than a
  7B model at Q4_K_M with 2K context.
  ✅ **The fix:** Calculate total memory usage: weights + KV cache. Use the
  formula in the Theory section. Profile with `ollama ps` which shows live
  VRAM and RAM usage.

- ❌ **The mistake:** Using overly aggressive quantisation (Q2_K) to fit a
  larger model, then blaming the model for poor responses.
  ✅ **The fix:** Prefer Q4_K_M or Q5_K_M. If you must use Q2_K, benchmark
  quality on your specific task before deploying.

- ❌ **The mistake:** Using a large context window (32K+) for simple tasks,
  consuming VRAM for KV cache that could hold more concurrent requests.
  ✅ **The fix:** Set `num_ctx` in the Ollama Modelfile to the actual context
  your task needs, not the maximum the model supports.

- ❌ **The mistake:** Exposing the Ollama API on `0.0.0.0` (all interfaces)
  on a shared network without authentication. Anyone on the network can use
  your compute.
  ✅ **The fix:** Bind to `127.0.0.1` for local-only access, or add an
  authenticated reverse proxy (nginx + HTTP Basic Auth) before exposing
  externally.

- ❌ **The mistake:** Not batching requests when running offline inference
  over a large dataset. Single-request-at-a-time throughput wastes the
  prefill parallelism.
  ✅ **The fix:** Use llama.cpp's server with `--parallel` flag or the
  `vllm` framework for batched offline inference at scale.

## 6. Knowledge Check

1. **Conceptual:** LLM generation speed is memory-bandwidth-bound, not
   compute-bound. Explain why using the KV cache formula in the Theory section
   and the decode phase description. When would generation become compute-bound?

2. **Conceptual:** GGUF Q4_K_M quantises weights to 4 bits. During inference,
   weights must be dequantised to FP16 for matrix multiplication. Why does this
   not eliminate the memory savings if dequantisation is required anyway?

3. **Coding challenge:** Using the `from_scratch` client in Section 3, extend
   `chat_completion` to support streaming responses by parsing the
   `text/event-stream` format and yielding each token as it arrives.

## References

1. Gerganov, G. (2023). llama.cpp — LLM inference in C/C++. GitHub: https://github.com/ggerganov/llama.cpp
2. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*. *(Context for quantisation quality tradeoffs.)*
3. Pope, R., et al. (2023). Efficiently Scaling Transformer Inference. *MLSys 2023*. *(KV cache and prefill/decode analysis.)*
4. GGUF format specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
