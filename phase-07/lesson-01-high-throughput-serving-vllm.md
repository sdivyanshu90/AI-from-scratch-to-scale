# High-Throughput LLM Serving with vLLM

High-throughput serving is the engineering discipline of turning a trained
generative model into a system that answers hundreds of concurrent requests
efficiently. Naive per-request generation leaves modern GPUs badly
underutilised because each request monopolises GPU memory for its own
key-value (KV) cache. vLLM (Kwon et al., 2023) attacks this with
**PagedAttention** and **continuous batching**, achieving 10-24x higher
throughput than a standard HuggingFace generation loop on the same hardware.

## 1. The Core Intuition (The "Why")

A single generation request with a 2,048-token context window reserves a
contiguous memory block for its KV cache for the entire duration of decoding
— even though the actual KV values grow token by token and the memory is
mostly empty at the start. For 100 concurrent requests, this wastes
gigabytes of GPU memory on padding and fragmentation.

Kwon et al. (2023) observed that this is exactly the same problem operating
systems solved for CPU memory in the 1960s: **virtual memory paging**.
Instead of allocating one large contiguous block per process, the OS divides
memory into small fixed-size pages and maps virtual addresses to physical
pages on demand. vLLM applies the same idea to the KV cache.

The second insight is **continuous batching** (Yu et al., 2022): instead of
waiting for all requests in a batch to finish before starting new ones, the
server adds new requests to the batch as slots free up. This keeps GPU
utilisation high even when requests have widely varying lengths.

## 2. The Theoretical Underpinning

### KV Cache Memory Requirement

For a Transformer with $L$ layers, $H$ attention heads, head dimension $d_h$,
and sequence length $T$, the KV cache requires:

$$M_\text{KV} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot \text{sizeof}(\text{dtype})$$

The factor of 2 is for keys and values. For LLaMA-3-8B with
$L=32$, $H=32$, $d_h=128$, $T=4096$, dtype=fp16 (2 bytes):

$$M_\text{KV} = 2 \cdot 32 \cdot 32 \cdot 128 \cdot 4096 \cdot 2 = 2\,\text{GB per request}$$

With 40 GB of GPU memory and ~15 GB for model weights, you can serve at
most ~12 requests simultaneously with naive allocation.

### PagedAttention

vLLM partitions the KV cache into fixed-size **pages** of $B$ tokens each.
Each request is allocated pages on demand; pages from different requests
can be freely interleaved in physical GPU memory.

The attention computation over paged memory reads KV values from a
**block table** mapping logical token positions to physical page addresses:

$$\text{Attention}(Q, K_\text{paged}, V_\text{paged}) = \text{softmax}\!\left(\frac{Q K_\text{paged}^\top}{\sqrt{d_h}}\right) V_\text{paged}$$

PagedAttention is implemented as a custom CUDA kernel that follows the block
table to gather KV pages without materialising a contiguous KV matrix.

### Continuous Batching Throughput

Let request $i$ have prompt length $p_i$ and generation length $g_i$.
Aggregate throughput is:

$$\text{TPS} = \frac{\sum_{i=1}^N g_i}{T_\text{total}}$$

With continuous batching, the active batch $B_t$ at decode step $t$ changes
dynamically: finished requests are removed and new arrivals are added.
The latency model for request $i$ is:

$$L_i = L_\text{prefill}(p_i) + \sum_{t=1}^{g_i} \frac{1}{R(B_t)}$$

where $R(B_t)$ is the token generation rate for the active batch at step $t$.
Continuous batching keeps $|B_t|$ as large as possible, maximising $R(B_t)$.

### Speculative Decoding

Leviathan et al. (2022) showed that a small draft model can propose
$\gamma$ tokens per step; the target model verifies all $\gamma$ tokens in
a single parallel forward pass. Expected accepted tokens per step:

$$\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

where $\alpha$ is the probability the draft matches the target. For $\alpha=0.8$,
$\gamma=4$: expected acceptance $\approx 3.36$ tokens per forward pass instead
of 1 — a ~3x latency reduction.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    """State of one generation request in the scheduler.

    Args:
        request_id:       Unique request identifier.
        prompt_tokens:    Number of tokens in the prompt (prefill).
        max_new_tokens:   Maximum tokens to generate.
        generated_tokens: Tokens generated so far.
    """
    request_id:       str
    prompt_tokens:    int
    max_new_tokens:   int
    generated_tokens: int = 0

    @property
    def is_done(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens


class ContinuousBatchScheduler:
    """Simulate continuous batching: fill the batch as requests finish.

    Unlike static batching (wait for all to finish), continuous batching
    swaps in new requests as soon as a slot opens. This is the key
    scheduling insight behind vLLM's throughput gains.
    """

    def __init__(self, max_batch_size: int, gpu_memory_pages: int,
                 page_size: int = 16) -> None:
        self.max_batch_size   = max_batch_size
        self.gpu_memory_pages = gpu_memory_pages
        self.page_size        = page_size
        self.active:  list[Request] = []
        self.waiting: list[Request] = []
        self._page_table: dict[str, int] = {}   # request_id -> pages allocated

    def _pages_needed(self, req: Request) -> int:
        total_tokens = req.prompt_tokens + req.max_new_tokens
        return -(-total_tokens // self.page_size)   # Ceiling division

    def _pages_used(self) -> int:
        return sum(self._page_table.values())

    def admit(self, req: Request) -> None:
        """Add a new request to the waiting queue."""
        self.waiting.append(req)

    def schedule(self) -> None:
        """Move waiting requests into the active batch if capacity allows.

        Continuous batching: admit new requests whenever memory and batch
        size allow, rather than waiting for all active requests to finish.
        """
        while self.waiting:
            candidate = self.waiting[0]
            pages     = self._pages_needed(candidate)
            has_batch = len(self.active) < self.max_batch_size
            has_mem   = self._pages_used() + pages <= self.gpu_memory_pages
            if has_batch and has_mem:
                self.active.append(self.waiting.pop(0))
                self._page_table[candidate.request_id] = pages
            else:
                break

    def decode_step(self) -> list[str]:
        """Advance each active request by one token. Return finished IDs."""
        finished = []
        for req in self.active:
            req.generated_tokens += 1
            if req.is_done:
                finished.append(req.request_id)
        # Remove finished requests and free their pages
        for rid in finished:
            self.active = [r for r in self.active if r.request_id != rid]
            del self._page_table[rid]
        return finished

    def run_until_empty(self) -> dict[str, int]:
        """Run the scheduler until all requests are served.

        Returns:
            Dict mapping request_id to total decode steps taken.
        """
        steps: dict[str, int] = {}
        while self.active or self.waiting:
            self.schedule()
            finished = self.decode_step()
            for rid in finished:
                steps[rid] = steps.get(rid, 0) + 1
        return steps
```

## 4. Implementation: Production-Grade (vLLM)

```python
from __future__ import annotations

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


def sync_batch_generate(
    model_name: str,
    prompts:    list[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> list[str]:
    """Generate text for a batch of prompts with vLLM (synchronous).

    vLLM automatically uses PagedAttention and continuous batching.
    Setting temperature=0.0 uses greedy decoding (deterministic).

    Args:
        model_name:  HuggingFace model ID or local path.
        prompts:     List of input prompts.
        max_tokens:  Maximum new tokens per prompt.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        List of generated text strings, one per prompt.
    """
    llm    = LLM(
        model              = model_name,
        tensor_parallel_size = 1,          # Set to num GPUs for multi-GPU
        gpu_memory_utilization = 0.90,     # Fraction of GPU memory for KV cache
        max_model_len      = 4096,
    )
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    out    = llm.generate(prompts, params)
    return [o.outputs[0].text for o in out]


async def async_generate_stream(
    model_name: str,
    prompt:     str,
    max_tokens: int = 256,
) -> list[str]:
    """Stream tokens from vLLM asynchronous engine (production API serving).

    The async engine is designed for FastAPI/aiohttp serving where many
    requests arrive concurrently. It exposes streaming via an async generator.

    Args:
        model_name: HuggingFace model ID or local path.
        prompt:     Input prompt.
        max_tokens: Maximum new tokens to generate.

    Returns:
        List of streamed token strings (for testing; in prod, yield them).
    """
    engine_args = AsyncEngineArgs(model=model_name)
    engine      = AsyncLLMEngine.from_engine_args(engine_args)
    params      = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    tokens = []
    async for output in engine.generate(prompt, params, request_id="req-0"):
        if output.outputs:
            tokens.append(output.outputs[0].text)
    return tokens
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Benchmarking throughput with a single request. Single-request
  latency looks fine, but the server is idle 90% of the time. This
  measurement does not reflect real concurrent traffic.

  **Fix:** Use a load testing tool (e.g., `vllm benchmark_serving`) with
  mixed prompt lengths and concurrent users matching your expected peak
  traffic. Report TPS and P95 latency under load.

- **Mistake:** Setting `gpu_memory_utilization = 1.0`. vLLM allocates
  (utilisation * total_GPU_memory) for the KV cache at startup. Leaving no
  headroom causes OOM errors under sudden traffic spikes.

  **Fix:** Use `gpu_memory_utilization = 0.85-0.90`. Monitor GPU memory
  usage with `nvidia-smi` under peak load and tune accordingly.

- **Mistake:** Using vLLM for tasks where low latency for a single request
  is critical (e.g., interactive chat where the user waits). vLLM optimises
  for throughput; this can increase P50 latency for individual requests.

  **Fix:** For latency-sensitive single-request workloads, use speculative
  decoding or a smaller model. For throughput-sensitive batch workloads
  (offline inference, data pipelines), vLLM is the right choice.

- **Mistake:** Not setting `max_model_len` appropriately. vLLM pre-allocates
  KV cache pages for the maximum sequence length. Leaving this at the
  model default (often 128k) wastes huge amounts of GPU memory.

  **Fix:** Set `max_model_len` to the 99th percentile of your actual
  input + output lengths. This dramatically increases the number of
  concurrent requests that fit in GPU memory.

- **Mistake:** Deploying vLLM without a request queue in front of it.
  Under traffic spikes, vLLM can run out of KV cache pages, causing
  requests to fail with OOM errors rather than queueing gracefully.

  **Fix:** Add a queue (Redis, Celery, or a simple async queue) that
  holds requests until vLLM has capacity. Return HTTP 503 with a
  Retry-After header when the queue is full.

## 6. Knowledge Check

1. **Conceptual:** Explain how PagedAttention eliminates KV cache memory
   fragmentation. Why does allocating one large contiguous block per request
   waste memory? What is the analogy to OS virtual memory paging?

2. **Conceptual:** Derive the KV cache memory requirement for a Transformer
   layer. For a model with 32 layers, 32 heads, and head dimension 128,
   how much GPU memory does a single 4,096-token request consume? How does
   this change with 4-bit KV cache quantisation?

3. **Coding challenge:** Implement a `ContinuousBatchScheduler` from scratch.
   Create 10 requests with random prompt lengths (50-200 tokens) and
   generation lengths (50-200 tokens). Simulate decoding with max batch size
   4 and measure total decode steps. Compare to the static batching baseline
   (all 10 served sequentially in 3 batches of ~3).

## References

1. Kwon, W., Li, Z., Zhuang, S., et al. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP 2023*. arXiv:2309.06180.
2. Yu, G., Kim, J., Jeong, H., et al. (2022). Orca: A distributed serving system for Transformer-based generative models. *OSDI 2022*.
3. Leviathan, Y., Kalman, M., & Matias, Y. (2022). Fast inference from transformers via speculative decoding. *ICML 2023*. arXiv:2211.17192.
4. Pope, R., Douglas, S., Chowdhery, A., et al. (2023). Efficiently scaling transformer inference. *MLSys 2023*. arXiv:2211.05102.
