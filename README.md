# AI - From Scratch to Scale

> Open-source notes and implementations covering the full AI engineering stack:
> from mathematical foundations to production-grade multi-agent systems.

Each lesson follows a consistent six-section structure — **why it exists**, **the theory**, **from-scratch implementation**, **production code**, **pitfalls**, and **knowledge-check questions** — so the material stays grounded and actionable at every level.

---

## Learning Roadmap

```
Phase 1  ──────────────────────────────────────────────────────────────────
  Math primitives (linear algebra, calculus, probability)
  + Developer tooling (Git, Conda/uv, Hugging Face, Ollama, Pandas, Polars)
         │
         ▼
Phase 2  ──────────────────────────────────────────────────────────────────
  Classical ML (regression, trees, boosting, clustering, PCA)
  + Deep learning stack (PyTorch → MLP → backprop → AdamW → CNN → RNN)
         │
         ▼
Phase 3  ──────────────────────────────────────────────────────────────────
  Self-attention → Transformer families (BERT / GPT / Llama)
  → Linear alternatives (Mamba, Jamba) → Multimodality (ViT, SD, Flux)
         │
    ┌────┴────┐
    ▼         ▼
Phase 4    Phase 5
Applied    Customisation
Gen AI     & Training
(RAG,      (LoRA, DPO,
 Tools,     Distillation)
 Reasoning) │
    │        │
    └────┬───┘
         ▼
Phase 6  ──────────────────────────────────────────────────────────────────
  Multi-agent systems (LangGraph, AutoGen, CrewAI)
  + State management + Human-in-the-Loop
         │
         ▼
Phase 7  ──────────────────────────────────────────────────────────────────
  MLOps & deployment (vLLM, quantization, AI security, Docker/K8s/Ray)
```

**Navigation tips**

| Goal                                          | Start here                         |
| --------------------------------------------- | ---------------------------------- |
| Complete beginner                             | Phase 1 → 2 → 3 → 4 → 5 → 6 → 7    |
| Know ML, new to DL                            | Phase 2 lesson 6 (PyTorch) onwards |
| Know DL, new to LLMs                          | Phase 3 lesson 1 (Self-Attention)  |
| Interview prep (ML fundamentals)              | Phase 2 lessons 1–5                |
| Interview prep (deep learning / transformers) | Phase 2 lessons 6–11 + Phase 3     |
| Building RAG / agent systems                  | Phase 4 + Phase 6                  |
| Fine-tuning & alignment                       | Phase 5                            |
| Production / MLOps                            | Phase 7                            |

---

## Phase 1 — The Primitives

> Build the mathematical intuition and engineering toolbox that everything else depends on.
> The first three lessons cover the three pillars of ML mathematics.
> The remaining six cover the tools every practitioner reaches for daily.

```
Linear Algebra ──► Calculus ──► Probability
      │                │              │
      └────────────────┴──────────────┘
                       │
              Git  ◄───┘
               │
            Conda/uv
               │
         Hugging Face Hub
               │
         Ollama / LM Studio
               │
        Pandas ──► Polars
```

| #   | Lesson                                                       | What You Learn                                                                                                                                          |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Linear Algebra](phase-01/lesson-01-linear-algebra.md)       | Vectors, matrices, dot products, eigenvalues, and SVD. Every neural layer is $y = Wx + b$; this is the language it speaks.                              |
| 2   | [Calculus](phase-01/lesson-02-calculus.md)                   | Derivatives, the chain rule, and automatic differentiation. Tells you which direction is downhill so gradient descent can work.                         |
| 3   | [Probability](phase-01/lesson-03-probability.md)             | Bayes' theorem, distributions, entropy, and KL divergence. Lets models reason under uncertainty instead of giving brittle yes/no answers.               |
| 4   | [Git](phase-01/lesson-04-git.md)                             | Content-addressed snapshots, branching, and merge strategies. Every experiment needs a stable identity and reproducible history.                        |
| 5   | [Conda / uv](phase-01/lesson-05-conda-uv.md)                 | Dependency graphs, lockfiles, and virtual environment isolation. Prevents "works on my machine" from reaching production.                               |
| 6   | [Hugging Face Hub](phase-01/lesson-06-hugging-face-hub.md)   | Model cards, snapshot downloads, dataset versioning, and the `transformers` / `datasets` APIs. Models are versioned software artifacts, not just files. |
| 7   | [Ollama / LM Studio](phase-01/lesson-07-ollama-lm-studio.md) | Local LLM serving, quantised model formats, and REST APIs. Run models on your own hardware for fast iteration and privacy.                              |
| 8   | [Pandas](phase-01/lesson-08-pandas.md)                       | DataFrames, merges, groupby, and time-series ops. The bridge from raw CSVs to model-ready features in most pipelines.                                   |
| 9   | [Polars](phase-01/lesson-09-polars.md)                       | Lazy query plans, columnar Arrow layout, and parallel execution. Outperforms Pandas on large files by reading only the columns each query needs.        |

---

## Phase 2 — Core Models

> Go from the oldest formal ML technique all the way to deep networks.
> The arc: understand how models learn (regression → trees → boosting),
> understand geometry (k-means → PCA), then build the deep learning stack
> from first principles (PyTorch → MLP → backprop → AdamW → CNN → RNN).

```
Classical ML track                Deep Learning track
─────────────────                 ────────────────────
Regression                        PyTorch Basics
    │                                    │
Decision Trees                          MLP
    │                                    │
XGBoost / LightGBM               Backpropagation
                                         │
Unsupervised track                     AdamW
──────────────────                       │
K-Means                              CNN / ResNet
    │                                    │
   PCA                                 RNNs
```

| #   | Lesson                                                       | What You Learn                                                                                                                                                                |
| --- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Regression](phase-02/lesson-01-regression.md)               | Least-squares fitting, normal equations, regularisation (Ridge/Lasso), and the bias-variance tradeoff — the foundation every other supervised method extends.                 |
| 2   | [Decision Trees](phase-02/lesson-02-decision-trees.md)       | Greedy splits via information gain / Gini impurity, pruning, and the overfit–prune cycle. The atomic unit inside random forests and gradient boosting.                        |
| 3   | [XGBoost / LightGBM](phase-02/lesson-03-xgboost-lightgbm.md) | Functional gradient descent: each new tree corrects the residuals of the current ensemble. Newton-step leaf values, column/row subsampling, and histogram binning.            |
| 4   | [K-Means](phase-02/lesson-04-k-means.md)                     | Lloyd's fixed-point iteration (assign → update centroid → repeat), k-means++ initialisation, elbow method. Foundation of vector quantisation codebooks in embedding search.   |
| 5   | [PCA](phase-02/lesson-05-pca.md)                             | SVD finds the axes of greatest variance; truncated SVD gives the low-rank approximation. Used for visualisation, noise filtering, whitening, and compression.                 |
| 6   | [PyTorch Basics](phase-02/lesson-06-pytorch-basics.md)       | Tensors, device placement, the autograd tape, `.backward()`, and the module API. Define-by-run means the graph is built as Python executes — debugging is natural.            |
| 7   | [MLPs](phase-02/lesson-07-mlps.md)                           | Stacked affine + nonlinear layers, activation functions (ReLU, GELU, SiLU), batch normalisation, and dropout. The Universal Approximation Theorem motivates depth.            |
| 8   | [Backpropagation](phase-02/lesson-08-backprop.md)            | Reuse intermediate activations so one backward pass computes every parameter gradient in O(forward) time. Werbos (1974) → Rumelhart et al. (1986) history.                    |
| 9   | [AdamW](phase-02/lesson-09-adamw.md)                         | Exponential moving averages of gradient (momentum) and squared gradient (scale) give per-parameter adaptive rates. Decoupled weight decay fixes Adam's L2 regularisation bug. |
| 10  | [CNNs / ResNet](phase-02/lesson-10-cnns-resnet.md)           | Local connectivity and weight sharing exploit spatial structure; residual connections ($y = F(x) + x$) solve the degradation problem for very deep networks.                  |
| 11  | [RNNs](phase-02/lesson-11-rnns.md)                           | Rolling hidden state for sequential data; LSTM/GRU gates mitigate vanishing gradients. Understanding why long-range dependencies still fail motivates transformers.           |

---

## Phase 3 — Transformers & Modern Architectures

> The architecture revolution. Self-attention allows every token to interact with every other
> in one shot, replacing the position-by-position bottleneck of RNNs.
> This phase covers the mechanism, the two dominant families, the linear-time
> alternative, and multimodal systems.

```
Self-Attention  (Q·Kᵀ / √d) · V
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
BERT-style Encoder               GPT / Llama Decoder
(bidirectional, MLM)             (causal, next-token pred)
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
          Multimodality
          ViT (images as patches)
          Stable Diffusion (latent denoising)
          Flux (flow matching)

SSMs / Mamba / Jamba  ──►  linear-time alternative to attention
```

| #   | Lesson                                                                                                       | What You Learn                                                                                                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Self-Attention](phase-03/lesson-01-self-attention.md)                                                       | Project tokens into Q, K, V; compute $\text{softmax}(QK^T/\sqrt{d_k})V$; stack as multi-head attention. Quadratic memory cost in sequence length is the key tradeoff.                       |
| 2   | [BERT vs GPT / Llama](phase-03/lesson-02-bert-vs-gpt-llama.md)                                               | Bidirectional masked language modelling (BERT) for understanding tasks vs. causal left-to-right prediction (GPT/Llama) for generation. Choosing the wrong family wastes fine-tuning budget. |
| 3   | [SSMs — Mamba / Jamba](phase-03/lesson-03-ssms-mamba-jamba.md)                                               | State-space models compress history into a recurrent state with linear-time scanning. Mamba adds input-selective updates; Jamba mixes SSM layers with sparse attention and MoE routing.     |
| 4   | [Multimodality — ViT, Stable Diffusion, Flux](phase-03/lesson-04-multimodality-vit-stable-diffusion-flux.md) | Treat images as non-overlapping patch tokens (ViT); denoise in a learned latent space conditioned on text (Stable Diffusion); replace DDPM with continuous flow matching (Flux).            |

---

## Phase 4 — Applied Generative AI & Engineering

> Turn a capable model into a reliable system.
> Prompts are programs (DSPy), retrieval is routing + ranking (Advanced RAG),
> models need structured external actions (Tool Use / MCP), and hard problems
> deserve more inference compute (Reasoning Models).

```
User query
    │
    ▼
Systematic Prompting (DSPy)     ◄─── compile prompts against a metric
    │
    ▼
Advanced RAG
  ├── Semantic routing ──► pick the right collection / tool
  ├── GraphRAG         ──► expand from retrieved nodes via relationships
  └── Reranking        ──► cross-encoder rescores top-k candidates
    │
    ▼
Tool Use / MCP              ◄─── function schemas, validation, observability
    │
    ▼
Reasoning Models (o1, R1)   ◄─── allocate more test-time compute for hard steps
```

| #   | Lesson                                                                         | What You Learn                                                                                                                                                                                                             |
| --- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Systematic Prompting — DSPy](phase-04/lesson-01-systematic-prompting-dspy.md) | Prompts are part of the model program. DSPy declares modules and optimises prompt structure, reasoning style, and few-shot examples against a measurable metric — no more manual rewriting.                                |
| 2   | [Advanced RAG](phase-04/lesson-02-advanced-rag.md)                             | Naive nearest-chunk retrieval breaks on heterogeneous corpora. Semantic routing directs queries to the right pipeline; GraphRAG expands through entity relationships; cross-encoder reranking rescores for true relevance. |
| 3   | [Tool Use / MCP](phase-04/lesson-03-tool-use-mcp.md)                           | Models call external functions via typed JSON schemas. The Model Context Protocol standardises tool discovery so tools and models interoperate across clients and servers without bespoke integrations.                    |
| 4   | [Reasoning Models](phase-04/lesson-04-reasoning-models.md)                     | Chain-of-thought asks for intermediate steps; o1 and DeepSeek-R1 go further by training on process-reward signals. Visible reasoning traces are not a correctness guarantee — they can still be wrong and expensive.       |

---

## Phase 5 — Customisation & Training

> Adapt large pretrained models cheaply, align them to human preferences,
> and expand training data without more human annotation.

```
Pretrained base model
        │
        ├─── PEFT (LoRA / QLoRA / DoRA)
        │       Freeze base weights; train tiny low-rank adapters.
        │       QLoRA keeps the frozen base in 4-bit NF4.
        │       DoRA splits each weight update into magnitude + direction.
        │
        ├─── Alignment (RLHF → DPO)
        │       Collect preference pairs (chosen vs. rejected).
        │       DPO turns that into a supervised classification loss —
        │       no reward model, no PPO rollouts needed.
        │
        └─── Synthetic Data & Distillation
                Teacher generates instruction-response pairs.
                Student trains on teacher's soft probability outputs,
                not just hard labels — transfers richer knowledge.
```

| #   | Lesson                                                                                        | What You Learn                                                                                                                                                                                                                                                           |
| --- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | [PEFT — LoRA / QLoRA / DoRA](phase-05/lesson-01-peft-lora-qlora-dora.md)                      | Most task adaptation is a small correction to pretrained weights, not a full rewrite. LoRA adds a rank-$r$ update matrix; QLoRA loads the frozen base in 4-bit NF4 while training the adapters in BF16; DoRA separates magnitude from direction for better expressivity. |
| 2   | [Alignment & RLHF / DPO](phase-05/lesson-02-alignment-rlhf-dpo.md)                            | RLHF trains a reward model on human preferences then optimises policy via PPO. DPO reparameterises the objective so preference learning reduces to a binary cross-entropy loss on chosen vs. rejected pairs — simpler and stabler.                                       |
| 3   | [Synthetic Data & Distillation](phase-05/lesson-03-synthetic-data-generation-distillation.md) | Teacher models create additional training examples; students learn from soft token distributions (knowledge distillation), not hard labels. Bad synthetic data amplifies teacher bias — data quality evaluation is not optional.                                         |

---

## Phase 6 — Multi-Agent Systems

> Build systems where multiple LLM-powered agents collaborate, maintain durable
> state across long workflows, and hand off to humans when the cost of a wrong
> action is too high.

```
Incoming task
      │
      ▼
Agentic Framework  (LangGraph / AutoGen / CrewAI)
  ├── Planner agent   ──► decomposes goal into sub-tasks
  ├── Executor agent  ──► calls tools, writes code, fetches data
  └── Critic agent    ──► verifies results before committing
      │
      ▼
State Management
  ├── Ephemeral context   (in-memory, current turn)
  ├── Durable workflow    (checkpointed, survives restart)
  └── External record     (database, source of truth)
      │
      ▼
Human-in-the-Loop gate
  ├── Auto-approve  ──► low-risk, high-confidence actions
  ├── Human review  ──► high-risk or uncertain actions
  └── Escalate      ──► ambiguous or out-of-scope tasks
```

| #   | Lesson                                                         | What You Learn                                                                                                                                                                                                                                                                           |
| --- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | [Agentic Frameworks](phase-06/lesson-01-agentic-frameworks.md) | Multi-step reasoning is workflow orchestration, not a longer prompt. LangGraph models agent behaviour as an explicit state graph; AutoGen uses agent conversation; CrewAI assigns role-based crews. Many tasks are better solved with one good tool call than with five agents talking.  |
| 2   | [State Management](phase-06/lesson-02-state-management.md)     | Without explicit state, agents forget context, duplicate work, and become impossible to debug. Separate short-lived working memory from durable checkpoint facts and external system-of-record data. Design for recovery, branching, and audit trails from the start.                    |
| 3   | [Human-in-the-Loop](phase-06/lesson-03-hitl.md)                | Some actions should never be taken without review: irreversible writes, financial transactions, public communications. Place explicit approval gates at high-risk steps. Too little oversight creates errors at scale; too much destroys the speed advantage that made the agent useful. |

---

## Phase 7 — MLOps, Security & Edge Deployment

> Put everything into production reliably.
> Maximise GPU utilisation (vLLM), reduce memory for edge hardware (quantisation),
> defend against model-specific attack vectors (AI security),
> and package everything for repeatable deployment (Docker / K8s / Ray).

```
Trained model
      │
      ├─── High-Throughput Serving (vLLM)
      │       Paged attention: KV cache lives in non-contiguous pages.
      │       Continuous batching: new requests join mid-flight batches.
      │       Result: near-100% GPU utilisation across many concurrent users.
      │
      ├─── Quantization (GGUF / AWQ)
      │       GGUF: llama.cpp-compatible format for CPU / edge inference.
      │       AWQ: calibration identifies salient weight channels and
      │            protects them during 4-bit quantisation.
      │
      ├─── AI Security
      │       Prompt injection: untrusted input hijacks model instructions.
      │       Data exfiltration: model leaks context through tool calls.
      │       NeMo Guardrails: structured rail policies, not just prose.
      │       Defence-in-depth: input validation + output filtering + audit.
      │
      └─── Orchestration (Docker → Kubernetes → Ray)
              Docker:     reproducible image = reproducible runtime.
              Kubernetes: replica sets, rolling deploys, health probes.
              Ray:        distributed Python for training & serving jobs
                          that don't fit on one node.
```

| #   | Lesson                                                                                   | What You Learn                                                                                                                                                                                                                                                           |
| --- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | [High-Throughput Serving — vLLM](phase-07/lesson-01-high-throughput-serving-vllm.md)     | Generation is a scheduling problem. Paged attention eliminates KV cache fragmentation; continuous batching keeps GPUs saturated across concurrent requests instead of waiting for each to finish.                                                                        |
| 2   | [Quantization — GGUF / AWQ](phase-07/lesson-02-quantization-gguf-awq.md)                 | Most weights don't need full 16-bit precision at inference time. GGUF packages quantised models for llama.cpp-style CPU/edge runtimes; AWQ identifies and protects the small subset of weight channels that matter most for quality.                                     |
| 3   | [AI Security](phase-07/lesson-03-ai-security.md)                                         | Prompt injection, jailbreaks, and tool misuse are model-specific attack surfaces on top of ordinary software risk. NeMo Guardrails enforces structured rail policies at the colang layer. No single control is sufficient — defence in depth is required.                |
| 4   | [Orchestration — Docker / K8s / Ray](phase-07/lesson-04-orchestration-docker-k8s-ray.md) | Docker makes the runtime portable and reproducible. Kubernetes manages replica sets, rolling deploys, and self-healing across a cluster. Ray coordinates Python workloads — distributed training, hyperparameter search, and scalable model serving — across many nodes. |

---

## Lesson Structure

Every file in this repo follows the same six-section format:

```
1. The Core Intuition       Why this exists, what predecessor it replaced, and
                            where the common hype about it fails.

2. Theoretical Underpinning Key equations, derivations, and proofs with
                            intuitive explanations of each symbol.

3. From-Scratch Implementation
                            NumPy / pure Python code that keeps the math
                            visible — no magic, no framework abstractions.

4. Production Implementation
                            PyTorch / HuggingFace / framework code you would
                            actually ship, with the non-obvious choices called out.

5. Pitfalls & Pro-Tips      Common mistakes, their symptoms, and concrete fixes.

6. Knowledge Check          Conceptual questions and coding challenges at the
                            level of a technical interview or paper review.
```
