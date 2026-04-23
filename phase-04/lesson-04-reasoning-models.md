# Reasoning Models: Process Rewards, MCTS, and RLVR

Reasoning models are LLMs specifically trained or prompted to generate
explicit multi-step reasoning before producing a final answer. They
represent a shift from scaling model size (pretraining compute) to scaling
inference compute: allocating more tokens (and thus more forward passes) to
deliberation before answering. OpenAI's o1, DeepSeek-R1, and related systems
achieve state-of-the-art performance on mathematical olympiad and competitive
programming benchmarks by learning to "think longer" on hard problems.

## 1. The Core Intuition (The "Why")

GPT-4 answers in a single pass: each token is generated from a single
forward pass through the network. For a question like "Prove that there are
infinitely many prime numbers", a single pass is insufficient — the correct
answer requires constructing a mathematical proof step by step, checking
each step, and backtracking if a step fails.

Humans solve hard problems through deliberate, extended reasoning. Snell et
al. (2024) showed empirically that at test time, allocating more compute
(more tokens for reasoning) can match the performance of training a model
that is many times larger. This is the **inference-time compute** hypothesis:
for reasoning-intensive tasks, thinking more is better than being bigger.

The key technical challenge is teaching the model *when* and *how* to reason
deeply, and providing a training signal that rewards correct reasoning
processes, not just correct final answers.

## 2. The Theoretical Underpinning

### Outcome Reward Models (ORM) vs Process Reward Models (PRM)

An **Outcome Reward Model (ORM)** assigns a scalar reward to the final
answer only:

$$R_\text{ORM}(x, y) = \text{score}(y_\text{final} \mid x)$$

This is binary for verifiable tasks: 1 if correct, 0 if not. ORMs are easy
to train (labels are cheap) but provide no signal about which reasoning
steps were good or bad.

A **Process Reward Model (PRM)** assigns a reward to each reasoning step:

$$R_\text{PRM}(x, y) = \sum_{t=1}^{T} r_t(s_t)$$

where $s_t$ is the $t$-th reasoning step. Lightman et al. (2023) showed that
PRMs outperform ORMs on math reasoning tasks because they provide dense
training signal: even wrong final answers may contain useful intermediate steps.

### Monte Carlo Tree Search (MCTS) for Reasoning

MCTS explores the tree of possible reasoning steps by:
1. **Selection**: traverse the tree using UCB1 to balance exploration vs exploitation
2. **Expansion**: expand the selected node by generating the next reasoning step
3. **Simulation**: roll out to a terminal state (final answer)
4. **Backpropagation**: update node values with the terminal reward

$$\text{UCB1}(v) = \bar{r}(v) + c \sqrt{\frac{\ln N(v_\text{parent})}{N(v)}}$$

where $\bar{r}(v)$ is the mean reward from node $v$, $N(v)$ is the visit
count, and $c$ is the exploration constant.

MCTS allows the model to explore multiple reasoning paths in parallel and
select the best one at inference time, trading latency for accuracy.

### RLVR: Reinforcement Learning from Verifiable Rewards

DeepSeek-AI (2025) introduced RLVR, training DeepSeek-R1 using GRPO
(Group Relative Policy Optimisation) with a reward function defined on
verifiable outcomes (math answers, code that passes tests):

$$R(s, a) = \begin{cases} +1 & \text{if answer is correct} \\ -1 & \text{if answer is wrong or format is broken} \end{cases}$$

GRPO computes the policy gradient relative to a group of outputs:

$$\mathcal{L}_\text{GRPO}(\theta) = -\mathbb{E}_{q \sim P, \{o_i\}_{i=1}^G \sim \pi_\theta^{\text{old}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G A_i \min\!\left(\frac{\pi_\theta(o_i|q)}{\pi_\theta^{\text{old}}(o_i|q)}, \text{clip}(\cdot, 1\pm\epsilon)\right) \right]$$

where $A_i = r_i - \text{mean}(\{r_j\})$ is the advantage of output $i$
relative to the group mean reward.

The striking result: models trained with RLVR on math problems spontaneously
developed **self-verification** and **backtracking** behaviours — they
learned to check their own work without being explicitly taught to.

### Chain-of-Thought Scaling Laws

Wei et al. (2022) showed that CoT only emerges in models above ~100B
parameters. However, Snell et al. (2024) showed that with sufficient
inference-time compute budget $T$ (number of tokens), smaller models with
CoT can match larger models without CoT:

$$\text{accuracy}(n_\text{params}, T) \approx f\!\left(n_\text{params} \cdot T^\alpha\right)$$

This is the basis for **best-of-N** sampling: generate $N$ answers, verify
each with an ORM or PRM, return the best one. For verifiable tasks (math,
code), best-of-N can close the gap between a 7B and a 70B model at the
cost of $N\times$ more compute.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import math
import random
from typing import Callable


class MCTSNode:
    """A node in the MCTS tree representing a partial reasoning chain."""

    def __init__(
        self,
        state:  str,
        parent: MCTSNode | None = None,
    ) -> None:
        self.state    = state     # Partial reasoning text
        self.parent   = parent
        self.children: list[MCTSNode] = []
        self.visits   = 0
        self.value    = 0.0       # Accumulated reward

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb1(self, c: float = 1.41) -> float:
        """Upper Confidence Bound for tree exploration."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return (self.value / self.visits) + c * math.sqrt(
            math.log(parent_visits) / self.visits
        )


def mcts_reasoning(
    question:    str,
    llm_step:    Callable[[str], list[str]],
    reward_fn:   Callable[[str], float],
    n_iters:     int = 50,
    rollout_len: int = 5,
) -> str:
    """Use MCTS to search for the best reasoning chain.

    Args:
        question:    The question to answer.
        llm_step:    Function(context) -> list of next reasoning steps.
        reward_fn:   Function(full_chain) -> float reward.
        n_iters:     Number of MCTS iterations.
        rollout_len: Max steps per rollout simulation.

    Returns:
        The best reasoning chain found.
    """
    root = MCTSNode(state=f"Question: {question}\n")

    for _ in range(n_iters):
        # Selection: traverse to the best leaf
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda c: c.ucb1())

        # Expansion: generate next steps
        next_steps = llm_step(node.state)
        for step in next_steps:
            node.children.append(MCTSNode(state=node.state + step, parent=node))

        # Simulation: random rollout from a random child
        if node.children:
            sim_node = random.choice(node.children)
        else:
            sim_node = node
        sim_state = sim_node.state
        for _ in range(rollout_len):
            steps = llm_step(sim_state)
            if not steps:
                break
            sim_state += random.choice(steps)

        # Evaluation
        reward = reward_fn(sim_state)

        # Backpropagation: update all ancestors
        backtrack = sim_node
        while backtrack is not None:
            backtrack.visits += 1
            backtrack.value  += reward
            backtrack         = backtrack.parent

    # Return the path of highest-value children
    node = root
    path = root.state
    while node.children:
        node  = max(node.children, key=lambda c: c.value / max(c.visits, 1))
        path += node.state
    return path
```

## 4. Implementation: Production-Grade (Best-of-N with PRM)

```python
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class ProcessRewardModel:
    """Wrapper for a HuggingFace process reward model.

    A PRM scores each step in a chain-of-thought. We aggregate step scores
    by taking the minimum (weakest-link): the chain is only as strong as
    its weakest step.
    """

    def __init__(self, model_name: str = "openai/process-reward-model") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score_step(self, question: str, steps_so_far: str) -> float:
        """Score a partial reasoning chain."""
        text   = f"Question: {question}\n\nReasoning: {steps_so_far}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logits = self.model(**inputs).logits
        return float(torch.softmax(logits, dim=-1)[0, 1])   # Probability of 'good' step

    def score_chain(self, question: str, chain: list[str]) -> float:
        """Score a full chain by taking the minimum step score (weakest-link)."""
        scores = [self.score_step(question, "\n".join(chain[:i+1])) for i in range(len(chain))]
        return min(scores) if scores else 0.0


def best_of_n(
    question:  str,
    llm_call:  callable,
    prm:       ProcessRewardModel,
    n:         int = 16,
) -> str:
    """Generate N candidate answers and return the one with the best PRM score.

    This is the simplest form of inference-time compute scaling:
    trade N-fold more compute for higher accuracy on verifiable tasks.

    Args:
        question: The question to answer.
        llm_call: Function(question) -> (chain: list[str], answer: str).
        prm:      Process reward model for scoring reasoning chains.
        n:        Number of candidate answers to generate.

    Returns:
        The answer with the highest PRM score.
    """
    best_answer, best_score = "", float("-inf")
    for _ in range(n):
        chain, answer = llm_call(question)
        score = prm.score_chain(question, chain)
        if score > best_score:
            best_score  = score
            best_answer = answer
    return best_answer
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using an Outcome Reward Model to train on tasks where the
  ground truth is expensive to verify (e.g., open-ended writing). ORMs
  trained on correct/incorrect labels only work reliably for tasks with
  clear verifiable answers (math, code tests).

  **Fix:** For open-ended tasks, use a PRM or RLHF with human preference
  labels. Reserve ORM/RLVR for verifiable tasks.

- **Mistake:** Generating very long reasoning chains and charging the
  user for all tokens. Reasoning models can generate 10,000+ tokens per
  query; this may cost 20x more than a direct answer.

  **Fix:** Set a `max_reasoning_tokens` budget. For simple queries,
  skip extended reasoning. Classify query difficulty first.

- **Mistake:** Assuming best-of-N always outperforms a single longer
  chain. Best-of-N with a weak reward model can select a confidently-wrong
  answer.

  **Fix:** Validate your reward model on a held-out set before deploying
  best-of-N. Prefer verified reward functions (unit test pass rate for
  code, symbolic math checkers for math).

- **Mistake:** Using MCTS at production inference time for latency-sensitive
  applications. A single MCTS query with 50 iterations and 5 steps per
  rollout requires 250+ LM forward passes.

  **Fix:** Use MCTS during training to generate high-quality reasoning
  traces (as in AlphaProof). At inference time, use the trained model's
  greedy decoding or best-of-N.

- **Mistake:** Fine-tuning on reasoning traces from a stronger teacher
  model and expecting the student to generalise to new problem types.
  The student may learn to mimic the teacher's format without learning
  to reason.

  **Fix:** Mix reasoning traces with novel problems the teacher has never
  seen. Evaluate on distribution-shifted test sets, not just problems
  similar to training.

## 6. Knowledge Check

1. **Conceptual:** Explain the difference between an Outcome Reward Model
   and a Process Reward Model. For which types of tasks is each appropriate?
   What is the "weakest-link" property of PRM scoring?

2. **Conceptual:** What is the inference-time compute hypothesis? How does
   best-of-N sampling trade computation for accuracy? What limits the
   effectiveness of scaling inference compute indefinitely?

3. **Coding challenge:** Implement best-of-N from scratch: call an LLM
   10 times on a math problem, extract the final numeric answer from each
   response, and return the majority-vote answer. Compare this to a single
   greedy response on 10 math problems from the MATH benchmark.

## References

1. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*. arXiv:2201.11903.
2. Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's verify step by step. *ICLR 2024*. arXiv:2305.20050. *(Process Reward Models.)*
3. Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv:2408.03314.
4. DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948. *(GRPO / RLVR.)*
5. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature* 529, 484–489. *(MCTS foundations.)*
