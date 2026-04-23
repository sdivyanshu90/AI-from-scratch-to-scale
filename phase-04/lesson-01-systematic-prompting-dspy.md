# Systematic Prompting and DSPy

Prompting is the practice of constructing inputs that elicit desired
outputs from a language model without modifying its weights. Systematic
prompting — using structured techniques like Chain-of-Thought, few-shot
exemplars, and self-consistency — consistently improves LLM performance on
complex tasks. DSPy (Khattab et al., 2023) goes further, treating prompt
engineering as an optimisation problem: given a metric, DSPy automatically
discovers the best instructions and few-shot exemplars for your task.

## 1. The Core Intuition (The "Why")

GPT-3 (Brown et al., 2020) demonstrated that a single prompt could make a
model perform tasks it was never explicitly trained on (few-shot learning).
However, prompt performance varied wildly with small wording changes.

Wei et al. (2022) showed that asking a model to "think step by step" before
answering — **Chain-of-Thought prompting** — dramatically improved performance
on multi-step reasoning tasks. The mechanism: by generating intermediate
reasoning steps, the model allocates more computational tokens to the
problem, effectively increasing the "reasoning depth" available.

Khattab et al. (2023) addressed the fragility of hand-crafted prompts with
DSPy: a framework that treats prompts as learnable program parameters and
optimises them using a feedback loop (compiling the program). This moved
prompting from an art to an engineering discipline.

## 2. The Theoretical Underpinning

### In-Context Learning (ICL)

A few-shot prompt provides $k$ (input, output) demonstrations followed by
the test input:

$$\text{Prompt} = [(x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k), x_{\text{test}}]$$

The model's output is conditioned on all preceding context. In-context
learning is not gradient descent — the weights are not updated. The
demonstrations act as a **prior** that shifts the output distribution.

### Chain-of-Thought (CoT)

Standard prompting: $P(y \mid x)$

Chain-of-Thought: $P(y \mid x, z)$ where $z$ is a chain of reasoning steps

$$P(y \mid x) = \sum_z P(y \mid z, x) P(z \mid x)$$

By making the model generate $z$ explicitly before $y$, we marginalise over
reasoning paths. Each token generated in $z$ conditions the subsequent tokens,
allowing the model to build up complex answers incrementally.

### Self-Consistency

Wang et al. (2022) showed that sampling multiple reasoning paths and taking
the majority vote of the final answers (self-consistency) outperforms
greedy CoT decoding:

$$y^* = \arg\max_{y} \sum_{z \in \mathcal{Z}} P(z, y \mid x)$$

where $\mathcal{Z}$ is a set of sampled reasoning chains. Empirically,
self-consistency with 40 samples reduces error rates by 10-30% on math
and reasoning benchmarks.

### DSPy: Prompts as Learnable Parameters

DSPy defines a program as a computation graph of **modules** (Predict,
ChainOfThought, ReAct). Each module has a signature: a typed input-output
specification.

The **DSPy compiler** (teleprompter) optimises the program by:
1. Collecting (input, output, metric_score) examples by running the program
   on a small development set.
2. Running a prompt optimiser (e.g., BootstrapFewShot) that selects the
   best few-shot examples by gradient-free search.
3. Optionally, generating or refining instructions for each module.

The optimised program is saved as a compiled JSON that can be loaded at
inference time.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import json
import re
from typing import Any, Callable


def zero_shot_cot(question: str) -> str:
    """Build a zero-shot Chain-of-Thought prompt.

    The phrase "Let's think step by step" was empirically shown to
    elicit reasoning chains in large language models (Kojima et al., 2022).
    """
    return (
        f"Question: {question}\n"
        "Let's think step by step.\n"
        "Reasoning:"
    )


def few_shot_prompt(
    examples: list[dict[str, str]],
    query:    dict[str, str],
    schema:   str = "Question: {question}\nAnswer: {answer}",
) -> str:
    """Build a few-shot prompt by formatting examples using a schema.

    Args:
        examples: List of dicts with keys matching schema placeholders.
        query:    Dict with input keys (answer key is absent).
        schema:   Format string with {key} placeholders.

    Returns:
        Formatted few-shot prompt string.
    """
    parts = []
    for ex in examples:
        parts.append(schema.format(**ex))
    # For the query, only format the input keys
    query_line = schema.split("\n")[0].format(**query)
    parts.append(query_line)
    return "\n\n".join(parts)


class SelfConsistencyAggregator:
    """Sample N CoT responses and return the most common final answer."""

    def __init__(
        self,
        llm_call: Callable[[str], str],
        n_samples: int = 10,
    ) -> None:
        self.llm_call = llm_call
        self.n_samples = n_samples

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from a CoT response.
        Looks for 'The answer is X' pattern.
        """
        match = re.search(r"[Tt]he answer is[:\s]+([^\n.]+)", response)
        return match.group(1).strip() if match else response.strip()

    def __call__(self, prompt: str) -> str:
        """Generate n_samples responses and return the majority answer."""
        answers: dict[str, int] = {}
        for _ in range(self.n_samples):
            response = self.llm_call(prompt)
            answer   = self._extract_answer(response)
            answers[answer] = answers.get(answer, 0) + 1
        return max(answers, key=answers.get)
```

## 4. Implementation: Production-Grade (DSPy)

```python
from __future__ import annotations

import dspy
from dspy.teleprompt import BootstrapFewShot


def setup_dspy(model_name: str = "openai/gpt-4o-mini") -> None:
    """Configure DSPy with a language model backend.

    DSPy supports OpenAI, Anthropic, HuggingFace, and Ollama backends.
    """
    lm = dspy.LM(model_name)
    dspy.configure(lm=lm)


class MultiHopQA(dspy.Module):
    """Multi-hop question answering with Chain-of-Thought.

    DSPy modules define computation graphs. Each Predict/ChainOfThought
    call generates a prompt from its signature and calls the LM.
    The compiler will optimise the instructions and examples for each call.
    """

    def __init__(self) -> None:
        super().__init__()
        # First hop: decompose the question into simpler sub-questions
        self.decompose = dspy.ChainOfThought(
            "question -> sub_questions: list[str]"
        )
        # Second hop: answer each sub-question
        self.answer_sub = dspy.Predict(
            "question, context -> answer: str"
        )
        # Final aggregation
        self.aggregate = dspy.ChainOfThought(
            "question, sub_answers: list[str] -> final_answer: str"
        )

    def forward(self, question: str) -> dspy.Prediction:
        # Decompose the question
        sub_qs = self.decompose(question=question).sub_questions
        # Answer each sub-question
        sub_answers = [
            self.answer_sub(question=q, context="").answer
            for q in sub_qs
        ]
        # Aggregate into final answer
        return self.aggregate(question=question, sub_answers=sub_answers)


def compile_with_bootstrap(
    program:  dspy.Module,
    trainset: list[dspy.Example],
    metric:   callable,
    max_bootstrapped_demos: int = 4,
) -> dspy.Module:
    """Compile (optimise) a DSPy program using BootstrapFewShot.

    BootstrapFewShot generates candidate few-shot examples by running
    the program on the training set, then selects the examples that
    maximise the metric.

    Args:
        program:  The DSPy module to compile.
        trainset: List of training examples.
        metric:   Function(example, prediction) -> float.
        max_bootstrapped_demos: Max few-shot examples per module.

    Returns:
        The compiled (optimised) program.
    """
    teleprompter = BootstrapFewShot(
        metric = metric,
        max_bootstrapped_demos = max_bootstrapped_demos,
    )
    return teleprompter.compile(program, trainset=trainset)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Hand-tuning prompts on the same examples used to evaluate
  them. This is overfitting in prompt space: you end up with a prompt
  that works for those examples but fails on new inputs.

  **Fix:** Maintain a separate held-out evaluation set. Optimise prompts
  only on the development set.

- **Mistake:** Using temperature=0 (greedy decoding) for complex reasoning
  tasks. Greedy decoding commits to the most likely first token, which
  may not lead to the best reasoning path.

  **Fix:** For accuracy-critical tasks, use self-consistency with
  temperature=0.7. For deterministic APIs, use n=10 if the API supports it.

- **Mistake:** Assuming that longer prompts with more examples always
  help. Very long contexts can dilute the model's attention on the actual
  question.

  **Fix:** Use 3-8 well-selected, diverse few-shot examples. Measure
  performance empirically.

- **Mistake:** Not specifying an output format in the prompt. LLMs produce
  free-form text by default; parsing it programmatically is fragile.

  **Fix:** Specify output format explicitly: "Answer in JSON with keys
  'reasoning' and 'answer'." Use DSPy signatures for automatic output
  parsing.

- **Mistake:** Using DSPy without a reliable metric. The compiler needs
  a metric that accurately reflects task quality to bootstrap good examples.

  **Fix:** Define a rigorous metric (exact match, F1, or LLM-as-judge)
  before compiling. Validate the metric on a small manual sample.

## 6. Knowledge Check

1. **Conceptual:** Explain why Chain-of-Thought prompting improves
   performance on multi-step reasoning tasks. Is CoT doing gradient descent?
   What "computation" is the model performing when generating reasoning steps?

2. **Conceptual:** DSPy treats prompt engineering as an optimisation problem.
   What is the objective being optimised, and what are the "parameters"?
   How does this differ from fine-tuning the model weights?

3. **Coding challenge:** Implement self-consistency from scratch: call an
   LLM 10 times with temperature=0.7 on the same CoT prompt, extract the
   final answer from each response using regex, and return the majority
   vote. Evaluate this vs greedy decoding on 5 math word problems.

## References

1. Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *NeurIPS 2020*. arXiv:2005.14165.
2. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*. arXiv:2201.11903.
3. Wang, X., Wei, J., Schuurmans, D., et al. (2022). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*. arXiv:2203.11171.
4. Khattab, O., Singhvi, A., Maheshwari, P., et al. (2023). DSPy: Compiling declarative language model calls into self-improving pipelines. arXiv:2310.03714.
5. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. *NeurIPS 2022*. arXiv:2205.11916.
