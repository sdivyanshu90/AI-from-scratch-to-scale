# Human-in-the-Loop (HITL) Design

Human-in-the-loop (HITL) design places explicit review, correction, or
approval steps into agent workflows at the points where the cost of an
autonomous mistake exceeds the cost of human intervention. HITL is not a
fallback for broken agents — it is a principled engineering decision about
where the human provides irreplaceable value: judgment on ambiguous cases,
accountability for high-stakes actions, and ground-truth correction for
learning systems.

## 1. The Core Intuition (The "Why")

Fully autonomous execution fails whenever a wrong action is expensive to
reverse: sending a message to 10,000 users, executing a financial
transaction, modifying production infrastructure. The naive response is to
add a human review step to everything — but that destroys the speed
advantage of automation.

Kahneman (2011) distinguishes System 1 (fast, automatic, pattern-matching)
from System 2 (slow, deliberate, rule-following) thinking. Autonomous LLM
agents excel at System 1 tasks; they struggle at edge cases requiring
System 2 judgment. HITL routes exactly those edge cases to humans.

The engineering insight: model **uncertainty** or **impact** as a scalar
score. Route to human review when the score crosses a threshold; automate
everywhere else. This makes HITL a control-flow mechanism, not a prompt
instruction.

## 2. The Theoretical Underpinning

### Gating Model

Let the agent propose action $a_t$ in state $s_t$. Associate each proposal
with a risk-or-uncertainty score $u_t \in [0, 1]$:

$$u_t = \sigma\!\left(\text{risk\_model}(a_t, s_t)\right)$$

A binary gating policy routes to human review when uncertainty exceeds
threshold $\tau$:

$$g_t = \mathbf{1}[u_t > \tau]$$

When $g_t = 1$, a human reviewer applies a correction function:

$$\tilde{a}_t = H(a_t, s_t)$$

When $g_t = 0$, the action executes directly: $\tilde{a}_t = a_t$.

### Throughput-Risk Trade-off

If fraction $p$ of actions require human review taking time $T_H$, while
autonomous execution takes $T_A \ll T_H$, the mean action latency is:

$$\bar{L} = p \cdot T_H + (1 - p) \cdot T_A$$

Calibrating $\tau$ adjusts $p$. Higher $\tau$ reduces $p$ and lowers
latency but increases the rate of unreviewed risky actions.

### Interrupt / Resume Semantics

In LangGraph, HITL is implemented via `interrupt()` — execution pauses,
the state is serialised to a checkpoint, and the workflow waits for an
external signal (human approval). On resume:

$$s_t \xrightarrow{\text{interrupt}} \text{checkpoint} \xrightarrow{H} \tilde{s}_t \xrightarrow{\text{resume}} s_{t+1}$$

This requires durable state management: the checkpoint must survive
indefinitely until the human responds.

### Active Learning Integration

HITL correction is high-quality training signal. If the human edits
action $a_t$ to $a_t^*$, the pair $(s_t, a_t^*)$ is a labelled example:

$$\mathcal{D}_\text{HITL} = \{(s_t, a_t^*) : g_t = 1\}$$

Ziegler et al. (2019) showed that even a small amount of such preference
data substantially improves policy alignment.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ActionProposal:
    """A proposed agent action awaiting possible review.

    Args:
        action:     Action name or serialised command.
        confidence: Model confidence in [0, 1].
        context:    Relevant context for the reviewer.
        metadata:   Arbitrary key-value pairs for auditing.
    """
    action:     str
    confidence: float
    context:    str
    metadata:   dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewDecision:
    """Auditable record of a human review decision."""
    proposal:      ActionProposal
    approved:      bool
    edited_action: str | None    # Human override, if any
    reviewer_id:   str
    reviewed_at:   float         # Unix timestamp


def risk_score(proposal: ActionProposal) -> float:
    """Compute a risk score for a proposed action.

    Combines model uncertainty with action-type risk.
    In production this would call a trained risk classifier.

    Args:
        proposal: Action proposal to score.

    Returns:
        Risk score in [0, 1]. Higher means more risky.
    """
    uncertainty = 1.0 - proposal.confidence
    high_risk   = proposal.action in {"delete", "send_email", "transfer_funds"}
    action_risk = 0.3 if high_risk else 0.0
    return min(1.0, uncertainty + action_risk)


def execute_with_hitl(
    proposal:    ActionProposal,
    threshold:   float,
    reviewer_fn: Callable[[ActionProposal], ReviewDecision],
) -> tuple[str, bool, ReviewDecision | None]:
    """Execute a proposal, routing to human review if risk exceeds threshold.

    Args:
        proposal:    Action to execute.
        threshold:   Risk threshold above which human review is required.
        reviewer_fn: Callback that presents the proposal to a human and
                     returns their ReviewDecision.

    Returns:
        (final_action, was_reviewed, review_decision_or_None)
    """
    score = risk_score(proposal)
    if score > threshold:
        decision = reviewer_fn(proposal)
        final    = decision.edited_action or proposal.action
        return final, True, decision
    return proposal.action, False, None


class AuditLog:
    """Append-only record of all HITL decisions for compliance."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(self, decision: ReviewDecision) -> None:
        self._entries.append({
            "action":      decision.proposal.action,
            "approved":    decision.approved,
            "edited_to":   decision.edited_action,
            "reviewer":    decision.reviewer_id,
            "reviewed_at": decision.reviewed_at,
        })

    def export(self) -> list[dict[str, Any]]:
        return list(self._entries)
```

## 4. Implementation: Production-Grade (LangGraph interrupt)

```python
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt


class ApprovalState(TypedDict):
    """State for a workflow requiring human approval before execution."""
    task:            str
    proposed_action: str
    risk_score:      float
    approved:        bool
    final_action:    str


REVIEW_THRESHOLD = 0.5


def propose_action(state: ApprovalState) -> ApprovalState:
    """Generate an action proposal and compute its risk score."""
    proposed = f"execute: {state['task']}"
    score    = 0.8 if "delete" in state["task"] else 0.2
    return {**state, "proposed_action": proposed, "risk_score": score}


def maybe_interrupt(state: ApprovalState) -> ApprovalState:
    """Interrupt for human review if the risk score exceeds the threshold.

    interrupt() pauses execution, serialises state to a checkpoint, and
    waits for the caller to provide a decision via graph.update_state().
    The workflow resumes from the exact point of interruption.

    Args:
        state: Current workflow state.

    Returns:
        State updated with the approval decision.
    """
    if state["risk_score"] > REVIEW_THRESHOLD:
        decision = interrupt({
            "proposed_action": state["proposed_action"],
            "risk_score":      state["risk_score"],
            "reason":          "Risk exceeds auto-execution threshold.",
        })
        approved = bool(decision.get("approved"))
        final    = decision.get("override") or state["proposed_action"]
        return {**state, "approved": approved, "final_action": final}
    return {**state, "approved": True, "final_action": state["proposed_action"]}


def execute_action(state: ApprovalState) -> ApprovalState:
    """Execute the (possibly human-modified) final action."""
    if not state["approved"]:
        return {**state, "final_action": "[REJECTED BY REVIEWER]"}
    return state


def build_hitl_graph() -> Any:
    """Build a LangGraph workflow with HITL approval."""
    g = StateGraph(ApprovalState)
    g.add_node("propose", propose_action)
    g.add_node("review",  maybe_interrupt)
    g.add_node("execute", execute_action)
    g.set_entry_point("propose")
    g.add_edge("propose", "review")
    g.add_edge("review",  "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=MemorySaver())
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Sending every action through human review. When approval
  latency is 5 minutes and 95% of actions are low-risk, the review queue
  becomes a bottleneck that eliminates the value of automation.

  **Fix:** Calibrate $\tau$ by measuring the historical distribution of
  risk scores. Target a review rate matching your reviewer capacity (e.g.,
  review only the top 5% by risk score).

- **Mistake:** Presenting the reviewer with only the proposed action, without
  the context that led to it. The reviewer cannot make a good decision
  without the task, the agent reasoning, and the data used.

  **Fix:** The approval payload should include: the proposed action, the
  triggering user request, key evidence or retrieved documents, and a
  confidence/risk score with a brief explanation.

- **Mistake:** Logging that approval happened without recording what action
  was actually executed. Compliance audits require the full decision trail.

  **Fix:** Log the proposal, the approval/rejection decision, any human
  edits to the action, the reviewer identity, and the timestamp in an
  append-only audit log that cannot be retroactively edited.

- **Mistake:** Using `interrupt()` without durable checkpointing. If the
  process restarts while waiting for human approval, the workflow is lost.

  **Fix:** Always pair `interrupt()` with a persistent checkpointer
  (`PostgresSaver` or `RedisSaver`). Test recovery by simulating a crash
  during the approval wait.

- **Mistake:** Discarding HITL correction data after each session. Human
  corrections are high-quality labelled examples for improving the policy.

  **Fix:** Store every $(s_t, a_t^\text{proposed}, a_t^*)$ triple in a
  labelled dataset and use it for periodic DPO fine-tuning or preference
  alignment of the agent model.

## 6. Knowledge Check

1. **Conceptual:** Express the throughput-risk trade-off mathematically.
   If 20% of actions require human review (3 min each) and autonomous
   execution takes 2 seconds, what is the mean action latency? How does
   this change if you lower the review rate to 10%?

2. **Conceptual:** What does LangGraph's `interrupt()` actually do at the
   implementation level? How does it pause graph execution, and what must
   be in place for the graph to resume correctly after a process restart?

3. **Coding challenge:** Implement `execute_with_hitl` from scratch. Define
   a `risk_score` function that combines model confidence and action type.
   Write a mock reviewer that approves high-confidence actions (> 0.7)
   and rejects others with an edited fallback. Verify that the audit log
   records every review decision, including reviewer identity and timestamp.

## References

1. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
2. Amershi, S., Weld, D., Vorvoreanu, M., et al. (2019). Software engineering for machine learning: A case study. *ICSE 2019*.
3. Ziegler, D. M., Stiennon, N., Wu, J., et al. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.
4. LangGraph (2024). Human-in-the-loop. https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/.
