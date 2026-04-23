# State Management in Agent Systems

State management is the discipline of keeping agent workflows coherent across
multiple steps, users, tool calls, retries, and potential failures. Without
explicit, durable state, agents forget critical context, duplicate work, and
become impossible to debug or resume. The reducer-event pattern (borrowed from
functional programming and Redux) and event sourcing provide the theoretical
foundation; LangGraph's checkpointers and vector memory stores provide the
production implementation.

## 1. The Core Intuition (The "Why")

Early chatbots managed "state" as an append-only list of messages. This breaks
the moment a workflow branches, pauses for a tool call, requires human approval,
or needs to be resumed after a crash.

The insight from functional state management is that **state transitions should
be explicit, pure functions**: given the current state and an event, produce the
next state. This makes state changes:
- **Reproducible**: the same event sequence always produces the same final state.
- **Debuggable**: you can replay any event log to reconstruct any past state.
- **Resumable**: a checkpoint at any step allows restarting from there.

Inspired by Redux (Abramov, 2015) and event sourcing (Fowler, 2005), modern
agent frameworks like LangGraph adopt this pattern explicitly.

## 2. The Theoretical Underpinning

### The Reducer Model

Let $s_t$ be the workflow state at step $t$ and $e_t$ be the event (node
output or external signal) at that step. A **reducer** is a pure function:

$$s_{t+1} = R(s_t, e_t)$$

The full state at any point can be reconstructed from the initial state
$s_0$ and the complete event log $E_{1:T} = (e_1, \ldots, e_T)$:

$$s_T = R(R(\cdots R(s_0, e_1) \cdots, e_{T-1}), e_T)$$

This is the **event sourcing** pattern: the event log is the source of truth;
the current state is a derived view.

### Checkpointing

For long-running workflows, replaying from $s_0$ is expensive. A checkpoint
stores a compact snapshot $c_k = s_k$ at step $k$, so replay starts from
$c_k$ instead of $s_0$:

$$s_T = R(R(\cdots R(c_k, e_{k+1}) \cdots, e_{T-1}), e_T)$$

The trade-off: more checkpoints mean faster recovery but more storage.
LangGraph's `MemorySaver` and `PostgresSaver` implement this pattern.

### Memory Types in LLM Agents

Agents need three qualitatively different memory stores:

| Type | Description | Lifetime | Example |
|------|-------------|----------|---------|
| **Ephemeral** | In-context window | One turn | Recent messages |
| **Episodic** | Per-session checkpoints | One session | LangGraph state |
| **Semantic** | Long-term vector memory | Persistent | RAG knowledge base |

The distinction matters because different retrieval mechanisms suit each
type: context concatenation for ephemeral, checkpoint restore for episodic,
embedding similarity for semantic.

### LangGraph State Annotations

LangGraph uses Python `Annotated` types to declare how state fields are
updated when multiple nodes write to the same field:

```python
from typing import Annotated
import operator

messages: Annotated[list, operator.add]  # Appends rather than overwrites
```

This is a declarative **merge function** — equivalent to specifying the
reducer $R$ at the field level rather than the state level.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkflowState:
    """Minimal workflow state for an agent run."""
    messages:        list[str]  = field(default_factory=list)
    completed_steps: list[str]  = field(default_factory=list)
    tool_results:    dict[str, Any] = field(default_factory=dict)
    approved:        bool       = False


@dataclass
class Event:
    """A typed event applied to the workflow state."""
    kind:    str           # "message" | "step_done" | "tool_result" | "approval"
    payload: dict[str, Any]


def reduce_state(state: WorkflowState, event: Event) -> WorkflowState:
    """Pure reducer: given state + event, produce next state.

    This function must be pure — no side effects, no I/O.
    The same inputs must always produce the same output.

    Args:
        state: Current workflow state.
        event: Incoming state-change event.

    Returns:
        Updated workflow state.

    Raises:
        ValueError: If the event kind is unknown.
    """
    # Shallow-copy to preserve immutability
    next_state = WorkflowState(
        messages        = list(state.messages),
        completed_steps = list(state.completed_steps),
        tool_results    = dict(state.tool_results),
        approved        = state.approved,
    )
    if event.kind == "message":
        next_state.messages.append(str(event.payload["text"]))
    elif event.kind == "step_done":
        next_state.completed_steps.append(str(event.payload["step"]))
    elif event.kind == "tool_result":
        next_state.tool_results[event.payload["tool"]] = event.payload["result"]
    elif event.kind == "approval":
        next_state.approved = bool(event.payload["granted"])
    else:
        raise ValueError(f"Unknown event kind: {event.kind!r}")
    return next_state


def replay(initial: WorkflowState, event_log: list[Event]) -> WorkflowState:
    """Reconstruct final state by replaying an event log.

    s_T = R(R(... R(s_0, e_1) ..., e_{T-1}), e_T)

    Args:
        initial:   Starting state s_0.
        event_log: Ordered list of events.

    Returns:
        Reconstructed state.
    """
    state = initial
    for event in event_log:
        state = reduce_state(state, event)
    return state


def save_checkpoint(state: WorkflowState, path: Path) -> None:
    """Serialise a state snapshot to JSON.

    Args:
        state: Workflow state to persist.
        path:  Destination file path.
    """
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))


def load_checkpoint(path: Path) -> WorkflowState:
    """Load a previously saved checkpoint.

    Args:
        path: Path to the checkpoint JSON file.

    Returns:
        Restored WorkflowState.
    """
    return WorkflowState(**json.loads(path.read_text()))
```

## 4. Implementation: Production-Grade (LangGraph + PostgreSQL)

```python
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Typed LangGraph state with reducer annotations.

    The Annotated[list, operator.add] annotation tells LangGraph to
    merge (append) new messages rather than replace the field.
    This prevents nodes from accidentally clobbering each other's output.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    step_count:    int
    last_tool:     str
    approved:      bool


def increment_step(state: AgentState) -> AgentState:
    """Increment the step counter — demonstrates an arithmetic state update."""
    return {"step_count": state["step_count"] + 1}


def build_stateful_graph(use_postgres: bool = False) -> Any:
    """Build a LangGraph workflow with durable checkpointing.

    Args:
        use_postgres: If True, use PostgresSaver for durable storage.
                      If False, use in-memory MemorySaver (dev only).

    Returns:
        Compiled graph with checkpointer attached.
    """
    if use_postgres:
        # Production: checkpoints survive process restarts
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver.from_conn_string(
            "postgresql://user:pass@localhost:5432/agent_db"
        )
    else:
        # Development: in-memory only
        checkpointer = MemorySaver()

    builder = StateGraph(AgentState)
    builder.add_node("step", increment_step)
    builder.set_entry_point("step")
    builder.add_edge("step", END)
    return builder.compile(checkpointer=checkpointer)


def resume_from_checkpoint(
    graph: Any,
    thread_id: str,
    new_messages: list[BaseMessage],
) -> dict:
    """Continue a paused workflow from its saved checkpoint.

    LangGraph uses thread_id to identify and load the correct checkpoint.
    This enables pause-resume semantics without explicit replay logic.

    Args:
        graph:        Compiled LangGraph workflow.
        thread_id:    Unique identifier for the conversation/run.
        new_messages: New inputs to inject into the resumed run.

    Returns:
        Final state after resumed execution.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": new_messages}, config=config)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Mutating the state dictionary in place inside a node handler.
  When two nodes share a reference to the same list and both mutate it,
  the final state depends on execution order — a race condition in disguise.

  **Fix:** Always copy before modifying: `new_state = dict(state); new_state["key"] = ...`.
  Use `Annotated` field reducers in LangGraph so the framework merges
  concurrent writes correctly.

- **Mistake:** Mixing ephemeral context (in-progress LLM thoughts) with
  durable workflow state (completed task results) in the same state dict.
  Checkpointing ephemeral data wastes storage; losing durable data breaks
  recovery.

  **Fix:** Separate fields: ephemeral fields hold only what the next step
  needs; durable fields hold results that must survive a restart.

- **Mistake:** Not versioning the state schema before deploying to production.
  If the schema changes (a field renamed, a type changed), old checkpoints
  become unrestorable.

  **Fix:** Add a `schema_version: int` field to every checkpointed state.
  Write a migration function for each schema change before deploying.

- **Mistake:** Relying on `MemorySaver` in production. In-memory checkpoints
  are lost on process restart, making the system non-resumable after a crash
  or deployment.

  **Fix:** Use `PostgresSaver` or `RedisSaver` for any workflow that must
  survive restarts. Test recovery by intentionally killing the process
  mid-workflow.

- **Mistake:** Storing raw LLM output (full message text) in the checkpointed
  state for every step of a long conversation. This causes checkpoints to
  grow linearly in size and load times to grow accordingly.

  **Fix:** Apply a message trimmer or summariser to compact the message list
  before checkpointing. LangChain provides `trim_messages` for this purpose.

## 6. Knowledge Check

1. **Conceptual:** Explain why the reducer pattern (pure state transitions from
   events) makes agent workflows more debuggable than maintaining a mutable
   shared state object. What is the "event sourcing" relationship between the
   event log and the current state?

2. **Conceptual:** What is the difference between ephemeral, episodic, and
   semantic memory in an agent? Give a concrete example of data that belongs
   in each type, and explain why putting episodic data in semantic memory
   (or vice versa) causes problems.

3. **Coding challenge:** Implement the reducer/replay pattern from scratch.
   Define a `WorkflowState` dataclass and an `Event` dataclass. Write a pure
   `reduce_state(state, event) -> state` function. Then write `replay(s0, events)`
   that reconstructs the final state. Write a test that verifies replay
   produces the same result regardless of whether you use `MemorySaver` or
   replay from the event log.

## References

1. Abramov, D. (2015). Redux: A predictable state container for JavaScript apps. https://redux.js.org.
2. Fowler, M. (2005). Event sourcing. https://martinfowler.com/eaaDev/EventSourcing.html.
3. LangGraph (2024). Persistence and checkpointing. https://langchain-ai.github.io/langgraph/concepts/persistence/.
4. Bobrow, D. G. (1985). If Prolog is the answer, what is the question? *IJCAI 1985*. *(Foundational discussion of declarative state.)*
