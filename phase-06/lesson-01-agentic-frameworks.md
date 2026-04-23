# Agentic Frameworks: LangGraph, AutoGen, and CrewAI

Agentic frameworks turn multi-step LLM workflows into explicit programs with
typed state, deterministic routing, and observable transitions. Single-shot
prompting fails when a task requires planning, tool use, branching, and retry
logic — each of which demands code-level control rather than prompt-level
control. LangGraph, AutoGen, and CrewAI represent three different design
philosophies: graph-based state machines, multi-agent conversation, and
role-based crew coordination.

## 1. The Core Intuition (The "Why")

Early "agents" were glorified while-loops: call the LLM, parse its output,
call a tool, repeat. This works for demos, then collapses when you need error
recovery, parallel sub-tasks, human approvals, or reproducible debugging.

Shinn et al. (2023) showed with **Reflexion** that agents can self-improve by
maintaining an explicit memory of past failures and reflecting on them before
the next attempt. This required structured state that a simple loop cannot
provide.

The key insight is that an agent system is a **stateful directed graph**:
nodes are functions (LLM calls, tool calls, routers), edges are transitions
conditioned on the current state, and the entire execution trace is
inspectable. This makes testing and debugging tractable — you can replay any
execution from any checkpoint.

## 2. The Theoretical Underpinning

### Workflow as a Directed Graph

Model the workflow as $G = (V, E)$ where each node $v \in V$ is a state
transition function and each edge $(u, v) \in E$ is a valid routing decision.

The current state at step $t$ is $s_t \in \mathcal{S}$. The workflow policy
selects the next node:

$$v_t = \pi(s_t)$$

Each node applies a state transition:

$$s_{t+1} = F_{v_t}(s_t)$$

The run terminates when a halting predicate $h(s_t) = 1$ is satisfied.

The full execution trace $\tau = [v_0, s_0, v_1, s_1, \ldots, v_T, s_T]$
is the object needed for debugging, auditing, and replay.

### Conditional Routing

A **conditional edge** evaluates a function over the current state and
routes to one of several nodes:

$$v_{t+1} = \text{router}(s_t) = \begin{cases} v_A & \text{if condition}_A(s_t) \\ v_B & \text{if condition}_B(s_t) \\ v_\text{END} & \text{otherwise} \end{cases}$$

This is the mechanism behind LangGraph's `add_conditional_edges`. The key
correctness requirement: the routing function must be deterministic given
the state — it must not make its own LLM call.

### Multi-Agent Coordination

AutoGen models coordination as a **conversation protocol** between agents.
Each agent $A_i$ has a role (system prompt) and a reply function:

$$m_{t+1} = A_i(m_{1:t})$$

Agents take turns until a termination condition is met (e.g., a message
contains "TERMINATE" or a max-round limit is hit). The message history is
the shared state.

### Crew-Based Role Assignment

CrewAI assigns agents **roles, goals, and backstories** and uses a manager
agent to delegate tasks:

$$\text{task}_i \to A_{\text{role}(i)}$$

The manager agent uses function calling to delegate tasks to the right
role-specialist agent. This is useful when sub-tasks require genuinely
different expertise (e.g., researcher + writer + critic).

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


State = dict[str, Any]


@dataclass
class Node:
    """A node in the agent workflow graph.

    Args:
        name:    Unique node identifier.
        handler: Function that receives and returns the state.
    """
    name:    str
    handler: Callable[[State], State]


@dataclass
class WorkflowGraph:
    """Minimal graph executor for agent workflows.

    Nodes hold state-transition logic. Edges are either unconditional
    (str -> str) or conditional (str -> Callable[[State], str]).
    """

    nodes:       dict[str, Node]   = field(default_factory=dict)
    edges:       dict[str, Any]    = field(default_factory=dict)  # str | Callable
    entry_point: str               = ""
    MAX_STEPS:   int               = 50

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str) -> None:
        """Add an unconditional edge."""
        self.edges[src] = dst

    def add_conditional_edge(self, src: str, router: Callable[[State], str]) -> None:
        """Add a conditional edge whose target depends on the state."""
        self.edges[src] = router

    def _next_node(self, current: str, state: State) -> str | None:
        edge = self.edges.get(current)
        if edge is None:
            return None
        if callable(edge):
            return edge(state)   # Conditional routing
        return edge              # Unconditional routing

    def run(self, initial_state: State) -> tuple[State, list[str]]:
        """Execute the graph from the entry point.

        Args:
            initial_state: Starting workflow state.

        Returns:
            (final_state, visited_nodes) for debugging.
        """
        if self.entry_point not in self.nodes:
            raise ValueError(f"Unknown entry point: {self.entry_point!r}")

        state   = dict(initial_state)
        current = self.entry_point
        history: list[str] = []

        for _ in range(self.MAX_STEPS):
            history.append(current)
            # Apply the node's state transition: s_{t+1} = F_v(s_t)
            state   = self.nodes[current].handler(state)
            # Halt when the predicate h(s_t) = 1
            if state.get("__end__"):
                break
            nxt = self._next_node(current, state)
            if nxt is None or nxt == "__end__":
                break
            current = nxt

        return state, history
```

## 4. Implementation: Production-Grade (LangGraph)

```python
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph


class ResearchState(TypedDict):
    """Typed state for a research-and-write workflow."""
    question:    str
    search_results: str
    draft:       str
    critique:    str
    final:       str
    next_step:   Literal["search", "write", "critique", "revise", "finish"]


def router(state: ResearchState) -> str:
    """Conditional routing function.

    Reads state["next_step"] and routes to the appropriate node.
    This function must be pure (no LLM calls).
    """
    return state["next_step"]


def search_node(state: ResearchState) -> ResearchState:
    """Simulated web search step."""
    return {**state,
            "search_results": f"[Results for: {state['question']}]",
            "next_step": "write"}


def write_node(state: ResearchState) -> ResearchState:
    """Draft an answer from search results."""
    return {**state,
            "draft": f"Draft based on: {state['search_results']}",
            "next_step": "critique"}


def critique_node(state: ResearchState) -> ResearchState:
    """Critique the draft and decide whether to revise."""
    # In production: LLM evaluates draft quality
    needs_revision = len(state["draft"]) < 50
    return {**state,
            "critique": "Too short." if needs_revision else "Good.",
            "next_step": "revise" if needs_revision else "finish"}


def revise_node(state: ResearchState) -> ResearchState:
    """Revise the draft based on the critique."""
    return {**state,
            "draft": state["draft"] + " [expanded]",
            "next_step": "finish"}


def build_research_graph() -> Any:
    """Compile the research-and-write LangGraph workflow."""
    g = StateGraph(ResearchState)
    for name, fn in [
        ("search", search_node), ("write", write_node),
        ("critique", critique_node), ("revise", revise_node),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("search")
    # Conditional routing: the router reads state["next_step"]
    g.add_conditional_edges(
        "critique",
        router,
        {"revise": "revise", "finish": END},
    )
    g.add_edge("search",  "write")
    g.add_edge("write",   "critique")
    g.add_edge("revise",  END)
    return g.compile()
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using a multi-agent system for tasks that require a single
  tool call followed by a single formatting step. Agent overhead (extra LLM
  calls, state serialisation, routing) adds latency and cost.

  **Fix:** Start with the simplest pipeline that solves the task. Add agents
  only when the task genuinely requires branching, delegation, or parallel
  sub-tasks that deliver measurable value.

- **Mistake:** Putting routing logic inside prompts ("decide whether to search
  or answer directly"). Routing buried in prompts is invisible, untestable,
  and fragile to wording changes.

  **Fix:** Keep routing in code as explicit conditional functions. The LLM
  output determines a `next_step` field; a Python function reads it and
  routes. This makes routing unit-testable.

- **Mistake:** Running agent loops without a hard `MAX_STEPS` limit or cost
  budget. A misconfigured tool or a self-referential subtask can cause
  the agent to loop indefinitely and exhaust API credits.

  **Fix:** Set `MAX_STEPS = 15` (or similar) at the graph level. Add per-run
  cost tracking and alert when a run exceeds a token budget threshold.

- **Mistake:** Designing state as a flat dictionary with no schema. As the
  workflow grows, undocumented keys collide and type errors surface at
  runtime.

  **Fix:** Use `TypedDict` or Pydantic models for state. LangGraph enforces
  typed state, making schema violations a compile-time error rather than a
  runtime mystery.

- **Mistake:** Choosing between LangGraph, AutoGen, and CrewAI based on
  hype rather than the actual control-flow requirements of the task.

  **Fix:** Use LangGraph when you need explicit typed-state graphs and
  deterministic routing. Use AutoGen when the task is fundamentally a
  multi-party conversation. Use CrewAI when you want role-based delegation
  with minimal orchestration boilerplate.

## 6. Knowledge Check

1. **Conceptual:** Explain why placing routing logic inside LLM prompts is
   inferior to encoding it in explicit conditional edge functions. What
   testing and debugging properties are lost when routing is prompt-based?

2. **Conceptual:** Compare the AutoGen conversation protocol with the
   LangGraph state-graph model. In which scenario is each better suited?
   What happens when you need both structured state and multi-agent
   conversation?

3. **Coding challenge:** Implement the `WorkflowGraph` class from scratch
   and build a three-node workflow: a `triage` node that sets
   `state["topic"]`, a conditional router that routes `"math"` queries to
   a `calculator` node and all others to a `search` node, and both
   terminals that set `state["answer"]`. Write unit tests for each routing
   path.

## References

1. Shinn, N., Cassano, F., Labash, B., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS 2023*. arXiv:2303.11366.
2. Wu, Q., Bansal, G., Zhang, J., et al. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. arXiv:2308.08155.
3. Chase, H. (2022). LangChain. https://github.com/langchain-ai/langchain.
4. LangGraph (2024). LangGraph: Build stateful, multi-actor applications with LLMs. https://langchain-ai.github.io/langgraph/.
5. Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. arXiv:2210.03629.
