# Tool Use and the Model Context Protocol (MCP)

Language models excel at reasoning but lack access to real-time information
and cannot execute code or interact with external systems. Tool use gives
LMs the ability to call external functions — web search, code execution,
database queries, API calls — bridging the gap between language reasoning
and real-world action. The Model Context Protocol (MCP) standardises how
LMs discover and invoke tools, turning ad-hoc integrations into a universal
interface.

## 1. The Core Intuition (The "Why")

Early LMs were "closed" systems: knowledge was frozen at training time, and
the only way to get updated information was to retrain. Nakano et al. (2021)
showed with WebGPT that a model that could browse the web outperformed GPT-3
on knowledge-intensive tasks by a large margin — not because it was smarter,
but because it could look things up.

Schick et al. (2023) formalised tool use with **Toolformer**: a model trained
in a self-supervised way to decide when to call tools (calculator, calendar,
search) and how to incorporate tool outputs into its reasoning. OpenAI
extended this to **function calling**, where the model returns structured
JSON describing the function it wants to call, and the application layer
executes it.

The **ReAct** framework (Yao et al., 2022) interleaves reasoning (Thought)
with action (Act) and observation (Obs) in a loop, allowing the model to
plan multi-step tool-use sequences.

## 2. The Theoretical Underpinning

### ReAct: Reason + Act

The ReAct loop is a sequence of tokens:

$$\tau = [\text{Thought}_1, \text{Act}_1, \text{Obs}_1, \text{Thought}_2, \text{Act}_2, \text{Obs}_2, \ldots, \text{Answer}]$$

At each step, the model generates a **Thought** (free-form reasoning about
what to do), selects an **Action** (tool call with arguments), receives an
**Observation** (tool output), and continues.

Yao et al. (2022) showed that combining reasoning and acting outperformed
pure reasoning (CoT) and pure acting (MRKL/API calls) on fact-verification
and web navigation tasks. The key benefit: reasoning grounds the action
selection; actions bring in external evidence that grounds the reasoning.

### Function Calling Schema

A function is described by a JSON schema:

```json
{
  "name": "search_web",
  "description": "Search the web for recent information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"},
      "max_results": {"type": "integer", "default": 5}
    },
    "required": ["query"]
  }
}
```

The model generates a JSON object matching this schema when it decides to
call the tool. The application parses the JSON, executes the function, and
appends the result to the conversation as a tool message.

### The Model Context Protocol (MCP)

The Model Context Protocol (Anthropic, 2024) standardises the client-server
communication between an LM application (the client) and external services
that expose tools (the server). The protocol defines:

- **Resources**: data sources the server exposes (files, database rows)
- **Tools**: callable functions with JSON schemas
- **Prompts**: reusable prompt templates

An MCP server exposes its capabilities via a standard interface; any
compliant MCP client (Claude Desktop, VS Code Copilot, custom agent) can
connect to any MCP server without custom integration code.

The flow:
1. Client connects to MCP server and requests a list of available tools.
2. Client includes tool schemas in the LM's system prompt or tool list.
3. LM generates a tool call; client forwards it to the MCP server.
4. MCP server executes the tool and returns the result.
5. Client appends the result to the LM's context and continues.

### Parallel Tool Calls

Modern APIs support requesting multiple tool calls in a single LM turn.
The LM can emit a list of tool calls to be executed in parallel:

```json
[
  {"name": "search_web",     "arguments": {"query": "RAG research 2024"}},
  {"name": "read_file",      "arguments": {"path": "notes.md"}}
]
```

This reduces latency from $O(n)$ sequential calls to $O(1)$ when tool
calls are independent.

## 3. Implementation: From Scratch (Python)

```python
from __future__ import annotations

import inspect
import json
from typing import Any, Callable


class Tool:
    """Wraps a Python function as a tool with a JSON schema."""

    def __init__(self, fn: Callable, description: str) -> None:
        self.fn          = fn
        self.name        = fn.__name__
        self.description = description
        self.schema      = self._build_schema()

    def _build_schema(self) -> dict:
        """Build a minimal JSON schema from function annotations."""
        sig    = inspect.signature(self.fn)
        props  = {}
        req    = []
        for name, param in sig.parameters.items():
            ann   = param.annotation
            ptype = {str: "string", int: "integer", float: "number"}.get(ann, "string")
            props[name] = {"type": ptype}
            if param.default is inspect.Parameter.empty:
                req.append(name)
        return {
            "name":        self.name,
            "description": self.description,
            "parameters":  {
                "type":       "object",
                "properties": props,
                "required":   req,
            },
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class ReActAgent:
    """A minimal ReAct agent that interleaves reasoning and tool calls.

    Each 'step' the agent calls the LM, parses the output for a tool call,
    executes the tool, and appends the observation to the conversation.
    """

    MAX_STEPS = 10  # Safety limit to prevent infinite loops

    def __init__(
        self,
        tools:    list[Tool],
        llm_call: Callable[[list[dict]], str],
    ) -> None:
        self.tools    = {t.name: t for t in tools}
        self.llm_call = llm_call

    def _system_prompt(self) -> str:
        schemas = json.dumps([t.schema for t in self.tools.values()], indent=2)
        return (
            "You are an agent that can use tools to answer questions.\n"
            f"Available tools:\n{schemas}\n\n"
            "To call a tool, output: TOOL_CALL: {\"name\": ..., \"args\": {...}}\n"
            "When you have the final answer, output: ANSWER: <answer>"
        )

    def run(self, question: str) -> str:
        """Run the ReAct loop until the agent produces a final answer."""
        messages = [
            {"role": "system",  "content": self._system_prompt()},
            {"role": "user",    "content": question},
        ]
        for _ in range(self.MAX_STEPS):
            response = self.llm_call(messages)
            messages.append({"role": "assistant", "content": response})

            if "ANSWER:" in response:
                return response.split("ANSWER:")[-1].strip()

            if "TOOL_CALL:" in response:
                raw  = response.split("TOOL_CALL:")[-1].strip()
                call = json.loads(raw)
                tool = self.tools.get(call["name"])
                obs  = str(tool(**call["args"])) if tool else "Tool not found."
                messages.append({"role": "tool", "content": obs})

        return "Max steps reached without an answer."
```

## 4. Implementation: Production-Grade (OpenAI Function Calling)

```python
from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI


def define_tools() -> list[dict]:
    """Define tools using the OpenAI function calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name":        "search_web",
                "description": "Search the web for real-time information.",
                "parameters":  {
                    "type":       "object",
                    "properties": {
                        "query":       {"type": "string"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name":        "run_python",
                "description": "Execute a Python snippet and return the output.",
                "parameters":  {
                    "type":       "object",
                    "properties": {
                        "code": {"type": "string"},
                    },
                    "required": ["code"],
                },
            },
        },
    ]


def tool_router(name: str, args: dict) -> str:
    """Route a tool call to its implementation.

    In production, each tool should be validated and sandboxed.
    Code execution must happen in a restricted environment (e.g., E2B).
    """
    if name == "search_web":
        # Replace with a real search API (Tavily, Serper, etc.)
        return f"[Search results for '{args['query']}' would appear here]"
    if name == "run_python":
        # WARNING: Never exec() untrusted code in production.
        # Use a sandboxed environment like E2B or a subprocess with restrictions.
        import io, sys
        buf = io.StringIO()
        try:
            exec(args["code"], {}, {})    # noqa: S102 — dev/demo only
        except Exception as e:
            return f"Error: {e}"
        return buf.getvalue() or "(no output)"
    return "Unknown tool."


def run_agent(question: str, max_turns: int = 6) -> str:
    """Run a multi-turn tool-use agent with OpenAI function calling.

    Each turn: call the API, check if it wants to use a tool,
    execute the tool, append the result, and continue.

    Args:
        question:  The user's question.
        max_turns: Maximum number of tool-use turns.

    Returns:
        The model's final answer.
    """
    client   = OpenAI()
    messages = [{"role": "user", "content": question}]
    tools    = define_tools()

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = messages,
            tools    = tools,
        )
        msg = response.choices[0].message

        # No tool call: we have the final answer
        if not msg.tool_calls:
            return msg.content or ""

        messages.append(msg)

        # Execute each tool call and append observations
        for tc in msg.tool_calls:
            result = tool_router(tc.function.name, json.loads(tc.function.arguments))
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    return "Max turns reached."
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Executing code from tool calls in an unsandboxed Python
  environment. The model can be prompted to run malicious code.

  **Fix:** Use a sandboxed execution environment (E2B, Docker with
  resource limits, or a restricted subprocess). Never `exec()` LLM-generated
  code in your production process.

- **Mistake:** Not setting a maximum iteration limit in the ReAct loop.
  A malformed tool response can cause the model to loop indefinitely.

  **Fix:** Always set `MAX_STEPS` (e.g., 10). Log and alert when the limit
  is hit so you can debug runaway agents.

- **Mistake:** Providing too many tools in a single system prompt.
  Models struggle to select the right tool when given 50+ options.

  **Fix:** Use tool retrieval: embed tool descriptions and retrieve the
  most relevant tools for the current query. Expose only 5–10 tools per turn.

- **Mistake:** Ignoring tool call errors. If a tool returns an error,
  the model may silently hallucinate a reasonable-sounding answer.

  **Fix:** Return structured error messages from tools. Instruct the model
  explicitly: "If a tool returns an error, say so and ask the user to clarify."

- **Mistake:** Using string parsing to detect tool calls rather than
  structured outputs. Regex-based parsing is fragile to wording changes.

  **Fix:** Use the provider's native function calling API (OpenAI, Anthropic)
  which guarantees structured JSON output for tool calls.

## 6. Knowledge Check

1. **Conceptual:** Explain the ReAct framework. Why does interleaving
   reasoning (Thought) with action (Act) outperform pure reasoning (CoT)
   or pure action (direct API calls) on complex tasks?

2. **Conceptual:** What does the Model Context Protocol standardise?
   What problem does it solve that ad-hoc function calling implementations
   don't?

3. **Coding challenge:** Build a minimal ReAct agent that answers arithmetic
   questions by calling a `calculate(expression: str) -> float` tool.
   Implement the loop from scratch: parse TOOL_CALL JSON from the LM output,
   execute the tool, and append the observation. Test on "What is (17 * 23)
   + sqrt(144)?".

## References

1. Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. arXiv:2210.03629.
2. Schick, T., Dwivedi-Yu, J., Dessì, R., et al. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS 2023*. arXiv:2302.04761.
3. Nakano, R., Hilton, J., Balwit, A., et al. (2021). WebGPT: Browser-assisted question-answering with human feedback. arXiv:2112.09332.
4. Anthropic. (2024). Model Context Protocol specification. https://modelcontextprotocol.io/specification.
