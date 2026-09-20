# LangGraph from scratch: a tool-calling agent in five notebooks

A hands-on introduction for someone who has never used LangGraph. Each notebook
hits a wall that the next one solves, ending at a graph with two nodes and one
cycle that answers questions by calling three tools — sequencing them itself.

Built to be opened in a **Databricks workspace** as a git folder.

All of it lives in [`langgraph-demo/`](langgraph-demo).

## Running it in Databricks

1. **Repos / Git folders → Add git folder**, pointing at this repository.
2. Open [`langgraph-demo/01_plain_llm.ipynb`](langgraph-demo/01_plain_llm.ipynb)
   and attach it to a cluster (DBR 13.3 LTS or later).
3. Run the first cell. If the imports fail, run this in a cell above it, then
   restart Python:
   ```
   %pip install langgraph databricks-langchain grandalf
   ```
4. Work through the notebooks in order.

No API keys, tokens, or secret scopes are needed. `ChatDatabricks` uses the
notebook's own credentials.

### Model endpoint

The notebooks call the serving endpoint named in
[`langgraph-demo/common.py`](langgraph-demo/common.py):

```python
MODEL_ENDPOINT = "databricks-claude-sonnet-5"
```

If your workspace does not have it, list what you do have and change that one
line:

```python
from databricks.sdk import WorkspaceClient
print([e.name for e in WorkspaceClient().serving_endpoints.list()])
```

Any tool-calling chat endpoint works — `databricks-claude-sonnet-4-6`,
`databricks-gpt-5`, and similar are all fine.

## The lessons

| Notebook | The idea | The wall it hits |
|---|---|---|
| [1. A model alone](langgraph-demo/01_plain_llm.ipynb) | A model is text-in, text-out | No clock; arithmetic costs >1000 tokens |
| [2. Tool calling](langgraph-demo/02_tool_calling.ipynb) | The five-step tool handshake, no framework | One round trip is not enough |
| [3. The manual loop](langgraph-demo/03_manual_loop.ipynb) | The agent loop in ~15 lines of plain Python | Works, but production features tangle it |
| [4. Your first graph](langgraph-demo/04_first_graph.ipynb) | State, nodes, edges, reducers | Every edge is fixed — no decisions |
| [5. The full agent](langgraph-demo/05_tool_agent.ipynb) | The full agent: conditional edge + cycle | — |

Supporting files, imported by the notebooks rather than opened directly:

- [**`common.py`**](langgraph-demo/common.py) — `make_llm()` and the `text_of()` message helper
- [**`tools.py`**](langgraph-demo/tools.py) — the three tools

The notebooks are the source of truth: edit them directly, in Databricks or
anywhere else. They ship with **no stored outputs**, so you watch cells run
rather than inheriting someone else's stale results, and diffs stay readable.
If you edit a notebook in the workspace, clear its outputs before committing to
keep that true.

## The one thing to understand first

**The model never runs any code.** It only emits text. When we say "the model
called the calculator", what actually happened is:

1. We sent the model descriptions of the available tools.
2. The model replied with a structured request: a name and arguments.
3. **Our Python code** read that request and ran the function.
4. We sent the return value back as a new message.
5. The model read the result and wrote its answer.

Steps 3 and 4 are yours to write. Lesson 2 does all five by hand so this is
concrete before any framework hides it.

## The shape you are building toward

```
       agent  ──tool calls?──>  tools
         ^                        │
         └────── results ─────────┘
         │
         └──no tool calls──> END
```

Two nodes. The conditional edge out of `agent` is the only real decision; the
edge from `tools` back to `agent` is the loop.

This is **not** a DAG — the cycle is the entire point, and allowing cycles is
what separates LangGraph from a plain DAG pipeline. Looping is bounded by
`recursion_limit`, not by the graph's shape.

## What lesson 5 actually does

Real output from the default question:

```
  1. [agent] wants: lookup_employee, get_current_time
    [tool] lookup_employee({'name': 'Ada Lovelace'})
    [tool] get_current_time({'timezone_name': 'UTC'})
  2. [tools] returned 2 result(s)
  3. [agent] wants: calculator, calculator
    [tool] calculator({'expression': '2026 - 2019'})
    [tool] calculator({'expression': '7 * 365.25'})
  4. [tools] returned 2 result(s)
  5. [agent] final answer ready

Ada Lovelace started in 2019, and as of today (September 20, 2026), her tenure
is 7 whole years. Expressed in days using 365.25 days per year, that's
2,556.75 days.
```

The loop turns **twice**, and that is the part to study. The model could not call
`calculator` on the first pass — it did not yet know the start year or the
current year. So it fired the two independent lookups together, waited, then
computed. Nobody wrote that plan; it falls out of the cycle.

## Notes for whoever maintains this

**Reasoning cannot be disabled** on `databricks-claude-sonnet-5`. Passing
`thinking={"type": "disabled"}` is silently ignored and
`extra_params={"reasoning_effort": ...}` returns a 400. Lesson 1 is written
around this: it demonstrates that arithmetic *costs* ~1,400 reasoning tokens
rather than that the model gets it wrong.

**`text_of()` exists for a real reason.** On this endpoint a reasoning reply
arrives as a *string containing JSON content blocks*, so printing `.content`
dumps a wall of JSON including base64 reasoning signatures. `text_of()` handles
all three shapes `.content` can take (plain string, block list, JSON-string
blocks). Do not replace it with `.content`.

**Endpoint naming.** Serving endpoints are `databricks-claude-*`. The
`system.ai.claude-*` form is a Unity Catalog model name and returns a 404 from
`ChatDatabricks` — it only works through an AI Gateway.

**Tool safety.** [`tools.py`](langgraph-demo/tools.py) parses arithmetic with
`ast` instead of `eval()`, and
caps exponent size. Both matter: `eval()` would run
`__import__("os").system(...)` if the model were induced to send it, and
`9**9**9` — five characters — wedges the interpreter computing a
370-million-digit number. **A tool is a code path reachable from model output**,
so arguments are untrusted input.

Tool errors are *returned as strings*, not raised. A raised exception kills the
graph; a returned error lets the model read what went wrong and retry.
