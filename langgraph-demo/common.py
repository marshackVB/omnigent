"""Shared setup for the lesson notebooks.

Two helpers live here so the notebooks can stay focused on LangGraph:

  * `make_llm()`  -- builds the chat model
  * `text_of()`   -- pulls readable text out of a message

Why `text_of` exists
--------------------
`response.content` is a plain string for an ordinary answer:

    "Mars is the fourth planet."

But when the model decides to call a tool, `content` becomes a *list of
blocks*, because the reply carries more than prose:

    [{"type": "text", "text": "Let me calculate that."},
     {"type": "tool_use", "name": "calculator", "input": {...}}]

So printing `.content` directly suddenly dumps a pile of JSON. That is not a
bug and not something LangGraph did -- it is the shape of the underlying API.
`text_of()` just grabs the human-readable part.
"""

import json
import os

# The Databricks model serving endpoint these notebooks use.
#
# Verified present via `w.serving_endpoints.list()`. Note the name: serving
# endpoints are `databricks-claude-*`. The `system.ai.*` form you may see
# elsewhere is a Unity Catalog model name and gets a 404 here.
MODEL_ENDPOINT = "databricks-claude-sonnet-5"


def make_llm(**kwargs):
    """Return a chat model.

    Inside a Databricks workspace this is just:

        from databricks_langchain import ChatDatabricks
        ChatDatabricks(endpoint="system.ai.claude-sonnet-5", max_tokens=1024)

    No API keys and no tokens -- the notebook's own credentials are used
    automatically, which is the main reason these notebooks prefer it.

    The fallback branch below exists only so the same code also runs outside
    Databricks (e.g. a laptop or CI box) by talking to an AI Gateway with a
    bearer token. If you are reading this in a workspace, ignore it.
    """
    max_tokens = kwargs.pop("max_tokens", 1024)

    try:
        from databricks_langchain import ChatDatabricks
    except ImportError:
        pass
    else:
        # `endpoint=` is the documented kwarg; internally the field is `model`
        # with `endpoint` as its alias, so either spelling works.
        return ChatDatabricks(
            endpoint=MODEL_ENDPOINT, max_tokens=max_tokens, **kwargs
        )

    # --- Off-Databricks fallback: AI Gateway + bearer token ------------------
    from langchain_anthropic import ChatAnthropic

    token = os.environ.get("DATABRICKS_BEARER") or os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        raise SystemExit(
            "No model backend available.\n"
            "In a Databricks workspace: pip install databricks-langchain\n"
            "Outside one: set DATABRICKS_BEARER (gateway) or ANTHROPIC_API_KEY."
        )
    # The gateway addresses the same model by its Unity Catalog name, which
    # differs from the serving-endpoint name used above.
    return ChatAnthropic(
        model="system.ai.claude-sonnet-5",
        api_key="unused-see-default-headers",
        default_headers={"Authorization": f"Bearer {token}"},
        max_tokens=max_tokens,
        **kwargs,
    )


def _text_from_blocks(blocks) -> str:
    """Join the 'text' blocks, skipping reasoning/tool_use/etc."""
    parts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def text_of(message) -> str:
    """Extract just the readable text from a message.

    Handles the three shapes `.content` can actually take:

      1. a plain string                      -- the ordinary case
      2. a list of blocks                    -- when tools or reasoning are used
      3. a *string containing JSON blocks*   -- what ChatDatabricks returns for
         a reasoning model, which will print as a wall of JSON if you do not
         parse it. Verified against databricks-claude-sonnet-5.
    """
    content = message.content

    if isinstance(content, list):
        return _text_from_blocks(content)

    if isinstance(content, str):
        # Case 3: looks like a JSON array of content blocks.
        stripped = content.lstrip()
        if stripped.startswith("[{"):
            try:
                blocks = json.loads(stripped)
            except json.JSONDecodeError:
                return content  # genuinely just text that happens to start oddly
            # Only treat it as blocks if it really looks like them, so a string
            # that merely happens to be a JSON list is left alone.
            if isinstance(blocks, list) and all(
                isinstance(b, dict) and "type" in b for b in blocks
            ):
                # Deliberately NOT `or content`: a reply can be all reasoning
                # and no text (the model spent its whole budget thinking). The
                # honest answer then is "no text", not a wall of JSON.
                return _text_from_blocks(blocks)
        return content

    return str(content)
