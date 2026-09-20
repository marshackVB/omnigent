"""The three tools the final agent can call.

A "tool" is just a Python function the model is allowed to ask for by name.
The `@tool` decorator turns a function into something a model can see.

Two things get sent to the model, so both matter a lot:

  * the function **name**
  * the **docstring** and type hints

That is the entire spec the model gets. A vague docstring is the single most
common reason an agent calls the wrong tool or passes bad arguments. Write
docstrings for the model, not for yourself.

Note these tools are deliberately boring and offline -- no API keys, no
network, same answer every time (well, except the clock). That keeps the focus
on LangGraph rather than on debugging someone else's service.
"""

import ast
import operator
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Tool 1: a calculator
# ---------------------------------------------------------------------------

# Only these operations are permitted inside an expression.
def _safe_pow(base, exponent):
    """Exponentiation, refusing inputs that would take effectively forever.

    `9 ** 9 ** 9` is only five characters of model output, but evaluating it
    wedges the interpreter computing a number with ~370 million digits. A tool
    is a code path reachable from model output, so "the model asked for
    something absurd" has to be a handled case, not a hang.
    """
    if abs(base) > 1 and abs(exponent) > 1024:
        raise ValueError("exponent too large for this calculator")
    return operator.pow(base, exponent)


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    """Recursively evaluate a parsed arithmetic expression."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError("expression is too complex for this calculator")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Use this for any arithmetic instead of computing the answer yourself.
    Supports + - * / // % ** and parentheses. Numbers only, no variables.

    Args:
        expression: An arithmetic expression, for example "(17 * 23) / 4".
    """
    # We parse the expression and walk the tree rather than calling eval(),
    # which would happily run `__import__("os").system(...)` if the model were
    # ever talked into sending it. Tools are real code paths reachable from
    # model output, so they get validated like any other untrusted input.
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError) as exc:
        # Returning the error as a string (rather than raising) lets the model
        # read what went wrong and try again. Raising would crash the graph.
        return f"Error: could not evaluate {expression!r} ({exc})."

    # Trim the pointless ".0" on whole-number floats.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# Tool 2: the clock
# ---------------------------------------------------------------------------


@tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current date and time in a given timezone.

    The model has no clock of its own, so it must call this to know the date.

    Args:
        timezone_name: An IANA timezone name such as "UTC",
            "America/New_York", or "Asia/Tokyo". Defaults to "UTC".
    """
    try:
        tz = timezone.utc if timezone_name.upper() == "UTC" else ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return (
            f"Error: unknown timezone {timezone_name!r}. "
            'Use an IANA name like "UTC" or "Europe/Paris".'
        )
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


# ---------------------------------------------------------------------------
# Tool 3: an employee directory lookup
# ---------------------------------------------------------------------------

# Stands in for the database or HTTP call you would really make.
_DIRECTORY = {
    "ada lovelace": {
        "title": "Principal Engineer",
        "department": "Platform",
        "manager": "Grace Hopper",
        "location": "London",
        "start_year": 2019,
    },
    "grace hopper": {
        "title": "VP of Engineering",
        "department": "Platform",
        "manager": None,
        "location": "New York",
        "start_year": 2015,
    },
    "alan turing": {
        "title": "Staff Research Scientist",
        "department": "Research",
        "manager": "Grace Hopper",
        "location": "Cambridge",
        "start_year": 2021,
    },
    "katherine johnson": {
        "title": "Senior Data Scientist",
        "department": "Analytics",
        "manager": "Ada Lovelace",
        "location": "Houston",
        "start_year": 2022,
    },
}


@tool
def lookup_employee(name: str) -> str:
    """Look up an employee's role, department, manager, location and start year.

    Use this for any question about a specific person at the company.

    Args:
        name: The employee's full name, for example "Ada Lovelace".
    """
    record = _DIRECTORY.get(name.strip().lower())
    if record is None:
        known = ", ".join(n.title() for n in _DIRECTORY)
        return f"No employee named {name!r} was found. Known employees: {known}."

    manager = record["manager"] or "(none - top of reporting chain)"
    return (
        f"{name.strip().title()}\n"
        f"  Title: {record['title']}\n"
        f"  Department: {record['department']}\n"
        f"  Manager: {manager}\n"
        f"  Location: {record['location']}\n"
        f"  Started: {record['start_year']}"
    )


# The list every later script imports.
ALL_TOOLS = [calculator, get_current_time, lookup_employee]
