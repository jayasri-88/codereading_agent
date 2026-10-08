"""
Restricted Python sandbox for executing AI-generated code.

Safety features:
- Timeout (default 15s)
- Restricted builtins
- Blocked dangerous modules (os, sys, subprocess, shutil, socket, etc.)
- Only pandas, numpy, matplotlib, plotly, sklearn, and a few safe libs allowed
- Captures stdout / stderr
- Returns structured result or error message
"""

from __future__ import annotations

import ast
import contextlib
import io
import threading
import sys
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional imports – keep sandbox light
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Safety: block dangerous modules & names
# ---------------------------------------------------------------------------

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "pathlib",
    "importlib", "ctypes", "multiprocessing", "threading",
    "pickle", "shelve", "marshal", "builtins", "__builtins__",
    "pty", "fcntl", "resource", "signal", "tempfile", "glob",
    "http", "urllib", "requests", "ftplib", "telnetlib",
    "webbrowser", "code", "codeop", "compile", "exec", "eval",
}

BLOCKED_BUILTINS = {
    "open", "exec", "eval", "compile", "input", "__import__",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "breakpoint", "exit", "quit",
}


class TimeoutException(Exception):
    pass


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom __import__ that blocks dangerous modules."""
    root = name.split(".")[0]
    if root in BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
    return __import__(name, globals, locals, fromlist, level)


def _make_safe_builtins() -> dict[str, Any]:
    """Create a restricted builtins dict."""
    safe = {}
    import builtins as _b

    allowed = {
        "abs", "all", "any", "bin", "bool", "bytearray", "bytes",
        "chr", "complex", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "hash", "hex", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "print", "range",
        "repr", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip", "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "ZeroDivisionError",
        "StopIteration", "AssertionError",
    }

    for name in allowed:
        if hasattr(_b, name):
            safe[name] = getattr(_b, name)

    # Override dangerous ones
    safe["__import__"] = _safe_import
    safe["open"] = None  # hard block
    return safe


def _validate_code(code: str) -> None:
    """Static check: reject obviously dangerous AST nodes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for n in names:
                if n in BLOCKED_MODULES:
                    raise ValueError(f"Import of blocked module '{n}' is not allowed")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile", "__import__"}:
                raise ValueError(f"Call to '{node.func.id}' is not allowed")


def run_sandboxed(
    code: str,
    df: pd.DataFrame,
    timeout: int = 15,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute user/AI code safely.

    Returns:
        {
            "success": bool,
            "stdout": str,
            "error": str | None,
            "figures": list[bytes],   # PNG images if any
            "result": Any,            # last expression value if useful
        }
    """
    _validate_code(code)

    # Prepare restricted globals
    safe_builtins = _make_safe_builtins()
    restricted_globals: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": df.copy(),          # work on a copy
        "pd": pd,
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
    }

    if HAS_PLOTLY:
        restricted_globals["px"] = px
        restricted_globals["go"] = go

    if HAS_SKLEARN:
        restricted_globals.update({
            "train_test_split": train_test_split,
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "LogisticRegression": LogisticRegression,
            "LinearRegression": LinearRegression,
            "accuracy_score": accuracy_score,
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
            "classification_report": classification_report,
        })

    if extra_globals:
        restricted_globals.update(extra_globals)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    figures: list[bytes] = []

    # Capture matplotlib figures
    original_show = plt.show

    def _capture_show(*args, **kwargs):
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        figures.append(buf.read())
        plt.close(fig)

    plt.show = _capture_show

    result = None
    error_msg = None
    success = False
    timed_out = False

    def _run():
        nonlocal result, error_msg, success
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec_locals: dict[str, Any] = {}
                exec(code, restricted_globals, exec_locals)
                if "result" in exec_locals:
                    result = exec_locals["result"]
                elif "output" in exec_locals:
                    result = exec_locals["output"]
            success = True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)

    try:
        if thread.is_alive():
            timed_out = True
            error_msg = f"Execution timed out after {timeout} seconds"
    finally:
        plt.show = original_show
        plt.close("all")

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    if stderr and not error_msg:
        error_msg = stderr

    return {
        "success": success,
        "stdout": stdout,
        "error": error_msg,
        "figures": figures,
        "result": result,
    }
