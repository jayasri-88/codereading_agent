"""
Tool definitions and implementations for the DataPilot agent.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .sandbox import run_sandboxed
from utils.data_loader import profile_dataframe, profile_to_text


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / Gemini compatible format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_schema",
        "description": (
            "Get a detailed profile of the currently loaded dataframe: "
            "column names, dtypes, missing values, numeric stats, and 5 sample rows. "
            "Always call this first when you need to understand the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute Python code on the dataframe `df` (a pandas DataFrame). "
            "You can use pandas as `pd`, numpy as `np`, matplotlib.pyplot as `plt`. "
            "Print results with print(). For charts, create them with plt and call plt.show(). "
            "The code runs in a restricted sandbox (no os, sys, file I/O, network). "
            "Return value of the last expression or variables named `result` / `output` are captured."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute. Must be valid Python.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of what this code does (shown to user).",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "make_chart",
        "description": (
            "Create a chart from the dataframe and return it as an image. "
            "Supported chart types: bar, line, scatter, hist, box, pie, heatmap. "
            "Prefer this tool for simple charts; use run_python + matplotlib for complex ones."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "hist", "box", "pie", "heatmap"],
                    "description": "Type of chart to create",
                },
                "x": {
                    "type": "string",
                    "description": "Column name for x-axis (or labels for pie)",
                },
                "y": {
                    "type": "string",
                    "description": "Column name for y-axis (optional for hist/pie)",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title",
                },
                "hue": {
                    "type": "string",
                    "description": "Optional column for color grouping",
                },
                "agg": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "median", "max", "min"],
                    "description": "Aggregation if needed (for bar/line)",
                },
            },
            "required": ["chart_type", "title"],
        },
    },
    {
        "name": "train_model",
        "description": (
            "Train a quick scikit-learn model (classification or regression) and return metrics. "
            "Useful for questions like 'predict churn' or 'what features matter most'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Name of the target column to predict",
                },
                "task": {
                    "type": "string",
                    "enum": ["classification", "regression", "auto"],
                    "description": "Type of ML task. 'auto' detects from target dtype.",
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of feature columns. If omitted, uses all other columns.",
                },
                "test_size": {
                    "type": "number",
                    "description": "Fraction for test set (default 0.2)",
                },
            },
            "required": ["target"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def execute_tool(
    name: str,
    arguments: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Dispatch a tool call and return a structured result that can be
    fed back to the LLM and also rendered in the UI.
    """
    try:
        if name == "get_schema":
            return _tool_get_schema(df)
        elif name == "run_python":
            return _tool_run_python(df, arguments)
        elif name == "make_chart":
            return _tool_make_chart(df, arguments)
        elif name == "train_model":
            return _tool_train_model(df, arguments)
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {name}",
                "content": f"Error: Unknown tool '{name}'",
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": f"Tool execution failed: {type(e).__name__}: {e}",
        }


def _tool_get_schema(df: pd.DataFrame) -> dict[str, Any]:
    profile = profile_dataframe(df)
    text = profile_to_text(profile)
    return {
        "success": True,
        "content": text,
        "profile": profile,  # for UI if needed
    }


def _tool_run_python(df: pd.DataFrame, args: dict[str, Any]) -> dict[str, Any]:
    code = args.get("code", "")
    explanation = args.get("explanation", "")

    if not code.strip():
        return {"success": False, "content": "Error: empty code provided", "error": "empty code"}

    result = run_sandboxed(code, df, timeout=20)

    # Build human + LLM readable content
    parts = []
    if explanation:
        parts.append(f"Explanation: {explanation}")
    parts.append("```python\n" + code.strip() + "\n```")

    if result["stdout"]:
        parts.append("Output:\n" + result["stdout"].rstrip())

    if result["error"]:
        parts.append("ERROR:\n" + result["error"])

    if result["result"] is not None:
        try:
            parts.append(f"Result: {result['result']}")
        except Exception:
            parts.append("Result: <unprintable>")

    content = "\n\n".join(parts)

    # Encode figures as base64 for the UI
    images_b64 = []
    for fig_bytes in result.get("figures", []):
        images_b64.append(base64.b64encode(fig_bytes).decode("utf-8"))

    return {
        "success": result["success"],
        "content": content,
        "error": result["error"],
        "images": images_b64,
        "code": code,
        "explanation": explanation,
    }


def _tool_make_chart(df: pd.DataFrame, args: dict[str, Any]) -> dict[str, Any]:
    chart_type = args.get("chart_type", "bar")
    x = args.get("x")
    y = args.get("y")
    title = args.get("title", "Chart")
    hue = args.get("hue")
    agg = args.get("agg")

    fig, ax = plt.subplots(figsize=(9, 5))

    try:
        work = df.copy()

        if chart_type in ("bar", "line") and x and y:
            if agg and y in work.columns:
                grouped = work.groupby(x, dropna=False)[y]
                if agg == "sum":
                    data = grouped.sum()
                elif agg == "mean":
                    data = grouped.mean()
                elif agg == "count":
                    data = grouped.count()
                elif agg == "median":
                    data = grouped.median()
                elif agg == "max":
                    data = grouped.max()
                elif agg == "min":
                    data = grouped.min()
                else:
                    data = grouped.sum()
                if chart_type == "bar":
                    data.plot(kind="bar", ax=ax)
                else:
                    data.plot(kind="line", ax=ax, marker="o")
            else:
                if chart_type == "bar":
                    work.plot(x=x, y=y, kind="bar", ax=ax)
                else:
                    work.plot(x=x, y=y, kind="line", ax=ax, marker="o")

        elif chart_type == "scatter" and x and y:
            if hue and hue in work.columns:
                for name, group in work.groupby(hue):
                    ax.scatter(group[x], group[y], label=str(name), alpha=0.7)
                ax.legend()
            else:
                ax.scatter(work[x], work[y], alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        elif chart_type == "hist":
            col = y or x
            if col and col in work.columns:
                work[col].dropna().plot(kind="hist", bins=30, ax=ax, edgecolor="black")
                ax.set_xlabel(col)

        elif chart_type == "box":
            col = y or x
            if col and col in work.columns:
                work[[col]].boxplot(ax=ax)

        elif chart_type == "pie" and x:
            counts = work[x].value_counts()
            counts.plot(kind="pie", ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")

        elif chart_type == "heatmap":
            numeric = work.select_dtypes(include="number")
            if numeric.shape[1] >= 2:
                corr = numeric.corr()
                im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.columns)
                fig.colorbar(im, ax=ax)
            else:
                raise ValueError("Need at least 2 numeric columns for heatmap")

        else:
            raise ValueError(f"Unsupported chart configuration: type={chart_type}, x={x}, y={y}")

        ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "success": True,
            "content": f"Created {chart_type} chart: {title}",
            "images": [img_b64],
            "title": title,
        }

    except Exception as e:
        plt.close(fig)
        return {
            "success": False,
            "content": f"Failed to create chart: {e}",
            "error": str(e),
            "images": [],
        }


def _tool_train_model(df: pd.DataFrame, args: dict[str, Any]) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
    import numpy as np

    target = args.get("target")
    task = args.get("task", "auto")
    features = args.get("features")
    test_size = float(args.get("test_size", 0.2))

    if target not in df.columns:
        return {"success": False, "content": f"Target column '{target}' not found", "error": "missing target"}

    work = df.copy()
    y = work[target]

    if features:
        X = work[features]
    else:
        X = work.drop(columns=[target])

    # Simple preprocessing: drop non-numeric for now (can be improved)
    X = X.select_dtypes(include=["number", "bool"])
    if X.empty:
        return {
            "success": False,
            "content": "No numeric features available after dropping non-numeric columns. "
                       "Use run_python to encode categoricals first.",
            "error": "no numeric features",
        }

    # Align and drop NaNs
    data = pd.concat([X, y], axis=1).dropna()
    X = data[X.columns]
    y = data[target]

    if len(data) < 20:
        return {"success": False, "content": "Not enough rows after dropping NaNs", "error": "too few rows"}

    # Detect task
    if task == "auto":
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
            task = "regression"
        else:
            task = "classification"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        importances = sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda t: t[1],
            reverse=True,
        )[:10]

        content = (
            f"Classification model (RandomForest) on target '{target}'\n"
            f"Features used: {list(X.columns)}\n"
            f"Train size: {len(X_train)}, Test size: {len(X_test)}\n"
            f"Accuracy: {acc:.4f}\n\n"
            f"Classification Report:\n{report}\n"
            f"Top feature importances:\n"
            + "\n".join(f"  {name}: {imp:.4f}" for name, imp in importances)
        )
        metrics = {"task": "classification", "accuracy": float(acc)}
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        importances = sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda t: t[1],
            reverse=True,
        )[:10]

        content = (
            f"Regression model (RandomForest) on target '{target}'\n"
            f"Features used: {list(X.columns)}\n"
            f"Train size: {len(X_train)}, Test size: {len(X_test)}\n"
            f"MSE: {mse:.4f}\n"
            f"R²: {r2:.4f}\n\n"
            f"Top feature importances:\n"
            + "\n".join(f"  {name}: {imp:.4f}" for name, imp in importances)
        )
        metrics = {"task": "regression", "mse": float(mse), "r2": float(r2)}

    return {
        "success": True,
        "content": content,
        "metrics": metrics,
    }
