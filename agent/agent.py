"""
DataPilot Agent – core reasoning loop with tool use and self-correction.
Powered by Gemini (Google AI Studio free tier).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Generator

import pandas as pd
from dotenv import load_dotenv

from .tools import TOOL_DEFINITIONS, execute_tool

load_dotenv()


SYSTEM_PROMPT = """You are DataPilot, an expert AI Data Analyst Agent.

You have access to a pandas DataFrame called `df` and the following tools:
- get_schema: always start with this to understand columns, types, missing values and sample rows.
- run_python: write and execute Python code (pandas, numpy, matplotlib). Use print() for output. Call plt.show() for charts.
- make_chart: quick helper for common charts (bar, line, scatter, hist, box, pie, heatmap).
- train_model: train a quick RandomForest classifier or regressor and get metrics + feature importances.

Rules:
1. Always call get_schema first if you do not yet know the columns.
2. Prefer run_python for analysis and complex charts. Keep code short and focused.
3. When using run_python, put a short "explanation" of what the code does.
4. If code fails, carefully read the error and fix it. You have limited retries.
5. Never invent columns that do not exist. Check the schema.
6. For final answers, be clear, concise and insightful. Show key numbers.
7. When you are done, respond with a normal message (no more tool calls) that summarizes the findings.
8. Security: you cannot access the internet, files outside the dataframe, or dangerous modules.

Current dataframe is already loaded. Do not try to read files.
"""


class DataPilotAgent:
    def __init__(
        self,
        df: pd.DataFrame,
        provider: str | None = None,
        max_retries: int = 4,
        max_steps: int = 12,
    ):
        self.df = df
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.history: list[dict[str, Any]] = []
        self._client: Any = None
        self._model_name: str = "gemini-3.8-flash"
        self._init_llm()

    def _init_llm(self):
        self.provider = "gemini"
        self._init_gemini()

    def _init_gemini(self):
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/app/apikey "
                "and put it in a .env file."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self._types = types

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> Generator[dict[str, Any], None, None]:
        """
        Run the agent loop. Yields events the UI can render:

        {"type": "status", "message": "..."}
        {"type": "tool_call", "name": "...", "args": {...}}
        {"type": "tool_result", "name": "...", "result": {...}}
        {"type": "code", "code": "...", "explanation": "..."}
        {"type": "image", "image_b64": "...", "title": "..."}
        {"type": "final", "content": "..."}
        {"type": "error", "message": "..."}
        """
        self.history.append({"role": "user", "content": user_message})

        steps = 0
        consecutive_errors = 0

        while steps < self.max_steps:
            steps += 1
            yield {"type": "status", "message": f"Thinking (step {steps})..."}

            try:
                response = self._call_llm()
            except Exception as e:
                yield {"type": "error", "message": f"LLM call failed: {e}"}
                return

            # Parse response into either tool calls or final text
            tool_calls, text = self._parse_response(response)

            if not tool_calls:
                # Final answer
                final = text or "I could not produce an answer."
                self.history.append({"role": "assistant", "content": final})
                yield {"type": "final", "content": final}
                return

            # Execute tools
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})
                yield {"type": "tool_call", "name": name, "args": args}

                if name == "run_python":
                    yield {
                        "type": "code",
                        "code": args.get("code", ""),
                        "explanation": args.get("explanation", ""),
                    }

                result = execute_tool(name, args, self.df)

                # Surface images
                for img in result.get("images", []):
                    yield {
                        "type": "image",
                        "image_b64": img,
                        "title": result.get("title") or args.get("title") or name,
                    }

                yield {"type": "tool_result", "name": name, "result": result}

                # Feed observation back into history
                observation = result.get("content", str(result))
                if not result.get("success"):
                    consecutive_errors += 1
                    observation = f"TOOL ERROR (attempt related):\n{observation}"
                else:
                    consecutive_errors = 0

                self.history.append({
                    "role": "tool",
                    "name": name,
                    "content": observation,
                })

                if consecutive_errors >= self.max_retries:
                    yield {
                        "type": "final",
                        "content": (
                            "I hit too many consecutive errors while trying to run code. "
                            "Please rephrase the question or ask something simpler.\n\n"
                            f"Last error:\n{result.get('error') or result.get('content')}"
                        ),
                    }
                    return

        yield {
            "type": "final",
            "content": "Reached maximum reasoning steps. Please try a more focused question.",
        }

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    def _call_llm(self) -> Any:
        last_err: Exception = RuntimeError("LLM call failed")
        for attempt in range(3):
            try:
                return self._call_gemini()
            except Exception as e:
                last_err = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                    time.sleep(wait)
                else:
                    raise
        raise last_err

    def _call_gemini(self):
        types = self._types

        # Convert our tool defs to Gemini format
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                    )
                    for t in TOOL_DEFINITIONS
                ]
            )
        ]

        # Build contents from history
        contents = []
        for msg in self.history:
            role = msg["role"]
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
            elif role == "tool":
                # Represent tool result as user message for simplicity
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"[Tool {msg.get('name')} result]\n{msg['content']}")],
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=gemini_tools,
            temperature=0.1,
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        return response

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> tuple[list[dict], str | None]:
        """Return (list of tool_calls, final_text)."""
        return self._parse_gemini(response)

    def _parse_gemini(self, response) -> tuple[list[dict], str | None]:
        tool_calls = []
        text_parts = []

        try:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append({"name": fc.name, "arguments": args})
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        except Exception:
            # Fallback: try response.text
            try:
                text_parts.append(response.text)
            except Exception:
                pass

        text = "\n".join(text_parts).strip() if text_parts else None
        return tool_calls, text


