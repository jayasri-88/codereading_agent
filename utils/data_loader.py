"""Data loading and basic profiling utilities."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


def load_dataframe(file) -> pd.DataFrame:
    """Load CSV or Excel file into a pandas DataFrame."""
    name = getattr(file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    # Default to CSV (handles .csv and unknown)
    return pd.read_csv(file)


def profile_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> dict[str, Any]:
    """Return a rich but compact schema + profile for the LLM."""
    profile: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {},
        "sample_rows": df.head(sample_rows).to_dict(orient="records"),
        "missing_summary": {},
        "numeric_summary": {},
    }

    for col in df.columns:
        series = df[col]
        col_info = {
            "dtype": str(series.dtype),
            "non_null": int(series.count()),
            "null_count": int(series.isna().sum()),
            "null_pct": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info["min"] = float(series.min()) if series.count() > 0 else None
            col_info["max"] = float(series.max()) if series.count() > 0 else None
            col_info["mean"] = float(series.mean()) if series.count() > 0 else None
            col_info["std"] = float(series.std()) if series.count() > 0 else None
            profile["numeric_summary"][col] = {
                "min": col_info["min"],
                "max": col_info["max"],
                "mean": round(col_info["mean"], 4) if col_info["mean"] is not None else None,
                "std": round(col_info["std"], 4) if col_info["std"] is not None else None,
            }
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["min"] = str(series.min()) if series.count() > 0 else None
            col_info["max"] = str(series.max()) if series.count() > 0 else None
        else:
            # categorical / object
            top = series.value_counts(dropna=True).head(5)
            col_info["top_values"] = {str(k): int(v) for k, v in top.items()}

        profile["columns"][col] = col_info
        if col_info["null_count"] > 0:
            profile["missing_summary"][col] = col_info["null_pct"]

    return profile


def profile_to_text(profile: dict[str, Any]) -> str:
    """Human / LLM friendly text version of the profile."""
    lines = [
        f"Dataset shape: {profile['shape']['rows']} rows × {profile['shape']['columns']} columns",
        "",
        "Columns:",
    ]
    for col, info in profile["columns"].items():
        line = f"  - {col} ({info['dtype']}): non-null={info['non_null']}, nulls={info['null_count']} ({info['null_pct']}%)"
        if "mean" in info and info["mean"] is not None:
            line += f", mean={info['mean']:.4g}, min={info['min']}, max={info['max']}"
        if "top_values" in info:
            tops = ", ".join(f"{k}:{v}" for k, v in list(info["top_values"].items())[:3])
            line += f", top=[{tops}]"
        lines.append(line)

    if profile["missing_summary"]:
        lines.append("")
        lines.append("Columns with missing values:")
        for col, pct in profile["missing_summary"].items():
            lines.append(f"  - {col}: {pct}%")

    lines.append("")
    lines.append("Sample rows (first 5):")
    for i, row in enumerate(profile["sample_rows"]):
        lines.append(f"  [{i}] {row}")

    return "\n".join(lines)
