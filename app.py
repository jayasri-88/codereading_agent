"""
DataPilot – AI Data Analyst Agent
Upload a CSV / Excel file and ask questions in plain English.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.agent import DataPilotAgent
from utils.data_loader import load_dataframe, profile_dataframe, profile_to_text

load_dotenv()
st.set_page_config(
    page_title="DataPilot – AI Data Analyst Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown(
    """
<style>
    .stApp { max-width: 1400px; }
    .tool-box {
        background: #0e1117;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .code-box {
        background: #1e1e1e;
        border-left: 4px solid #00d4aa;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
        overflow-x: auto;
    }
    .final-answer {
        background: linear-gradient(135deg, #0f2027, #203a43);
        border: 1px solid #00d4aa;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("DataPilot")
    st.caption("AI Data Analyst Agent")

    st.markdown("---")
    st.subheader("1. Upload Data")
    uploaded = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Your data never leaves this session except the schema sent to the LLM.",
    )

    st.markdown("---")
    st.subheader("2. API Key")
    key_set = bool(os.getenv("GEMINI_API_KEY"))
    st.write("Gemini key:", "set" if key_set else "missing (add to .env)")

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        """
**What is this?**
A real **code-executing agent**:
- Writes pandas code
- Runs it in a sandbox
- Fixes its own errors
- Creates charts
- Can train quick ML models

Built with free models only.
        """
    )

st.title("DataPilot")
st.markdown("Upload a dataset and ask questions in plain English. The agent writes & runs code for you.")

if uploaded is None:
    st.info("Upload a CSV or Excel file in the sidebar to get started.")
    st.markdown(
        """
### Example questions you can ask
- Which month had the highest sales?
- Show me revenue by region as a bar chart
- What is the correlation between age and income?
- Predict whether a customer will churn
- Clean the data and summarize missing values
- Plot a histogram of the price column
        """
    )
    st.stop()

# Load data once
@st.cache_data(show_spinner="Loading data...")
def _load(file_bytes: bytes, name: str) -> pd.DataFrame:
    import io
    bio = io.BytesIO(file_bytes)
    bio.name = name
    return load_dataframe(bio)


file_bytes = uploaded.getvalue()
df = _load(file_bytes, uploaded.name)

# Quick profile
with st.expander("📋 Dataset Profile (click to expand)", expanded=False):
    profile = profile_dataframe(df)
    st.text(profile_to_text(profile))
    st.dataframe(df.head(10), use_container_width=True)

st.markdown(f"**Loaded:** `{uploaded.name}` — **{df.shape[0]:,} rows × {df.shape[1]} columns**")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("images"):
            for img_b64 in msg["images"]:
                st.image(base64.b64decode(img_b64), use_container_width=True)
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        result_placeholder = st.container()
        final_placeholder = st.empty()

        images_collected = []
        final_text = ""

        try:
            agent = DataPilotAgent(df=df)

            for event in agent.run(prompt):
                etype = event["type"]

                if etype == "status":
                    status_placeholder.info(event["message"])

                elif etype == "tool_call":
                    status_placeholder.info(f"🛠️ Calling tool: **{event['name']}**")

                elif etype == "tool_result":
                    res = event["result"]
                    if not res.get("success"):
                        st.warning(f"Tool error (agent will try to fix):\n\n{res.get('error') or res.get('content')}")

                elif etype == "image":
                    img_b64 = event["image_b64"]
                    images_collected.append(img_b64)
                    st.image(base64.b64decode(img_b64), caption=event.get("title", "Chart"), use_container_width=True)

                elif etype == "final":
                    final_text = event["content"]
                    status_placeholder.empty()
                    with final_placeholder:
                        st.markdown(
                            f'<div class="final-answer">{final_text}</div>',
                            unsafe_allow_html=True,
                        )

                elif etype == "error":
                    st.error(event["message"])
                    final_text = event["message"]

        except Exception as e:
            st.error(f"Agent failed to start: {e}")
            final_text = str(e)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text,
            "images": images_collected,
        })
