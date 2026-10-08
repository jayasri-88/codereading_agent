# DataPilot – AI Data Analyst Agent

A real code-executing AI agent that lets you upload any CSV/Excel file and ask questions in plain English.

The agent:
- Profiles your data
- Writes pandas / matplotlib / scikit-learn code
- Executes it inside a restricted sandbox
- Reads errors and **self-corrects** (up to 4 retries)
- Returns charts, tables and clear insights

Built entirely with the **Gemini free tier** (Google AI Studio).

---

## Why this project matters

Most student projects are simple RAG chatbots.  
DataPilot demonstrates the actual agent pattern used in production:

```
User question
    → LLM decides which tool to call
    → Tool runs (code / chart / model)
    → Observation goes back to LLM
    → LLM may fix code or call another tool
    → Final answer
```

This is the same loop behind modern coding agents, data copilots and internal ops agents.

---

## Quick Start (5 minutes)

### 1. Clone / open the folder
```bash
cd datapilot
```

### 2. Create virtual environment & install
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get a free Gemini API key
Go to https://aistudio.google.com/app/apikey and create a key — no credit card needed.

### 4. Create `.env`
```bash
cp .env.example .env
```
Edit `.env` and paste your key:
```
GEMINI_API_KEY=AIza...
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
```

### 5. Run
```bash
streamlit run app.py
```

Open the URL shown in the terminal, upload any CSV/Excel, and start asking questions.

---

## Example questions

- Which product category has the highest revenue?
- Plot monthly sales as a line chart
- Show the correlation heatmap of numeric columns
- How many missing values are there per column?
- Predict the target column and tell me the most important features
- Clean the data (fill missing values with median) and summarize

---

## Project Structure

```
datapilot/
├── app.py                  # Streamlit UI
├── agent/
│   ├── agent.py            # Core agent loop (Gemini)
│   ├── tools.py            # Tool definitions + implementations
│   └── sandbox.py          # Restricted Python executor
├── utils/
│   └── data_loader.py      # CSV/Excel loading + profiling
├── requirements.txt
├── .env.example
└── README.md
```

---

## Safety (Sandbox)

AI-written code runs with:
- 15–20 second timeout
- Blocked modules: `os`, `sys`, `subprocess`, `shutil`, `socket`, `pathlib`, etc.
- No file system access outside the dataframe
- No network
- Restricted builtins (`open`, `exec`, `eval` blocked)

Always review the generated code in the UI.

---

## LLM Provider

| Provider | Model                | Notes                        |
|----------|----------------------|------------------------------|
| Gemini   | gemini-2.0-flash     | Free tier, no credit card    |

Configure via `GEMINI_MODEL` in `.env` to switch between Gemini models (e.g. `gemini-1.5-pro`).

---

## Features

1. Auto-profiling (shape, dtypes, missing %, sample rows)
2. Natural language → pandas code
3. Charts (bar, line, scatter, hist, box, pie, heatmap)
4. Self-correction loop on code errors
5. Quick ML (RandomForest classification / regression + feature importance)
6. Clean Streamlit UI with live status and images

---

## Limitations & Honest Notes

- Gemini free tier has rate limits (~15 RPM). For heavy use consider upgrading to a paid tier.
- Only schema + sample rows are sent to the LLM (privacy friendly).
- Large files (>100k rows) work but charts/ML may be slower.
- Complex multi-step analysis may need a few retries.

---

## Next Steps (if you want to level it up)

- Add FastAPI backend + React frontend
- Persist chat history
- Support Plotly interactive charts
- Evaluation suite (20 fixed questions on Titanic / sales data)
- Deploy on Hugging Face Spaces or Streamlit Cloud

---

## License

MIT – use it, modify it, put it in your portfolio.
