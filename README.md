# Crypto Trader Behavioral Analysis

An end-to-end data science project analyzing algorithmic trading performance against the crypto Fear & Greed sentiment index. Covers data preparation, behavioral analysis, trader segmentation, a predictive model, and an interactive dashboard.

---

[![Live App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen?logo=streamlit)](https://trader-performance-vs-market-sentiment-dinesh.streamlit.app)

## 🌐 Live Demo

🚀 **Interactive Streamlit Dashboard**  
🔗 https://trader-performance-vs-market-sentiment-dinesh.streamlit.app

## Project Files

| File | Description |
|---|---|
| `eda_notebook.ipynb` | **Main deliverable** — full walkthrough, Part A → B → C → Bonus + write-up |
| `model.py` | Standalone script: trains the Random Forest model and saves outputs |
| `clustering.py` | Standalone script: runs KMeans and exports trader segments |
| `eda_analysis.py` | Lightweight EDA script for quick exploration |
| `dashboard.py` | **Streamlit dashboard** — interactive exploration of all results |
| `README.md` | This file |

**Generated outputs (after running scripts):**

| File | What it is |
|---|---|
| `daily_features.csv` | 480-row daily feature table used by the model |
| `trader_segments.csv` | Per-account cluster assignments |
| `pnl_model.pkl` | Saved Random Forest model |
| `summary_metrics.csv` | Top-level KPI table |
| `*.png` | Chart exports from the analysis |

---

## Setup

You need Python 3.8+. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib jupyter nbformat
```

Place both source files in the same folder as the scripts:
- `fear_greed_index.csv`
- `historical_data.csv`

---

## How to Run

### Option 1 — Jupyter Notebook (recommended for submission)

```bash
jupyter notebook eda_notebook.ipynb
```

Run all cells top to bottom. No other files needed — the notebook is fully self-contained.

### Option 2 — Dashboard (interactive exploration)

First generate the data files the dashboard reads:

```bash
python model.py
python clustering.py
```

Then launch the dashboard:

```bash
python -m streamlit run dashboard.py
```

Opens at `http://localhost:8501`

### Option 3 — Standalone scripts

```bash
python eda_analysis.py    # basic EDA + charts
python model.py           # predictive model
python clustering.py      # trader archetypes
```

---

## What's Covered

| Part | Coverage |
|---|---|
| **A — Data Prep** | Load, document, clean, align timestamps, key metrics |
| **B — Analysis** | Fear vs Greed performance, behavioral shifts, 3 segments, 3 backed insights |
| **C — Strategy** | 2 actionable rules of thumb with evidence |
| **Bonus** | RF classifier (82% accuracy), KMeans archetypes, Streamlit dashboard |
