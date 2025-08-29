# EVWorkplaceChargingPublic

Code for https://ev-workplace-charging.streamlit.app/

## Local setup

1. Install dependencies with uv

Install uv by following the install instructions for your OS on their [website](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

2. Download input data Excel file and put it under `data/input_data`

3. Install gurobi solver: [Getting Started with Gurobi Optimizer](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)

## Run streamlit dashboard locally

```bash
uv run streamlit run ev_workplace_charging/dashboard.py
```
