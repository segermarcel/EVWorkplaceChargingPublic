# EVWorkplaceChargingPublic

## Setup

1. Set up Python 3.12 (e.g. with Miniconda)

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init zsh
```

2. Install dependencies with [Poetry](https://python-poetry.org)

Install Poetry by following the install instructions for your OS on their [website](https://python-poetry.org/docs/#installation).
Then run the following commands to install the dependecies:

```bash
poetry install
```

3. Download input data Excel file and put it under `data/input_data`

4. Install gurobi solver: [Getting Started with Gurobi Optimizer](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)

## Run streamlit dashboard

```bash
poetry run streamlit run ev_workplace_charging/dashboard.py
```

## Run optimization models for all days of February 2023 and all scenarios

```bash
poetry run python ev_workplace_charging/run_models.py
```

## Visualize input data

```bash
poetry run streamlit run ev_workplace_charging/visualize_input_data.py
```

## Visualize metrics

```bash
poetry run streamlit run ev_workplace_charging/visualize_metrics.py
```
