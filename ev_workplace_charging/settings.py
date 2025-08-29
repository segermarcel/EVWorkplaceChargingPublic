from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"

CHARGING_COSTS_DIR = DATA_DIR / "charging_costs"
FIGURES_DIR = DATA_DIR / "figures"
GRID_CARBON_INTENSITY_DIR = DATA_DIR / "grid_carbon_intensity"
INPUT_DATA_DIR = DATA_DIR / "input_data"
METRICS_DATA_DIR = DATA_DIR / "metrics"
MODEL_OUTPUTS_DIR = DATA_DIR / "model_outputs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CHARGER_OUTPUT_POWER = 11
SOLVER_TYPE = "gurobi"

MODEL_TYPES = {
    "ps": "Peak Minimisation & Valley Filling (PM-VF)",
    "ccm": "Charging Cost Minimisation (CCM)",
    "cem": "Carbon Emission Minimisation (CEM)",
}

METRICS = [
    "Max. Peak",
    "Charging Costs",
    "Carbon Emissions",
]

COLUMN_NAMES = {
    "model_type": "Model Type",
    "ev_portion": "EV Portion",
    "max_peak": METRICS[0],
    "charging_costs": METRICS[1],
    "carbon_emissions": METRICS[2],
}

EV_PORTIONS = [15, 30, 50, 80, 100]
