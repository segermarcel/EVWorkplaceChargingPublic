from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
INPUT_DATA_DIR = DATA_DIR / "input_data"
METRICS_DATA_DIR = DATA_DIR / "metrics"
MODEL_OUTPUTS_DIR = DATA_DIR / "model_outputs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DATA_FILE_PATH = INPUT_DATA_DIR / "EVWorkplaceCharging-Input-Data.xlsx"

# Input data sheet names
EV_PARAMETERS_SHEET_NAME = "EV Parameters"
EV_PARKING_MATRIX_SHEET_NAME = "EV Parking Matrix"
POWER_DATA_SHEET_NAME = "Electricity Consumption Profile"
ELECTRICITY_COSTS_SHEET_NAME = "Electricity Costs"
GRID_CARBON_INTENSITY_SHEET_NAME = "Grid Carbon Intensity"
UCC_SUMMARY_SHEET_NAME = "UCC Summary"

CHARGER_OUTPUT_POWER = 11
SOLVER_TYPE = "gurobi"

N_CARS = {
    "15%": 142,
    "30%": 322,
    "50%": 563,
    "80%": 871,
    "100%": 1100,
}

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
