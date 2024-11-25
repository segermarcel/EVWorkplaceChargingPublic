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

INPUT_DATA_FILE_PATH = INPUT_DATA_DIR / "EVWorkplaceCharging-Input-Data_V2.xlsx"

# Input data sheet names
EV_PARAMETERS_SHEET_NAME = "EV Parameters"
EV_PARKING_MATRIX_SHEET_NAME = "EV Parking Matrix"
POWER_DATA_PB_SHEET_NAME = "Power Data Pb_original"
ELECTRICITY_COSTS_SHEET_NAME = "Electricity Costs"
GRID_CARBON_INTENSITY_SHEET_NAME = "Grid Carbon Intensity"
UNCONTROLLED_CHARGING_SHEET_NAME = "Uncontrolled Charging"

CHARGER_OUTPUT_POWER = 11
SOLVER_TYPE = "gurobi"
