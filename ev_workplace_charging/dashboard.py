import datetime
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = DATA_DIR / "figures"
INPUT_DATA_DIR = DATA_DIR / "input_data"
METRICS_DATA_DIR = DATA_DIR / "metrics"
MODEL_OUTPUTS_DIR = DATA_DIR / "model_outputs"

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

CHARGING_TYPES = {
    "sc": "Smart Charging",
    "bdc": "Bidirectional Charging",
}

METRICS = [
    "Max. Peak",
    "Charging Costs",
    "Carbon Emissions",
]

COLUMN_NAMES = {
    "model_type": "Model Type",
    "charging_type": "Charging Type",
    "ev_portion": "EV Portion",
    "max_peak": METRICS[0],
    "charging_costs": METRICS[1],
    "carbon_emissions": METRICS[2],
}


# Enable wide layout
st.set_page_config(layout="wide")

sns.set_theme(
    context="notebook",
    font_scale=1.5,
)


def dashboard():
    # Sidebar to get all input parameters
    with st.sidebar:
        st.write("## Input Parameters")

        # Shift Patterns
        with st.expander("Shift Patterns", expanded=False):
            shift_1 = setup_shift("First Shift", start_h=8, end_h=16, num_cars=700)
            shift_2 = setup_shift("Second Shift", start_h=16, end_h=0, num_cars=300)
            shift_office = setup_shift("Office Hours", start_h=8, end_h=18, num_cars=100)
            shifts = (shift_1, shift_2, shift_office)

        # EV Battery Capacities
        with st.expander("EV Battery Capacities", expanded=False):
            batteries = setup_battery_capacity(num_batteries=3)

        # Power Profile
        uploaded_power_profile = st.file_uploader("Upload building power profile", type=["csv", "xlsx"])

        # Other parameters
        with st.container():
            col1, col2 = st.columns(2)
            date = col1.date_input("Date", datetime.date(2023, 2, 1))
            solver_type = col2.selectbox("Solver", [SOLVER_TYPE])
        with st.container():
            col1, col2 = st.columns(2)
            charger_output_power = col1.number_input(
                "Charger output power (kW)", value=CHARGER_OUTPUT_POWER, min_value=1, max_value=100
            )
            ev_portion = col2.selectbox("EV Portion", list(N_CARS.keys()))
            n_cars = N_CARS[ev_portion]

    # Setup dataframes
    df_parking_matrix = generate_parking_matrix(shifts, n_cars)
    df_ev_parameters = generate_ev_parameters(batteries, n_cars)
    df_power_profile, mean_power = process_uploaded_power_profile(uploaded_power_profile, date)
    df_charging_costs = load_charging_costs_from_api(date)
    df_grid_carbon_intensity = load_grid_carbon_intensity_from_api(date)
    df_dumb_charging = load_dumb_charging_data_cached(ev_portion, df_power_profile)

    # st.write("# Parking Matrix", df_parking_matrix)
    # st.write("# EV Parameters", df_ev_parameters)
    # st.write("# Power Profile")
    # st.line_chart(df_power_profile.values)
    # st.write("# Charging Costs")
    # st.line_chart(df_charging_costs.values)
    # st.write("# Grid Carbon Intensity")
    # st.line_chart(df_grid_carbon_intensity.values)
    # st.write("# Dumb Charging")
    # st.line_chart(df_dumb_charging.values)

    # Uncontrolled charging metrics
    max_peak_ucc = compute_max_peak(df_dumb_charging)
    charging_costs_ucc = compute_total_charging_costs(df_dumb_charging, df_power_profile, df_charging_costs)
    carbon_emissions_ucc = compute_total_carbon_emissions(df_dumb_charging, df_power_profile, df_grid_carbon_intensity)

    # Main page
    _, col2, _ = st.columns([1, 2, 1])
    col2.write("# EV Workplace Charging Dashboard")

    # Add some space
    st.container(height=32, border=False)

    column_spec = [4, 3]

    # Run models
    for model_type, model_type_long in MODEL_TYPES.items():
        _, col2, _ = st.columns([1, 2, 1])
        col2.write(f"## {model_type_long}")

        # Collect optimization outputs and metrics for each charging type
        metrics = []
        outputs = {}
        for charging_type in ["sc"]:
            vehicle_to_building = True if charging_type == "bdc" else False

            # Run optimization if output file does not already exist, else load output
            output_file_path = MODEL_OUTPUTS_DIR / f"{date}_{model_type}_{charging_type}_{ev_portion}.csv"
            if not output_file_path.exists():
                if model_type == "ps":
                    model = setup_and_solve_ps_model(
                        df_parking_matrix=df_parking_matrix,
                        df_ev_parameters=df_ev_parameters,
                        df_power_profile=df_power_profile,
                        charger_output_power=charger_output_power,
                        vehicle_to_building=vehicle_to_building,
                        solver_type=solver_type,
                    )
                elif model_type == "ccm":
                    model = setup_and_solve_ccm_model(
                        df_parking_matrix=df_parking_matrix,
                        df_ev_parameters=df_ev_parameters,
                        df_power_profile=df_power_profile,
                        df_charging_costs=df_charging_costs,
                        charger_output_power=charger_output_power,
                        vehicle_to_building=vehicle_to_building,
                        solver_type=solver_type,
                    )
                elif model_type == "cem":
                    model = setup_and_solve_cem_model(
                        df_parking_matrix=df_parking_matrix,
                        df_ev_parameters=df_ev_parameters,
                        df_power_profile=df_power_profile,
                        df_grid_carbon_intensity=df_grid_carbon_intensity,
                        charger_output_power=charger_output_power,
                        vehicle_to_building=vehicle_to_building,
                        solver_type=solver_type,
                    )

                df_output = save_model_output(model, date, model_type, charging_type, ev_portion)
            else:
                df_output = pd.read_csv(output_file_path)

            outputs[charging_type] = df_output

            # Compute metrics
            mp = compute_max_peak(df_output["Tc"])
            cc = compute_total_charging_costs(df_output["Tc"], df_power_profile, df_charging_costs)
            ce = compute_total_carbon_emissions(df_output["Tc"], df_power_profile, df_grid_carbon_intensity)

            metrics.append(
                {
                    "date": date,
                    "model_type": model_type,
                    "charging_type": charging_type,
                    "ev_portion": ev_portion,
                    "max_peak": compute_relative_change(mp, max_peak_ucc),
                    "charging_costs": compute_relative_change(cc, charging_costs_ucc),
                    "carbon_emissions": compute_relative_change(ce, carbon_emissions_ucc),
                }
            )

        # Plot optimization outputs and metrics
        with st.container():
            col1, col2 = st.columns(column_spec)

            # Plot outputs
            with col1:
                if model_type == "ps":
                    fig = create_output_fig(
                        df_output_sc=outputs["sc"],
                        df_dumb_charging=df_dumb_charging,
                        color_sc=sns.color_palette("tab10")[0],
                        # df_output_bdc=outputs["bdc"],
                        mean_power=mean_power,
                    )
                elif model_type == "ccm":
                    fig = create_output_fig(
                        df_output_sc=outputs["sc"],
                        df_dumb_charging=df_dumb_charging,
                        color_sc=sns.color_palette("tab10")[1],
                        df_electricity_costs=df_charging_costs,
                        mean_power=mean_power,
                    )
                elif model_type == "cem":
                    fig = create_output_fig(
                        df_output_sc=outputs["sc"],
                        df_dumb_charging=df_dumb_charging,
                        color_sc=sns.color_palette("tab10")[2],
                        df_grid_carbon_intensity=df_grid_carbon_intensity,
                        mean_power=mean_power,
                    )

                plt.tight_layout(pad=2.0)
                fig.savefig(
                    FIGURES_DIR / f"power_profiles_{date}_{model_type}_{charging_type}_{ev_portion}.png", dpi=300
                )
                st.write(fig)

            # Plot metrics
            with col2:
                fig = create_metrics_fig(metrics)
                # Add padding to left side of figure to prevent label cutoff
                plt.tight_layout(pad=2.0)
                fig.savefig(FIGURES_DIR / f"metrics_{date}_{model_type}_{charging_type}_{ev_portion}.png", dpi=300)
                st.write(fig)


# Sidebar setup functions
def setup_shift(title, start_h, end_h, num_cars):
    st.write(title)

    col1, col2, col3 = st.columns(3)

    start = col1.time_input("Start", value=datetime.time(start_h, 0), key=f"{title}_start")
    end = col2.time_input("End", value=datetime.time(end_h, 0), key=f"{title}_end")
    num_cars = col3.number_input("# Cars", value=num_cars, key=f"{title}_cars")

    shift = {"start": start, "end": end, "num_cars": num_cars}

    return shift


def setup_battery_capacity(num_batteries):
    batteries = defaultdict(list)

    for battery_id in range(num_batteries):
        col1, col2 = st.columns(2)

        capacity = col1.number_input("kwH", value=60, min_value=1, max_value=100, key=f"{battery_id}_capacity")
        relative_portion = col2.number_input(
            "Relative %", value=100 // num_batteries, min_value=0, max_value=100, key=f"{battery_id}_portion"
        )

        batteries["capacity"].append(capacity)
        batteries["relative_portion"].append(relative_portion)

    return batteries


@st.cache_data
def generate_parking_matrix(shifts, n_cars):
    # TODO: Implement with given shifts. Use Input Data for now.
    df_parking_matrix = load_parking_matrix(n_cars)
    return df_parking_matrix


@st.cache_data
def generate_ev_parameters(batteries, n_cars):
    # TODO: Implement with given batteries. Use Input Data for now.
    df_ev_parameters = load_ev_parameters(n_cars)
    return df_ev_parameters


@st.cache_data
def process_uploaded_power_profile(uploaded_power_profile=None, date=None):
    # TODO: Implement with power profile. Use Input Data for now.
    df_power_profile, mean_power = load_and_process_power_profile(date)
    return df_power_profile, mean_power


@st.cache_data
def load_charging_costs_from_api(date=None):
    # TODO: Load from API. Use Input data for now.
    df_charging_costs = load_and_process_charging_costs(date)
    return df_charging_costs


@st.cache_data
def load_grid_carbon_intensity_from_api(date=None):
    # TODO: Load from API. Use Input data for now.
    df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date)
    return df_grid_carbon_intensity


@st.cache_data
def load_dumb_charging_data_cached(ev_portion, df_power_profile):
    df_dumb_charging = load_and_process_dumb_charging_data(ev_portion, df_power_profile)
    return df_dumb_charging


@st.cache_resource(show_spinner="Solving Peak Shaving Model...")
def setup_and_solve_ps_model(
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    charger_output_power,
    vehicle_to_building,
    solver_type,
):
    ps_model = setup_model(
        model_type="ps",
        df_parking_matrix=df_parking_matrix,
        df_ev_parameters=df_ev_parameters,
        df_power_profile=df_power_profile,
        df_charging_costs=None,
        df_grid_carbon_intensity=None,
        charger_output_power=charger_output_power,
        vehicle_to_building=vehicle_to_building,
    )

    solve_model(solver_type, ps_model)

    return ps_model


@st.cache_resource(show_spinner="Solving Charging Cost Minimization Model...")
def setup_and_solve_ccm_model(
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    df_charging_costs,
    charger_output_power,
    vehicle_to_building,
    solver_type,
):
    ccm_model = setup_model(
        model_type="ccm",
        df_parking_matrix=df_parking_matrix,
        df_ev_parameters=df_ev_parameters,
        df_power_profile=df_power_profile,
        df_charging_costs=df_charging_costs,
        df_grid_carbon_intensity=None,
        charger_output_power=charger_output_power,
        vehicle_to_building=vehicle_to_building,
    )

    solve_model(solver_type, ccm_model)

    return ccm_model


@st.cache_resource(show_spinner="Solving Carbon Emission Minimization Model...")
def setup_and_solve_cem_model(
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    df_grid_carbon_intensity,
    charger_output_power,
    vehicle_to_building,
    solver_type,
):
    cem_model = setup_model(
        model_type="cem",
        df_parking_matrix=df_parking_matrix,
        df_ev_parameters=df_ev_parameters,
        df_power_profile=df_power_profile,
        df_charging_costs=None,
        df_grid_carbon_intensity=df_grid_carbon_intensity,
        charger_output_power=charger_output_power,
        vehicle_to_building=vehicle_to_building,
    )

    solve_model(solver_type, cem_model)

    return cem_model


@st.cache_data
def create_output_fig(
    df_output_sc,
    df_dumb_charging,
    mean_power,
    color_sc,
    df_output_bdc=None,
    df_electricity_costs=None,
    df_grid_carbon_intensity=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use n as index
    df_output_sc.index = df_output_sc["n"]
    df_dumb_charging.index = df_output_sc.index

    # Divide by mean_power
    df_output_sc["Pb"] = df_output_sc["Pb"] / mean_power
    df_dumb_charging = df_dumb_charging / mean_power
    df_output_sc["Tc"] = df_output_sc["Tc"] / mean_power

    const_C = (min(df_output_sc["Pb"]) + max(df_output_sc["Pb"])) / 2

    g = sns.lineplot(
        data=df_output_sc,
        x=df_output_sc.index,
        y="Pb",
        label="Power Curve Industrial Site",
        color="gray",
        linewidth=1.5,
        ax=ax,
    )
    sns.lineplot(
        data=df_dumb_charging,
        label="Uncontrolled charging (UCC)",
        # linestyle="dotted",
        color=sns.color_palette("tab10")[3],
        linewidth=2.5,
        ax=ax,
    )
    sns.lineplot(
        data=df_output_sc,
        x=df_output_sc.index,
        y="Tc",
        label="Smart Charging (SC)",
        color=color_sc,
        linewidth=2.5,
        ax=ax,
    )
    ax.axhline(
        y=const_C,
        color="gray",
        linestyle="dotted",
        linewidth=1.5,
        label="Constant C",
    )

    if df_output_bdc is not None:
        df_output_bdc["Tc"] = df_output_bdc["Tc"] / mean_power
        df_output_bdc.index = df_output_sc.index
        sns.lineplot(
            data=df_output_bdc,
            x=df_output_bdc.index,
            y="Tc",
            label="Bidirectional charging (BDC)",
            color="goldenrod",
            ax=ax,
        )

    if df_electricity_costs is not None:
        df_electricity_costs.index = df_output_sc.index

        ax2 = ax.twinx()
        sns.lineplot(
            data=df_electricity_costs,
            label="Electricity Costs",
            linestyle="--",
            color="gray",
            linewidth=1.5,
            ax=ax2,
        )

        ax2.set_ylabel("Electricity Costs (p/kWh)")
        ax2.tick_params(axis="y")

        sns.move_legend(ax2, "upper right")

    if df_grid_carbon_intensity is not None:
        df_grid_carbon_intensity.index = df_output_sc.index

        # Use second y-axis
        ax2 = ax.twinx()

        sns.lineplot(
            data=df_grid_carbon_intensity,
            label="Grid Carbon Intensity",
            linestyle="--",
            linewidth=1.5,
            color="gray",
            ax=ax2,
        )

        ax2.set_ylabel("Grid Carbon Intensity (gCO2/kWh)")
        ax2.tick_params(axis="y")

        sns.move_legend(ax2, "upper right")

    g.set_xticks(range(0, 97, 4))
    g.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 1)], rotation=45)

    sns.move_legend(ax, "lower right")

    ax.set_xlabel("")
    ax.set_xlim(0, 97)
    ax.tick_params(axis="y")
    ax.set_ylabel("Relative Power (normalized to Feb. 2023)")
    # ax.set_ylim(1000, 5000)

    return fig


@st.cache_data
def create_metrics_fig(metrics):
    metrics_df = pd.DataFrame(metrics)

    # Filter out charging_typ bdc
    metrics_df = metrics_df[metrics_df["charging_type"] == "sc"]

    # # Drop charging_type column
    # metrics_df = metrics_df.drop(columns=["charging_type"])

    # Translate model_type
    metrics_df["model_type"] = metrics_df["model_type"].map(MODEL_TYPES)

    # Translate column names
    metrics_df = metrics_df.rename(columns=COLUMN_NAMES)

    # Melt
    metrics_df = metrics_df.melt(value_vars=METRICS, var_name="Metric", value_name="Value")

    # Create horizontal bar chart of metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=metrics_df,
        x="Value",
        y="Metric",
        ax=ax,
        orient="h",
        palette=sns.color_palette("tab10"),
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="{:0.1f}%")
    # Add 15% more to xlimits in both directions
    ax.set_xlim(ax.get_xlim()[0] * 1.15, ax.get_xlim()[1] * 1.15)
    ax.set_xticklabels([f"{int(x)}%" for x in ax.get_xticks()])
    ax.set_xlabel("Relative Change (SC - UCC)")
    ax.tick_params(axis="y")
    ax.set_ylabel("")

    return fig


# LOAD FUNCTIONS
def load_parking_matrix(n_cars):
    df_parking_matrix = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=EV_PARKING_MATRIX_SHEET_NAME,
        index_col=0,
        nrows=n_cars,
    )
    return df_parking_matrix


def load_ev_parameters(n_cars):
    df_ev_parameters = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=EV_PARAMETERS_SHEET_NAME,
        index_col=0,
        usecols="A:L",
        nrows=n_cars,
    )
    return df_ev_parameters


def load_power_profile():
    df_power_profile = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=POWER_DATA_PB_SHEET_NAME,
        usecols="B:CU",
        skiprows=5,
        nrows=28,
    )
    return df_power_profile


def load_charging_costs():
    df_charging_costs = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=ELECTRICITY_COSTS_SHEET_NAME,
        usecols="B:AY",
        skiprows=5,
        nrows=28,
    )
    return df_charging_costs


def load_grid_carbon_intensity():
    df_grid_carbon_intensity = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=GRID_CARBON_INTENSITY_SHEET_NAME,
        usecols="B:AY",
        skiprows=5,
        nrows=28,
    )
    return df_grid_carbon_intensity


def load_dumb_charging_data():
    df_dumb_charging = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=UNCONTROLLED_CHARGING_SHEET_NAME,
        index_col=0,
        usecols="C:CU",
        skiprows=3,
    )
    return df_dumb_charging


# LOAD AND PROCESS FUNCTIONS
def load_and_process_power_profile(date):
    df_power_profile = load_power_profile()
    df_power_profile = process_date_df(df_power_profile, date)
    mean_power = df_power_profile.mean().mean()

    return df_power_profile, mean_power


def load_and_process_charging_costs(date):
    df_charging_costs = load_charging_costs()
    df_charging_costs = process_date_df(df_charging_costs, date, interpolation="nearest")

    df_charging_costs.loc[pd.Timestamp("1900-01-01 23:45:00")] = df_charging_costs.loc[
        pd.Timestamp("1900-01-01 23:30:00")
    ]

    return df_charging_costs


def load_and_process_grid_carbon_intensity(date):
    df_grid_carbon_intensity = load_grid_carbon_intensity()
    df_grid_carbon_intensity = process_date_df(df_grid_carbon_intensity, date)
    df_grid_carbon_intensity.loc[pd.Timestamp("1900-01-01 23:45:00")] = df_grid_carbon_intensity.loc[
        pd.Timestamp("1900-01-01 23:30:00")
    ]

    return df_grid_carbon_intensity


def load_and_process_dumb_charging_data(ev_portion, df_power_profile):
    df_dumb_charging = load_dumb_charging_data()

    ev_portion = int(ev_portion[:-1]) / 100
    df_dumb_charging = df_dumb_charging.loc[ev_portion]

    df_dumb_charging = convert_to_datetime_index_and_resample(df_dumb_charging)

    df_dumb_charging += df_power_profile

    return df_dumb_charging


# PROCESS FUNCTIONS
def process_date_df(df, date=None, interpolation="linear"):
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Use Date as index
    df.set_index("Date", inplace=True)

    # Drop Weekday column
    df.drop(columns="Weekday", inplace=True)

    if date is not None:
        # Select only row with the given date
        df = df.loc[pd.to_datetime(date)]

    # Transpose so timestamps are index
    df = df.T

    df = convert_to_datetime_index_and_resample(df, interpolation=interpolation)

    return df


def convert_to_datetime_index_and_resample(df, interval="15min", interpolation="linear"):
    # Convert index to datetime
    df.index = pd.to_datetime(df.index, format="%H:%M:%S")

    # Resample to 15 minute intervals
    df = df.resample(interval).interpolate(interpolation)

    return df


def compute_max_peak(df_with_peaks):
    return df_with_peaks.max()


def compute_total_charging_costs(df_charging, df_power_profile, df_charging_costs):
    charging_costs_total = ((df_charging.values - df_power_profile.values) * df_charging_costs.values / 100).sum()

    return charging_costs_total


def compute_total_carbon_emissions(df_charging, df_power_profile, df_grid_carbon_intensity):
    carbon_emissions_total = (
        (df_charging.values - df_power_profile.values) * df_grid_carbon_intensity.values / 1000
    ).sum()

    return carbon_emissions_total


def compute_relative_change(new, old):
    return (new - old) / old * 100


def setup_model(
    model_type,
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    df_charging_costs,
    df_grid_carbon_intensity,
    charger_output_power,
    vehicle_to_building,
):
    # Process parking matrix
    num_cars, num_timesteps = df_parking_matrix.shape
    M = range(1, num_cars + 1)
    N = range(1, num_timesteps + 1)
    f = {(r, c): df_parking_matrix.values[r - 1, c - 1] for r in M for c in N}

    # Process EV parameters
    E_cap = {r: df_ev_parameters["E_cap"].values[r - 1] for r in M}
    E_ini = {r: df_ev_parameters["E_ini"].values[r - 1] for r in M}
    E_next = {r: df_ev_parameters["E_next"].values[r - 1] for r in M}

    # Process Power Profile
    Pb = {n: df_power_profile.values[n - 1] for n in N}
    C = (min(df_power_profile) + max(df_power_profile)) / 2

    # Process charging costs
    if model_type == "ccm":
        p = {n: df_charging_costs.values[n - 1] for n in N}
    else:
        p = None

    # Process carbon intensity
    if model_type == "cem":
        gci = {n: df_grid_carbon_intensity.values[n - 1] for n in N}
    else:
        gci = None

    t_interval = 15 / 60
    P_MAX = charger_output_power * t_interval
    P_MIN = -P_MAX if vehicle_to_building else 0
    TAU = 1

    model = create_model(model_type, M, N, E_next, E_cap, E_ini, Pb, C, P_MAX, P_MIN, TAU, f, p, gci)

    return model


def solve_model(solver_type, model):
    solver = pyo.SolverFactory(solver_type)
    res = solver.solve(model)  # tee = True   to see detailed solver output
    pyo.assert_optimal_termination(res)


def save_model_output(model, date, model_type, charging_type, ev_portion):
    df_output = pd.DataFrame(
        {
            "n": [n for n in model.N],
            "y": [pyo.value(model.y[n]) for n in model.N],
            "Pb": [pyo.value(model.Pb[n]) for n in model.N],
            "Tc": [pyo.value(model.Pb[n]) + pyo.value(model.y[n]) for n in model.N],
        }
    )

    output_file_path = Path(MODEL_OUTPUTS_DIR / f"{date}_{model_type}_{charging_type}_{ev_portion}.csv")
    df_output.to_csv(output_file_path)

    return df_output


def create_model(model_type, M, N, E_next, E_cap, E_ini, Pb, C, P_MAX, P_MIN, tau, f, p=None, gci=None):
    model = pyo.ConcreteModel(name=model_type)

    # Sets
    model.M = pyo.Set(initialize=M, name="EVs")
    model.N = pyo.Set(initialize=N, name="time intervals")

    # Define and initialise parameters
    model.E_next = pyo.Param(M, initialize=E_next)
    model.E_cap = pyo.Param(M, initialize=E_cap)
    model.E_ini = pyo.Param(M, initialize=E_ini)
    model.Pb = pyo.Param(N, initialize=Pb)
    model.C = pyo.Param(initialize=C)
    model.P_MAX = pyo.Param(initialize=P_MAX, mutable=True)
    model.P_MIN = pyo.Param(initialize=P_MIN, mutable=True)
    model.tau = pyo.Param(initialize=tau)
    model.f = pyo.Param(M, N, initialize=f)

    if model_type == "ccm" and p is not None:
        model.p = pyo.Param(N, initialize=p)

    if model_type == "cem" and gci is not None:
        model.gci = pyo.Param(N, initialize=gci)

    # Define and initialise decision variables
    def x_bounds(mdl, m, n):
        if mdl.f[m, n] == 0:
            return (0, 0)
        else:
            return (P_MIN, P_MAX)

    def x_init(mdl, m, n):
        if mdl.f[m, n] == 0:
            return 0
        else:
            return P_MIN

    model.x = pyo.Var(M, N, initialize=x_init, bounds=x_bounds)  # Charging/ discharging power of EV m in interval i
    model.y = pyo.Var(
        N, initialize=0, within=pyo.Reals
    )  # Total load for charging/discharging the available EVs in interval i
    model.E_fin = pyo.Var(M, initialize=0, within=pyo.NonNegativeReals)

    # Objective function
    if model_type == "ps":
        rule = obj_rule_peak_shaving
    elif model_type == "ccm":
        rule = obj_rule_charging_cost_minimization
    elif model_type == "cem":
        rule = obj_rule_carbon_emission_minimization
    else:
        raise ValueError(f"Model type {model_type} not recognized")

    model.obj = pyo.Objective(rule=rule)  # Default rule: minimize

    def evchargingload_rule(mdl, n):
        return sum(mdl.x[m, n] * mdl.f[m, n] for m in mdl.M) == mdl.y[n]

    model.evchargingload = pyo.Constraint(N, rule=evchargingload_rule)

    def lbcharge_rule(mdl, m, n):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, k] * mdl.f[m, k] for k in range(1, n + 1)) >= 0

    model.lbcharge = pyo.Constraint(M, N, rule=lbcharge_rule)

    def ubcharge_rule(mdl, m, n):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, k] * mdl.f[m, k] for k in range(1, n + 1)) <= mdl.E_cap[m]

    model.ubcharge = pyo.Constraint(M, N, rule=ubcharge_rule)

    def SoCfinal_rule(mdl, m):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, n] * mdl.f[m, n] for n in mdl.N) == mdl.E_fin[m]

    model.SoCfinal = pyo.Constraint(M, rule=SoCfinal_rule)

    def SoCnext_rule(mdl, m):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, n] * mdl.f[m, n] for n in mdl.N) >= mdl.E_next[m]

    model.SoCnext = pyo.Constraint(M, rule=SoCnext_rule)

    return model


def obj_rule_peak_shaving(mdl):
    return sum(pow((mdl.Pb[n] + mdl.y[n] - mdl.C), 2) for n in mdl.N)


def obj_rule_charging_cost_minimization(mdl):
    return sum(mdl.y[n] * mdl.p[n] for n in mdl.N)


def obj_rule_carbon_emission_minimization(mdl):
    return sum(mdl.y[n] * mdl.gci[n] for n in mdl.N)


if __name__ == "__main__":
    dashboard()
