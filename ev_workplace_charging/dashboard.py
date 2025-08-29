import datetime
from collections import defaultdict

import pandas as pd
import seaborn as sns
import streamlit as st

from ev_workplace_charging.settings import (
    CHARGER_OUTPUT_POWER,
    FIGURES_DIR,
    METRICS_DATA_DIR,
    MODEL_OUTPUTS_DIR,
    MODEL_TYPES,
    N_CARS,
    SOLVER_TYPE,
)
from ev_workplace_charging.utils.data_loading import (
    generate_ev_parameters,
    generate_parking_matrix,
    load_and_process_uncontrolled_charging_data,
    load_charging_costs_from_api,
    load_grid_carbon_intensity_from_api,
    process_uploaded_power_profile,
)
from ev_workplace_charging.utils.metrics_computation import (
    compute_max_peak,
    compute_relative_change,
    compute_total_carbon_emissions,
    compute_total_charging_costs,
)
from ev_workplace_charging.utils.model_solving import (
    save_model_output,
    setup_and_solve_ccm_model,
    setup_and_solve_cem_model,
    setup_and_solve_ps_model,
)
from ev_workplace_charging.utils.plotting import (
    create_metrics_fig,
    create_output_fig,
    save_and_write_fig,
)

# Enable wide layout
st.set_page_config(layout="wide")

sns.set_theme(
    context="notebook",
    palette="tab10",
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
            shift_office = setup_shift(
                "Office Hours", start_h=8, end_h=18, num_cars=100
            )
            shifts = (shift_1, shift_2, shift_office)

        # EV Battery Capacities
        with st.expander("EV Battery Capacities", expanded=False):
            batteries = setup_battery_capacity(num_batteries=3)

        # Power Profile
        uploaded_power_profile = st.file_uploader(
            "Upload building power profile", type=["csv", "xlsx"]
        )

        # Other parameters
        with st.container():
            col1, col2 = st.columns(2)
            date = col1.date_input("Date", datetime.date(2023, 2, 1))
            solver_type = col2.selectbox("Solver", [SOLVER_TYPE])
        with st.container():
            col1, col2 = st.columns(2)
            charger_output_power = col1.number_input(
                "Charger output power (kW)",
                value=CHARGER_OUTPUT_POWER,
                min_value=1,
                max_value=100,
            )
            ev_portion = col2.selectbox("EV Portion", list(N_CARS.keys()))
            n_cars = N_CARS[ev_portion]

    # Main page
    st.write("# EV Workplace Charging Dashboard")

    # Add some space
    st.container(height=32, border=False)

    # Run models
    for model_type, model_type_long in MODEL_TYPES.items():
        st.write(f"## {model_type_long}")

        # Run optimization if output file does not already exist, else load output
        output_file_name = f"{date}_{model_type}_{ev_portion}"
        output_file_path = MODEL_OUTPUTS_DIR / f"{output_file_name}.csv"
        metrics_file_path = METRICS_DATA_DIR / f"{output_file_name}.csv"

        if not output_file_path.exists():
            # Setup dataframes
            df_parking_matrix = generate_parking_matrix(shifts, n_cars)
            df_ev_parameters = generate_ev_parameters(batteries, n_cars)
            df_power_profile, mean_power = process_uploaded_power_profile(
                uploaded_power_profile, date
            )
            df_charging_costs = load_charging_costs_from_api(date)
            df_grid_carbon_intensity = load_grid_carbon_intensity_from_api(date)
            df_ucc = load_and_process_uncontrolled_charging_data(
                ev_portion, df_power_profile
            )

            if model_type == "ps":
                model = setup_and_solve_ps_model(
                    df_parking_matrix=df_parking_matrix,
                    df_ev_parameters=df_ev_parameters,
                    df_power_profile=df_power_profile,
                    charger_output_power=charger_output_power,
                    solver_type=solver_type,
                )
            elif model_type == "ccm":
                model = setup_and_solve_ccm_model(
                    df_parking_matrix=df_parking_matrix,
                    df_ev_parameters=df_ev_parameters,
                    df_power_profile=df_power_profile,
                    df_charging_costs=df_charging_costs,
                    charger_output_power=charger_output_power,
                    solver_type=solver_type,
                )
            elif model_type == "cem":
                model = setup_and_solve_cem_model(
                    df_parking_matrix=df_parking_matrix,
                    df_ev_parameters=df_ev_parameters,
                    df_power_profile=df_power_profile,
                    df_grid_carbon_intensity=df_grid_carbon_intensity,
                    charger_output_power=charger_output_power,
                    solver_type=solver_type,
                )

            df_output = save_model_output(
                model=model,
                df_uncontrolled_charging=df_ucc,
                df_charging_costs=df_charging_costs,
                df_grid_carbon_intensity=df_grid_carbon_intensity,
                mean_power=mean_power,
                output_file_path=output_file_path,
            )

            # Compute metrics
            mp_ucc = compute_max_peak(df_output["UCC"])
            cc_ucc = compute_total_charging_costs(
                df_output["UCC"], df_output["Pb"], df_output["charging_costs"]
            )
            ce_ucc = compute_total_carbon_emissions(
                df_output["UCC"], df_output["Pb"], df_output["grid_carbon_intensity"]
            )

            mp_model = compute_max_peak(df_output["Tc"])
            cc_model = compute_total_charging_costs(
                df_output["Tc"], df_output["Pb"], df_output["charging_costs"]
            )
            ce_model = compute_total_carbon_emissions(
                df_output["Tc"], df_output["Pb"], df_output["grid_carbon_intensity"]
            )

            metrics = {
                "date": date,
                "model_type": model_type,
                "ev_portion": ev_portion,
                "max_peak": compute_relative_change(mp_model, mp_ucc),
                "charging_costs": compute_relative_change(cc_model, cc_ucc),
                "carbon_emissions": compute_relative_change(ce_model, ce_ucc),
            }

            df_metrics = pd.DataFrame(metrics, index=[0])
            df_metrics.to_csv(metrics_file_path, index=False)
        else:
            df_output = pd.read_csv(output_file_path)
            df_metrics = pd.read_csv(metrics_file_path)

        # Plot optimization outputs and metrics
        with st.container():
            col1, col2 = st.columns([4, 3])

            # Plot outputs
            with col1:
                if model_type == "ps":
                    fig = create_output_fig(
                        df_output=df_output,
                        color=sns.color_palette("tab10")[0],
                    )
                elif model_type == "ccm":
                    fig = create_output_fig(
                        df_output=df_output,
                        color=sns.color_palette("tab10")[1],
                        electricity_costs=df_output["charging_costs"],
                    )
                elif model_type == "cem":
                    fig = create_output_fig(
                        df_output=df_output,
                        color=sns.color_palette("tab10")[2],
                        grid_carbon_intensity=df_output["grid_carbon_intensity"],
                    )

                figure_path = (
                    FIGURES_DIR
                    / f"electricity_consumption_profiles_{output_file_name}.svg"
                )
                save_and_write_fig(fig, figure_path)

            # Plot metrics
            with col2:
                fig = create_metrics_fig(df_metrics)
                figure_path = FIGURES_DIR / f"metrics_{output_file_name}.svg"
                save_and_write_fig(fig, figure_path)


# Sidebar setup functions
def setup_shift(title, start_h, end_h, num_cars):
    st.write(title)

    col1, col2, col3 = st.columns(3)

    start = col1.time_input(
        "Start", value=datetime.time(start_h, 0), key=f"{title}_start"
    )
    end = col2.time_input("End", value=datetime.time(end_h, 0), key=f"{title}_end")
    num_cars = col3.number_input("# Cars", value=num_cars, key=f"{title}_cars")

    shift = {"start": start, "end": end, "num_cars": num_cars}

    return shift


def setup_battery_capacity(num_batteries):
    batteries = defaultdict(list)

    for battery_id in range(num_batteries):
        col1, col2 = st.columns(2)

        capacity = col1.number_input(
            "kWh", value=60, min_value=1, max_value=100, key=f"{battery_id}_capacity"
        )
        relative_portion = col2.number_input(
            "Relative %",
            value=100 // num_batteries,
            min_value=0,
            max_value=100,
            key=f"{battery_id}_portion",
        )

        batteries["capacity"].append(capacity)
        batteries["relative_portion"].append(relative_portion)

    return batteries


if __name__ == "__main__":
    dashboard()
