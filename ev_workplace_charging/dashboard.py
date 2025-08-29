import datetime

import pandas as pd
import seaborn as sns
import streamlit as st

from ev_workplace_charging.company_configs import COMPANY_CONFIGS
from ev_workplace_charging.settings import (
    CHARGER_OUTPUT_POWER,
    EV_PORTIONS,
    FIGURES_DIR,
    METRICS_DATA_DIR,
    MODEL_OUTPUTS_DIR,
    MODEL_TYPES,
    SOLVER_TYPE,
)
from ev_workplace_charging.utils.data_loading import (
    compute_uncontrolled_charging_data,
    generate_ev_parameters,
    generate_parking_matrix,
    load_charging_costs_from_api,
    load_grid_carbon_intensity_from_api,
)
from ev_workplace_charging.utils.metrics_computation import (
    compute_max_peak,
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
    font_scale=1.1,
)


def dashboard():
    # Sidebar to get all input parameters
    with st.sidebar:
        st.markdown(
            "## Input Parameters", help="Set the input parameters for the optimization."
        )

        company = st.selectbox("Company", options=COMPANY_CONFIGS.keys())
        company_config = COMPANY_CONFIGS[company]

        # Shift Patterns
        with st.expander("Shift Patterns", expanded=False):
            shifts = [setup_shift(**shift) for shift in company_config["shifts"]]

        # EV Battery Capacities
        with st.expander("EV Battery Capacities", expanded=False):
            col1, _ = st.columns(2)
            battery_1 = setup_battery(1, 48, 33)
            battery_2 = setup_battery(2, 71, 33)
            battery_3 = setup_battery(3, 100, 34)
            batteries = (battery_1, battery_2, battery_3)

        # Other parameters
        with st.container():
            col1, col2 = st.columns(2)
            date = col1.date_input(
                "Date",
                datetime.date(2024, 2, 1),
                min_value=datetime.date(2023, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                help="Set the starting date for the analysis.",
            )
            analysis_period = col2.selectbox(
                "Analysis Period",
                options=["1 Day", "1 Week", "1 Month"],
                index=1,
                help="Set the analysis period for the optimization.",
            )
            date_range = get_date_range(date, analysis_period)
        with st.container():
            col1, col2 = st.columns(2)
            charger_output_power = col1.selectbox(
                "Charger output power (kW)",
                options=[CHARGER_OUTPUT_POWER],
                help="Set the charger output power for the optimization.",
            )
            ev_portion = col2.selectbox(
                "EV Portion (%)",
                options=EV_PORTIONS,
                index=0,
                help="Set the EV portion for the optimization.",
            )

            show_electricity_costs = col1.checkbox(
                "Show Electricity Costs",
                value=True if analysis_period == "1 Day" else False,
                help="Set whether to show electricity costs in the output figure of the CCM model.",
            )
            show_grid_carbon_intensity = col2.checkbox(
                "Show Grid Carbon Intensity",
                value=True if analysis_period == "1 Day" else False,
                help="Set whether to show grid carbon intensity in the output figure of the CEM model.",
            )
            show_absolute_metrics = col1.checkbox(
                "Show Absolute Metrics",
                value=False,
                help="Set whether to show absolute metrics in the metrics figure.",
            )

    # Main page
    st.write("# EV Workplace Charging Dashboard")

    # Run models
    for model_type, model_type_long in MODEL_TYPES.items():
        st.write(f"## {model_type_long}")

        output_dfs, metrics_dfs = [], []
        with st.spinner(f"Running {model_type_long} models ..."):
            for date in date_range:
                # Run models and compute metrics
                output_df, metrics_df, output_file_name = (
                    setup_data_run_models_compute_metrics(
                        batteries,
                        charger_output_power,
                        company,
                        date,
                        ev_portion,
                        model_type,
                        shifts,
                    )
                )
                output_dfs.append(output_df)
                metrics_dfs.append(metrics_df)

        # Combine outputs and metrics
        output_df = pd.concat(output_dfs, ignore_index=True)
        metrics_df = pd.concat(metrics_dfs, ignore_index=True)

        # Plot optimization outputs and metrics
        with st.container():
            col1, col2 = st.columns([4, 3])

            # Plot outputs
            with col1:
                if model_type == "ps":
                    fig = create_output_fig(
                        output_df=output_df,
                        color=sns.color_palette("tab10")[0],
                    )
                elif model_type == "ccm":
                    fig = create_output_fig(
                        output_df=output_df,
                        color=sns.color_palette("tab10")[1],
                        electricity_costs=output_df["charging_costs"]
                        if show_electricity_costs
                        else None,
                    )
                elif model_type == "cem":
                    fig = create_output_fig(
                        output_df=output_df,
                        color=sns.color_palette("tab10")[2],
                        grid_carbon_intensity=(
                            output_df["grid_carbon_intensity"]
                            if show_grid_carbon_intensity
                            else None
                        ),
                    )

                figure_path = FIGURES_DIR / f"power_profiles_{output_file_name}.png"
                save_and_write_fig(fig, figure_path)

            # Plot metrics
            with col2:
                fig = create_metrics_fig(metrics_df, relative=not show_absolute_metrics)
                figure_path = FIGURES_DIR / f"metrics_{output_file_name}.png"
                save_and_write_fig(fig, figure_path)


def setup_shift(name, start_h, end_h, num_cars):
    st.markdown(name, help="Set the start, end, and number of cars for the shift.")

    col1, col2, col3 = st.columns(3)

    start = col1.time_input(
        "Start", value=datetime.time(start_h, 0), key=f"{name}_start"
    )
    end = col2.time_input("End", value=datetime.time(end_h, 0), key=f"{name}_end")
    num_cars = col3.number_input("# Cars", value=num_cars, key=f"{name}_cars")

    shift = {
        "start": start,
        "end": end,
        "num_cars": num_cars,
    }

    return shift


def setup_battery(id, capacity, relative_portion):
    st.markdown(
        f"Battery Type {id}",
        help="Set the capacity and relative portion of the battery type.",
    )

    col1, col2 = st.columns(2)

    capacity = col1.number_input(
        "kWh", value=capacity, min_value=1, max_value=100, key=f"{id}_capacity"
    )
    relative_portion = col2.number_input(
        "Relative %",
        value=relative_portion,
        min_value=0,
        max_value=100,
        key=f"{id}_portion",
    )

    return {"capacity": capacity, "relative_portion": relative_portion}


def get_date_range(date, analysis_period):
    # Calculate date range based on selected analysis period
    date_range = []
    if analysis_period == "1 Day":
        date_range = [date]
    elif analysis_period == "1 Week":
        for i in range(7):
            current_date = date + datetime.timedelta(days=i)
            date_range.append(current_date)
    elif analysis_period == "1 Month":
        # Get dates for exactly one month (from start date to same date next month)
        next_month = (
            date.replace(month=date.month + 1)
            if date.month < 12
            else date.replace(year=date.year + 1, month=1)
        )
        current_date = date
        while current_date < next_month:
            date_range.append(current_date)
            current_date = current_date + datetime.timedelta(days=1)

    return date_range


def setup_data_run_models_compute_metrics(
    batteries,
    charger_output_power,
    company_name,
    date,
    ev_portion,
    model_type,
    shifts,
):
    # Run optimization if output file does not already exist, else load output
    output_file_name = (
        f"{company_name}_{date}_{model_type}_{ev_portion}_{charger_output_power}"
    )
    output_file_path = MODEL_OUTPUTS_DIR / f"{output_file_name}.csv"
    metrics_file_path = METRICS_DATA_DIR / f"{output_file_name}.csv"

    if not output_file_path.exists():
        load_power_profile_fn = COMPANY_CONFIGS[company_name]["load_fn"]
        is_kw = COMPANY_CONFIGS[company_name]["is_kw"]

        # Setup dataframes
        df_ev_parameters = generate_ev_parameters(shifts, ev_portion, batteries)
        df_parking_matrix = generate_parking_matrix(df_ev_parameters)
        df_power_profile = load_power_profile_fn(date)
        df_charging_costs = load_charging_costs_from_api(date)
        df_grid_carbon_intensity = load_grid_carbon_intensity_from_api(date)
        df_ucc = compute_uncontrolled_charging_data(
            df_ev_parameters=df_ev_parameters,
            df_power_profile=df_power_profile,
            max_charger_output_power=charger_output_power,
            is_kw=is_kw,
        )

        # Convert to kWh
        if is_kw:
            df_ucc = df_ucc / 4
            df_power_profile = df_power_profile / 4

        if model_type == "ps":
            model = setup_and_solve_ps_model(
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                charger_output_power=charger_output_power,
                solver_type=SOLVER_TYPE,
            )
        elif model_type == "ccm":
            model = setup_and_solve_ccm_model(
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_charging_costs=df_charging_costs,
                charger_output_power=charger_output_power,
                solver_type=SOLVER_TYPE,
            )
        elif model_type == "cem":
            model = setup_and_solve_cem_model(
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_grid_carbon_intensity=df_grid_carbon_intensity,
                charger_output_power=charger_output_power,
                solver_type=SOLVER_TYPE,
            )

        df_output = save_model_output(
            date=date,
            model=model,
            output_file_path=output_file_path,
            df_uncontrolled_charging=df_ucc,
            df_charging_costs=df_charging_costs,
            df_grid_carbon_intensity=df_grid_carbon_intensity,
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
            "mp_model": mp_model,
            "cc_model": cc_model,
            "ce_model": ce_model,
            "mp_ucc": mp_ucc,
            "cc_ucc": cc_ucc,
            "ce_ucc": ce_ucc,
        }

        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(metrics_file_path, index=False)
    else:
        df_output = pd.read_csv(output_file_path)
        df_metrics = pd.read_csv(metrics_file_path)

    return df_output, df_metrics, output_file_name


if __name__ == "__main__":
    dashboard()
