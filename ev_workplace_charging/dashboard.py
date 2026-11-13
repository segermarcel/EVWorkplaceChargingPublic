import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from ev_workplace_charging.company_configs import COMPANY_CONFIGS
from ev_workplace_charging.settings import (
    CHARGER_OUTPUT_POWER,
    EV_PORTIONS,
    METRICS_DATA_DIR,
    MODEL_OUTPUTS_DIR,
    MODEL_TYPES,
    SLOTS_PER_DAY,
    SOLVER_TYPE,
)
from ev_workplace_charging.utils.data_loading import (
    charging_per_slot,
    compute_uncontrolled_charging,
    generate_ev_parameters,
    load_period_inputs,
    parked_slots,
)
from ev_workplace_charging.utils.metrics_computation import (
    compute_max_peak,
    compute_total_carbon_emissions,
    compute_total_charging_costs,
)
from ev_workplace_charging.utils.model_solving import solve_charging
from ev_workplace_charging.utils.plotting import (
    configure_plot_style,
    create_metrics_fig,
    create_output_fig,
    save_and_write_fig,
)

# Enable wide layout
st.set_page_config(layout="wide")

configure_plot_style()


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
            analysis_period = col2.selectbox(
                "Analysis Period",
                options=["1 Day", "1 Week", "1 Month"],
                index=1,
                help="Set the analysis period for the optimization.",
            )
            # Results are precomputed for February 2024
            max_date = {
                "1 Day": datetime.date(2024, 2, 29),
                "1 Week": datetime.date(2024, 2, 23),
                "1 Month": datetime.date(2024, 2, 1),
            }[analysis_period]
            date = col1.date_input(
                "Date",
                datetime.date(2024, 2, 1),
                min_value=datetime.date(2024, 2, 1),
                max_value=max_date,
                help="Set the starting date for the analysis.",
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

        with st.spinner(f"Running {model_type_long} models ..."):
            output_df, metrics_df = run_strategy_for_period(
                batteries,
                charger_output_power,
                company,
                date_range,
                ev_portion,
                model_type,
                shifts,
            )

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

                save_and_write_fig(fig)

            # Plot metrics
            with col2:
                fig = create_metrics_fig(metrics_df, relative=not show_absolute_metrics)
                save_and_write_fig(fig)


def setup_shift(name, start, end, num_cars):
    st.markdown(name, help="Set the start, end, and number of cars for the shift.")

    col1, col2, col3 = st.columns(3)

    start = col1.time_input("Start", value=start, key=f"{name}_start")
    end = col2.time_input("End", value=end, key=f"{name}_end")
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


def run_strategy_for_period(
    batteries,
    charger_output_power,
    company_name,
    date_range,
    ev_portion,
    model_type,
    shifts,
):
    """Outputs and metrics of one strategy for each day of `date_range`.

    The days are optimised as one continuous horizon, so overnight stays charge in the
    early hours of the following day. The fleet of the day before the range is included
    so that the first day carries its share of overnight charging, and the day after the
    range closes the stays of the last fleet. Results are cached per day; see
    `compute_day_metrics` for how a day's metrics are attributed.
    """
    file_names = [
        f"{company_name}_{date}_{model_type}_{ev_portion}_{charger_output_power}.csv"
        for date in date_range
    ]
    output_paths = [MODEL_OUTPUTS_DIR / name for name in file_names]
    metrics_paths = [METRICS_DATA_DIR / name for name in file_names]

    if all(path.exists() for path in [*output_paths, *metrics_paths]):
        return (
            pd.concat(map(pd.read_csv, output_paths), ignore_index=True),
            pd.concat(map(pd.read_csv, metrics_paths), ignore_index=True),
        )

    one_day = datetime.timedelta(days=1)
    if any(b - a != one_day for a, b in zip(date_range, date_range[1:])):
        raise ValueError("The days of a period must be consecutive")
    days = [date_range[0] - one_day, *date_range, date_range[-1] + one_day]

    company_config = COMPANY_CONFIGS[company_name]
    inputs = load_period_inputs(
        company_config["load_fn"], company_config["is_kw"], days
    )
    fleets = {
        d: generate_ev_parameters(shifts, ev_portion, batteries, d) for d in days[:-1]
    }
    fleet = pd.concat(fleets.values(), ignore_index=True)
    fleet_dates = np.repeat(list(fleets), [len(f) for f in fleets.values()])
    parked = parked_slots(fleet, inputs.index)
    energy = (fleet["E_next"] - fleet["E_ini"]).to_numpy()

    # Model and metrics work in energy per slot [kWh]
    charging = {
        "model": solve_charging(
            model_type, parked, energy, inputs, charger_output_power, SOLVER_TYPE
        ),
        "ucc": compute_uncontrolled_charging(parked, energy, charger_output_power),
    }
    load = {
        key: inputs["Pb"].to_numpy() + charging_per_slot(parked, c, len(inputs))
        for key, c in charging.items()
    }
    df_output = pd.DataFrame(
        {
            "date": inputs.index.date,
            "n": np.tile(np.arange(1, SLOTS_PER_DAY + 1), len(days)),
            "y": load["model"] - inputs["Pb"].to_numpy(),
            "Pb": inputs["Pb"].to_numpy(),
            "Tc": load["model"],
            "UCC": load["ucc"],
            "charging_costs": inputs["price"].to_numpy(),
            "grid_carbon_intensity": inputs["carbon"].to_numpy(),
        }
    )

    output_dfs, metrics_dfs = [], []
    for date, output_path, metrics_path in zip(date_range, output_paths, metrics_paths):
        df_day = df_output[df_output["date"] == date].reset_index(drop=True)
        fleet_charging = {
            key: [(s, e) for s, e, d in zip(parked, c, fleet_dates) if d == date]
            for key, c in charging.items()
        }
        df_metrics = compute_day_metrics(
            date, model_type, ev_portion, df_day, fleet_charging, inputs
        )
        df_day.to_csv(output_path, index=False)
        df_metrics.to_csv(metrics_path, index=False)
        output_dfs.append(df_day)
        metrics_dfs.append(df_metrics)

    return (
        pd.concat(output_dfs, ignore_index=True),
        pd.concat(metrics_dfs, ignore_index=True),
    )


def compute_day_metrics(date, model_type, ev_portion, df_day, fleet_charging, inputs):
    """Metrics of one day, with the strategy ("model") and uncontrolled charging ("ucc").

    The peak [kW] is the site's maximum load on that calendar day, including overnight
    charging of the previous day's fleet. Costs [EUR] and emissions [kg CO2] belong to
    the fleet that arrives on that day, including its charging after midnight, so both
    sides of the comparison charge the same EVs with the same energy.
    """
    price, carbon = inputs["price"].to_numpy(), inputs["carbon"].to_numpy()
    metrics = {"date": date, "model_type": model_type, "ev_portion": ev_portion}
    for suffix, column in (("model", "Tc"), ("ucc", "UCC")):
        evs = fleet_charging[suffix]
        metrics[f"mp_{suffix}"] = compute_max_peak(df_day[column])
        metrics[f"cc_{suffix}"] = sum(
            compute_total_charging_costs(e, price[s]) for s, e in evs
        )
        metrics[f"ce_{suffix}"] = sum(
            compute_total_carbon_emissions(e, carbon[s]) for s, e in evs
        )
    columns = ["date", "model_type", "ev_portion"] + [
        f"{metric}_{suffix}"
        for suffix in ("model", "ucc")
        for metric in ("mp", "cc", "ce")
    ]
    return pd.DataFrame(metrics, index=[0])[columns]


if __name__ == "__main__":
    dashboard()
