import datetime
import warnings

import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ev_workplace_charging.settings import (
    CHARGER_OUTPUT_POWER,
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
    load_power_profile_01,
    load_power_profile_02,
    load_power_profile_03,
    load_power_profile_04,
    load_power_profile_05,
    load_power_profile_06,
    load_power_profile_07,
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

warnings.filterwarnings("ignore")


def setup_shift(title, start_h, end_h, num_cars):
    # st.markdown(title, help="Set the start, end, and number of cars for the shift.")

    # col1, col2, col3 = st.columns(3)

    # start = col1.time_input(
    #     "Start", value=datetime.time(start_h, 0), key=f"{title}_start"
    # )
    # end = col2.time_input("End", value=datetime.time(end_h, 0), key=f"{title}_end")
    # num_cars = col3.number_input("# Cars", value=num_cars, key=f"{title}_cars")

    shift = {
        "start": datetime.time(start_h, 0),
        "end": datetime.time(end_h, 0),
        "num_cars": num_cars,
    }

    return shift


COMPANY_CONFIGS = {
    "01": {
        "shifts": (
            setup_shift("Early Shift", start_h=6, end_h=14, num_cars=90),
            setup_shift("Late Shift", start_h=14, end_h=22, num_cars=80),
            setup_shift("Night Shift", start_h=22, end_h=6, num_cars=60),
        ),
        "is_kw": True,
        "load_fn": load_power_profile_01,
    },
    "02": {
        "shifts": (setup_shift("Office Hours", start_h=8, end_h=16, num_cars=50),),
        "is_kw": False,
        "load_fn": load_power_profile_02,
    },
    "03": {
        "shifts": (setup_shift("Office Hours", start_h=8, end_h=16, num_cars=50),),
        "is_kw": False,
        "load_fn": load_power_profile_03,
    },
    "04": {
        "shifts": (
            setup_shift("Early Shift", start_h=6, end_h=14, num_cars=100),
            setup_shift("Late Shift", start_h=14, end_h=22, num_cars=150),
            setup_shift("Night Shift", start_h=22, end_h=6, num_cars=80),
            setup_shift("Office Hours", start_h=8, end_h=16, num_cars=300),
        ),
        "is_kw": False,
        "load_fn": load_power_profile_04,
    },
    "05": {
        "shifts": (
            setup_shift("Early Shift", start_h=6, end_h=14, num_cars=250),
            setup_shift("Late Shift", start_h=14, end_h=22, num_cars=175),
            setup_shift("Night Shift", start_h=22, end_h=6, num_cars=80),
            setup_shift("Office Hours", start_h=8, end_h=16, num_cars=60),
        ),
        "is_kw": False,
        "load_fn": load_power_profile_05,
    },
    "06": {
        "shifts": (
            setup_shift("Early Shift", start_h=6, end_h=14, num_cars=100),
            setup_shift("Late Shift", start_h=14, end_h=22, num_cars=70),
            setup_shift("Office Hours", start_h=8, end_h=16, num_cars=100),
        ),
        "is_kw": True,
        "load_fn": load_power_profile_06,
    },
    "08": {
        "shifts": (
            setup_shift("Early Shift", start_h=6, end_h=14, num_cars=170),
            setup_shift("Late Shift", start_h=14, end_h=22, num_cars=30),
            setup_shift("Office Hours", start_h=8, end_h=16, num_cars=140),
        ),
        "is_kw": False,
        "load_fn": load_power_profile_07,
    },
}


def dashboard(company, ev_portion):
    # Shift Patterns
    shifts = COMPANY_CONFIGS[company]["shifts"]

    battery_1 = setup_battery(1, 48, 33)
    battery_2 = setup_battery(2, 71, 33)
    battery_3 = setup_battery(3, 100, 34)
    batteries = (battery_1, battery_2, battery_3)

    date = datetime.date(2024, 2, 1)
    analysis_period = "1 Week"
    date_range = get_date_range(date, analysis_period)

    charger_output_power = CHARGER_OUTPUT_POWER

    # Run models
    for model_type, model_type_long in tqdm(
        MODEL_TYPES.items(),
        desc="Iterating over model types",
        total=len(MODEL_TYPES),
        leave=False,
    ):
        output_dfs, metrics_dfs = [], []
        for date in tqdm(
            date_range,
            desc="Iterating over dates",
            total=len(date_range),
            leave=False,
        ):
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
        if model_type == "ps":
            fig = create_output_fig(
                df_output=output_df,
                color=sns.color_palette("tab10")[0],
            )
        elif model_type == "ccm":
            fig = create_output_fig(
                df_output=output_df,
                color=sns.color_palette("tab10")[1],
                electricity_costs=None,
            )
        elif model_type == "cem":
            fig = create_output_fig(
                df_output=output_df,
                color=sns.color_palette("tab10")[2],
                grid_carbon_intensity=None,
            )

        figure_path = FIGURES_DIR / f"power_profiles_{output_file_name}.png"
        save_and_write_fig(fig, figure_path)

        fig = create_metrics_fig(metrics_df, relative=False)
        figure_path = FIGURES_DIR / f"metrics_{output_file_name}.png"
        save_and_write_fig(fig, figure_path)


def setup_battery(id, capacity, relative_portion):
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

    # if not output_file_path.exists():
    if 1:
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

        # st.write("Vebrauch kWh: ", df_power_profile.describe())

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
    for company in tqdm(
        COMPANY_CONFIGS, desc="Running companies", total=len(COMPANY_CONFIGS)
    ):
        for ev_portion in tqdm(
            [15, 30, 50, 80, 100],
            desc="Running ev_portions",
        ):
            print(f"Running {company} with {ev_portion}% of EVs")
            dashboard(company, ev_portion)
