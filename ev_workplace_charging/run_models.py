import datetime

import fire
import pandas as pd
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from ev_workplace_charging.settings import CHARGER_OUTPUT_POWER
from ev_workplace_charging.settings import METRICS_DATA_DIR
from ev_workplace_charging.settings import MODEL_OUTPUTS_DIR
from ev_workplace_charging.settings import MODEL_TYPES
from ev_workplace_charging.settings import N_CARS
from ev_workplace_charging.settings import SOLVER_TYPE
from ev_workplace_charging.utils.data_loading import load_and_process_charging_costs
from ev_workplace_charging.utils.data_loading import load_and_process_grid_carbon_intensity
from ev_workplace_charging.utils.data_loading import load_and_process_power_profile
from ev_workplace_charging.utils.data_loading import load_and_process_uncontrolled_charging_data
from ev_workplace_charging.utils.data_loading import load_ev_parameters
from ev_workplace_charging.utils.data_loading import load_parking_matrix
from ev_workplace_charging.utils.metrics_computation import compute_max_peak
from ev_workplace_charging.utils.metrics_computation import compute_relative_change
from ev_workplace_charging.utils.metrics_computation import compute_total_carbon_emissions
from ev_workplace_charging.utils.metrics_computation import compute_total_charging_costs
from ev_workplace_charging.utils.model_solving import save_model_output
from ev_workplace_charging.utils.model_solving import setup_and_solve_ccm_model
from ev_workplace_charging.utils.model_solving import setup_and_solve_cem_model
from ev_workplace_charging.utils.model_solving import setup_and_solve_ps_model


def main():
    fire.Fire(run_models)


def run_models(n_jobs: int = 1):
    # Prepare all tasks
    tasks = []
    for day in range(1, 29):
        for ev_portion, n_cars in N_CARS.items():
            for model_type in MODEL_TYPES:
                tasks.append((day, ev_portion, n_cars, model_type))

    # Run tasks in parallel
    Parallel(n_jobs=n_jobs)(delayed(run_models_for_task)(task) for task in tqdm(tasks, desc="Processing tasks"))


def run_models_for_task(task):
    day, ev_portion, n_cars, model_type = task
    date = datetime.date(2023, 2, day)

    # Setup dataframes
    df_parking_matrix = load_parking_matrix(n_cars)
    df_ev_parameters = load_ev_parameters(n_cars)
    df_power_profile, mean_power = load_and_process_power_profile(date)
    df_charging_costs = load_and_process_charging_costs(date)
    df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date)
    df_ucc = load_and_process_uncontrolled_charging_data(ev_portion, df_power_profile)

    output_file_name = f"{date}_{model_type}_{ev_portion}.csv"
    output_file_path = MODEL_OUTPUTS_DIR / output_file_name
    metrics_file_path = METRICS_DATA_DIR / output_file_name

    if not output_file_path.exists():
        if model_type == "ps":
            model = setup_and_solve_ps_model(
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                charger_output_power=CHARGER_OUTPUT_POWER,
                solver_type=SOLVER_TYPE,
            )
        elif model_type == "ccm":
            model = setup_and_solve_ccm_model(
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_charging_costs=df_charging_costs,
                charger_output_power=CHARGER_OUTPUT_POWER,
                solver_type=SOLVER_TYPE,
            )
        elif model_type == "cem":
            model = setup_and_solve_cem_model(
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_grid_carbon_intensity=df_grid_carbon_intensity,
                charger_output_power=CHARGER_OUTPUT_POWER,
                solver_type=SOLVER_TYPE,
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
        cc_ucc = compute_total_charging_costs(df_output["UCC"], df_output["Pb"], df_output["charging_costs"])
        ce_ucc = compute_total_carbon_emissions(df_output["UCC"], df_output["Pb"], df_output["grid_carbon_intensity"])

        mp_model = compute_max_peak(df_output["Tc"])
        cc_model = compute_total_charging_costs(df_output["Tc"], df_output["Pb"], df_output["charging_costs"])
        ce_model = compute_total_carbon_emissions(df_output["Tc"], df_output["Pb"], df_output["grid_carbon_intensity"])

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


if __name__ == "__main__":
    main()
