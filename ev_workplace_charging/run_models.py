import datetime

from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from ev_workplace_charging.dashboard import CHARGER_OUTPUT_POWER
from ev_workplace_charging.dashboard import CHARGING_TYPES
from ev_workplace_charging.dashboard import load_and_process_charging_costs
from ev_workplace_charging.dashboard import load_and_process_grid_carbon_intensity
from ev_workplace_charging.dashboard import load_and_process_power_profile
from ev_workplace_charging.dashboard import load_ev_parameters
from ev_workplace_charging.dashboard import load_parking_matrix
from ev_workplace_charging.dashboard import MODEL_OUTPUTS_DIR
from ev_workplace_charging.dashboard import MODEL_TYPES
from ev_workplace_charging.dashboard import N_CARS
from ev_workplace_charging.dashboard import save_model_output
from ev_workplace_charging.dashboard import setup_model
from ev_workplace_charging.dashboard import solve_model
from ev_workplace_charging.dashboard import SOLVER_TYPE


def main():
    # Prepare all tasks
    tasks = []
    for day in range(1, 29):
        for ev_portion in N_CARS.keys():
            n_cars = N_CARS[ev_portion]
            for model_type in MODEL_TYPES:
                for charging_type in CHARGING_TYPES:
                    tasks.append((day, ev_portion, n_cars, model_type, charging_type))

    # Run tasks in parallel
    Parallel(n_jobs=10)(delayed(run_models_for_task)(task) for task in tqdm(tasks, desc="Processing tasks"))


def run_models_for_task(task):
    day, ev_portion, n_cars, model_type, charging_type = task
    date = datetime.date(2023, 2, day)
    vehicle_to_building = charging_type == "bdc"

    # Setup dataframes
    df_parking_matrix = load_parking_matrix(n_cars)
    df_ev_parameters = load_ev_parameters(n_cars)
    df_power_profile, _ = load_and_process_power_profile(date)
    df_charging_costs = load_and_process_charging_costs(date)
    df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date)

    output_file_path = MODEL_OUTPUTS_DIR / f"{date}_{model_type}_{charging_type}_{ev_portion}.csv"
    if not output_file_path.exists():
        if model_type == "ps":
            model = setup_model(
                model_type="ps",
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_charging_costs=None,
                df_grid_carbon_intensity=None,
                charger_output_power=CHARGER_OUTPUT_POWER,
                vehicle_to_building=vehicle_to_building,
            )
            solve_model(SOLVER_TYPE, model)

        elif model_type == "ccm":
            model = setup_model(
                model_type="ccm",
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_charging_costs=df_charging_costs,
                df_grid_carbon_intensity=None,
                charger_output_power=CHARGER_OUTPUT_POWER,
                vehicle_to_building=vehicle_to_building,
            )
            solve_model(SOLVER_TYPE, model)

        elif model_type == "cem":
            model = setup_model(
                model_type="cem",
                df_parking_matrix=df_parking_matrix,
                df_ev_parameters=df_ev_parameters,
                df_power_profile=df_power_profile,
                df_charging_costs=None,
                df_grid_carbon_intensity=df_grid_carbon_intensity,
                charger_output_power=CHARGER_OUTPUT_POWER,
                vehicle_to_building=vehicle_to_building,
            )

            solve_model(SOLVER_TYPE, model)

        save_model_output(model, date, model_type, charging_type, ev_portion)


if __name__ == "__main__":
    main()
