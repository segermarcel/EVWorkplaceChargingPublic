import datetime

from joblib import Parallel, delayed
from tqdm import tqdm

from ev_workplace_charging.company_configs import COMPANY_CONFIGS
from ev_workplace_charging.dashboard import (
    get_date_range,
    setup_battery,
    setup_data_run_models_compute_metrics,
    setup_shift,
)
from ev_workplace_charging.settings import (
    CHARGER_OUTPUT_POWER,
    EV_PORTIONS,
    MODEL_TYPES,
)


def run_for_one(company, ev_portion):
    company_config = COMPANY_CONFIGS[company]

    # Shift Patterns
    shifts = [setup_shift(**shift) for shift in company_config["shifts"]]

    # EV Battery Capacities
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


if __name__ == "__main__":
    # Create all combinations of company and ev_portion
    tasks = [
        (company, ev_portion)
        for company in COMPANY_CONFIGS
        for ev_portion in EV_PORTIONS
    ]

    # Run in parallel
    Parallel(n_jobs=1)(
        delayed(run_for_one)(company, ev_portion)
        for company, ev_portion in tqdm(tasks, desc="Running companies and ev_portions")
    )
