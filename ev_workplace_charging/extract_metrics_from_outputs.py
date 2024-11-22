import datetime

import pandas as pd
import streamlit as st
from tqdm import trange

from ev_workplace_charging.dashboard import CHARGING_TYPES
from ev_workplace_charging.dashboard import compute_max_peak
from ev_workplace_charging.dashboard import compute_relative_change
from ev_workplace_charging.dashboard import compute_total_carbon_emissions
from ev_workplace_charging.dashboard import compute_total_charging_costs
from ev_workplace_charging.dashboard import load_and_process_charging_costs
from ev_workplace_charging.dashboard import load_and_process_dumb_charging_data
from ev_workplace_charging.dashboard import load_and_process_grid_carbon_intensity
from ev_workplace_charging.dashboard import load_and_process_power_profile
from ev_workplace_charging.dashboard import METRICS_DATA_DIR
from ev_workplace_charging.dashboard import MODEL_OUTPUTS_DIR
from ev_workplace_charging.dashboard import MODEL_TYPES
from ev_workplace_charging.dashboard import N_CARS


def main():
    metrics = []

    for day in trange(1, 29, desc="Iterating over days"):
        date = datetime.date(2023, 2, day)

        df_power_profile, _ = load_and_process_power_profile(date)
        df_charging_costs = load_and_process_charging_costs(date)
        df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date)

        for ev_portion in N_CARS.keys():
            df_dumb_charging = load_and_process_dumb_charging_data(ev_portion, df_power_profile)

            max_peak_ucc = compute_max_peak(df_dumb_charging)
            charging_costs_ucc = compute_total_charging_costs(df_dumb_charging, df_power_profile, df_charging_costs)
            carbon_emissions_ucc = compute_total_carbon_emissions(
                df_dumb_charging, df_power_profile, df_grid_carbon_intensity
            )

            for model_type in MODEL_TYPES:
                for charging_type in CHARGING_TYPES:

                    output_file_path = MODEL_OUTPUTS_DIR / f"{date}_{model_type}_{charging_type}_{ev_portion}.csv"

                    output_df = pd.read_csv(output_file_path)

                    mp = compute_max_peak(output_df["Tc"])
                    cc = compute_total_charging_costs(output_df["Tc"], df_power_profile, df_charging_costs)
                    ce = compute_total_carbon_emissions(output_df["Tc"], df_power_profile, df_grid_carbon_intensity)

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

    metrics_df = pd.DataFrame(metrics)
    st.write(metrics_df)

    # Save
    metrics_df.to_csv(METRICS_DATA_DIR / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
