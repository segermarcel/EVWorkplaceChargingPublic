import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from ev_workplace_charging.settings import FIGURES_DIR
from ev_workplace_charging.utils.data_loading import load_and_process_charging_costs
from ev_workplace_charging.utils.data_loading import load_and_process_grid_carbon_intensity
from ev_workplace_charging.utils.data_loading import load_and_process_power_profile
from ev_workplace_charging.utils.plotting import save_and_write_fig

sns.set_theme(
    context="notebook",
    palette="tab10",
    font_scale=1.5,
)


def main():
    st.write("# Visualise Input Data")

    st.write("## Electricity consumption profile")
    df_power_profile, mean_power = load_and_process_power_profile(date=None)

    # Divide each row by the mean of all values in df
    df_power_profile = df_power_profile.div(mean_power, axis=0)

    # Turn into long-form
    df_power_profile = process_df_to_long_form(df_power_profile)

    fig, ax = plot_long_form_df(df_power_profile)
    ax.set_ylabel("Electricity consumption [kWh] (normalised)")

    save_and_write_fig(fig, FIGURES_DIR / "electricity_consumption_profile.svg")

    st.write("## Charging costs")
    df_charging_costs = load_and_process_charging_costs(date=None)
    df_charging_costs = process_df_to_long_form(df_charging_costs)

    fig, ax = plot_long_form_df(df_charging_costs)
    ax.set_ylabel("Electricity costs [p/kWh]")

    save_and_write_fig(fig, FIGURES_DIR / "electricity_costs.svg")

    st.write("## Grid carbon intensity")
    df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date=None)
    df_grid_carbon_intensity = process_df_to_long_form(df_grid_carbon_intensity)

    fig, ax = plot_long_form_df(df_grid_carbon_intensity)
    ax.set_ylabel("Carbon intensity [gCO2/kWh]")

    save_and_write_fig(fig, FIGURES_DIR / "carbon_intensity.svg")


def process_df_to_long_form(df):
    # Turn into long-form
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Time"}, inplace=True)
    df = pd.melt(df, id_vars="Time", var_name="Date", value_name="Value")

    # Convert Time col to string
    df["Time"] = df["Time"].astype(str)

    return df


def plot_long_form_df(df_power_profile):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_power_profile,
        x="Time",
        y="Value",
        ax=ax,
    )
    ax.set_xlim(0, 96)
    ax.set_xticks(range(0, 97, 4))
    ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 1)], rotation=45)

    return fig, ax


if __name__ == "__main__":
    main()
