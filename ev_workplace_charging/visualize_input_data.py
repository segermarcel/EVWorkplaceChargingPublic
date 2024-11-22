import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from ev_workplace_charging.dashboard import FIGURES_DIR
from ev_workplace_charging.dashboard import load_and_process_charging_costs
from ev_workplace_charging.dashboard import load_and_process_grid_carbon_intensity
from ev_workplace_charging.dashboard import load_and_process_power_profile

sns.set_theme(
    context="notebook",
    palette="tab10",
    font_scale=1.5,
)


def main():
    st.write("# Visualize Input Data")

    st.write("## Power Profile")
    df_power_profile, mean_power = load_and_process_power_profile(date=None)

    # Divide each row by the mean of all values in df
    df_power_profile = df_power_profile.div(mean_power, axis=0)

    # Turn into long-form
    df_power_profile = process_df_to_long_form(df_power_profile)

    fig, ax = plot_long_form_df(df_power_profile)
    ax.set_ylabel("Relative Power (normalized to mean of February 2023)")

    fig.savefig(FIGURES_DIR / "power_profile.png")
    st.write(fig)

    st.write("## Charging Costs")
    df_charging_costs = load_and_process_charging_costs(date=None)
    df_charging_costs = process_df_to_long_form(df_charging_costs)

    fig, ax = plot_long_form_df(df_charging_costs)
    ax.set_ylabel("Electricity Costs (p/kWh)")

    fig.savefig(FIGURES_DIR / "electricity_costs.png")
    st.write(fig)

    st.write("## Grid Carbon Intensity")
    df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date=None)
    df_grid_carbon_intensity = process_df_to_long_form(df_grid_carbon_intensity)

    fig, ax = plot_long_form_df(df_grid_carbon_intensity)
    ax.set_ylabel("Carbon Intensity (gCO2/kWh)")

    fig.savefig(FIGURES_DIR / "carbon_intensity.png")
    st.write(fig)


def process_df_to_long_form(df):
    # Turn into long-form
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Time"}, inplace=True)
    df = pd.melt(df, id_vars="Time", var_name="Date", value_name="Value")

    # Convert Time col to string
    df["Time"] = df["Time"].astype(str)

    return df


def plot_long_form_df(df_power_profile):
    fig, ax = plt.subplots(figsize=(15, 6))
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
