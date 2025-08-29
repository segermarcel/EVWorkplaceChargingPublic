import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from ev_workplace_charging.settings import COLUMN_NAMES, METRICS, MODEL_TYPES
from ev_workplace_charging.utils.metrics_computation import (
    compute_absolute_change,
    compute_relative_change,
)


def save_and_write_fig(fig, figure_path):
    plt.tight_layout(pad=0.0)
    fig.savefig(figure_path, dpi=300)
    st.write(fig)


def create_output_fig(
    df_output,
    color,
    electricity_costs=None,
    grid_carbon_intensity=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use n as index
    df_output.index = df_output["n"]

    const_C = (min(df_output["Pb"]) + max(df_output["Pb"])) / 2

    g = sns.lineplot(
        data=df_output,
        x=df_output.index,
        y="Pb",
        label="Energy Demand Curve Industrial Site",
        color="gray",
        linewidth=1.5,
        ax=ax,
    )
    sns.lineplot(
        data=df_output,
        x=df_output.index,
        y="UCC",
        label="Uncontrolled Charging (UCC)",
        color=sns.color_palette("tab10")[3],
        linewidth=2.5,
        ax=ax,
    )
    sns.lineplot(
        data=df_output,
        x=df_output.index,
        y="Tc",
        label="Smart Charging (SC)",
        color=color,
        linewidth=2.5,
        ax=ax,
    )
    ax.axhline(
        y=const_C,
        color="gray",
        linestyle="dotted",
        linewidth=1.5,
        label="Constant C",
    )

    if electricity_costs is not None:
        ax2 = ax.twinx()
        sns.lineplot(
            x=df_output.index,
            y=electricity_costs,
            label="Electricity Costs",
            linestyle="--",
            color="gray",
            linewidth=1.5,
            ax=ax2,
        )

        ax2.set_ylabel("Electricity Costs (p/kWh)")
        ax2.tick_params(axis="y")

        sns.move_legend(ax2, "upper right")

    if grid_carbon_intensity is not None:
        ax2 = ax.twinx()
        sns.lineplot(
            x=df_output.index,
            y=grid_carbon_intensity,
            label="Grid Carbon Intensity",
            linestyle="--",
            linewidth=1.5,
            color="gray",
            ax=ax2,
        )

        ax2.set_ylabel("Grid Carbon Intensity (gCO2/kWh)")
        ax2.tick_params(axis="y")

        sns.move_legend(ax2, "upper right")

    g.set_xticks(range(0, 97, 4))
    g.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 1)], rotation=45)

    sns.move_legend(ax, "lower right")

    ax.set_xlabel("")
    ax.set_xlim(0, 97)
    ax.tick_params(axis="y")
    ax.set_ylabel("Energy consumption (kWh)")
    # ax.set_ylim(1000, 5000)

    return fig


def create_metrics_fig(metrics_df, relative=False):
    if relative:
        change_fn = compute_relative_change
    else:
        change_fn = compute_absolute_change

    metrics_df["max_peak"] = change_fn(metrics_df["mp_model"], metrics_df["mp_ucc"])
    metrics_df["charging_costs"] = change_fn(
        metrics_df["cc_model"], metrics_df["cc_ucc"]
    )
    metrics_df["carbon_emissions"] = change_fn(
        metrics_df["ce_model"], metrics_df["ce_ucc"]
    )

    # Translate model_type
    metrics_df["model_type"] = metrics_df["model_type"].map(MODEL_TYPES)

    # Translate column names
    metrics_df = metrics_df.rename(columns=COLUMN_NAMES)

    # Melt
    metrics_df = metrics_df.melt(
        value_vars=METRICS, var_name="Metric", value_name="Value"
    )

    # Create horizontal bar chart of metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=metrics_df,
        x="Value",
        y="Metric",
        ax=ax,
        orient="h",
        palette=sns.color_palette("tab10"),
        errorbar=None,
    )
    for container in ax.containers:
        if relative:
            ax.bar_label(container, fmt="{:0.1f}% ")
        else:
            ax.bar_label(container, fmt="{:0.1f} ")

    # Add 15% more to xlimits in both directions
    ax.set_xlim(ax.get_xlim()[0] * 1.15, ax.get_xlim()[1] * 1.15)
    if relative:
        ax.set_xticklabels([f"{int(x)}%" for x in ax.get_xticks()])
        ax.set_xlabel("Relative Change (SC - UCC)")
    else:
        ax.set_xlabel("Absolute Change (SC - UCC)")
    ax.tick_params(axis="y")
    ax.set_ylabel("")

    return fig
