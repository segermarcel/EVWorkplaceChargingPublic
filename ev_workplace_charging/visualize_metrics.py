import datetime

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from ev_workplace_charging.settings import COLUMN_NAMES
from ev_workplace_charging.settings import FIGURES_DIR
from ev_workplace_charging.settings import METRICS
from ev_workplace_charging.settings import METRICS_DATA_DIR
from ev_workplace_charging.settings import MODEL_TYPES
from ev_workplace_charging.settings import N_CARS
from ev_workplace_charging.utils.plotting import save_and_write_fig

sns.set_theme(
    context="notebook",
    palette="tab10",
    font_scale=1.5,
)


def main():
    metrics_path = METRICS_DATA_DIR / "metrics.csv"

    if not metrics_path.exists():
        metrics = []
        for day in range(1, 29):
            date = datetime.date(2023, 2, day)
            for ev_portion, n_cars in N_CARS.items():
                for model_type in MODEL_TYPES:
                    metrics_path = METRICS_DATA_DIR / f"{date}_{model_type}_{ev_portion}.csv"
                    metrics_df = pd.read_csv(metrics_path)
                    metrics.append(metrics_df)

        metrics_df = pd.concat(metrics)
        metrics_df.to_csv(metrics_path, index=False)
    else:
        metrics_df = pd.read_csv(metrics_path)

    # Translate model_type
    metrics_df["model_type"] = metrics_df["model_type"].map(MODEL_TYPES)

    # Translate column names
    metrics_df = metrics_df.rename(columns=COLUMN_NAMES)

    st.write("# Metrics")
    st.write(metrics_df)

    for model_type_short, model_type in MODEL_TYPES.items():
        st.write(f"## {model_type}")
        daily_metrics = metrics_df[(metrics_df["Model Type"] == model_type)]

        # Lineplots for first day
        first_day_metrics = daily_metrics[daily_metrics["date"] == "2023-02-01"]
        first_day_metrics["EV Portion"] = first_day_metrics["EV Portion"].apply(lambda x: int(x[:-1]))
        first_day_metrics = first_day_metrics.melt(
            id_vars=["EV Portion"], value_vars=METRICS, var_name="", value_name="Value"
        )
        st.write(first_day_metrics)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=first_day_metrics,
            x="EV Portion",
            y="Value",
            hue="",
            marker="o",
            linewidth=2.0,
            ax=ax,
        )
        ax.set_xticklabels([f"{x:.0f}%" for x in ax.get_xticks()])
        ax.set_yticklabels([f"{y:.0f}%" for y in ax.get_yticks()])
        ax.set_xlabel("EV adoption rate [%]")
        ax.set_ylabel("VoSC [%∆]\n(01.02.2023)")

        save_and_write_fig(fig, FIGURES_DIR / f"lineplot_2023-02-01_{model_type_short}.png")

        # Boxplots
        daily_metrics = daily_metrics.melt(
            id_vars=["date", "EV Portion"], value_vars=METRICS, var_name="", value_name="Value"
        )
        st.write(daily_metrics)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=daily_metrics,
            x="EV Portion",
            y="Value",
            hue="",
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels())
        ax.set_yticklabels([f"{y:.0f}%" for y in ax.get_yticks()])
        ax.set_xlabel("EV adoption rate [%]")
        ax.set_ylabel("VoSC [%∆]")

        save_and_write_fig(fig, FIGURES_DIR / f"metrics_{model_type_short}.png")

    st.write("## Summary Statistics")
    summary_df = (
        metrics_df.groupby(["Model Type", "EV Portion"])
        .agg(
            mean_max_peak=(COLUMN_NAMES["max_peak"], "mean"),
            std_max_peak=(COLUMN_NAMES["max_peak"], "std"),
            median_max_peak=(COLUMN_NAMES["max_peak"], "median"),
            iqr_max_peak=(COLUMN_NAMES["max_peak"], lambda x: x.quantile(0.75) - x.quantile(0.25)),
            mean_charging_costs=(COLUMN_NAMES["charging_costs"], "mean"),
            std_charging_costs=(COLUMN_NAMES["charging_costs"], "std"),
            median_charging_costs=(COLUMN_NAMES["charging_costs"], "median"),
            iqr_charging_costs=(COLUMN_NAMES["charging_costs"], lambda x: x.quantile(0.75) - x.quantile(0.25)),
            mean_carbon_emissions=(COLUMN_NAMES["carbon_emissions"], "mean"),
            std_carbon_emissions=(COLUMN_NAMES["carbon_emissions"], "std"),
            median_carbon_emissions=(COLUMN_NAMES["carbon_emissions"], "median"),
            iqr_carbon_emissions=(COLUMN_NAMES["carbon_emissions"], lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .reset_index()
    )
    st.write(summary_df)

    for metric in METRICS:
        st.write(f"## {metric}")

        fig, ax = plt.subplots(figsize=(15, 9))

        x = "EV Portion"
        hue = "Model Type"
        order = ["15%", "30%", "50%", "80%", "100%"]
        sns.boxplot(data=metrics_df, x=x, y=metric, hue=hue, order=order, ax=ax)

        # pairs = [
        #     ((ev_portion, metric1), (ev_portion, metric2))
        #     for metric1, metric2 in [
        #         (MODEL_TYPES["ps"], MODEL_TYPES["ccm"]),
        #         (MODEL_TYPES["ps"], MODEL_TYPES["cem"]),
        #         (MODEL_TYPES["ccm"], MODEL_TYPES["cem"]),
        #     ]
        #     for ev_portion in order
        # ]

        # annot = Annotator(
        #     ax,
        #     pairs=pairs,
        #     data=metrics_df,
        #     x=x,
        #     y=metric,
        #     hue=hue,
        #     order=order,
        # )
        # annot.configure(
        #     test="Mann-Whitney",
        #     text_format="star",
        #     loc="outside",
        #     verbose=2,
        # )
        # annot.apply_test()
        # ax, test_results = annot.annotate()

        ax.set_xticklabels(ax.get_xticklabels())
        ax.set_yticklabels([f"{y:.0f}%" for y in ax.get_yticks()])
        ax.set_xlabel(ax.get_xlabel())
        ax.set_ylabel("VoSC [%∆]")

        save_and_write_fig(fig, FIGURES_DIR / f'boxplot_{metric.replace(" ", "_")}.png')

        fig, ax = plt.subplots(figsize=(15, 9))
        sns.lineplot(data=metrics_df, x="date", y=metric, hue="Model Type", ax=ax)
        ax.set_xticklabels((f"{date:02d}.02.2023" for date in range(1, 29)), rotation=45)
        ax.set_yticklabels([f"{y:.0f}%" for y in ax.get_yticks()])
        ax.set_xlabel("EV adoption rate [%]")
        ax.set_ylabel("VoSC [%∆]")

        save_and_write_fig(fig, FIGURES_DIR / f'lineplot_{metric.replace(" ", "_")}.png')


if __name__ == "__main__":
    main()
