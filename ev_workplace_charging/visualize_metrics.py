import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from ev_workplace_charging.dashboard import COLUMN_NAMES
from ev_workplace_charging.dashboard import FIGURES_DIR
from ev_workplace_charging.dashboard import METRICS
from ev_workplace_charging.dashboard import METRICS_DATA_DIR
from ev_workplace_charging.dashboard import MODEL_TYPES

sns.set_theme(
    context="notebook",
    palette="tab10",
    font_scale=1.5,
)


def main():

    metrics_df = pd.read_csv(METRICS_DATA_DIR / "metrics.csv")

    # Filter out charging_typ bdc
    metrics_df = metrics_df[metrics_df["charging_type"] == "sc"]

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

        fig, ax = plt.subplots()
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

        fig.savefig(FIGURES_DIR / f"lineplot_2023-02-01_{model_type_short}.png", dpi=300)
        st.write(fig)

        # Boxplots
        daily_metrics = daily_metrics.melt(
            id_vars=["date", "EV Portion"], value_vars=METRICS, var_name="", value_name="Value"
        )
        st.write(daily_metrics)

        fig, ax = plt.subplots()
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

        fig.savefig(FIGURES_DIR / f"metrics_{model_type_short}.png", dpi=300)
        st.write(fig)

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

        fig, ax = plt.subplots(figsize=(15, 8))

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
        ax.set_ylabel("Relative Change (SC - UCC)")

        fig.savefig(FIGURES_DIR / f'boxplot_{metric.replace(" ", "_")}.png', dpi=300)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(data=metrics_df, x="date", y=metric, hue="Model Type", ax=ax)
        ax.set_xticklabels((f"{date:02d}.02.2023" for date in range(1, 29)), rotation=45)
        ax.set_yticklabels([f"{y:.0f}%" for y in ax.get_yticks()])
        ax.set_xlabel("EV adoption rate [%]")
        ax.set_ylabel("VoSC [%∆]")

        fig.savefig(FIGURES_DIR / f'lineplot_{metric.replace(" ", "_")}.png', dpi=300)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
