"""Paper Figure 4 (firms 05 and 06) and Figure S3 (all firms): daily VoSC boxplots.

Reads the per-run metrics written by run_for_companies.py. Each box holds one value
per simulated day.

    uv run python scripts/make_paper_figures.py [--metrics-dir DIR] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ev_workplace_charging.settings import (
    COLUMN_NAMES,
    FIGURES_DIR,
    METRICS,
    METRICS_DATA_DIR,
    MODEL_TYPES,
)
from ev_workplace_charging.utils.metrics_computation import compute_relative_change

FIGURE_EV_PORTIONS = [15, 50, 100]


def load_vosc(metrics_dir):
    """One row per company, date, strategy and EV portion with VoSC [%]."""
    frames = []
    for path in sorted(Path(metrics_dir).glob("*_*_*_*_*.csv")):
        df = pd.read_csv(path)
        df["company"] = path.stem.split("_")[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["max_peak"] = compute_relative_change(df["mp_model"], df["mp_ucc"])
    df["charging_costs"] = compute_relative_change(df["cc_model"], df["cc_ucc"])
    df["carbon_emissions"] = compute_relative_change(df["ce_model"], df["ce_ucc"])
    return df


def plot_grid(vosc, companies, out_path):
    sns.set_theme(context="notebook", palette="tab10", font_scale=1.7)
    df = vosc[vosc["ev_portion"].isin(FIGURE_EV_PORTIONS)].rename(columns=COLUMN_NAMES)

    fig, axes = plt.subplots(
        len(MODEL_TYPES),
        len(companies),
        figsize=(10 * len(companies), 8 * len(MODEL_TYPES)),
        squeeze=False,
    )
    for row, model_type in enumerate(MODEL_TYPES):
        for col, company in enumerate(companies):
            panel = df[(df["company"] == company) & (df["Model Type"] == model_type)]
            panel = panel.melt(
                id_vars=["date", "EV Portion"],
                value_vars=METRICS,
                var_name="",
                value_name="Value",
            )
            ax = axes[row, col]
            sns.boxplot(data=panel, x="EV Portion", y="Value", hue="", ax=ax)
            ax.set_xticks(ax.get_xticks(), [f"{p}%" for p in FIGURE_EV_PORTIONS])
            ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
            ax.set_xlabel("EV adoption rate [%]")
            ax.set_ylabel("Value of smart charging (VoSC) [%∆]")
            if row == 0:
                ax.set_title(f"ID {int(company)}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vosc = load_vosc(args.metrics_dir)
    vosc.to_csv(args.out_dir / "vosc.csv", index=False)
    plot_grid(vosc, ["05", "06"], args.out_dir / "figure_4.png")
    plot_grid(vosc, sorted(vosc["company"].unique()), args.out_dir / "figure_s3.png")
    print(f"{len(vosc)} runs over {vosc['date'].nunique()} days -> {args.out_dir}")


if __name__ == "__main__":
    main()
