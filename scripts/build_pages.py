"""Build the paper's static scenario explorer without a Python web runtime.

Only precomputed model outputs and metrics are published. Weekly profiles are
means at each time of day; weekly metrics average the daily changes, matching
the Streamlit dashboard. Plot PNGs use the original shared Matplotlib/Seaborn
constructors, including their uncertainty bands and axis formatting.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
COMPANIES = ["01", "02", "03", "04", "05", "06", "07", "08"]
EV_PORTIONS = [15, 30, 50, 80, 100]
DATES = [f"2024-02-{day:02}" for day in range(1, 8)]
MODELS = {
    "ps": "Peak Minimisation & Valley Filling (PM-VF)",
    "ccm": "Charging Cost Minimisation (CCM)",
    "cem": "Carbon Emission Minimisation (CEM)",
}
PROFILE_COLUMNS = (
    "Pb",
    "UCC",
    "Tc",
    "charging_costs",
    "grid_carbon_intensity",
)
METRIC_COLUMNS = {
    "max_peak": ("mp_model", "mp_ucc"),
    "charging_costs": ("cc_model", "cc_ucc"),
    "carbon_emissions": ("ce_model", "ce_ucc"),
}

PLOT_VARIANTS = ("profile", "profile_overlay", "metrics_relative", "metrics_absolute")


def plot_filename(company, period, model, ev_portion, variant):
    """Return the browser-relative filename for one original Matplotlib plot."""
    if company not in COMPANIES or period not in [*DATES, "week"]:
        raise ValueError("Unknown company or analysis period")
    if model not in MODELS or ev_portion not in EV_PORTIONS:
        raise ValueError("Unknown model or EV portion")
    if variant not in PLOT_VARIANTS or (model == "ps" and variant == "profile_overlay"):
        raise ValueError("Unknown or unavailable plot variant")
    return f"plots/{company}_{period}_{model}_{ev_portion}_11_{variant}.png"


def _scenario_frames(data_dir, company, period, model, ev_portion):
    import pandas as pd

    dates = DATES if period == "week" else [period]
    names = [f"{company}_{date}_{model}_{ev_portion}_11.csv" for date in dates]
    profiles = [pd.read_csv(Path(data_dir) / "model_outputs" / name) for name in names]
    metrics = [pd.read_csv(Path(data_dir) / "metrics" / name) for name in names]
    return pd.concat(profiles, ignore_index=True), pd.concat(metrics, ignore_index=True)


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns
    from matplotlib import pyplot as plt

    from ev_workplace_charging.utils import plotting

    plotting.configure_plot_style()
    return plotting, sns, plt


def _figure_for_variant(profiles, metrics, model, variant):
    plotting, sns, _ = _plot_modules()
    if variant.startswith("metrics_"):
        return plotting.create_metrics_fig(
            metrics.copy(), relative=variant.endswith("relative")
        )
    options = {}
    if variant == "profile_overlay":
        column = "charging_costs" if model == "ccm" else "grid_carbon_intensity"
        argument = "electricity_costs" if model == "ccm" else "grid_carbon_intensity"
        options[argument] = profiles[column]
    color = sns.color_palette("tab10")[list(MODELS).index(model)]
    return plotting.create_output_fig(profiles.copy(), color=color, **options)


def _png_bytes(fig):
    from matplotlib import pyplot as plt

    try:
        fig.tight_layout(pad=0.0)
        buffer = io.BytesIO()
        # Exactly the options used by Streamlit's st.pyplot(..., dpi=100).
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        fig.clear()
        plt.close(fig)


def render_plot(data_dir, company, period, model, ev_portion, variant):
    """Render one PNG with the same constructors and options as Streamlit."""
    plot_filename(company, period, model, ev_portion, variant)
    profiles, metrics = _scenario_frames(data_dir, company, period, model, ev_portion)
    return _png_bytes(_figure_for_variant(profiles, metrics, model, variant))


def _renderer_fingerprint():
    digest = hashlib.sha256()
    for source in [
        Path(__file__),
        ROOT / "ev_workplace_charging/settings.py",
        ROOT / "ev_workplace_charging/utils/plotting.py",
        ROOT / "ev_workplace_charging/utils/metrics_computation.py",
    ]:
        digest.update(source.read_bytes())
    for dependency in [
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "pillow",
        "streamlit",
    ]:
        digest.update(f"{dependency}={version(dependency)}".encode())
    return digest.digest()


def _scenario_fingerprint(
    renderer_fingerprint, data_dir, company, period, model, portion
):
    digest = hashlib.sha256(renderer_fingerprint)
    for date in DATES if period == "week" else [period]:
        filename = f"{company}_{date}_{model}_{portion}_11.csv"
        for folder in ["model_outputs", "metrics"]:
            digest.update((Path(data_dir) / folder / filename).read_bytes())
    return digest.hexdigest()


def _render_scenario_job(job):
    output, data_dir, company, period, model, portion, variants, fingerprint = job
    profiles, metrics = _scenario_frames(data_dir, company, period, model, portion)
    results = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="ev_workplace_charging.utils.plotting")
        for variant in variants:
            data = _png_bytes(_figure_for_variant(profiles, metrics, model, variant))
            filename = plot_filename(company, period, model, portion, variant)
            destination = Path(output) / filename
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)
            results[filename] = {
                "input": fingerprint,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    return results


def render_plot_assets(output, data_dir=ROOT / "data", *, jobs=4, selections=None):
    """Build/cache original plot PNGs; selections limits scenario groups for QA."""
    if jobs < 1:
        raise ValueError("Plot worker count must be positive")
    output, data_dir = Path(output), Path(data_dir)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    cache_path = output.parent / f".{output.name}-plot-cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    renderer_fingerprint = _renderer_fingerprint()
    if selections is None:
        selections = [
            (company, period, model, portion)
            for company in COMPANIES
            for period in [*DATES, "week"]
            for model in MODELS
            for portion in EV_PORTIONS
        ]
    pending, retained = [], {}
    for company, period, model, portion in selections:
        variants = [
            variant
            for variant in PLOT_VARIANTS
            if model != "ps" or variant != "profile_overlay"
        ]
        fingerprint = _scenario_fingerprint(
            renderer_fingerprint, data_dir, company, period, model, portion
        )
        missing = []
        for variant in variants:
            filename = plot_filename(company, period, model, portion, variant)
            destination = output / filename
            entry = cache.get(filename, {})
            if (
                entry.get("input") == fingerprint
                and destination.is_file()
                and hashlib.sha256(destination.read_bytes()).hexdigest()
                == entry.get("sha256")
            ):
                retained[filename] = entry
            else:
                missing.append(variant)
        if missing:
            pending.append(
                (
                    output,
                    data_dir,
                    company,
                    period,
                    model,
                    portion,
                    missing,
                    fingerprint,
                )
            )
    expected_count = len(retained) + sum(len(job[-2]) for job in pending)
    completed = 0

    def record(results):
        nonlocal completed
        retained.update(results)
        completed += 1
        if completed % 50 == 0 or completed == len(pending):
            print(f"Plot assets: {len(retained)}/{expected_count} ready", flush=True)

    if jobs == 1:
        for job in pending:
            record(_render_scenario_job(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=min(jobs, os.cpu_count() or 1)) as pool:
            for result in pool.map(_render_scenario_job, pending):
                record(result)
    cache.update(retained)
    cache_path.write_text(
        json.dumps(cache, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return sorted(retained)


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row, column, path):
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid numeric column {column}") from error
    if not math.isfinite(value):
        raise ValueError(f"{path}: non-finite value in {column}")
    return value


def _load_day(data_dir, name, date, model, ev_portion):
    output_path = data_dir / "model_outputs" / name
    rows = _read_rows(output_path)
    if len(rows) != 96:
        raise ValueError(f"{output_path}: expected 96 time intervals")
    profile = []
    for expected_n, row in enumerate(rows, start=1):
        if row.get("date") != date or _number(row, "n", output_path) != expected_n:
            raise ValueError(f"{output_path}: incorrect date or time interval")
        _number(row, "y", output_path)
        profile.append(
            {"n": expected_n}
            | {column: _number(row, column, output_path) for column in PROFILE_COLUMNS}
        )

    metrics_path = data_dir / "metrics" / name
    rows = _read_rows(metrics_path)
    if len(rows) != 1:
        raise ValueError(f"{metrics_path}: expected exactly one metrics row")
    row = rows[0]
    if (
        row.get("date") != date
        or row.get("model_type") != model
        or _number(row, "ev_portion", metrics_path) != ev_portion
    ):
        raise ValueError(f"{metrics_path}: scenario metadata mismatch")
    metrics = {}
    for metric, (new_column, old_column) in METRIC_COLUMNS.items():
        new = _number(row, new_column, metrics_path)
        old = _number(row, old_column, metrics_path)
        if old == 0:
            raise ValueError(f"{metrics_path}: zero baseline for {metric}")
        absolute = new - old
        relative = absolute / old * 100
        if not math.isfinite(absolute) or not math.isfinite(relative):
            raise ValueError(f"{metrics_path}: non-finite change for {metric}")
        metrics[metric] = {"absolute": absolute, "relative": relative}
    powers = [point["Pb"] for point in profile]
    return {
        "profile": profile,
        "metrics": metrics,
        "constant_c": (min(powers) + max(powers)) / 2,
    }


def _week(days):
    daily = list(days.values())
    profile = [
        {"n": n + 1}
        | {
            column: fmean(day["profile"][n][column] for day in daily)
            for column in PROFILE_COLUMNS
        }
        for n in range(96)
    ]
    metrics = {
        metric: {
            mode: fmean(day["metrics"][metric][mode] for day in daily)
            for mode in ("absolute", "relative")
        }
        for metric in METRIC_COLUMNS
    }
    powers = [point["Pb"] for day in daily for point in day["profile"]]
    return {
        "profile": profile,
        "metrics": metrics,
        "constant_c": (min(powers) + max(powers)) / 2,
    }


def build_payload(data_dir=ROOT / "data"):
    """Validate and export every precomputed run of 1-7 February; never omit one.

    The batch run covers all of February 2024; the explorer shows the first week.
    """
    data_dir = Path(data_dir)
    expected = {
        f"{company}_{date}_{model}_{ev_portion}_11.csv"
        for company in COMPANIES
        for date in DATES
        for model in MODELS
        for ev_portion in EV_PORTIONS
    }
    for folder in ("model_outputs", "metrics"):
        actual = {path.name for path in (data_dir / folder).glob("*.csv")}
        if missing := sorted(expected - actual):
            raise ValueError(
                f"{folder}: incomplete scenario matrix; missing={missing[:3]}"
            )
    scenarios = {}
    for company in COMPANIES:
        for ev_portion in EV_PORTIONS:
            models = {}
            for model in MODELS:
                days = {
                    date: _load_day(
                        data_dir,
                        f"{company}_{date}_{model}_{ev_portion}_11.csv",
                        date,
                        model,
                        ev_portion,
                    )
                    for date in DATES
                }
                models[model] = {"days": days, "week": _week(days)}
            scenarios[f"{company}_{ev_portion}"] = models
    return {
        "companies": COMPANIES,
        "ev_portions": EV_PORTIONS,
        "dates": DATES,
        "models": MODELS,
        "scenarios": scenarios,
    }


def build_pages(
    output=ROOT / "build" / "pages",
    data_dir=ROOT / "data",
    web_dir=ROOT / "web",
    *,
    jobs=4,
    render_plots=True,
):
    """Copy the browser app and validated data to a publishable directory."""
    output, web_dir = Path(output), Path(web_dir)
    payload = build_payload(data_dir)
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
    if not (web_dir / "index.html").is_file():
        raise ValueError(f"{web_dir}: missing static app index.html")
    shutil.copytree(web_dir, output, dirs_exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "data" / "scenarios.json").write_text(serialized, encoding="utf-8")
    if render_plots:
        render_plot_assets(output, data_dir, jobs=jobs)
    (output / ".nojekyll").touch()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "pages")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--jobs", type=int, default=4, help="Parallel plot-rendering workers"
    )
    args = parser.parse_args()
    destination = build_pages(args.output, args.data_dir, jobs=args.jobs)
    runs = len(COMPANIES) * len(DATES) * len(MODELS) * len(EV_PORTIONS)
    print(f"Built {runs} precomputed runs in {destination}")


if __name__ == "__main__":
    main()
