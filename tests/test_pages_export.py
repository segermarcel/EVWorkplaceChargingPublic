import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import seaborn as sns
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_pages", ROOT / "scripts/build_pages.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.fixture(scope="module")
def payload():
    return exporter.build_payload()


def test_complete_matrix_matches_every_source_profile_and_metric(payload):
    assert payload["companies"] == exporter.COMPANIES
    assert len(payload["scenarios"]) == len(exporter.COMPANIES) * len(
        exporter.EV_PORTIONS
    )
    count = 0
    for company in exporter.COMPANIES:
        for portion in exporter.EV_PORTIONS:
            for model in exporter.MODELS:
                exported = payload["scenarios"][f"{company}_{portion}"][model]
                profiles, metrics = [], []
                assert list(exported["days"]) == exporter.DATES
                for date in exporter.DATES:
                    name = f"{company}_{date}_{model}_{portion}_11.csv"
                    profile = pd.read_csv(ROOT / "data/model_outputs" / name)
                    metric = pd.read_csv(ROOT / "data/metrics" / name)
                    day = exported["days"][date]
                    assert [point["n"] for point in day["profile"]] == list(
                        range(1, 97)
                    )
                    for column in exporter.PROFILE_COLUMNS:
                        assert [
                            point[column] for point in day["profile"]
                        ] == pytest.approx(profile[column].tolist())
                    assert day["constant_c"] == pytest.approx(
                        (profile.Pb.min() + profile.Pb.max()) / 2
                    )
                    for name, (new, old) in exporter.METRIC_COLUMNS.items():
                        absolute = metric[new] - metric[old]
                        relative = absolute / metric[old] * 100
                        assert day["metrics"][name]["absolute"] == pytest.approx(
                            absolute.iloc[0]
                        )
                        assert day["metrics"][name]["relative"] == pytest.approx(
                            relative.iloc[0]
                        )
                    profiles.append(profile)
                    metrics.append(metric)
                    count += 1
                all_profiles = pd.concat(profiles)
                all_metrics = pd.concat(metrics)
                week = exported["week"]
                means = all_profiles.groupby("n")[list(exporter.PROFILE_COLUMNS)].mean()
                for column in exporter.PROFILE_COLUMNS:
                    assert [
                        point[column] for point in week["profile"]
                    ] == pytest.approx(means[column].tolist())
                assert week["constant_c"] == pytest.approx(
                    (all_profiles.Pb.min() + all_profiles.Pb.max()) / 2
                )
                for name, (new, old) in exporter.METRIC_COLUMNS.items():
                    absolute = all_metrics[new] - all_metrics[old]
                    relative = absolute / all_metrics[old] * 100
                    assert week["metrics"][name]["absolute"] == pytest.approx(
                        absolute.mean()
                    )
                    assert week["metrics"][name]["relative"] == pytest.approx(
                        relative.mean()
                    )
    # One run per company, first-week day, model and EV portion
    assert count == len(exporter.COMPANIES) * len(exporter.DATES) * len(
        exporter.MODELS
    ) * len(exporter.EV_PORTIONS)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
@pytest.mark.parametrize("model", ["ps", "ccm", "cem"])
def test_weekly_metrics_match_actual_seaborn_bar_heights(payload, mode, model):
    daily = pd.concat(
        [
            pd.read_csv(ROOT / f"data/metrics/05_{date}_{model}_80_11.csv")
            for date in exporter.DATES
        ]
    )
    changes = pd.DataFrame(
        {
            name: (daily[new] - daily[old])
            * (100 / daily[old] if mode == "relative" else 1)
            for name, (new, old) in exporter.METRIC_COLUMNS.items()
        }
    ).melt(var_name="Metric", value_name="Value")
    fig, ax = plt.subplots()
    try:
        sns.barplot(data=changes, x="Value", y="Metric", errorbar=None, ax=ax)
        expected = payload["scenarios"]["05_80"][model]["week"]["metrics"]
        assert [patch.get_width() for patch in ax.patches] == pytest.approx(
            [expected[metric][mode] for metric in exporter.METRIC_COLUMNS]
        )
    finally:
        plt.close(fig)


@pytest.fixture
def source_data(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT / "data/model_outputs", source / "model_outputs")
    shutil.copytree(ROOT / "data/metrics", source / "metrics")
    return source


def test_missing_scenario_fails_before_publishing(source_data, tmp_path):
    next((source_data / "metrics").glob("*_2024-02-01_*.csv")).unlink()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="incomplete scenario matrix"):
        exporter.build_pages(output, source_data, render_plots=False)
    assert not output.exists()


@pytest.mark.parametrize("invalid", ["nan", "inf", "invalid"])
def test_corrupt_numeric_data_fails(source_data, invalid):
    path = source_data / "model_outputs/01_2024-02-01_ps_15_11.csv"
    rows = path.read_text().splitlines()
    fields = rows[1].split(",")
    fields[3] = invalid
    rows[1] = ",".join(fields)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="finite|numeric"):
        exporter.build_payload(source_data)


def test_publication_is_deterministic_and_excludes_research_inputs(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>Explorer</title>")
    output = exporter.build_pages(tmp_path / "output", web_dir=web, render_plots=False)
    first = (output / "data/scenarios.json").read_bytes()
    exporter.build_pages(output, web_dir=web, render_plots=False)
    assert (output / "data/scenarios.json").read_bytes() == first
    assert len(json.loads(first)["scenarios"]) == len(exporter.COMPANIES) * len(
        exporter.EV_PORTIONS
    )
    assert sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    ) == [
        ".nojekyll",
        "data/scenarios.json",
        "index.html",
    ]
