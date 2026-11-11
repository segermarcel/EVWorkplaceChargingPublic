import gc
import importlib.util
import io
import shutil
from pathlib import Path

import pytest
import streamlit.elements.pyplot as streamlit_pyplot
from matplotlib import pyplot as plt
from PIL import Image

from ev_workplace_charging.utils import plotting

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "static_plot_exporter", ROOT / "scripts/build_pages.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


@pytest.mark.parametrize(
    ("period", "model", "variant"),
    [
        ("2024-02-01", "ccm", "profile_overlay"),
        ("week", "ps", "profile"),
        ("week", "cem", "profile_overlay"),
        ("2024-02-01", "ps", "metrics_relative"),
        ("week", "ccm", "metrics_absolute"),
    ],
)
def test_static_png_is_identical_to_real_streamlit_png(
    monkeypatch, period, model, variant
):
    """Capture Streamlit's real marshalling buffer before media storage."""
    captured = []
    monkeypatch.setattr(
        streamlit_pyplot,
        "marshall_images",
        lambda **kwargs: captured.append(kwargs["image"].getvalue()),
    )
    profiles, metrics = exporter._scenario_frames(
        ROOT / "data", "01", period, model, 15
    )
    figure = exporter._figure_for_variant(profiles, metrics, model, variant)
    plotting.save_and_write_fig(figure)
    actual = exporter.render_plot(ROOT / "data", "01", period, model, 15, variant)
    assert len(captured) == 1
    assert actual == captured[0]
    with Image.open(io.BytesIO(actual)) as image:
        assert image.format == "PNG"
        assert image.width > 900
        assert image.height > 500
    assert plt.get_fignums() == []
    gc.collect()
    gc.collect()


@pytest.fixture
def day_sources(tmp_path):
    destination = tmp_path / "source"
    for folder in ["model_outputs", "metrics"]:
        (destination / folder).mkdir(parents=True)
        shutil.copyfile(
            ROOT / "data" / folder / "01_2024-02-01_ccm_15_11.csv",
            destination / folder / "01_2024-02-01_ccm_15_11.csv",
        )
    return destination


def test_plot_cache_reuses_files_and_repairs_only_missing_or_corrupt_assets(
    tmp_path, day_sources, monkeypatch
):
    output = tmp_path / "site"
    selection = [("01", "2024-02-01", "ccm", 15)]
    names = exporter.render_plot_assets(
        output, day_sources, jobs=1, selections=selection
    )
    assert len(names) == 4
    initial = {name: (output / name).stat().st_mtime_ns for name in names}
    original = exporter._render_scenario_job

    def unexpected_render(job):
        pytest.fail("An unchanged cached plot was rendered again")

    monkeypatch.setattr(exporter, "_render_scenario_job", unexpected_render)
    assert (
        exporter.render_plot_assets(output, day_sources, jobs=1, selections=selection)
        == names
    )
    assert {name: (output / name).stat().st_mtime_ns for name in names} == initial
    corrupt, missing = names[0], names[-1]
    (output / corrupt).write_bytes(b"truncated PNG")
    (output / missing).unlink()
    rendered = []

    def record_render(job):
        rendered.extend(job[-2])
        return original(job)

    monkeypatch.setattr(exporter, "_render_scenario_job", record_render)
    exporter.render_plot_assets(output, day_sources, jobs=1, selections=selection)
    assert len(rendered) == 2
    for name in names:
        with Image.open(output / name) as image:
            image.verify()
        if name not in [corrupt, missing]:
            assert (output / name).stat().st_mtime_ns == initial[name]


def test_changed_source_invalidates_cached_plot_group(
    tmp_path, day_sources, monkeypatch
):
    output = tmp_path / "site"
    selection = [("01", "2024-02-01", "ccm", 15)]
    exporter.render_plot_assets(output, day_sources, jobs=1, selections=selection)
    path = day_sources / "metrics/01_2024-02-01_ccm_15_11.csv"
    # Even a source-byte change invalidates the scenario's cached images.
    path.write_text(path.read_text() + "\n")
    original = exporter._render_scenario_job
    rendered = []

    def record_render(job):
        rendered.extend(job[-2])
        return original(job)

    monkeypatch.setattr(exporter, "_render_scenario_job", record_render)
    exporter.render_plot_assets(output, day_sources, jobs=1, selections=selection)
    assert set(rendered) == set(exporter.PLOT_VARIANTS)


def test_changed_plot_labels_invalidate_renderer_fingerprint(tmp_path, monkeypatch):
    for relative in [
        "ev_workplace_charging/settings.py",
        "ev_workplace_charging/utils/plotting.py",
        "ev_workplace_charging/utils/metrics_computation.py",
    ]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    initial = exporter._renderer_fingerprint()
    settings = tmp_path / "ev_workplace_charging/settings.py"
    settings.write_text(settings.read_text().replace('"Max. Peak"', '"Peak Load"'))
    assert exporter._renderer_fingerprint() != initial


def test_output_filenames_cover_exactly_all_supported_variants():
    names = {
        exporter.plot_filename(company, period, model, portion, variant)
        for company in exporter.COMPANIES
        for period in [*exporter.DATES, "week"]
        for model in exporter.MODELS
        for portion in exporter.EV_PORTIONS
        for variant in exporter.PLOT_VARIANTS
        if model != "ps" or variant != "profile_overlay"
    }
    assert len(names) == 440 * len(exporter.COMPANIES)
    assert "plots/05_week_ccm_80_11_profile_overlay.png" in names
    assert "plots/01_2024-02-01_ps_15_11_metrics_relative.png" in names
    with pytest.raises(ValueError, match="variant"):
        exporter.plot_filename("01", "week", "ps", 15, "profile_overlay")
