import gc
import weakref

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ev_workplace_charging.utils.plotting import create_output_fig, save_and_write_fig


@pytest.mark.parametrize(
    "overlay", [None, "electricity_costs", "grid_carbon_intensity"]
)
def test_rendered_profiles_are_not_retained_by_legend_caches(monkeypatch, overlay):
    monkeypatch.setattr(
        "ev_workplace_charging.utils.plotting.st.pyplot",
        lambda fig, **kwargs: fig.clear(),
    )
    profile = pd.DataFrame(
        {"n": [1, 2, 3], "Pb": [10, 12, 11], "UCC": [12, 15, 13], "Tc": [13, 13, 13]}
    )

    def render_profile():
        options = {overlay: pd.Series([1, 2, 3])} if overlay else {}
        fig = create_output_fig(profile.copy(), color="blue", **options)
        reference = weakref.ref(fig)
        save_and_write_fig(fig)
        return reference

    references = [render_profile() for _ in range(3)]
    # Collect again after Matplotlib's weakref callbacks release shared axes.
    # A global cache retaining a legend survives both collections.
    gc.collect()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_repeated_renders_release_figures_without_writing_files(monkeypatch):
    initial_figures = plt.get_fignums()
    rendered = []
    monkeypatch.setattr(
        "ev_workplace_charging.utils.plotting.st.pyplot",
        lambda fig, **kwargs: rendered.append(kwargs),
    )
    for _ in range(24):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        monkeypatch.setattr(
            fig, "savefig", lambda *args, **kwargs: pytest.fail("Unexpected export")
        )
        save_and_write_fig(fig)
        assert plt.get_fignums() == initial_figures
    assert len(rendered) == 24
    assert all(options["clear_figure"] for options in rendered)


def test_render_failure_still_releases_figure(monkeypatch):
    initial_figures = plt.get_fignums()
    fig, _ = plt.subplots()

    def fail(*args, **kwargs):
        raise RuntimeError("Rendering interrupted")

    monkeypatch.setattr("ev_workplace_charging.utils.plotting.st.pyplot", fail)
    with pytest.raises(RuntimeError, match="Rendering interrupted"):
        save_and_write_fig(fig)
    assert plt.get_fignums() == initial_figures


def test_explicit_png_export_is_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ev_workplace_charging.utils.plotting.st.pyplot", lambda *args, **kwargs: None
    )
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    destination = tmp_path / "figure.png"
    save_and_write_fig(fig, destination)
    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not plt.fignum_exists(fig.number)
