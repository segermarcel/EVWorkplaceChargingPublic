from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from streamlit.testing.v1 import AppTest


def test_dashboard_reruns_do_not_retain_figures(monkeypatch):
    original_savefig = Figure.savefig

    def render_in_memory(fig, destination, *args, **kwargs):
        assert not isinstance(destination, (str, Path)), "Unexpected file export"
        return original_savefig(fig, destination, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", render_in_memory)
    app = AppTest.from_file("ev_workplace_charging/dashboard.py", default_timeout=60)
    initial_figures = plt.get_fignums()
    app.run()
    assert not app.exception
    assert len(app.get("imgs")) == 6
    assert plt.get_fignums() == initial_figures
    for company in ["02", "05", "08", "01"]:
        app.selectbox[0].select(company).run()
        assert not app.exception
        assert len(app.get("imgs")) == 6
        assert plt.get_fignums() == initial_figures
    app.selectbox[1].select("1 Day").run()
    assert not app.exception
    assert plt.get_fignums() == initial_figures
