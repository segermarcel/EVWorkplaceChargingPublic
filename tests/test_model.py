"""Verification of the optimisation model and metrics on small, hand-checkable cases."""

import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from ev_workplace_charging import dashboard
from ev_workplace_charging.settings import SLOT_HOURS
from ev_workplace_charging.utils.data_loading import (
    compute_uncontrolled_charging_data,
    generate_parking_matrix,
)
from ev_workplace_charging.utils.metrics_computation import (
    compute_max_peak,
    compute_total_carbon_emissions,
    compute_total_charging_costs,
)
from ev_workplace_charging.utils.model_solving import setup_model, solve_model

CHARGER_KW = 11
SLOTS = 96


def ev_parameters():
    return pd.DataFrame(
        {
            "ev_idx": [0, 1],
            "arrival_time": [datetime.time(8, 0), datetime.time(21, 50)],
            "departure_time": [datetime.time(16, 0), datetime.time(6, 10)],
            "E_cap": [60.0, 60.0],
            "E_ini": [20.0, 10.0],
            "E_next": [50.0, 54.0],
        }
    )


def site_load(values=None):
    values = np.full(SLOTS, 100.0) if values is None else values
    return pd.DataFrame({"Verbrauch": values})


def solve(model_type, df_ev, df_load, prices=None, intensities=None):
    model = setup_model(
        model_type=model_type,
        df_parking_matrix=generate_parking_matrix(df_ev),
        df_ev_parameters=df_ev,
        df_power_profile=df_load,
        df_charging_costs=None if prices is None else pd.DataFrame({"Price": prices}),
        df_grid_carbon_intensity=None
        if intensities is None
        else pd.Series(intensities),
        charger_output_power=CHARGER_KW,
    )
    solve_model("highs", model)
    return model


def requested_energy(df_ev):
    return (df_ev["E_next"] - df_ev["E_ini"]).sum()


def test_uncontrolled_charging_delivers_exactly_the_requested_energy():
    df_ev = ev_parameters()
    df_ucc = compute_uncontrolled_charging_data(
        df_ev, generate_parking_matrix(df_ev), site_load(np.zeros(SLOTS)), CHARGER_KW
    )
    charging = df_ucc["Verbrauch"].to_numpy()

    assert charging.sum() == pytest.approx(requested_energy(df_ev))
    assert charging.max() <= CHARGER_KW * SLOT_HOURS + 1e-9
    # The overnight EV charges from arrival (22:00) first; its 44 kWh take 16 slots
    assert charging[88:96].sum() == pytest.approx(8 * CHARGER_KW * SLOT_HOURS)
    assert charging[0:8].sum() > 0


@pytest.mark.parametrize("model_type", ["ps", "ccm", "cem"])
def test_strategies_deliver_the_same_energy_within_charger_limits(model_type):
    df_ev = ev_parameters()
    # A load valley and negative prices both reward charging beyond the request
    load = np.full(SLOTS, 100.0)
    load[36:60] = 0.0
    signal = np.linspace(-50, 150, SLOTS)
    model = solve(model_type, df_ev, site_load(load), prices=signal, intensities=signal)

    parked = generate_parking_matrix(df_ev).to_numpy()
    x = np.array([[pyo.value(model.x[m, n]) for n in model.N] for m in model.M])

    assert x.sum() == pytest.approx(requested_energy(df_ev))
    assert x.max() <= CHARGER_KW * SLOT_HOURS + 1e-6
    assert np.abs(x[parked == 0]).max() == pytest.approx(0)


def test_cost_minimisation_charges_in_the_cheapest_parked_slot():
    df_ev = ev_parameters().iloc[[0]]
    df_ev.loc[0, "E_next"] = df_ev.loc[0, "E_ini"] + CHARGER_KW * SLOT_HOURS
    prices = np.full(SLOTS, 100.0)
    prices[40] = 10.0  # 10:00, inside the 08:00-16:00 parking window
    model = solve("ccm", df_ev, site_load(), prices=prices)

    assert pyo.value(model.y[41]) == pytest.approx(CHARGER_KW * SLOT_HOURS)


def test_metric_units():
    load = pd.Series([0.0, 0.0])
    with_ev = pd.Series([600.0, 400.0])  # kWh per slot

    assert compute_max_peak(with_ev) == pytest.approx(2400.0)  # kW
    assert compute_total_charging_costs(
        with_ev, load, pd.Series([100.0, 100.0])
    ) == pytest.approx(100.0)  # 1 MWh at 100 EUR/MWh
    assert compute_total_carbon_emissions(
        with_ev, load, pd.Series([400.0, 400.0])
    ) == pytest.approx(400.0)  # 1 MWh at 400 gCO2/kWh, in kg


@pytest.mark.skipif(
    not pyo.SolverFactory("gurobi").available(exception_flag=False),
    reason="Gurobi not installed",
)
@pytest.mark.parametrize("model_type", ["ps", "ccm", "cem"])
def test_gurobi_and_highs_agree(model_type):
    df_ev = ev_parameters()
    signal = np.sin(np.linspace(0, 6, SLOTS)) * 50 + 100
    args = dict(prices=signal, intensities=signal)
    highs = solve(model_type, df_ev, site_load(signal), **args)

    gurobi = setup_model(
        model_type,
        generate_parking_matrix(df_ev),
        df_ev,
        site_load(signal),
        pd.DataFrame({"Price": signal}),
        pd.Series(signal),
        CHARGER_KW,
    )
    solve_model("gurobi", gurobi)

    assert pyo.value(gurobi.obj) == pytest.approx(pyo.value(highs.obj), rel=1e-5)


def test_dashboard_run_uses_consistent_units(monkeypatch, tmp_path):
    """End to end: a kW load profile, UCC and the model share kWh per slot."""
    load_kw = np.full(SLOTS, 400.0)
    monkeypatch.setitem(
        dashboard.COMPANY_CONFIGS,
        "99",
        {"is_kw": True, "load_fn": lambda date: site_load(load_kw)},
    )
    monkeypatch.setattr(dashboard, "MODEL_OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "METRICS_DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "SOLVER_TYPE", "highs")
    prices = pd.DataFrame({"Price": np.linspace(120, 20, SLOTS)})  # UCC pays more
    monkeypatch.setattr(dashboard, "load_charging_costs_from_api", lambda d: prices)
    monkeypatch.setattr(
        dashboard,
        "load_grid_carbon_intensity_from_api",
        lambda d: pd.Series(np.full(SLOTS, 300.0)),
    )
    shifts = [{"start": datetime.time(8), "end": datetime.time(16), "num_cars": 10}]
    batteries = [{"capacity": 60, "relative_portion": 100}]

    output, metrics, _ = dashboard.setup_data_run_models_compute_metrics(
        batteries, CHARGER_KW, "99", datetime.date(2024, 2, 1), 100, "ccm", shifts
    )
    metrics = metrics.iloc[0]
    ucc_energy = (output["UCC"] - output["Pb"]).sum()

    assert output["Pb"].to_numpy() == pytest.approx(load_kw * SLOT_HOURS)
    assert output["y"].sum() == pytest.approx(ucc_energy)
    assert metrics["mp_ucc"] == pytest.approx(output["UCC"].max() / SLOT_HOURS)
    assert metrics["ce_ucc"] == pytest.approx(ucc_energy * 300 / 1000)
    assert metrics["cc_model"] < metrics["cc_ucc"]
