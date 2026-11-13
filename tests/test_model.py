"""Verification of the optimisation model and metrics on small, hand-checkable cases."""

import datetime

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

from ev_workplace_charging import dashboard
from ev_workplace_charging.settings import SLOT_HOURS, SLOTS_PER_DAY
from ev_workplace_charging.utils.data_loading import (
    charging_per_slot,
    compute_uncontrolled_charging,
    parked_slots,
)
from ev_workplace_charging.utils.metrics_computation import (
    compute_max_peak,
    compute_total_carbon_emissions,
    compute_total_charging_costs,
)
from ev_workplace_charging.utils.model_solving import (
    setup_model,
    solve_charging,
    solve_model,
)

CHARGER_KW = 11
MAX_ENERGY_PER_SLOT = CHARGER_KW * SLOT_HOURS
DAY = datetime.datetime(2024, 2, 1)
# Two days, so that the overnight stay can end on the second one
SLOTS = pd.date_range(DAY, periods=2 * SLOTS_PER_DAY, freq="15min")


def ev_parameters():
    """A day-shift EV and one parked overnight from 21:50 to 06:10 the next day."""
    return pd.DataFrame(
        {
            "arrival": [DAY.replace(hour=8), DAY.replace(hour=21, minute=50)],
            "departure": [
                DAY.replace(hour=16),
                DAY.replace(hour=6, minute=10) + datetime.timedelta(days=1),
            ],
            "E_cap": [60.0, 60.0],
            "E_ini": [20.0, 10.0],
            "E_next": [50.0, 54.0],
        }
    )


def inputs(site_load=None, prices=None, intensities=None):
    def series(values, default):
        return np.full(len(SLOTS), default) if values is None else values

    return pd.DataFrame(
        {
            "Pb": series(site_load, 100.0),
            "price": series(prices, 50.0),
            "carbon": series(intensities, 300.0),
        },
        index=SLOTS,
    )


def energy(df_ev):
    return (df_ev["E_next"] - df_ev["E_ini"]).to_numpy()


def solve(model_type, df_ev, df_inputs):
    """Total EV charging per slot [kWh] of the strategy."""
    parked = parked_slots(df_ev, SLOTS)
    charging = solve_charging(
        model_type, parked, energy(df_ev), df_inputs, CHARGER_KW, "highs"
    )
    return charging_per_slot(parked, charging, len(SLOTS))


def test_overnight_stays_end_on_the_next_day():
    parked = parked_slots(ev_parameters(), SLOTS)

    assert list(parked[0]) == list(range(32, 65))  # 08:00 to the 16:00 slot
    assert list(parked[1]) == list(range(88, 121))  # 22:00 to 06:00 the next day


def test_parking_windows_must_lie_within_the_horizon():
    df_ev = ev_parameters()
    df_ev.loc[1, "departure"] += datetime.timedelta(days=1)

    with pytest.raises(ValueError, match="within the modelled slots"):
        parked_slots(df_ev, SLOTS)


def test_uncontrolled_charging_delivers_exactly_the_requested_energy():
    df_ev = ev_parameters()
    parked = parked_slots(df_ev, SLOTS)
    per_ev = compute_uncontrolled_charging(parked, energy(df_ev), CHARGER_KW)
    charging = charging_per_slot(parked, per_ev, len(SLOTS))

    assert charging.sum() == pytest.approx(energy(df_ev).sum())
    assert charging.max() <= MAX_ENERGY_PER_SLOT + 1e-9
    # The overnight EV's 44 kWh take 16 full slots from 22:00, so until 02:00 next day
    assert charging[88:104] == pytest.approx(np.full(16, MAX_ENERGY_PER_SLOT))
    assert charging[:32].sum() == 0
    assert charging[104:].sum() == 0


@pytest.mark.parametrize("model_type", ["ps", "ccm", "cem"])
def test_strategies_deliver_the_same_energy_within_charger_limits(model_type):
    df_ev = ev_parameters()
    parked = parked_slots(df_ev, SLOTS)
    # A load valley and negative prices both reward charging beyond the request
    load = np.full(len(SLOTS), 100.0)
    load[36:60] = 0.0
    signal = np.linspace(-50, 150, len(SLOTS))
    model = setup_model(
        model_type, parked, energy(df_ev), inputs(load, signal, signal), CHARGER_KW
    )
    solve_model("highs", model)

    x = {(m, n): pyo.value(model.x[m, n]) for m, n in model.P}

    assert set(x) == {(m, n) for m, slots in enumerate(parked) for n in slots}
    assert sum(x.values()) == pytest.approx(energy(df_ev).sum())
    assert max(x.values()) <= MAX_ENERGY_PER_SLOT + 1e-6
    for m, slots in enumerate(parked):
        assert sum(x[m, n] for n in slots) == pytest.approx(energy(df_ev)[m])


def test_cost_minimisation_charges_in_the_cheapest_parked_slot():
    df_ev = ev_parameters().iloc[[0]]
    df_ev.loc[0, "E_next"] = df_ev.loc[0, "E_ini"] + MAX_ENERGY_PER_SLOT
    prices = np.full(len(SLOTS), 100.0)
    prices[40] = 10.0  # 10:00, inside the 08:00-16:00 parking window
    charging = solve("ccm", df_ev, inputs(prices=prices))

    assert charging[40] == pytest.approx(MAX_ENERGY_PER_SLOT)


def test_overnight_charging_uses_the_next_mornings_prices():
    """Cheap hours on the morning of the arrival day are out of reach; the cheapest
    slots the overnight EV can use are those of the following morning."""
    df_ev = ev_parameters().iloc[[1]].reset_index(drop=True)
    prices = np.full(len(SLOTS), 50.0)
    prices[:24] = 1.0  # 00:00-06:00 of the arrival day
    prices[SLOTS_PER_DAY + 8 : SLOTS_PER_DAY + 24] = 20.0  # 02:00-06:00 next day
    charging = solve("ccm", df_ev, inputs(prices=prices))

    assert charging[:SLOTS_PER_DAY].sum() == pytest.approx(0)
    assert charging[SLOTS_PER_DAY + 8 : SLOTS_PER_DAY + 24] == pytest.approx(
        np.full(16, MAX_ENERGY_PER_SLOT)
    )


def test_peak_shaving_targets_each_days_own_load_midpoint():
    """With a low first day and a high second day, an EV that can charge on either day
    fills the valley of the second day's profile, not that of a two-day average."""
    df_ev = ev_parameters().iloc[[1]].reset_index(drop=True)
    load = np.r_[np.full(SLOTS_PER_DAY, 10.0), np.full(SLOTS_PER_DAY, 200.0)]
    load[SLOTS_PER_DAY : SLOTS_PER_DAY + 4] = 100.0  # 00:00-01:00, second day's valley
    load[-1] = 300.0
    charging = solve("ps", df_ev, inputs(site_load=load))

    # Target on day 2 is (100 + 300) / 2 = 200: charging lifts the valley towards it
    assert charging[SLOTS_PER_DAY : SLOTS_PER_DAY + 4] == pytest.approx(
        np.full(4, MAX_ENERGY_PER_SLOT)
    )


def test_metric_units():
    ev_energy = np.array([600.0, 400.0])  # kWh per slot

    assert compute_max_peak(pd.Series(ev_energy)) == pytest.approx(2400.0)  # kW
    assert compute_total_charging_costs(
        ev_energy, np.array([100.0, 100.0])
    ) == pytest.approx(100.0)  # 1 MWh at 100 EUR/MWh
    assert compute_total_carbon_emissions(
        ev_energy, np.array([400.0, 400.0])
    ) == pytest.approx(400.0)  # 1 MWh at 400 gCO2/kWh, in kg


@pytest.mark.skipif(
    not pyo.SolverFactory("gurobi").available(exception_flag=False),
    reason="Gurobi not installed",
)
@pytest.mark.parametrize("model_type", ["ps", "ccm", "cem"])
def test_gurobi_and_highs_agree(model_type):
    df_ev = ev_parameters()
    signal = np.sin(np.linspace(0, 6, len(SLOTS))) * 50 + 100
    parked = parked_slots(df_ev, SLOTS)
    models = {}
    for solver_type in ("highs", "gurobi"):
        models[solver_type] = setup_model(
            model_type,
            parked,
            energy(df_ev),
            inputs(signal, signal, signal),
            CHARGER_KW,
        )
        solve_model(solver_type, models[solver_type])

    assert pyo.value(models["gurobi"].obj) == pytest.approx(
        pyo.value(models["highs"].obj), rel=1e-5
    )


@pytest.fixture
def synthetic_company(monkeypatch, tmp_path):
    """Company "99" with a constant 400 kW site load, falling prices and constant
    carbon intensity; results are written to `tmp_path`. Returns the site load and the
    price setter."""
    load_kw = np.full(SLOTS_PER_DAY, 400.0)
    monkeypatch.setitem(
        dashboard.COMPANY_CONFIGS,
        "99",
        {"is_kw": True, "load_fn": lambda date: pd.DataFrame({"Verbrauch": load_kw})},
    )
    monkeypatch.setattr(dashboard, "MODEL_OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "METRICS_DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "SOLVER_TYPE", "highs")
    data_loading = dashboard.load_period_inputs.__globals__

    def set_prices(prices):
        monkeypatch.setitem(
            data_loading,
            "load_charging_costs_from_api",
            lambda d: pd.DataFrame({"Price": prices}),
        )

    set_prices(np.linspace(120, 20, SLOTS_PER_DAY))
    monkeypatch.setitem(
        data_loading,
        "load_grid_carbon_intensity_from_api",
        lambda d: pd.Series(np.full(SLOTS_PER_DAY, 300.0)),
    )
    return load_kw, set_prices


def run(shifts, date_range):
    batteries = [{"capacity": 60, "relative_portion": 100}]
    return dashboard.run_strategy_for_period(
        batteries, CHARGER_KW, "99", date_range, 100, "ccm", shifts
    )


def test_dashboard_run_uses_consistent_units(synthetic_company):
    """End to end: a kW load profile, UCC and the model share kWh per slot."""
    shifts = [{"start": datetime.time(8), "end": datetime.time(16), "num_cars": 10}]
    output, metrics = run(shifts, [datetime.date(2024, 2, 1)])
    metrics = metrics.iloc[0]
    ucc_energy = (output["UCC"] - output["Pb"]).sum()

    assert len(output) == SLOTS_PER_DAY
    load_kw, _ = synthetic_company
    assert output["Pb"].to_numpy() == pytest.approx(load_kw * SLOT_HOURS)
    assert output["y"].sum() == pytest.approx(ucc_energy)
    assert metrics["mp_ucc"] == pytest.approx(output["UCC"].max() / SLOT_HOURS)
    assert metrics["ce_ucc"] == pytest.approx(ucc_energy * 300 / 1000)
    assert metrics["cc_model"] < metrics["cc_ucc"]


def test_each_day_carries_the_previous_nights_charging(synthetic_company):
    """Night-shift EVs arrive around 22:00 and charge into the next morning, so every
    day of the period, including the first, starts with charging after midnight."""
    shifts = [{"start": datetime.time(22), "end": datetime.time(6), "num_cars": 10}]
    dates = [datetime.date(2024, 2, 1), datetime.date(2024, 2, 2)]
    output, metrics = run(shifts, dates)

    for date in dates:
        day = output[output["date"] == date]
        assert len(day) == SLOTS_PER_DAY
        assert (day["UCC"] - day["Pb"]).iloc[:4].sum() > 0  # 00:00-01:00
    assert list(metrics["date"]) == dates


def test_costs_and_emissions_belong_to_the_arriving_fleet(synthetic_company):
    """At a constant price, smart and uncontrolled charging of the same fleet cost the
    same, although smart charging shifts much of it past midnight into the next day."""
    _, set_prices = synthetic_company
    prices = np.full(SLOTS_PER_DAY, 50.0)
    prices[:24] = 49.0  # nights are slightly cheaper, so CCM charges after midnight
    set_prices(prices)
    shifts = [{"start": datetime.time(22), "end": datetime.time(6), "num_cars": 10}]
    dates = [datetime.date(2024, 2, 1), datetime.date(2024, 2, 2)]
    output, metrics = run(shifts, dates)

    smart_before_midnight = output["y"].iloc[88:96].sum()
    ucc_before_midnight = (output["UCC"] - output["Pb"]).iloc[88:96].sum()
    assert smart_before_midnight < ucc_before_midnight / 2
    assert metrics["ce_model"].to_numpy() == pytest.approx(metrics["ce_ucc"].to_numpy())
    assert (metrics["cc_model"] <= metrics["cc_ucc"]).all()
    assert metrics["cc_model"].to_numpy() == pytest.approx(
        metrics["cc_ucc"].to_numpy(), rel=0.03
    )
