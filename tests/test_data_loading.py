"""Every input series is stamped by interval start in German local time."""

import datetime

import pandas as pd
import pytest

from ev_workplace_charging.company_configs import COMPANY_CONFIGS
from ev_workplace_charging.settings import (
    CHARGING_COSTS_DIR,
    GRID_CARBON_INTENSITY_DIR,
    INPUT_DATA_DIR,
)
from ev_workplace_charging.utils.data_loading import (
    generate_ev_parameters,
    load_charging_costs_from_api,
    load_grid_carbon_intensity_from_api,
)

SLOTS = 96
DATE = datetime.date(2024, 2, 1)


def test_each_day_has_its_own_reproducible_ev_fleet():
    shifts = [{"start": datetime.time(8), "end": datetime.time(16), "num_cars": 50}]
    batteries = [
        {"capacity": 48, "relative_portion": 50},
        {"capacity": 100, "relative_portion": 50},
    ]
    next_day = DATE + datetime.timedelta(days=1)

    fleet = generate_ev_parameters(shifts, 100, batteries, DATE)

    pd.testing.assert_frame_equal(
        fleet, generate_ev_parameters(shifts, 100, batteries, DATE)
    )
    assert not fleet.equals(generate_ev_parameters(shifts, 100, batteries, next_day))
    assert set(fleet["E_cap"]) == {48, 100}


def test_grid_carbon_intensity_is_in_local_time_and_holds_over_the_hour():
    raw = pd.read_csv(
        GRID_CARBON_INTENSITY_DIR / "DE_2024_hourly.csv", index_col="Datetime (UTC)"
    )["Carbon intensity gCO₂eq/kWh (direct)"]
    local = load_grid_carbon_intensity_from_api(DATE)

    assert len(local) == SLOTS
    assert local.index[0] == pd.Timestamp(DATE)
    # Berlin is UTC+1 in February: local hour h is UTC hour h-1
    for hour in range(24):
        utc = (pd.Timestamp(DATE) + pd.Timedelta(hours=hour - 1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assert (local.iloc[4 * hour : 4 * hour + 4] == raw[utc]).all()


def test_day_ahead_prices_hold_over_the_hour():
    raw = pd.read_csv(CHARGING_COSTS_DIR / f"{DATE}.csv", index_col=0)["Price"]
    prices = load_charging_costs_from_api(DATE)["Price"]

    assert len(raw) == 24
    assert len(prices) == SLOTS
    assert prices.index[0] == pd.Timestamp(DATE)
    assert prices.to_numpy().tolist() == raw.repeat(4).to_numpy().tolist()


@pytest.mark.skipif(
    not any(INPUT_DATA_DIR.iterdir()), reason="confidential load profiles not present"
)
@pytest.mark.parametrize("company", COMPANY_CONFIGS)
def test_load_profiles_are_stamped_by_interval_start(company):
    """Meter exports differ (some stamp interval ends, some skip header rows): after
    loading, each year starts with the 00:00-00:15 interval and has no gaps."""
    profile = COMPANY_CONFIGS[company]["load_fn"]()

    assert profile.index[0].time() == datetime.time(0, 0)
    assert profile.index[0].month == profile.index[0].day == 1
    steps = profile.index.to_series().diff().dropna()
    february = steps[steps.index.month == 2]
    assert (february == pd.Timedelta(minutes=15)).all()


@pytest.mark.skipif(
    not any(INPUT_DATA_DIR.iterdir()), reason="confidential load profiles not present"
)
@pytest.mark.parametrize("company", COMPANY_CONFIGS)
def test_load_profiles_match_the_weekday_of_the_modelled_date(company):
    """Load shapes differ between workdays and weekends, so a profile from another
    year must come from the same weekday as the prices and carbon intensities."""
    for date in pd.date_range("2024-02-01", "2024-02-29").date:
        profile = COMPANY_CONFIGS[company]["load_fn"](date)

        assert len(profile) == SLOTS
        assert profile.index[0].weekday() == date.weekday()
