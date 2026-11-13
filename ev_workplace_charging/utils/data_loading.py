import os
from collections import defaultdict
from datetime import datetime, timedelta
from functools import cache

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient

from ev_workplace_charging.settings import (
    CHARGING_COSTS_DIR,
    GRID_CARBON_INTENSITY_DIR,
    INPUT_DATA_DIR,
    SLOT_HOURS,
    SLOTS_PER_DAY,
)

# Load profiles and day-ahead prices are in German local time
LOCAL_TZ = "Europe/Berlin"

# ID 7 consumes 2,000 MWh per year; its standard load profile is normalised to 1 GWh
STANDARD_PROFILE_SCALING = 2


def generate_ev_parameters(
    shifts, ev_portion, batteries, date, arrival_lambda=10, departure_lambda=10
):
    """Simulated EV fleet for one day. Seeding by date gives every day its own fleet,
    while all strategies of that day are compared on the same fleet. A shift that ends
    before it starts (night shift, overnight fleet) ends on the next day."""
    rng = np.random.default_rng(date.toordinal())

    ev_parameters = defaultdict(list)
    for shift in shifts:
        shift_evs = int(shift["num_cars"] * ev_portion / 100)
        start = datetime.combine(date, shift["start"])
        end = datetime.combine(date, shift["end"])
        if end <= start:
            end += timedelta(days=1)

        for _ in range(shift_evs):
            arrival = start - timedelta(minutes=int(rng.poisson(arrival_lambda)))
            departure = end + timedelta(minutes=int(rng.poisson(departure_lambda)))

            battery = rng.choice(
                len(batteries), p=[b["relative_portion"] / 100 for b in batteries]
            )
            e_cap = batteries[battery]["capacity"]

            ev_parameters["arrival"].append(arrival)
            ev_parameters["departure"].append(departure)
            ev_parameters["E_cap"].append(e_cap)
            ev_parameters["E_ini"].append(rng.uniform(0.1 * e_cap, 0.8 * e_cap))
            ev_parameters["E_next"].append(rng.uniform(0.8 * e_cap, 1.0 * e_cap))

    return pd.DataFrame(ev_parameters)


def parked_slots(df_ev_parameters, slots):
    """For each EV, the indices of the `slots` (stamped by interval start) it is parked
    in: from the first slot starting at or after arrival to the last one starting at or
    before departure."""
    first = slots.searchsorted(df_ev_parameters["arrival"], side="left")
    last = slots.searchsorted(df_ev_parameters["departure"], side="right")
    if df_ev_parameters["arrival"].min() < slots[0] or last.max() == len(slots):
        raise ValueError("Parking windows must lie within the modelled slots")
    return [np.arange(a, b) for a, b in zip(first, last)]


def load_period_inputs(load_power_profile_fn, is_kw, days):
    """Site load [kWh per slot], day-ahead price [EUR/MWh] and grid carbon intensity
    [gCO2/kWh] for every slot of the consecutive `days`."""
    frames = []
    for day in days:
        load = load_power_profile_fn(day)["Verbrauch"].to_numpy(float)
        frames.append(
            pd.DataFrame(
                {
                    "Pb": load * SLOT_HOURS if is_kw else load,
                    "price": load_charging_costs_from_api(day)["Price"].to_numpy(),
                    "carbon": load_grid_carbon_intensity_from_api(day).to_numpy(),
                },
                index=pd.date_range(
                    pd.Timestamp(day), periods=SLOTS_PER_DAY, freq="15min"
                ),
            )
        )
    return pd.concat(frames)


def load_power_profile_01(date: datetime.date = None):
    year = 2024

    df = pd.read_excel(
        INPUT_DATA_DIR / "01.xlsx",
        sheet_name=f"{year} OG",
        names=["Datum", "Zeit", "Verbrauch"],
        usecols="A:C",
        skiprows=3,
    )

    # Combine Datum and Zeit
    df["Zeit"] = pd.to_datetime(df["Datum"].astype(str) + " " + df["Zeit"].astype(str))
    df = df.set_index("Zeit")
    df = df.drop(columns=["Datum"])
    df = _stamp_interval_start(df)

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_02(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "02.xlsx",
        header=None,
        names=["Zeit", "Verbrauch"],
        index_col=0,
        usecols="A:B",
        skiprows=3,
    )

    df.index = pd.to_datetime(df.index, dayfirst=True)

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_03(date: datetime.date = None):
    df = pd.read_csv(
        INPUT_DATA_DIR / "03.csv",
        sep=";",
        names=["Datum", "Verbrauch"],
        index_col=0,
        usecols=[0, 7],
        skiprows=1,
        skipfooter=6,
    )

    df.index = pd.to_datetime(df.index, dayfirst=True)

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_04(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "04.xlsx",
        names=["Zeit", "Verbrauch"],
        index_col=0,
        skiprows=1,
    )

    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = _stamp_interval_start(df)

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_05(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "05.xlsx",
        header=None,
        names=["Zeit", "Verbrauch"],
        index_col=0,
        skiprows=3,
        nrows=28804,
    )

    df.index = pd.to_datetime(df.index, dayfirst=True)

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_06(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "06.xlsx",
        names=["Zeit", "Verbrauch"],
        index_col=0,
    )

    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = _stamp_interval_start(df)

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_07(date: datetime.date = None):
    """BDEW standard load profile G25 (general commercial) scaled to the firm's
    2,000 MWh per year, in kWh per slot. It distinguishes the month and the day type
    (workday, Saturday, Sunday); public holidays are not modelled (none in February)."""
    days = pd.date_range("2024-01-01", "2024-12-31").date if date is None else [date]
    df = pd.concat([_standard_load_profile_day(day) for day in days])
    return df


@cache
def _standard_load_profile_sheet():
    return pd.read_excel(INPUT_DATA_DIR / "07.xlsx", sheet_name="G25", header=None)


def _standard_load_profile_day(day):
    sheet = _standard_load_profile_sheet()
    months = pd.to_datetime(sheet.iloc[2, 2:]).dt.month
    day_types = sheet.iloc[3, 2:]
    day_type = {5: "SA", 6: "FT"}.get(day.weekday(), "WT")
    column = months.index[(months == day.month) & (day_types == day_type)][0]
    # The profile is normalised to 1 million kWh per year
    energy = sheet.iloc[4:100, column].to_numpy(float) * STANDARD_PROFILE_SCALING
    index = pd.date_range(pd.Timestamp(day), periods=len(energy), freq="15min")
    return pd.DataFrame({"Verbrauch": energy}, index=index)


def load_power_profile_08(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "08.xlsx",
        names=["Datum", "Zeit", "Verbrauch"],
        usecols="A:C",
        skiprows=0,
    )

    # The export stamps interval ends and writes 24:00 as 00:00 of the same date
    stamps = pd.to_datetime(df["Datum"] + " " + df["Zeit"], dayfirst=True)
    stamps += pd.to_timedelta((df["Zeit"] == "00:00").astype(int), unit="D")
    df = df.set_index(stamps).drop(columns=["Datum", "Zeit"])
    df = _stamp_interval_start(df)

    if date is not None:
        # Only 2025 data exist: use the day 52 weeks later, on the same weekday
        df = df[df.index.date == date + timedelta(weeks=52)]

    return df


def load_charging_costs_from_api(date):
    """Day-ahead prices for Germany-Luxembourg [EUR/MWh] per slot of the local day."""
    # The CSV cache holds the prices as published by ENTSO-E
    charging_costs_csv_path = CHARGING_COSTS_DIR / f"{date}.csv"
    if charging_costs_csv_path.exists():
        prices = pd.read_csv(charging_costs_csv_path, index_col=0)["Price"]
        prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(LOCAL_TZ)
    else:
        client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
        start = pd.Timestamp(date, tz=LOCAL_TZ)
        prices = client.query_day_ahead_prices(
            "DE_LU", start=start, end=start + pd.Timedelta(days=1)
        ).rename("Price")
        prices.to_csv(charging_costs_csv_path)

    return _hold_over_slots(prices, date).to_frame()


def load_grid_carbon_intensity_from_api(date):
    df_grid_carbon_intensity = pd.read_csv(
        GRID_CARBON_INTENSITY_DIR / f"DE_{date.year}_hourly.csv",
        index_col="Datetime (UTC)",
    )["Carbon intensity gCO₂eq/kWh (direct)"]

    # Electricity Maps stamps are UTC; load profiles and prices are German local time
    df_grid_carbon_intensity.index = (
        pd.to_datetime(df_grid_carbon_intensity.index)
        .tz_localize("UTC")
        .tz_convert(LOCAL_TZ)
    )

    return _hold_over_slots(df_grid_carbon_intensity, date)


def compute_uncontrolled_charging(parked, energy, charger_output_power):
    """For each EV, the energy charged in each of its parked slots [kWh] when it charges
    at full power from arrival until it holds exactly E_next, the same energy the
    optimised strategies must deliver."""
    max_energy_per_slot = charger_output_power * SLOT_HOURS
    return [
        np.clip(e - max_energy_per_slot * np.arange(len(slots)), 0, max_energy_per_slot)
        for slots, e in zip(parked, energy)
    ]


def charging_per_slot(parked, charging, num_slots):
    """Total EV charging per slot [kWh] from the charging of each EV in its slots."""
    total = np.zeros(num_slots)
    for slots, energy in zip(parked, charging):
        total[slots] += energy
    return total


def _stamp_interval_start(df):
    """Restamp a meter export that labels each 15-minute interval by its end (00:15
    for 00:00-00:15), so that every series in the model is stamped by interval start."""
    df.index = df.index - pd.Timedelta(hours=SLOT_HOURS)
    return df


def _hold_over_slots(series, date):
    """Values of `series` (stamped by interval start, tz-aware) for the slots of the
    local `date`. Each value holds until the next stamp: an hourly price or carbon
    intensity applies to all four 15-minute slots of its hour."""
    slots = pd.date_range(
        pd.Timestamp(date, tz=LOCAL_TZ), periods=SLOTS_PER_DAY, freq="15min"
    )
    return series.sort_index().reindex(slots, method="ffill").tz_localize(None)
