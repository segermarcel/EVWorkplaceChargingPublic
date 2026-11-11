import os
from bisect import bisect_left
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
)

# Load profiles and day-ahead prices are in German local time
LOCAL_TZ = "Europe/Berlin"

# ID 7 consumes 2,000 MWh per year; its standard load profile is normalised to 1 GWh
STANDARD_PROFILE_SCALING = 2


def generate_ev_parameters(
    shifts, ev_portion, batteries, date, arrival_lambda=10, departure_lambda=10
):
    """Simulated EV fleet for one day. Seeding by date gives every day its own fleet,
    while all strategies of that day are compared on the same fleet."""
    rng = np.random.default_rng(date.toordinal())

    ev_parameters = defaultdict(list)
    ev_idx = 0
    # Iterate through each shift
    for shift in shifts:
        # Calculate number of EVs for this shift based on portion
        shift_evs = int(shift["num_cars"] * ev_portion / 100)

        # Generate parameters for EVs in this shift
        for _ in range(shift_evs):
            ev_parameters["ev_idx"].append(ev_idx)

            # Generate arrival and departure times
            arrival_time = (
                datetime.combine(datetime.today(), shift["start"])
                - timedelta(minutes=int(rng.poisson(arrival_lambda)))
            ).time()
            departure_time = (
                datetime.combine(datetime.today(), shift["end"])
                + timedelta(minutes=int(rng.poisson(departure_lambda)))
            ).time()

            ev_parameters["arrival_time"].append(arrival_time)
            ev_parameters["departure_time"].append(departure_time)

            # Assign a random battery type
            battery = rng.choice(
                len(batteries), p=[b["relative_portion"] / 100 for b in batteries]
            )
            e_cap = batteries[battery]["capacity"]
            e_ini = rng.uniform(0.1 * e_cap, 0.8 * e_cap)
            e_next = rng.uniform(0.8 * e_cap, 1.0 * e_cap)

            ev_parameters["E_cap"].append(e_cap)
            ev_parameters["E_ini"].append(e_ini)
            ev_parameters["E_next"].append(e_next)

            ev_idx += 1

    # Create the dataframe
    df_ev_parameters = pd.DataFrame(ev_parameters)

    return df_ev_parameters


def generate_parking_matrix(df_ev_parameters):
    # Generate timestamp columns for a 24-hour period with 15-minute intervals
    timestamps = pd.date_range(start="00:00", end="23:59", freq="15min").time

    # Initialize the dataframe
    num_evs = df_ev_parameters.shape[0]
    df_parking_matrix = pd.DataFrame(0, index=range(num_evs), columns=timestamps)

    for row in df_ev_parameters.itertuples():
        ev_idx = row.ev_idx

        arrival_time = _as_time(row.arrival_time)
        departure_time = _as_time(row.departure_time)

        # Handle overnight shifts
        if departure_time < arrival_time:
            df_parking_matrix.loc[ev_idx, arrival_time:] = 1
            df_parking_matrix.loc[ev_idx, :departure_time] = 1
        else:
            df_parking_matrix.loc[ev_idx, arrival_time:departure_time] = 1

    return df_parking_matrix


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


def compute_uncontrolled_charging_data(
    df_ev_parameters, df_parking_matrix, df_power_profile, charger_output_power
):
    """Add uncontrolled charging to the site load (both in kWh per slot).

    Each EV charges at full power from arrival until it holds exactly E_next, the
    same energy the optimised strategies must deliver.
    """
    max_energy_per_slot = charger_output_power * SLOT_HOURS
    timestamps = list(df_parking_matrix.columns)
    charging = np.zeros(df_parking_matrix.shape)

    for row in df_ev_parameters.itertuples():
        parked_slots = np.flatnonzero(df_parking_matrix.loc[row.ev_idx].to_numpy())
        arrival_slot = bisect_left(timestamps, _as_time(row.arrival_time))
        # Overnight stays wrap around midnight: charge from arrival, then from 00:00
        parked_slots = sorted(parked_slots, key=lambda n: (n < arrival_slot, n))

        remaining = row.E_next - row.E_ini
        for n in parked_slots:
            if remaining <= 0:
                break
            charging[row.ev_idx, n] = min(max_energy_per_slot, remaining)
            remaining -= charging[row.ev_idx, n]

    df_ucc = df_power_profile.copy()
    df_ucc["Verbrauch"] = df_ucc["Verbrauch"] + charging.sum(axis=0)

    return df_ucc


def _as_time(value):
    if isinstance(value, str):
        return datetime.strptime(value, "%H:%M:%S").time()
    return value


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
        pd.Timestamp(date, tz=LOCAL_TZ), periods=round(24 / SLOT_HOURS), freq="15min"
    )
    return series.sort_index().reindex(slots, method="ffill").tz_localize(None)
