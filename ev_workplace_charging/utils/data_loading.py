import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient

from ev_workplace_charging.settings import (
    CHARGING_COSTS_DIR,
    GRID_CARBON_INTENSITY_DIR,
    INPUT_DATA_DIR,
)


def generate_ev_parameters(
    shifts, ev_portion, batteries, arrival_lambda=10, departure_lambda=10
):
    # Set seed
    np.random.seed(42)

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
                - timedelta(minutes=np.random.poisson(arrival_lambda))
            ).time()
            departure_time = (
                datetime.combine(datetime.today(), shift["end"])
                + timedelta(minutes=np.random.poisson(departure_lambda))
            ).time()

            ev_parameters["arrival_time"].append(arrival_time)
            ev_parameters["departure_time"].append(departure_time)

            # Assign a random battery type
            e_cap = np.random.choice(
                batteries, p=[b["relative_portion"] / 100 for b in batteries]
            )["capacity"]
            e_ini = np.random.uniform(0.1 * e_cap, 0.8 * e_cap)
            e_next = np.random.uniform(0.8 * e_cap, 1.0 * e_cap)

            ev_parameters["E_cap"].append(e_cap)
            ev_parameters["E_ini"].append(e_ini)
            ev_parameters["E_next"].append(e_next)

            ev_idx += 1

    # Create the dataframe
    df_ev_parameters = pd.DataFrame(ev_parameters)

    return df_ev_parameters


def generate_parking_matrix(df_ev_parameters):
    # Generate timestamp columns for a 24-hour period with 15-minute intervals
    timestamps = pd.date_range(start="00:00", end="23:59", freq="15T").time

    # Initialize the dataframe
    num_evs = df_ev_parameters.shape[0]
    df_parking_matrix = pd.DataFrame(0, index=range(num_evs), columns=timestamps)

    for row in df_ev_parameters.itertuples():
        ev_idx = row.ev_idx

        arrival_time = row.arrival_time
        departure_time = row.departure_time
        if isinstance(arrival_time, str):
            arrival_time = datetime.strptime(arrival_time, "%H:%M:%S").time()
        if isinstance(departure_time, str):
            departure_time = datetime.strptime(departure_time, "%H:%M:%S").time()

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
        skiprows=4,
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
        skiprows=2,
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

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_05(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "05.xlsx",
        names=["Zeit", "Verbrauch"],
        index_col=0,
        skiprows=3,
        nrows=28803,
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

    if date is not None:
        df = df[df.index.date == date]

    return df


def load_power_profile_07(date: datetime.date = None):
    df = pd.read_excel(
        INPUT_DATA_DIR / "07.xlsx",
        names=["Datum", "Zeit", "Verbrauch"],
        usecols="A:C",
        skiprows=0,
    )

    # Combine Datum and Zeit
    df["Zeit"] = pd.to_datetime(df["Datum"] + " " + df["Zeit"], dayfirst=True)
    df = df.set_index("Zeit")
    df = df.drop(columns=["Datum"])

    if date is not None:
        date = date.replace(year=2025)
        df = df[df.index.date == date]

    return df


def load_charging_costs_from_api(date):
    # Load from CSV if exists
    charging_costs_csv_path = CHARGING_COSTS_DIR / f"{date}.csv"
    if charging_costs_csv_path.exists():
        return pd.read_csv(charging_costs_csv_path, index_col=0)

    # Load from API
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    start = pd.Timestamp(date, tz="Europe/Brussels")
    end = pd.Timestamp(date + timedelta(days=1), tz="Europe/Brussels")
    country_code = "DE_LU"  # Germany-Luxembourg
    df = client.query_day_ahead_prices(country_code, start=start, end=end)

    # Convert series to dataframe
    df = pd.DataFrame(df, columns=["Price"])

    # Remove timezone
    df.index = df.index.tz_localize(None)

    # Convert to datetime index and resample
    df = convert_to_datetime_index_and_resample(df)

    # Drop last row
    df = df[:-1]

    # Save to CSV
    df.to_csv(charging_costs_csv_path)

    return df


def load_grid_carbon_intensity_from_api(date):
    year = date.year
    df_grid_carbon_intensity = pd.read_csv(
        GRID_CARBON_INTENSITY_DIR / f"DE_{year}_hourly.csv"
    )

    # Use Datetime (UTC) col as index and convert to datetime
    df_grid_carbon_intensity.set_index("Datetime (UTC)", inplace=True)

    # Keep only Carbon intensity gCO₂eq/kWh (direct) column
    df_grid_carbon_intensity = df_grid_carbon_intensity[
        "Carbon intensity gCO₂eq/kWh (direct)"
    ]

    # Convert to datetime index and resample
    df_grid_carbon_intensity = convert_to_datetime_index_and_resample(
        df_grid_carbon_intensity
    )

    # Filter for the given date
    date_str = date.strftime("%Y-%m-%d")
    mask = df_grid_carbon_intensity.index.strftime("%Y-%m-%d") == date_str
    df_grid_carbon_intensity = df_grid_carbon_intensity[mask]

    return df_grid_carbon_intensity


def compute_uncontrolled_charging_data(
    df_ev_parameters, df_power_profile, max_charger_output_power, is_kw
):
    # Generate timestamp columns for a 24-hour period with 15-minute intervals
    timestamps = pd.date_range(start="00:00", end="23:59", freq="15T").time

    # Initialize the dataframe
    num_evs = df_ev_parameters.shape[0]
    df_charging_matrix = pd.DataFrame(0, index=range(num_evs), columns=timestamps)

    # Process each EV
    for row in df_ev_parameters.itertuples():
        ev_idx = row.ev_idx
        E_ini = row.E_ini
        E_next = row.E_next

        arrival_time = row.arrival_time
        departure_time = row.departure_time
        if isinstance(arrival_time, str):
            arrival_time = datetime.strptime(arrival_time, "%H:%M:%S").time()
        if isinstance(departure_time, str):
            departure_time = datetime.strptime(departure_time, "%H:%M:%S").time()

        # Compute how many 15-minute intervals this EV needs to be charged
        E_diff = E_next - E_ini
        num_intervals_to_charge = int(
            np.ceil(E_diff / (max_charger_output_power * 0.25))
        )  # * 0.25 to convert to kWh

        # Handle overnight shifts
        if departure_time < arrival_time:
            num_evening_intervals_parked = len(
                df_charging_matrix.loc[ev_idx, arrival_time:].index
            )

            # Charge first in the evening, then maybe continue in the morning
            if num_evening_intervals_parked >= num_intervals_to_charge:
                # Evening charging is enough
                df_charging_matrix.loc[ev_idx, arrival_time:].iloc[
                    :num_intervals_to_charge
                ] = max_charger_output_power
            else:
                # Evening charging is not enough, so charge a bit in the morning as well
                df_charging_matrix.loc[ev_idx, arrival_time:] = max_charger_output_power
                num_intervals_to_charge_morning = (
                    num_intervals_to_charge - num_evening_intervals_parked
                )
                df_charging_matrix.loc[ev_idx, :departure_time].iloc[
                    :num_intervals_to_charge_morning
                ] = max_charger_output_power
        else:
            # Normal case: Charge during the day
            df_charging_matrix.loc[ev_idx, arrival_time:departure_time].iloc[
                :num_intervals_to_charge
            ] = max_charger_output_power

    if is_kw:
        total_power = df_charging_matrix.sum()
    else:
        total_power = df_charging_matrix.sum() / 4

    df_ucc = df_power_profile.copy()
    df_ucc["Verbrauch"] = df_ucc["Verbrauch"] + total_power.values

    return df_ucc


def convert_to_datetime_index_and_resample(
    df, interval="15min", interpolation="linear"
):
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

    # Resample to 15 minute intervals
    df = df.resample(interval).interpolate(interpolation)

    return df
