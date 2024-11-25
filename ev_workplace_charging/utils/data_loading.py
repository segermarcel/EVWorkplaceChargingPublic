import pandas as pd

from ev_workplace_charging.settings import ELECTRICITY_COSTS_SHEET_NAME
from ev_workplace_charging.settings import EV_PARAMETERS_SHEET_NAME
from ev_workplace_charging.settings import EV_PARKING_MATRIX_SHEET_NAME
from ev_workplace_charging.settings import GRID_CARBON_INTENSITY_SHEET_NAME
from ev_workplace_charging.settings import UNCONTROLLED_CHARGING_SHEET_NAME


def generate_parking_matrix(shifts, n_cars):
    # TODO: Implement with given shifts. Use Input Data for now.
    df_parking_matrix = load_parking_matrix(n_cars)
    return df_parking_matrix


def generate_ev_parameters(batteries, n_cars):
    # TODO: Implement with given batteries. Use Input Data for now.
    df_ev_parameters = load_ev_parameters(n_cars)
    return df_ev_parameters


def process_uploaded_power_profile(uploaded_power_profile=None, date=None):
    # TODO: Implement with power profile. Use Input Data for now.
    df_power_profile, mean_power = load_and_process_power_profile(date)
    return df_power_profile, mean_power


def load_charging_costs_from_api(date=None):
    # TODO: Load from API. Use Input data for now.
    df_charging_costs = load_and_process_charging_costs(date)
    return df_charging_costs


def load_grid_carbon_intensity_from_api(date=None):
    # TODO: Load from API. Use Input data for now.
    df_grid_carbon_intensity = load_and_process_grid_carbon_intensity(date)
    return df_grid_carbon_intensity


def load_parking_matrix(n_cars):
    df_parking_matrix = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=EV_PARKING_MATRIX_SHEET_NAME,
        index_col=0,
        nrows=n_cars,
    )
    return df_parking_matrix


def load_ev_parameters(n_cars):
    df_ev_parameters = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=EV_PARAMETERS_SHEET_NAME,
        index_col=0,
        usecols="A:L",
        nrows=n_cars,
    )
    return df_ev_parameters


def load_power_profile():
    df_power_profile = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=POWER_DATA_PB_SHEET_NAME,
        usecols="B:CU",
        skiprows=5,
        nrows=28,
    )
    return df_power_profile


def load_charging_costs():
    df_charging_costs = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=ELECTRICITY_COSTS_SHEET_NAME,
        usecols="B:AY",
        skiprows=5,
        nrows=28,
    )
    return df_charging_costs


def load_grid_carbon_intensity():
    df_grid_carbon_intensity = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=GRID_CARBON_INTENSITY_SHEET_NAME,
        usecols="B:AY",
        skiprows=5,
        nrows=28,
    )
    return df_grid_carbon_intensity


def load_uncontrolled_charging_data():
    df_uncontrolled_charging = pd.read_excel(
        INPUT_DATA_FILE_PATH,
        sheet_name=UNCONTROLLED_CHARGING_SHEET_NAME,
        index_col=0,
        usecols="C:CU",
        skiprows=3,
    )
    return df_uncontrolled_charging


def load_and_process_power_profile(date):
    df_power_profile = load_power_profile()
    df_power_profile = process_date_df(df_power_profile, date)
    mean_power = df_power_profile.mean().mean()

    return df_power_profile, mean_power


def load_and_process_charging_costs(date):
    df_charging_costs = load_charging_costs()
    df_charging_costs = process_date_df(df_charging_costs, date, interpolation="nearest")

    df_charging_costs.loc[pd.Timestamp("1900-01-01 23:45:00")] = df_charging_costs.loc[
        pd.Timestamp("1900-01-01 23:30:00")
    ]

    return df_charging_costs


def load_and_process_grid_carbon_intensity(date):
    df_grid_carbon_intensity = load_grid_carbon_intensity()
    df_grid_carbon_intensity = process_date_df(df_grid_carbon_intensity, date)
    df_grid_carbon_intensity.loc[pd.Timestamp("1900-01-01 23:45:00")] = df_grid_carbon_intensity.loc[
        pd.Timestamp("1900-01-01 23:30:00")
    ]

    return df_grid_carbon_intensity


def load_and_process_uncontrolled_charging_data(ev_portion, df_power_profile):
    df_uncontrolled_charging = load_uncontrolled_charging_data()

    ev_portion = int(ev_portion[:-1]) / 100
    df_uncontrolled_charging = df_uncontrolled_charging.loc[ev_portion]

    df_uncontrolled_charging = convert_to_datetime_index_and_resample(df_uncontrolled_charging)

    df_uncontrolled_charging += df_power_profile

    return df_uncontrolled_charging


def process_date_df(df, date=None, interpolation="linear"):
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Use Date as index
    df.set_index("Date", inplace=True)

    # Drop Weekday column
    df.drop(columns="Weekday", inplace=True)

    if date is not None:
        # Select only row with the given date
        df = df.loc[pd.to_datetime(date)]

    # Transpose so timestamps are index
    df = df.T

    df = convert_to_datetime_index_and_resample(df, interpolation=interpolation)

    return df


def convert_to_datetime_index_and_resample(df, interval="15min", interpolation="linear"):
    # Convert index to datetime
    df.index = pd.to_datetime(df.index, format="%H:%M:%S")

    # Resample to 15 minute intervals
    df = df.resample(interval).interpolate(interpolation)

    return df
