from datetime import time

from ev_workplace_charging.utils.data_loading import (
    load_power_profile_01,
    load_power_profile_02,
    load_power_profile_03,
    load_power_profile_04,
    load_power_profile_05,
    load_power_profile_06,
    load_power_profile_07,
    load_power_profile_08,
)


def shift(name, start, end, num_cars):
    return {"name": name, "start": start, "end": end, "num_cars": num_cars}


# Shifts as in Table 1 of the paper
EARLY = (time(6), time(14))
LATE = (time(14), time(22))
NIGHT = (time(22), time(6))
OFFICE = (time(8), time(16))

COMPANY_CONFIGS = {
    "01": {
        "shifts": (
            shift("Early Shift", *EARLY, 90),
            shift("Late Shift", *LATE, 80),
            shift("Night Shift", *NIGHT, 60),
        ),
        "is_kw": True,
        "load_fn": load_power_profile_01,
    },
    "02": {
        "shifts": (shift("Office Hours", *OFFICE, 50),),
        "is_kw": False,
        "load_fn": load_power_profile_02,
    },
    "03": {
        "shifts": (shift("Fleet", time(16), time(7, 30), 50),),
        "is_kw": False,
        "load_fn": load_power_profile_03,
    },
    "04": {
        "shifts": (
            shift("Early Shift", *EARLY, 100),
            shift("Late Shift", *LATE, 150),
            shift("Night Shift", *NIGHT, 80),
            shift("Office Hours", *OFFICE, 300),
        ),
        "is_kw": False,
        "load_fn": load_power_profile_04,
    },
    "05": {
        "shifts": (
            shift("Early Shift", *EARLY, 250),
            shift("Late Shift", *LATE, 175),
            shift("Night Shift", *NIGHT, 80),
            shift("Office Hours", *OFFICE, 60),
        ),
        "is_kw": False,
        "load_fn": load_power_profile_05,
    },
    "06": {
        "shifts": (
            shift("Early Shift", *EARLY, 100),
            shift("Late Shift", *LATE, 70),
            shift("Office Hours", *OFFICE, 100),
        ),
        "is_kw": True,
        "load_fn": load_power_profile_06,
    },
    "07": {
        "shifts": (shift("Office Hours", time(7, 30), time(17), 500),),
        "is_kw": False,
        "load_fn": load_power_profile_07,
    },
    "08": {
        "shifts": (
            shift("Early Shift", *EARLY, 170),
            shift("Late Shift", *LATE, 30),
            shift("Office Hours", time(7), time(16), 140),
        ),
        "is_kw": False,
        "load_fn": load_power_profile_08,
    },
}
