from ev_workplace_charging.utils.data_loading import (
    load_power_profile_01,
    load_power_profile_02,
    load_power_profile_03,
    load_power_profile_04,
    load_power_profile_05,
    load_power_profile_06,
    load_power_profile_07,
)

COMPANY_CONFIGS = {
    "01": {
        "shifts": (
            {"name": "Early Shift", "start_h": 6, "end_h": 14, "num_cars": 90},
            {"name": "Late Shift", "start_h": 14, "end_h": 22, "num_cars": 80},
            {"name": "Night Shift", "start_h": 22, "end_h": 6, "num_cars": 60},
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 50},
        ),
        "is_kw": True,
        "load_fn": load_power_profile_01,
    },
    "02": {
        "shifts": (
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 50},
        ),
        "is_kw": False,
        "load_fn": load_power_profile_02,
    },
    "03": {
        "shifts": (
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 50},
        ),
        "is_kw": False,
        "load_fn": load_power_profile_03,
    },
    "04": {
        "shifts": (
            {"name": "Early Shift", "start_h": 6, "end_h": 14, "num_cars": 100},
            {"name": "Late Shift", "start_h": 14, "end_h": 22, "num_cars": 150},
            {"name": "Night Shift", "start_h": 22, "end_h": 6, "num_cars": 80},
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 300},
        ),
        "is_kw": False,
        "load_fn": load_power_profile_04,
    },
    "05": {
        "shifts": (
            {"name": "Early Shift", "start_h": 6, "end_h": 14, "num_cars": 250},
            {"name": "Late Shift", "start_h": 14, "end_h": 22, "num_cars": 175},
            {"name": "Night Shift", "start_h": 22, "end_h": 6, "num_cars": 80},
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 60},
        ),
        "is_kw": False,
        "load_fn": load_power_profile_05,
    },
    "06": {
        "shifts": (
            {"name": "Early Shift", "start_h": 6, "end_h": 14, "num_cars": 100},
            {"name": "Late Shift", "start_h": 14, "end_h": 22, "num_cars": 70},
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 100},
        ),
        "is_kw": True,
        "load_fn": load_power_profile_06,
    },
    "08": {
        "shifts": (
            {"name": "Early Shift", "start_h": 6, "end_h": 14, "num_cars": 170},
            {"name": "Late Shift", "start_h": 14, "end_h": 22, "num_cars": 30},
            {"name": "Office Hours", "start_h": 8, "end_h": 16, "num_cars": 140},
        ),
        "is_kw": False,
        "load_fn": load_power_profile_07,
    },
}
