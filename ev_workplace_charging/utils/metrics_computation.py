from ev_workplace_charging.settings import SLOT_HOURS


def compute_max_peak(energy_per_slot):
    """Peak site load [kW] from energy per 15-minute slot [kWh]."""
    return energy_per_slot.max() / SLOT_HOURS


def compute_total_charging_costs(
    charging_values, power_profile_values, charging_cost_values
):
    """EV charging costs [EUR] from energy [kWh] and day-ahead prices [EUR/MWh]."""
    return (
        (charging_values - power_profile_values) * charging_cost_values / 1000
    ).sum()


def compute_total_carbon_emissions(
    charging_values, power_profile_values, grid_carbon_intensity_values
):
    """EV charging emissions [kg CO2] from energy [kWh] and intensity [gCO2/kWh]."""
    return (
        (charging_values - power_profile_values) * grid_carbon_intensity_values / 1000
    ).sum()


def compute_relative_change(new, old):
    return compute_absolute_change(new, old) / old * 100


def compute_absolute_change(new, old):
    return new - old
