def compute_max_peak(df_with_peaks):
    return df_with_peaks.max()


def compute_total_charging_costs(charging_values, power_profile_values, charging_cost_values):
    return ((charging_values - power_profile_values) * charging_cost_values / 100).sum()


def compute_total_carbon_emissions(charging_values, power_profile_values, grid_carbon_intensity_values):
    return ((charging_values - power_profile_values) * grid_carbon_intensity_values / 1000).sum()


def compute_relative_change(new, old):
    return (new - old) / old * 100
