import pandas as pd
import pyomo.environ as pyo


def setup_and_solve_ps_model(
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    charger_output_power,
    solver_type,
):
    ps_model = setup_model(
        model_type="ps",
        df_parking_matrix=df_parking_matrix,
        df_ev_parameters=df_ev_parameters,
        df_power_profile=df_power_profile,
        df_charging_costs=None,
        df_grid_carbon_intensity=None,
        charger_output_power=charger_output_power,
    )

    solve_model(solver_type, ps_model)

    return ps_model


def setup_and_solve_ccm_model(
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    df_charging_costs,
    charger_output_power,
    solver_type,
):
    ccm_model = setup_model(
        model_type="ccm",
        df_parking_matrix=df_parking_matrix,
        df_ev_parameters=df_ev_parameters,
        df_power_profile=df_power_profile,
        df_charging_costs=df_charging_costs,
        df_grid_carbon_intensity=None,
        charger_output_power=charger_output_power,
    )

    solve_model(solver_type, ccm_model)

    return ccm_model


def setup_and_solve_cem_model(
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    df_grid_carbon_intensity,
    charger_output_power,
    solver_type,
):
    cem_model = setup_model(
        model_type="cem",
        df_parking_matrix=df_parking_matrix,
        df_ev_parameters=df_ev_parameters,
        df_power_profile=df_power_profile,
        df_charging_costs=None,
        df_grid_carbon_intensity=df_grid_carbon_intensity,
        charger_output_power=charger_output_power,
    )

    solve_model(solver_type, cem_model)

    return cem_model


def setup_model(
    model_type,
    df_parking_matrix,
    df_ev_parameters,
    df_power_profile,
    df_charging_costs,
    df_grid_carbon_intensity,
    charger_output_power,
):
    # Process parking matrix
    num_cars, num_timesteps = df_parking_matrix.shape
    M = range(1, num_cars + 1)
    N = range(1, num_timesteps + 1)
    f = {(r, c): df_parking_matrix.values[r - 1, c - 1] for r in M for c in N}

    # Process EV parameters
    E_cap = {r: df_ev_parameters["E_cap"].values[r - 1] for r in M}
    E_ini = {r: df_ev_parameters["E_ini"].values[r - 1] for r in M}
    E_next = {r: df_ev_parameters["E_next"].values[r - 1] for r in M}

    # Process Power Profile
    Pb = {n: df_power_profile.values[n - 1] for n in N}
    C = (min(df_power_profile) + max(df_power_profile)) / 2

    # Process charging costs
    if model_type == "ccm":
        p = {n: df_charging_costs.values[n - 1] for n in N}
    else:
        p = None

    # Process carbon intensity
    if model_type == "cem":
        gci = {n: df_grid_carbon_intensity.values[n - 1] for n in N}
    else:
        gci = None

    t_interval = 15 / 60
    P_MAX = charger_output_power * t_interval
    P_MIN = 0
    TAU = 1

    model = create_model(model_type, M, N, E_next, E_cap, E_ini, Pb, C, P_MAX, P_MIN, TAU, f, p, gci)

    return model


def solve_model(solver_type, model):
    solver = pyo.SolverFactory(solver_type)
    res = solver.solve(model)  # tee = True   to see detailed solver output
    pyo.assert_optimal_termination(res)


def save_model_output(
    model,
    df_uncontrolled_charging,
    df_charging_costs,
    df_grid_carbon_intensity,
    mean_power,
    output_file_path,
):
    # Create df
    df_output = pd.DataFrame(
        {
            "n": [n for n in model.N],
            "y": [pyo.value(model.y[n]) for n in model.N],
            "Pb": [pyo.value(model.Pb[n]) for n in model.N],
            "Tc": [pyo.value(model.Pb[n]) + pyo.value(model.y[n]) for n in model.N],
            "UCC": df_uncontrolled_charging.values,
            "charging_costs": df_charging_costs.values,
            "grid_carbon_intensity": df_grid_carbon_intensity.values,
        }
    )

    df_output_unscaled = df_output.copy()

    # Divide by mean_power
    df_output["Pb"] = df_output["Pb"] / mean_power
    df_output["Tc"] = df_output["Tc"] / mean_power
    df_output["UCC"] = df_output["UCC"] / mean_power

    # Save
    df_output.to_csv(output_file_path, index=False)

    return df_output_unscaled


def create_model(model_type, M, N, E_next, E_cap, E_ini, Pb, C, P_MAX, P_MIN, tau, f, p=None, gci=None):
    model = pyo.ConcreteModel(name=model_type)

    # Sets
    model.M = pyo.Set(initialize=M, name="EVs")
    model.N = pyo.Set(initialize=N, name="time intervals")

    # Define and initialise parameters
    model.E_next = pyo.Param(M, initialize=E_next)
    model.E_cap = pyo.Param(M, initialize=E_cap)
    model.E_ini = pyo.Param(M, initialize=E_ini)
    model.Pb = pyo.Param(N, initialize=Pb)
    model.C = pyo.Param(initialize=C)
    model.P_MAX = pyo.Param(initialize=P_MAX, mutable=True)
    model.P_MIN = pyo.Param(initialize=P_MIN, mutable=True)
    model.tau = pyo.Param(initialize=tau)
    model.f = pyo.Param(M, N, initialize=f)

    if model_type == "ccm" and p is not None:
        model.p = pyo.Param(N, initialize=p)

    if model_type == "cem" and gci is not None:
        model.gci = pyo.Param(N, initialize=gci)

    # Define and initialise decision variables
    def x_bounds(mdl, m, n):
        if mdl.f[m, n] == 0:
            return (0, 0)
        else:
            return (P_MIN, P_MAX)

    def x_init(mdl, m, n):
        if mdl.f[m, n] == 0:
            return 0
        else:
            return P_MIN

    model.x = pyo.Var(M, N, initialize=x_init, bounds=x_bounds)  # Charging/ discharging power of EV m in interval i
    # Total load for charging/discharging the available EVs in interval i
    model.y = pyo.Var(N, initialize=0, within=pyo.Reals)
    model.E_fin = pyo.Var(M, initialize=0, within=pyo.NonNegativeReals)

    # Objective function
    if model_type == "ps":
        rule = obj_rule_peak_shaving
    elif model_type == "ccm":
        rule = obj_rule_charging_cost_minimization
    elif model_type == "cem":
        rule = obj_rule_carbon_emission_minimization
    else:
        raise ValueError(f"Model type {model_type} not recognized")

    model.obj = pyo.Objective(rule=rule)  # Default rule: minimize

    def evchargingload_rule(mdl, n):
        return sum(mdl.x[m, n] * mdl.f[m, n] for m in mdl.M) == mdl.y[n]

    model.evchargingload = pyo.Constraint(N, rule=evchargingload_rule)

    def lbcharge_rule(mdl, m, n):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, k] * mdl.f[m, k] for k in range(1, n + 1)) >= 0

    model.lbcharge = pyo.Constraint(M, N, rule=lbcharge_rule)

    def ubcharge_rule(mdl, m, n):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, k] * mdl.f[m, k] for k in range(1, n + 1)) <= mdl.E_cap[m]

    model.ubcharge = pyo.Constraint(M, N, rule=ubcharge_rule)

    def SoCfinal_rule(mdl, m):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, n] * mdl.f[m, n] for n in mdl.N) == mdl.E_fin[m]

    model.SoCfinal = pyo.Constraint(M, rule=SoCfinal_rule)

    def SoCnext_rule(mdl, m):
        return mdl.E_ini[m] + sum(mdl.tau * mdl.x[m, n] * mdl.f[m, n] for n in mdl.N) >= mdl.E_next[m]

    model.SoCnext = pyo.Constraint(M, rule=SoCnext_rule)

    return model


def obj_rule_peak_shaving(mdl):
    return sum(pow((mdl.Pb[n] + mdl.y[n] - mdl.C), 2) for n in mdl.N)


def obj_rule_charging_cost_minimization(mdl):
    return sum(mdl.y[n] * mdl.p[n] for n in mdl.N)


def obj_rule_carbon_emission_minimization(mdl):
    return sum(mdl.y[n] * mdl.gci[n] for n in mdl.N)
