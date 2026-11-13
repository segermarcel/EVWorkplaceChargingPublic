from collections import defaultdict

import numpy as np
import pyomo.environ as pyo

from ev_workplace_charging.settings import GUROBI_THREADS, SLOT_HOURS


def solve_charging(
    model_type, parked, energy, inputs, charger_output_power, solver_type
):
    """For each EV, the energy the strategy `model_type` charges in each of its parked
    slots [kWh]."""
    model = setup_model(model_type, parked, energy, inputs, charger_output_power)
    solve_model(solver_type, model)
    return [
        np.array([pyo.value(model.x[m, n]) for n in slots])
        for m, slots in enumerate(parked)
    ]


def setup_model(model_type, parked, energy, inputs, charger_output_power):
    """Optimisation model over a continuous horizon of 15-minute slots.

    parked: for each EV, the indices of the slots it is parked in
    energy: for each EV, the energy it must be charged with [kWh] (E_next - E_ini)
    inputs: per slot, site load Pb [kWh], price [EUR/MWh] and carbon [gCO2/kWh]
    """
    evs_at = defaultdict(list)
    for m, slots in enumerate(parked):
        for n in slots:
            evs_at[n].append(m)

    model = pyo.ConcreteModel(name=model_type)
    model.M = pyo.Set(initialize=range(len(parked)), name="EVs")
    model.N = pyo.Set(initialize=sorted(evs_at), name="slots with a parked EV")
    model.P = pyo.Set(
        initialize=[(m, n) for m, slots in enumerate(parked) for n in slots],
        dimen=2,
        name="parked EV-slot pairs",
    )

    # Energy charged into EV m in slot n [kWh], up to the charger limit
    model.x = pyo.Var(model.P, bounds=(0, charger_output_power * SLOT_HOURS))
    # Total EV charging in slot n [kWh]
    model.y = pyo.Var(model.N)

    model.evchargingload = pyo.Constraint(
        model.N,
        rule=lambda mdl, n: mdl.y[n] == pyo.quicksum(mdl.x[m, n] for m in evs_at[n]),
    )
    # Every strategy, and uncontrolled charging, delivers exactly the requested energy,
    # so differences between them come from timing alone. As E_next <= E_cap and x >= 0,
    # the battery can never exceed its capacity, so no per-slot capacity bound is needed.
    model.energy = pyo.Constraint(
        model.M,
        rule=lambda mdl, m: pyo.quicksum(mdl.x[m, n] for n in parked[m]) == energy[m],
    )

    if model_type == "ps":
        site_load = inputs["Pb"]
        # Valley-filling target: the midpoint of each calendar day's site load
        target = site_load.groupby(site_load.index.date).transform(
            lambda day: (day.min() + day.max()) / 2
        )
        pb, c = site_load.to_numpy(), target.to_numpy()
        objective = pyo.quicksum((pb[n] + model.y[n] - c[n]) ** 2 for n in model.N)
    elif model_type in ("ccm", "cem"):
        signal = inputs["price" if model_type == "ccm" else "carbon"].to_numpy()
        objective = pyo.quicksum(signal[n] * model.y[n] for n in model.N)
    else:
        raise ValueError(f"Model type {model_type} not recognized")
    model.obj = pyo.Objective(expr=objective)  # minimise

    return model


def solve_model(solver_type, model):
    solver = pyo.SolverFactory(solver_type)
    if solver_type == "gurobi":
        solver.options["Threads"] = GUROBI_THREADS
    res = solver.solve(model)  # tee = True   to see detailed solver output
    pyo.assert_optimal_termination(res)
