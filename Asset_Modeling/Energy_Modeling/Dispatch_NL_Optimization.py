import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Logger.Logger import mylogger

COAL_COST = 1.4
COAL_STARTUPCOST = 2
GAS_COST = 15
GAS_STARTUPCOST = 5
NUC_COST = 6.5
NUC_STARTUPCOST = 10000
OIL_COST = 1.3
OIL_STARTUPCOST = 0
DEMAND = 1000
COAL_MAX = 1000
GAS_MAX = 600
NUC_MAX = 600
OIL_MAX = 1000

COAL_EXP = 1.4
OIL_EXP = 1.5
DEFAULT_INIT = DEMAND / 4

def find_energy_mix(initial_guesses=None):
    if initial_guesses is None:
        initial_guesses = [
            [0, 0, 0, 0],
            [DEFAULT_INIT, DEFAULT_INIT, DEFAULT_INIT, DEFAULT_INIT],
            [DEMAND, 0, 0, 0],
            [0, DEMAND, 0, 0],
            [0, 0, DEMAND, 0],
            [0, 0, 0, DEMAND],
        ]

    def objective(x):
        coal, gas, nuclear, oil = x
        startup_costs = (
            COAL_STARTUPCOST * (coal > 0),
            GAS_STARTUPCOST * (gas > 0),
            NUC_STARTUPCOST * (nuclear > 0),
            OIL_STARTUPCOST * (oil > 0),
        )
        return (
            sum(startup_costs)
            + COAL_COST * coal**COAL_EXP
            + GAS_COST * gas
            + NUC_COST * nuclear
            + OIL_COST * oil**OIL_EXP
        )

    def demand_constraint(x):
        return np.sum(x) - DEMAND

    bounds = [(0, COAL_MAX), (0, GAS_MAX), (0, NUC_MAX), (0, OIL_MAX)]
    constraints = [{"type": "eq", "fun": demand_constraint}]

    # Optimize for all initial guesses
    results = []
    for guess in initial_guesses:
        result = minimize(objective, guess, bounds=bounds, constraints=constraints, method="SLSQP")
        if result.success:
            results.append(result)

    if not results:
        mylogger.logger.error("All optimizations failed.")
        return

    # Choose the best result
    best_result = min(results, key=lambda r: r.fun)
    if best_result.success:
        log_results(best_result)
        plot_energy_mix(best_result.x)
    else:
        mylogger.logger.error("Optimization failed for all initial guesses.")

def log_results(result):
    sources = ["Coal", "Gas", "Nuclear", "Oil"]
    mylogger.logger.info("Optimization Successful")
    for i, source in enumerate(sources):
        mylogger.logger.info(f"{source} Power: {result.x[i]:.2f} MW")
    mylogger.logger.info(f"Total Cost: {result.fun:.2f}")

def screening_curves_mw():
    production_levels = np.linspace(0, DEMAND, 100)
    coal_costs = COAL_STARTUPCOST + COAL_COST * production_levels**COAL_EXP
    gas_costs = GAS_STARTUPCOST + GAS_COST * production_levels
    nuc_costs = NUC_STARTUPCOST + NUC_COST * production_levels
    oil_costs = OIL_STARTUPCOST + OIL_COST * production_levels**OIL_EXP

    plt.figure()
    plt.plot(production_levels, coal_costs, label="Coal")
    plt.plot(production_levels, gas_costs, label="Gas")
    plt.plot(production_levels, nuc_costs, label="Nuclear")
    plt.plot(production_levels, oil_costs, label="Oil")
    plt.title("Screening Curves (Cost vs MW Production)")
    plt.xlabel("MW Production")
    plt.ylabel("Cost ($)")
    plt.legend()
    plt.grid()
    plt.show()

def screening_curves_capacity():
    capacity_factors = np.linspace(0, 1, 100)
    coal_costs = COAL_STARTUPCOST + COAL_COST * (capacity_factors * COAL_MAX)**COAL_EXP
    gas_costs = GAS_STARTUPCOST + GAS_COST * (capacity_factors * GAS_MAX)
    nuc_costs = NUC_STARTUPCOST + NUC_COST * (capacity_factors * NUC_MAX)
    oil_costs = OIL_STARTUPCOST + OIL_COST * (capacity_factors * OIL_MAX)**OIL_EXP

    plt.figure()
    plt.plot(capacity_factors, coal_costs, label="Coal")
    plt.plot(capacity_factors, gas_costs, label="Gas")
    plt.plot(capacity_factors, nuc_costs, label="Nuclear")
    plt.plot(capacity_factors, oil_costs, label="Oil")
    plt.title("Screening Curves")
    plt.xlabel("Capacity Factor")
    plt.ylabel("Cost per MW ($)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_energy_mix(power_outputs):
    sources = ["Coal", "Gas", "Nuclear", "Oil"]
    max_productions = [COAL_MAX, GAS_MAX, NUC_MAX, OIL_MAX]
    percentages = [output / max_prod * 100 for output, max_prod in zip(power_outputs, max_productions)]

    plt.figure()
    plt.bar(sources, percentages, color="skyblue")
    plt.title("Energy Mix Allocation (Percentage of Max Production)")
    plt.xlabel("Energy Source")
    plt.ylabel("Percentage of Max Production (%)")
    plt.ylim(0, 100)
    plt.show()

# Run the optimization and plot the results
find_energy_mix()
screening_curves_mw()
screening_curves_capacity()
