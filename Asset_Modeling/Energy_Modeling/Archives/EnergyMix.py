import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from Logger.Logger import mylogger
# Constants
COAL_COST = 25
COAL_STARTUPCOST = 16
GAS_COST = 80
GAS_STARTUPCOST = 5
NUC_COST = 6.5
NUC_STARTUPCOST = 32
OIL_COST = 100
OIL_STARTUPCOST = 0
DEMAND = 600
COAL_MAX = 500
GAS_MAX = 200
NUC_MAX = 200
OIL_MAX = 1000

def FindEnergyMix():
    # Objective coefficients (costs)
    c = [
        COAL_COST, GAS_COST, NUC_COST, OIL_COST,  # Power production costs
        COAL_STARTUPCOST, GAS_STARTUPCOST, NUC_STARTUPCOST, OIL_STARTUPCOST  # Startup costs
    ]

    # Inequality constraints (A_ub * x <= b_ub)
    A_ub = [
        [1, 0, 0, 0, -COAL_MAX, 0, 0, 0],  # Coal max production with startup
        [0, 1, 0, 0, 0, -GAS_MAX, 0, 0],  # Gas max production with startup
        [0, 0, 1, 0, 0, 0, -NUC_MAX, 0],  # Nuclear max production with startup
        [0, 0, 0, 1, 0, 0, 0, -OIL_MAX]   # Oil max production with startup
    ]
    b_ub = [0, 0, 0, 0]

    # Equality constraint (A_eq * x = b_eq)
    A_eq = [[1, 1, 1, 1, 0, 0, 0, 0]]  # Sum of productions equals demand
    b_eq = [DEMAND]

    # Variable bounds
    bounds = [
        (0, COAL_MAX), (0, GAS_MAX), (0, NUC_MAX), (0, OIL_MAX),  # Power production bounds
        (0, 1), (0, 1), (0, 1), (0, 1)  # Binary startup approximations as [0, 1]
    ]

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        mylogger.logger.info("Optimization Successful")
        power_outputs = result.x[:4]
        startup_flags = result.x[4:]
        sources = ['Coal', 'Gas', 'Nuclear', 'Oil']
        for i, source in enumerate(sources):
            mylogger.logger.info(f"{source} Power: {power_outputs[i]:.2f} MW, Startup Flag: {startup_flags[i]:.2f}")
        mylogger.logger.info(f"Total Cost: {result.fun:.2f}")
        energy_mix_allocation(power_outputs)
    else:
        mylogger.logger.info("Optimization Failed")

    screening_curves()
    screening_curvesMW()

def screening_curvesMW():
    production_levels = np.linspace(0, max(COAL_MAX, GAS_MAX, NUC_MAX, OIL_MAX), 100)
    coal_costs = COAL_STARTUPCOST + COAL_COST * production_levels
    gas_costs = GAS_STARTUPCOST + GAS_COST * production_levels
    nuc_costs = NUC_STARTUPCOST + NUC_COST * production_levels
    oil_costs = OIL_STARTUPCOST + OIL_COST * production_levels
    plt.plot(production_levels, coal_costs, label='Coal')
    plt.plot(production_levels, gas_costs, label='Gas')
    plt.plot(production_levels, nuc_costs, label='Nuclear')
    plt.plot(production_levels, oil_costs, label='Oil')
    plt.title('Screening Curves (Cost vs MW Production)')
    plt.xlabel('MW Production')
    plt.ylabel('Cost ($)')
    plt.legend()
    plt.grid()
    plt.show()

def screening_curves():
    capacity_factors = np.linspace(0, 1, 100)
    coal_costs = COAL_STARTUPCOST + COAL_COST * capacity_factors
    gas_costs = GAS_STARTUPCOST + GAS_COST * capacity_factors
    nuc_costs = NUC_STARTUPCOST + NUC_COST * capacity_factors
    oil_costs = OIL_STARTUPCOST + OIL_COST * capacity_factors
    plt.plot(capacity_factors, coal_costs, label='Coal')
    plt.plot(capacity_factors, gas_costs, label='Gas')
    plt.plot(capacity_factors, nuc_costs, label='Nuclear')
    plt.plot(capacity_factors, oil_costs, label='Oil')
    plt.title('Screening Curves')
    plt.xlabel('Capacity Factor')
    plt.ylabel('Cost per MW ($)')
    plt.legend()
    plt.grid()
    plt.show()

def energy_mix_allocation(power_outputs):
    sources = ['Coal', 'Gas', 'Nuclear', 'Oil']
    max_productions = [COAL_MAX, GAS_MAX, NUC_MAX, OIL_MAX]
    percentages = [output / max_prod * 100 for output, max_prod in zip(power_outputs, max_productions)]
    plt.bar(sources, percentages, color='skyblue')
    plt.title('Energy Mix Allocation (Percentage of Max Production)')
    plt.xlabel('Energy Source')
    plt.ylabel('Percentage of Max Production (%)')
    plt.ylim(0, 100)
    plt.show()

FindEnergyMix()
