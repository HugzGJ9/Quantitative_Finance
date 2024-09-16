Quantitative_Finance is a project that began in Fall 2023, with its purpose and motivations evolving over time. Initially, the goal was to create a comprehensive project that integrated various academic approaches. However, only a few useful features from these approaches have been incorporated into the project.

The second motivation was to develop a personal Python library for quantitative finance, aimed at retrieving and visualizing metrics related to advanced options strategies.

The third objective was to study the evolution of an options book over time by simulating the underlying assets and analyzing how risks change.

The current motivation is to utilize this project as a tool for managing an options portfolio.

1/ASSET CLASS

In this project, the underlying asset of an option has been modeled as an object. The main purpose was to enable simulations and manage positions, specifically for hedging in option management.

2/OPTION CLASS

The instance creation first requires the creation of an Asset object, which is then used as a parameter for the Option_eu object. Below is the list of methods defined for the class:

DeltaRisk / Delta_DF / Delta_surface / GammaRisk / Gamma_DF / Gamma_surface / VegaRisk / Vega_DF / Vega_surface / ThetaRisk / Theta_DF / Theta_surface / PnlRisk / RiskAnalysis /  display_payoff_option / get_payoff_option / option_price_close_formulae / option_price_mc / run_Booking / simu_asset


![graphviz](https://github.com/user-attachments/assets/c7cde25a-0317-4ea7-84bd-7d4e8f35b87a)
