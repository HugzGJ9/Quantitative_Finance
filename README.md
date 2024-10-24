Quantitative_Finance is a project that began in Fall 2023, with its purpose and motivations evolving over time. Initially, the goal was to create a comprehensive project that integrated various academic approaches. However, only a few useful features from these approaches have been incorporated into the project.

The second motivation was to develop a personal Python library for quantitative finance, aimed at retrieving and visualizing metrics related to advanced options strategies.

The third objective was to study the evolution of an options book over time by simulating the underlying assets and analyzing how risks change.

The current motivation is to utilize this project as a tool for managing an options portfolio.

1/ASSET CLASS

In this project, the underlying asset of an option has been modeled as an object. The main purpose was to enable simulations and manage positions, specifically for hedging in option management.

2/OPTION CLASS

The instance creation first requires the creation of an Asset object, which is then used as a parameter for the Option_eu object. Below is the list of methods defined for the class:

DeltaRisk / Delta_DF / Delta_surface / GammaRisk / Gamma_DF / Gamma_surface / VegaRisk / Vega_DF / Vega_surface / ThetaRisk / Theta_DF / Theta_surface / PnlRisk / RiskAnalysis /  display_payoff_option / get_payoff_option / option_price_close_formulae / option_price_mc / run_Booking / simu_asset

3/ OPTION 1st GEN

The Option 1st Generation class is defined as a combination of European vanilla options, such as spreads, straddles, and strangles. An inheritance relationship exists between this class and the OptionEU class.

4/BOOK CLASS

Similar to the Option 1st Generation class, the Book class is defined as a combination of OptionEU or Option 1st Generation objects. For now, the Book class considers a single unique underlying asset.

5/BOOKING REQUEST

By creating an instance of a Booking Request that can take either an Option or Asset object as a parameter, you can update a booking Excel file used to manage option book positions.

![image](https://github.com/user-attachments/assets/409b6bb2-b3db-43cb-9c70-53fbab07939d)

From this, the system can compute the Mark-to-Market (MtM) value of the position and assess the risk exposure of the book.

![image](https://github.com/user-attachments/assets/f10715be-78a7-450b-a6b1-8fdcbdddbb9b)

Figure: Inheritance relationship between classes.

![graphviz](https://github.com/user-attachments/assets/c7cde25a-0317-4ea7-84bd-7d4e8f35b87a)
