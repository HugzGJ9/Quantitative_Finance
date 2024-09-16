Quantitative_Finance is a project that began in Fall 2023, with its purpose and motivations evolving over time. Initially, the goal was to create a comprehensive project that integrated various academic approaches. However, only a few useful features from these approaches have been incorporated into the project.

The second motivation was to develop a personal Python library for quantitative finance, aimed at retrieving and visualizing metrics related to advanced options strategies.

The third objective was to study the evolution of an options book over time by simulating the underlying assets and analyzing how risks change.

The current motivation is to utilize this project as a tool for managing an options portfolio.

1/ASSET CLASS

This snippet is simulating the price movement of a stock or asset over a time interval (1 day) using a BS stochastic model and plotting the result :

![image](https://github.com/user-attachments/assets/2b0b20e2-b570-4d22-83d9-4cd6b1e02f61)

![image](https://github.com/user-attachments/assets/76b76de9-ac24-467f-8596-00e9c771fd52)

2/OPTION CLASS

The instance creation first requires the creation of an Asset object, which is then used as a parameter for the Option_eu object. Below is the list of methods defined for the class:

DeltaRisk

Delta_DF

GammaRisk

Gamma_DF

PnlRisk

RiskAnalysis

ThetaRisk

Theta_DF

VegaRisk

Vega_DF

display_payoff_option

get_payoff_option

option_price_close_formulae

option_price_mc

run_Booking

simu_asset

![image](https://github.com/user-attachments/assets/7f6b2033-1fac-4330-9eab-6d2d4ba51a42)

![image](https://github.com/user-attachments/assets/d92ea114-02d1-41bc-b50a-c53e47f917ee)

![image](https://github.com/user-attachments/assets/781efd1d-483e-41d2-ab05-6152e34dbc84)

![image](https://github.com/user-attachments/assets/c96f2fb2-fb79-4d27-a2ba-0b73421431dd)

![image](https://github.com/user-attachments/assets/5e8c945e-8641-405c-b02a-c5484dd9c8eb)

![image](https://github.com/user-attachments/assets/faad6a5c-444a-4993-975d-35a3c12c4fdf)

![image](https://github.com/user-attachments/assets/04fb714f-a99e-42fd-a135-e33332b086a3)



![graphviz](https://github.com/user-attachments/assets/c7cde25a-0317-4ea7-84bd-7d4e8f35b87a)
