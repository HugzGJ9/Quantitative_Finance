# Quantitative Finance Library & Option Book Management Tool

Welcome to the **Quantitative Finance Library**, a Python-based toolkit for modeling, analyzing, and managing financial derivatives. This repository is the result of in-depth studies in quantitative finance, combining practical tools with advanced academic concepts.

I have also led parallel studies on:
- **Optimization** techniques for portfolio management.
- **Pricing CSSOs** (Compound Spread and Swing Options).
- **Swing Options** for energy markets.
- **DA Power Correlations** between countries.
- **Cross-Border Optimizations** in power markets.

This library integrates these research insights into actionable tools for advanced financial and energy market analysis.

---

## 📊 Features and Highlights

- **Comprehensive Risk Analysis**: Delta, Gamma, Vega, Theta, and PnL risk metrics.
- **Option Portfolio Management**: Analyze and manage books of European options.
- **Dynamic Simulations**: Simulate the evolution of underlying assets and risk profiles.
- **Volatility Surfaces**: Model volatility skew and term structures for accurate pricing.

---

## 🚀 Demo Code

Here's a demonstration of the tool in action:

![Demo Code Example](https://github.com/user-attachments/assets/98d13de1-a7b3-488e-9a02-847eeb43d9bd)

---

## 📈 Risk Exposure Analysis

This library provides a detailed breakdown of portfolio risk. Below are sample visualizations of risk metrics:

### Delta Risk Exposure
![Delta Risk](https://github.com/user-attachments/assets/7cf6c9cc-4741-4951-be15-0e719b4263c9)

### Vega Convexity
![image](https://github.com/user-attachments/assets/02be9ce0-67f1-41ea-8093-e4032d2fda39)

### Pnl Price Exposure

![image](https://github.com/user-attachments/assets/94a13926-a622-41f1-b062-ae56411f80b7)

---

## 🌀 Volatility Surface: Smile and Skew

### Volatility Surface Example
Volatility surfaces integrate skew and term structure, OTM option trading.

![image](https://github.com/user-attachments/assets/379b7880-fb12-4469-96e9-87a55c2d5e6b)

---

## 📚 Description of Classes

### 1. **Asset Class**
- Represents the underlying asset for options.
- Enables simulations and position management for hedging strategies.

### 2. **Option Class**
- Built on the Asset class to represent financial derivatives.
- **Key methods**:
  - Risk Metrics: `DeltaRisk`, `GammaRisk`, `VegaRisk`, `ThetaRisk`
  - Pricing: `option_price_mc`, `option_price_close_formulae`
  - Visualization: `display_payoff_option`, `RiskAnalysis`

### 3. **Option 1st Generation**
- Comprises European vanilla options (e.g., spreads, straddles, strangles).
- Inherits features from the `Option` class for advanced analysis.

### 4. **Book Class**
- Combines multiple options (European or 1st Generation) into a portfolio.
- Focused on a single underlying asset for simplicity.

### 5. **Booking Request**
- Updates a booking Excel file to manage option book positions.
- Computes Mark-to-Market (MtM) values and assesses risk exposure.

**Example Visualization:**

![Booking Request Visualization](https://github.com/user-attachments/assets/d3c5abc5-c9e0-4895-9f6f-3da11c368b95)


![image](https://github.com/user-attachments/assets/b466aacd-d2ba-44cd-b97d-140bcb5f6d19)

![image](https://github.com/user-attachments/assets/b876d322-39fc-418f-8bab-6ae6fce3206f)

---

## 🎯 Motivations

This project began in **Fall 2023** and has evolved with three main goals:
1. **Academic Exploration**: Integrate advanced quantitative finance approaches.
2. **Personal Library**: Develop a Python toolkit for advanced option strategies.
3. **Portfolio Management**: Study and manage the evolution of option books over time.

The **Quantitative Finance Library** is now a robust tool for risk analysis and option portfolio management, leveraging simulations and dynamic risk profiling.

---

## 🛠️ How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo'''
2. Install dependencies:
   ```bash
   pip install -r requirements.txt'''
3. Explore the demo code and customize it for your use case.

## 📥 Contributions

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

## 📧 Contact

For questions or feedback, contact: hugo.lambert.perso@gmail.com
