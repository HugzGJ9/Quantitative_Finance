# Quantitative Finance Library & Option Book Management Tool

Welcome to the **Quantitative Finance Library**, a Python-based toolkit for modeling, analyzing, and managing books of european options. This repository is the result of in-depth studies in quantitative finance, combining practical tools with advanced academic concepts.

---

## 📊 Features and Highlights

- **Comprehensive Risk Analysis**: Includes PnL, Greeks, Vega convexity, skew, and term structure risk metrics.  
- **Option Portfolio Management**: Analyzing and managing books of European options.  
- **Dynamic Simulations**: Simulating the evolution of underlying assets and risk profiles.  
- **Volatility Surfaces**: Modeling volatility as a surface for active volatility trading.  
- **Learning Tool**: A valuable application for an initial assessment of risk exposure in a new, advanced trading strategy.  

---

## 🚀 Demo Code

### 1. Demo script :
- **Visualizing Trading Strategies** – This tool is perfect for users who want to visualize new trading strategies. Users can easily access the theoretical price of a book, payoff, Greek exposure (in 2D or 3D), skew, and term structure. https://youtu.be/npcQdp4R_DU?si=ZkZCIkVludQWTVGu

### 2. Booking script :
- **Options Trading and Risk Management** – For more advanced users, this tool allows traders to save their positions and conveniently access risk metrics, making it highly useful for managing option trades. https://youtu.be/Wg5Euv6VoKg

---

## 📈 Risk Exposure Analysis

This library provides a detailed breakdown of portfolio risk. Below are sample visualizations of risk metrics:

### Payoff
![image](https://github.com/user-attachments/assets/5421f593-e3c2-40ca-9bb9-244a74bb38b0)

### Delta Risk Exposure
![image](https://github.com/user-attachments/assets/92879974-00b2-41ff-8d07-8c8b9886a0f5)

### Vega Convexity
![image](https://github.com/user-attachments/assets/ba038935-4c4c-480c-945d-37a51fa52fce)

### Pnl Price Exposure

![image](https://github.com/user-attachments/assets/692ced67-f981-4b83-815a-790db06d665c)

---

## 🌀 Volatility Surface: Smile and Skew

### Volatility Surface Example
Volatility surfaces integrate skew and term structure, required for OTM option trading.

![image](https://github.com/user-attachments/assets/379b7880-fb12-4469-96e9-87a55c2d5e6b)

---

## 📚 Description of Classes

### 1. **Asset Class**
- Represents the underlying asset for options.
- Enables simulations and position management for hedging strategies.

### 2. **Option Class**
- Built on the Asset class to represent financial derivatives.
- **Key methods**:
  - Risk Metrics: `DeltaRisk`, `GammaRisk`, `VegaRisk`, `ThetaRisk`, `VannaRisk`, `VolgaRisk`
  - Pricing: `option_price_mc`, `option_price_close_formulae`
  - Visualization: `display_payoff_option`, `RiskAnalysis`, `PnlRisk`
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

## :zap: Subprojects

I have also led parallel studies on:
- **Optimizations**.
- **DA Power Correlations** between countries.
- **Cross-Border Optimizations** in power markets.
- **Swing Options** for energy markets.
- **Pricing CSSOs**.

This library integrates these research insights into actionable tools for advanced financial and energy market analysis.

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
   git clone https://github.com/Quantitative_Finance.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Explore the demo code and customize it for your use case.

## 📥 Contributions

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

## 📧 Contact

For questions or feedback, contact: hugo.lambert.perso@gmail.com
