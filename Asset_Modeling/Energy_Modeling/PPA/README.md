# PPA Class – Renewable Energy Contract Modeling

## Purpose

The `PPA` class was developed to establish a structured system for managing Power Purchase Agreements (PPAs) and Renewable Energy Source (RES) portfolios. It helps simulate asset performance, evaluate risks, and estimate contract values under various scenarios.

---

## Attributes

Each `PPA` instance includes the following attributes:

- `id`: Unique identifier of the PPA  
- `site_name`: Name of the renewable asset  
- `start_date`: Contract start date  
- `end_date`: Contract end date  
- `capacity`: Installed capacity (MW)  
- `techno`: Technology type (e.g., solar, wind)  
- `pricing_type`: PPA pricing model (e.g., pay-as-produced, baseload)  
- `country`: Location of the asset  
- `proxy`: Time series representing estimated generation  
- `mark`: Valuation metric or reference price  
- `p50`: Expected annual generation at 50% probability (MWh)

---

## I. `buildProxy()` Method

The `buildProxy()` method is used to construct a proxy generation dataset for a renewable energy asset.

This proxy can be based on:

- **Historical production data** (cleaned and processed)
- **Synthetic generation data** (theoretical generation profiles based on weather inputs)

### Why is this important?

- The **value of a renewable energy park** depends on when it produces power relative to market prices.  
  A park generating during **high-price hours** has significantly more value than one producing during low or negative price hours.
- By analyzing a proxy (or historical) dataset, one can estimate:
  - The expected **p50 volume**
  - The **revenue potential** under different market scenarios

---

## II. Synthetic Generation Modeling

### A. Solar Generation

To model solar generation:

1. **Location**: Montluçon, France (as used in this project)  
   Source: [Global Solar Atlas](https://globalsolaratlas.info/detail?c=46.34003,2.607396,11&m=site&s=46.34003,2.607396)

   ![image](https://github.com/user-attachments/assets/7e19ce3f-4250-46fd-91cd-8afba313d7ed)


3. **Method**:
   - Download **average load factors** by month and hour for the site.
   - Merge this with monthly/hourly **solar radiation data** (weather-based).
   - Plot the scatter of radiation vs. load factor.

![image](https://github.com/user-attachments/assets/1b1ea9d5-82b8-476a-b7d1-82e969a5760e)


4. **Model**:
   - Fit a **polynomial regression** to capture the relationship between solar radiation and energy output.
   - This model is then used to simulate **historical generation** using past weather data.

![image](https://github.com/user-attachments/assets/05dac703-13a2-4eb1-9478-15c7580ca299)


### B. Wind Generation

For wind assets:

1. **Wind Resource Data**:  
   Source: [Global Wind Atlas](https://globalwindatlas.info/en/)

2. **Method**:
   - Use wind speed data as input.
   - Apply a wind power curve or generation model to convert wind speed into energy output.

![image](https://github.com/user-attachments/assets/2a158614-d506-4465-ac72-544bebe11ee7)


3. **Output**:
   - The resulting synthetic time series represents theoretical generation based on past weather conditions.

---

### C. Realism Enhancement

To simulate **real-world variability**:

- Add **normally distributed noise** around the modeled generation values.
- This mimics daily operational fluctuations and measurement uncertainties, making the synthetic data more suitable for backtesting.

---

## Summary

The `PPA` class offers a foundational tool for modeling RES contracts and generation behavior. By simulating realistic generation time series, users can:

- Estimate production risk (p50, p90)
- Assess market capture potential
- Support valuation and hedging strategies

---

## Contact

Feel free to reach out at : hugo.lambert.perso@gmail.com.
