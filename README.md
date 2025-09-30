# Electric Vehicles Market Size Analysis 🚗⚡

This project analyzes historical Electric Vehicle (EV) registration data in the United States to understand market trends, estimate growth, and forecast future EV registrations. Insights from this analysis can help manufacturers, policymakers, and investors make informed decisions regarding production, infrastructure, and policies.

## Project Overview

Electric vehicles are becoming increasingly popular due to environmental concerns and government incentives. Monitoring and forecasting EV market size is important for strategic planning. This project focuses on:

- Analyzing historical EV registration data  
- Identifying market trends and growth patterns  
- Forecasting future EV registrations using multiple predictive models  

## Predictive Models Evaluated

| Model                  | RMSE           |
|------------------------|---------------|
| Linear Regression      | 10,030.23     |
| Polynomial Regression  | 8,483.55      |
| Random Forest          | 4,218.15      |
| **XGBoost** ✅          | 0.003015      |
| Prophet                | 42,542.49     |

**Key Insight:** XGBoost achieved the lowest RMSE, making it the most accurate model for forecasting EV registrations.

## Conclusion

The analysis indicates strong growth trends in the US EV market. Using XGBoost, future registrations can be predicted accurately, providing valuable guidance for manufacturers, policymakers, and other stakeholders in planning production, infrastructure, and strategies for market expansion.
