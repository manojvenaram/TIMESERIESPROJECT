 # A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 12-04-2024

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python .
Here's a concise explanation of the ARIMA model in points, suitable for a GitHub post:

---

### ARIMA Model: A Detailed Overview

**ARIMA (AutoRegressive Integrated Moving Average)** is a popular time series forecasting technique. It combines three components to capture various aspects of time series data.

1. **Autoregressive (AR) Component**: 
   - Uses dependency between an observation and several lagged observations.
   - **Order (p)**: Number of lagged observations used in the model.
   
2. **Integrated (I) Component**: 
   - Represents the differencing of raw observations to make the time series stationary.
   - **Order (d)**: Number of differencing required to remove the trend or make the series stationary.

3. **Moving Average (MA) Component**: 
   - Uses dependency between an observation and residual errors from a moving average model applied to lagged observations.
   - **Order (q)**: Number of lagged forecast errors used to correct the prediction.

4. **Model Notation**: 
   - Expressed as **ARIMA(p, d, q)**, where:
     - **p**: Order of the autoregressive part.
     - **d**: Degree of differencing.
     - **q**: Order of the moving average part.

5. **Stationarity**:
   - ARIMA assumes that the time series is stationary. Differencing is applied to remove trends and seasonality for stationarity.

6. **Model Selection**:
   - The values of **p**, **d**, and **q** can be selected using ACF (Autocorrelation Function), PACF (Partial Autocorrelation Function), and AIC/BIC criteria.

7. **Seasonal ARIMA (SARIMA)**:
   - For seasonal time series, an extended version of ARIMA known as **SARIMA** can be used, represented as **SARIMA(p,d,q)(P,D,Q,m)** where **m** is the number of periods in a season.

8. **Applications**:
   - ARIMA is widely used in stock price forecasting, economic data analysis, and various other time series predictions.

---
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions

### DataSet Generation:
https://github.com/manojvenaram/TEMPERATUREDATA-using-API
### PROGRAM:
#### Import the neccessary packages
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```

#### Load the dataset
```
data = pd.read_csv("/content/seattle-weather.csv")
```
#### Convert 'Date' column to datetime format
```
data['date'] = pd.to_datetime(data['date'])
```
#### Set 'Date' column as index
```
data.set_index('date', inplace=True)
```
#### Arima Model
```
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

arima_model(data, 'temp_max', order=(5,1,0))

```
## Challenges Faced with ARIMA Model
### Stationarity Requirement:

ARIMA assumes the time series is stationary. Non-stationary data requires differencing or transformations, which can be complex.
### Parameter Selection:

Choosing the right values for p, d, and q can be difficult. Wrong values can lead to poor forecasting performance.
Requires careful tuning using ACF, PACF plots, and trial-and-error with metrics like AIC/BIC.
### Seasonality:

ARIMA cannot handle seasonality directly. Seasonal ARIMA (SARIMA) is needed, which adds complexity to the model.
### Overfitting:

Too many parameters may cause overfitting, where the model fits the noise rather than the actual signal, leading to poor generalization.
### Large Data Processing:

ARIMA can struggle with large datasets, as it’s a computationally intensive model due to the autoregressive nature and differencing.
### Sensitive to Outliers:

ARIMA is sensitive to outliers, which can distort predictions and affect the overall performance.
Assumes Linear Relationships:

ARIMA captures only linear relationships. If the data has nonlinear patterns, ARIMA may not perform well.
### Lagged Effects:

Forecasting far into the future can become inaccurate, as the model relies on past values and residuals. The further out, the more errors accumulate.
### OUTPUT:
![image](https://github.com/manojvenaram/TSA_EXP9/assets/94165064/1a62e960-f6ba-4c61-a8eb-c4482f3bba11)

