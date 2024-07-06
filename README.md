 # A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 12-04-2024

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python .
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
### OUTPUT:
![image](https://github.com/manojvenaram/TSA_EXP9/assets/94165064/1a62e960-f6ba-4c61-a8eb-c4482f3bba11)

