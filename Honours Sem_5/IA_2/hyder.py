import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import os

def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

def determine_arima_orders(series):
    # Difference the series once
    diff_series = series.diff().dropna()
    
    # Calculate ACF and PACF
    acf_values = acf(diff_series)
    pacf_values = pacf(diff_series)
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(acf_values)
    ax1.set_title('Autocorrelation Function')
    ax2.plot(pacf_values)
    ax2.set_title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    
    # Suggest orders based on ACF and PACF
    p = np.where(pacf_values > 0.2)[0][-1] if len(np.where(pacf_values > 0.2)[0]) > 0 else 1
    q = np.where(acf_values > 0.2)[0][-1] if len(np.where(acf_values > 0.2)[0]) > 0 else 1
    d = 1  # We've already differenced once
    
    return p, d, q

def analyze_file(file_path):
    print(f"\nAnalyzing file: {os.path.basename(file_path)}")
    data = pd.read_csv(file_path)
    
    print("Columns in the file:")
    print(data.columns)
    
    # Identify date column
    date_col = next((col for col in data.columns if 'date' in col.lower()), None)
    
    if not date_col:
        print("Suitable date column not found. Skipping this file.")
        return
    
    # Convert date and set as index
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    
    # Count appointments per day
    series = data.groupby(data[date_col].dt.date).size()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    
    target_col = 'Appointments Count'
    
    print(f"Analyzing: {target_col}")
    
    # Test for stationarity
    print("Stationarity Test Results:")
    test_stationarity(series)
    
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(f'Time Series of {target_col}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.show()
    
    # Determine ARIMA orders
    p, d, q = determine_arima_orders(series)
    print(f"Suggested ARIMA orders: p={p}, d={d}, q={q}")
    
    # Fit the ARIMA model
    arima_model = ARIMA(series, order=(p, d, q))
    results = arima_model.fit()
    print(results.summary())
    
    # Forecast
    forecast_steps = 30  # Forecast for the next 30 days
    forecast = results.forecast(steps=forecast_steps)
    
    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Observed')
    plt.plot(pd.date_range(start=series.index[-1], periods=forecast_steps+1, freq='D')[1:],
             forecast, color='red', label='Forecast')
    plt.fill_between(pd.date_range(start=series.index[-1], periods=forecast_steps+1, freq='D')[1:],
                     forecast - 1.96 * np.std(series),
                     forecast + 1.96 * np.std(series), color='pink', alpha=0.3)
    plt.title(f'ARIMA Forecast for {target_col}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

# Directory containing the CSV files
directory = r'C:\Users\hyder\Downloads\Exp_05'

# List of CSV files
files = ['Appointment.csv', 'Billing.csv', 'Doctor.csv', 'Medical Procedure.csv', 'Patient.csv']

for file in files:
    file_path = os.path.join(directory, file)
    if os.path.exists(file_path):
        analyze_file(file_path)
    else:
        print(f"File not found: {file_path}")

print("Analysis completed.")