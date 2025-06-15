from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt


def train_prophet(region_df, solar_or_wind):
    if solar_or_wind not in ['solar', 'wind']:
        raise ValueError("solar_or_wind must be either 'solar' or 'wind'")
    
    df = region_df.reset_index().rename(columns={'datetime': 'ds', solar_or_wind: 'y'})
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    # add regressors
    for col in df.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)

    model.fit(df)
    return model

def forecast_prophet(model, test_data, solar_or_wind, region):
    # add regressors
    test_data = test_data.reset_index().rename(columns={'datetime': 'ds', solar_or_wind: 'y'})
    future = test_data

    forecast = model.predict(future)

    plt.figure(figsize=(12,3))
    model.plot(forecast)
    plt.title(f'{region} {solar_or_wind.capitalize()} Prophet Forecast')
    #model.plot_components(forecast)

    return forecast[['ds', 'yhat']].set_index('ds')

