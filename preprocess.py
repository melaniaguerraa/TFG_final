import pandas as pd
import numpy as np
from feature_engine.creation import CyclicalFeatures
from sklearn.preprocessing import MinMaxScaler


def preprocess_solar(df:pd.DataFrame):
  df = df.copy()
  df = df.drop(['consumption', 'thermal', 'wind', 'hydraulic', 'bioenergies'], axis=1)
  df['datetime'] = pd.to_datetime(df['datetime'])
  df.set_index('datetime', inplace=True)

  # Create time features
  df['hour'] = df.index.hour
  df['month'] = df.index.month
  df['year'] = df.index.year
  # encode region
  df['region_code'] = df['region'].astype('category').cat.codes
  # Create radiation feature
  df['total_radiation'] = df['diffuse_radiation'] + df['direct_radiation']
  df['is_day'] = (df['sunshine_duration']>0).astype(int)
  df = df.drop(['diffuse_radiation', 'direct_radiation', 'global_tilted_irradiance'], axis=1)

  df = df.fillna(0)
  return df

def divide_energy_weather(df:pd.DataFrame, energy):
    if energy not in ['solar', 'wind']:
        raise ValueError(f"{energy} should be 'solar' or 'wind'")
    df = df.copy()

    if energy == 'solar':
        weather_features = ["total_radiation", 
                            "relative_humidity_2m", 
                            "is_day", 
                            "surface_pressure", 
                            "cloud_cover", 
                            "sunshine_duration"]

    else:
        weather_features = ['wind_speed_10m', 
                            'cloud_cover', 
                            'total_radiation', 
                            'is_day',
                            'wind_speed_gusts', 
                            'surface_pressure']

    # weather_features = ['temperature_2m',
    #                     'relative_humidity_2m',
    #                     'apparent_temperature',
    #                     'precipitation',
    #                     'surface_pressure',
    #                     'cloud_cover',
    #                     'wind_speed_10m',
    #                     'wind_direction_10m',
    #                     'wind_gusts_10m',
    #                     'is_day',
    #                     'sunshine_duration',
    #                     'total_radiation']

    energy = df[energy]
    weather = df[weather_features]

    return energy, weather

def preprocess_wind(df:pd.DataFrame):
  df = df.copy()
  df = df.drop(['consumption', 'thermal', 'solar', 'hydraulic', 'bioenergies'], axis=1)
  df['datetime'] = pd.to_datetime(df['datetime'])
  df.set_index('datetime', inplace=True)
  # Create time features
  df['hour'] = df.index.hour
  df['month'] = df.index.month
  df['year'] = df.index.year
  # encode region
  df['region_code'] = df['region'].astype('category').cat.codes
  df['is_day'] = (df['sunshine_duration']>0).astype(int)
  df['wind_speed_gusts'] = df['wind_speed_10m']*df['wind_gusts_10m']
  
  # Create radiation feature
  df['total_radiation'] = df['diffuse_radiation'] + df['direct_radiation']
  df = df.drop(['diffuse_radiation', 'direct_radiation', 'global_tilted_irradiance'], axis=1)

  df = df.fillna(0)
  return df

def preprocess_both(df:pd.DataFrame):
  df = df.copy()
  df = df.drop(['consumption', 'thermal', 'hydraulic', 'bioenergies', 'global_tilted_irradiance'], axis=1)
  df['datetime'] = pd.to_datetime(df['datetime'])
  df.set_index('datetime', inplace=True)
  # Create lag features
  df['solar_power_t-24'] = df.groupby('region')['solar'].shift(24)  # previous day, same hour
  df['solar_power_t-168'] = df.groupby('region')['solar'].shift(24 * 7)  # Previous week, same hour
  # Create time features
  df['wind_power_t-24'] = df.groupby('region')['wind'].shift(24)  # previous day, same hour
  df['wind_power_t-168'] = df.groupby('region')['wind'].shift(24 * 7)  # Previous week, same hour
  # Create time features
  df['hour'] = df.index.hour
  df['month'] = df.index.month
  df['year'] = df.index.year
  # encode region
  #df['region_code'] = df['region'].astype('category').cat.codes
  df['is_day'] = (df['sunshine_duration']>0).astype(int)
  df['total_radiation'] = df['diffuse_radiation'] + df['direct_radiation']

  df = df.fillna(0)
  return df


def encode_cyclical_features(df):
    df = df.copy()

    features_to_encode = ["month", "hour"]
    max_values = {"month":12, "hour":24}

    # Create cyclical features
    cyclical_encoder = CyclicalFeatures(
        variables=features_to_encode,
        max_values=max_values,
        drop_original=False
    )

    cyclical_encoder.fit(df)
    df_encoded = cyclical_encoder.transform(df)
    return df_encoded


def split_data_val(df:pd.DataFrame, train_end='30-06-2021 23:59:59', val_end='31-12-2021 23:59:59', test_end='01-01-2023'):
    # Split the data
    train = df[df.index < train_end]
    val = df[(df.index >= train_end) & (df.index < val_end)]
    test = df[(df.index >= val_end) & (df.index < test_end)]
    print(f"Train shape: {train.shape}, train period: {train.index.min()} to {train.index.max()}")
    print(f"Validation shape: {val.shape}, validation period: {val.index.min()} to {val.index.max()}")
    print(f"Test shape: {test.shape}, test period: {test.index.min()} to {test.index.max()}")
    return train, val, test

def split_data(df, train_end='31-12-2021 23:59:59', test_end='31-12-2022 23:59:59'):
    # Split the data
    train = df[df.index < train_end]
    test = df[(df.index >= train_end) & (df.index < test_end)]
    print(f"Train shape: {train.shape}, train period: {train.index.min()} to {train.index.max()}")
    print(f"Test shape: {test.shape}, test period: {test.index.min()} to {test.index.max()}")
    return train, test
   
def scale_data_val(X_train, X_val, X_test, scaler):
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled features back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_val_scaled = pd.DataFrame(X_val_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def scale_data(X_train, X_test, scaler):
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled features back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)
    
    return X_train_scaled, X_test_scaled


def create_daily_blocks(energy, weather, input_width=24, forecast_horizon=24):
    # Create daily blocks of data: past 24h of energy, future 24h of weather. (predicted)
    X_energy, X_weather, y = [], [], []

    for i in range(input_width, len(energy)-forecast_horizon+1, forecast_horizon):
        x_i = energy.iloc[i-input_width:i].values
        w_i = weather.iloc[i:i+forecast_horizon].values
        y_i = energy.iloc[i:i+forecast_horizon].values

        X_energy.append(x_i)
        X_weather.append(w_i)
        y.append(y_i)

    return np.array(X_energy), np.array(X_weather), np.array(y)


def add_weather_error(weather_data):
    # adds 1% of error to the weather data  
    error_df = weather_data * np.random.normal(loc=0, scale=0.01, size=weather_data.shape)

    weather_with_error = weather_data + error_df

    return weather_with_error