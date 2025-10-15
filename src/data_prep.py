# src/data_prep.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Bareilly approximate coords
BAREILLY_LAT = 28.3670
BAREILLY_LON = 79.4305
TIMEZONE = 'Asia/Kolkata'

def fetch_weather(lat, lon, start_date, end_date, hourly_vars=None, timezone=TIMEZONE):
    """
    Fetch hourly weather from Open-Meteo between start_date and end_date (inclusive).
    start_date, end_date: 'YYYY-MM-DD' strings or date-like.
    hourly_vars: list like ['temperature_2m','relative_humidity_2m']
    """
    if hourly_vars is None:
        hourly_vars = ['temperature_2m', 'relative_humidity_2m']
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ",".join(hourly_vars),
        'start_date': pd.to_datetime(start_date).strftime('%Y-%m-%d'),
        'end_date': pd.to_datetime(end_date).strftime('%Y-%m-%d'),
        'timezone': timezone
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if 'hourly' not in data:
        raise RuntimeError("Open-Meteo returned no hourly data: " + str(data))
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    # rename columns to shorter names if needed
    df = df.rename(columns={
        'temperature_2m': 'temp_2m',
        'relative_humidity_2m': 'rh_2m'
    })
    return df

def prepare_hourly(city='Bareilly', history_days=7, with_weather=False):
    """
    Loads and prepares the smart meter CSV and (optionally) merges hourly weather.
    Expects data/raw_smart_meter.csv with columns:
      x_Timestamp, t_kWh, z_Avg Voltage (Volt), z_Avg Current (Amp), y_Freq (Hz), meter
    Returns hourly dataframe with column 'hourly_kwh' and (if requested) weather columns.
    """
    # 1) Load raw CSV
    df = pd.read_csv('data/smart-meter-data.csv', parse_dates=['x_Timestamp'])
    df = df.rename(columns={'x_Timestamp': 'timestamp', 't_kWh': 'kwh'})
    df = df[['timestamp', 'kwh']].copy()

    # 2) Index, sort and resample to hourly
    df = df.sort_values('timestamp').set_index('timestamp')
    hourly = df.resample('h').sum()               # use 'h' frequency (future-proof)

    # 3) Small gaps: forward fill up to 2 hours
    hourly['kwh'] = hourly['kwh'].ffill(limit=2)

    # 4) Outlier capping: 1st–99th percentiles
    if hourly['kwh'].notna().sum() > 0:
        low, high = hourly['kwh'].quantile([0.01, 0.99])
        hourly['kwh'] = hourly['kwh'].clip(lower=low, upper=high)

    # 5) Ensure continuous index for the last history_days window (and include origin)
    end = hourly.index.max()
    start = end - pd.Timedelta(days=history_days)
    idx = pd.date_range(start=start.floor('h'), end=end.ceil('h'), freq='h')
    hourly = hourly.reindex(idx)

    # 6) Forecast origin + rename to match models
    forecast_origin = hourly.index.max()
    hourly = hourly.rename(columns={'kwh': 'hourly_kwh'})

    # 7) Optionally fetch and merge weather (historical for training and forecast for future)
    if with_weather:
        # fetch weather from start date to forecast_origin + 1 day (so we have T+1..T+24)
        start_date = (start.date()).isoformat()
        end_date = (forecast_origin + pd.Timedelta(days=1)).date().isoformat()
        try:
            weather = fetch_weather(BAREILLY_LAT, BAREILLY_LON, start_date, end_date,
                                    hourly_vars=['temperature_2m', 'relative_humidity_2m'],
                                    timezone=TIMEZONE)
            # align index names & merge
            weather = weather.reindex(hourly.index.union(weather.index)).reindex(hourly.index)  # ensure aligned index
            # Merge; we want hourly index -> left join on index
            hourly = hourly.join(weather[['temp_2m', 'rh_2m']], how='left')
            # small weather gaps: use forward/backfill up to 3 hours
            hourly['temp_2m'] = hourly['temp_2m'].ffill(limit=3).bfill(limit=3)
            hourly['rh_2m'] = hourly['rh_2m'].ffill(limit=3).bfill(limit=3)
        except Exception as e:
            print("Warning: failed to fetch weather:", e)
            # keep going without weather

    print(f"Prepared hourly data from {start.date()} to {end.date()} — total {len(hourly)} hours.")
    return hourly, forecast_origin
