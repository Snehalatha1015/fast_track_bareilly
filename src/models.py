# src/models.py
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def seasonal_naive_forecast(df_hourly, forecast_origin):
    preds = []
    for h in range(1, 25):
        src_time = forecast_origin - pd.Timedelta(hours=24) + pd.Timedelta(hours=h)
        # safe access: if missing, use nearest
        if src_time in df_hourly.index and pd.notna(df_hourly.loc[src_time, 'hourly_kwh']):
            preds.append(df_hourly.loc[src_time, 'hourly_kwh'])
        else:
            # fallback: mean of last available day same hour (simple fallback)
            alt = df_hourly['hourly_kwh'].groupby(df_hourly.index.hour).mean()
            preds.append(float(alt.get(src_time.hour, df_hourly['hourly_kwh'].mean())))
    idx = pd.date_range(start=forecast_origin + pd.Timedelta(hours=1), periods=24, freq='h')
    return pd.DataFrame({'timestamp': idx, 'yhat': preds})

def make_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # lags (demand)
    for lag in [1,2,3,24]:
        df[f'lag_{lag}'] = df['hourly_kwh'].shift(lag)

    # rolling features
    df['roll24'] = df['hourly_kwh'].rolling(window=24, min_periods=1).mean().shift(1)

    # include weather if available
    if 'temp_2m' in df.columns:
        df['temp'] = df['temp_2m']
        # simple temp lag capture
        df['temp_lag1'] = df['temp'].shift(1)
    if 'rh_2m' in df.columns:
        df['rh'] = df['rh_2m']
        df['rh_lag1'] = df['rh'].shift(1)

    # drop rows with NaN in essential features
    feature_cols = [c for c in df.columns if c not in ['hourly_kwh']]
    df = df.dropna(subset=['lag_1', 'lag_2', 'lag_3'])
    return df

def train_ridge_forecast(df_hourly, forecast_origin, with_weather=False):
    df_feats = make_features(df_hourly)
    train = df_feats[df_feats.index <= forecast_origin]

    # choose features
    base_feats = ['hour_sin','hour_cos','lag_1','lag_2','lag_3','roll24']
    # add dow dummies
    X = train[base_feats + [c for c in train.columns if c.startswith('temp') or c.startswith('rh')]].copy()
    X = pd.concat([X, pd.get_dummies(train['dow'], prefix='dow', drop_first=True)], axis=1)
    y = train['hourly_kwh']

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # forecast horizon features
    idx = pd.date_range(start=forecast_origin + pd.Timedelta(hours=1), periods=24, freq='h')
    fut = pd.DataFrame(index=idx)
    fut['hour'] = fut.index.hour
    fut['hour_sin'] = np.sin(2 * np.pi * fut['hour'] / 24)
    fut['hour_cos'] = np.cos(2 * np.pi * fut['hour'] / 24)
    fut['dow'] = fut.index.dayofweek

    # lags for future: use last known values from df_hourly
    last = df_hourly.copy()
    for lag in [1,2,3]:
        fut[f'lag_{lag}'] = last['hourly_kwh'].iloc[-lag]
    fut['roll24'] = last['hourly_kwh'].rolling(window=24).mean().iloc[-1]

    # weather for future: use actual forecasted weather if present in df_hourly (we merged T+1)
    if 'temp_2m' in df_hourly.columns:
        fut['temp'] = df_hourly['temp_2m'].reindex(fut.index)
        fut['temp_lag1'] = fut['temp'].shift(1).fillna(df_hourly['temp_2m'].iloc[-1])
    if 'rh_2m' in df_hourly.columns:
        fut['rh'] = df_hourly['rh_2m'].reindex(fut.index)
        fut['rh_lag1'] = fut['rh'].shift(1).fillna(df_hourly['rh_2m'].iloc[-1])

    Xf = fut[base_feats + [c for c in fut.columns if c.startswith('temp') or c.startswith('rh')]].copy()
    Xf = pd.concat([Xf, pd.get_dummies(fut['dow'], prefix='dow', drop_first=True)], axis=1)

    # align columns with training X
    for c in set(X.columns) - set(Xf.columns):
        Xf[c] = 0
    Xf = Xf[X.columns]

    yhat = model.predict(Xf)
    out = pd.DataFrame({'timestamp': idx, 'yhat': yhat})
    return out, model
