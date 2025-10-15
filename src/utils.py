import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def wape(y_true, y_pred):
    return np.sum(np.abs(np.array(y_true) - np.array(y_pred))) / (np.sum(np.abs(np.array(y_true))) + 1e-8)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(np.array(y_pred) - np.array(y_true)) / (np.abs(np.array(y_true)) + np.abs(np.array(y_pred)) + 1e-8))

# saving helpers (existing)
def save_forecast_csv(df_forecast, path):
    df_forecast.to_csv(path, index=False)

def save_metrics_csv(metrics_dict, path):
    if isinstance(metrics_dict, list):
        pd.DataFrame(metrics_dict).to_csv(path, index=False)
    else:
        pd.DataFrame([metrics_dict]).to_csv(path, index=False)

def ensure_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)

# plotting helper (existing)
def make_plots(df_hourly, yhat_df, naive_df, out_dir):
    last3_start = df_hourly.index.max() - pd.Timedelta(days=3) + pd.Timedelta(hours=1)
    actual = df_hourly.loc[last3_start:df_hourly.index.max()]['hourly_kwh']
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(yhat_df['timestamp'], yhat_df['yhat'], label='Ridge forecast')
    plt.plot(naive_df['timestamp'], naive_df['yhat'], label='Seasonal naive', alpha=0.7)
    plt.legend(); plt.title('Last 3 days actuals and final 24-hour forecast')
    plt.ylabel('kWh'); plt.xlabel('timestamp')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plots', 'actuals_and_forecast.png'))
    plt.close()
