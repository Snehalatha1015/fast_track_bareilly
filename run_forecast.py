#!/usr/bin/env python3
# run_forecast.py
import argparse
from src.data_prep import prepare_hourly
from src.models import seasonal_naive_forecast, train_ridge_forecast
from src.utils import save_forecast_csv, save_metrics_csv, make_plots, ensure_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='Bareilly')
    parser.add_argument('--history_window', default='days:7')  # parse as needed
    parser.add_argument('--with_weather', type=str, default='false')
    parser.add_argument('--make_plots', type=str, default='true')
    parser.add_argument('--save_report', type=str, default='true')
    args = parser.parse_args()

    out_dir = "results"
    ensure_dirs(out_dir)

    # 1) load & prepare data (expects raw CSV path or Kaggle file in working dir)
    df_hourly, forecast_origin = prepare_hourly(city=args.city, history_days=7, with_weather=(args.with_weather.lower()=='true'))

    # 2) seasonal-naive baseline forecast
    naive_forecast = seasonal_naive_forecast(df_hourly, forecast_origin)

    # 3) train a ridge model and produce forecast
    yhat, model = train_ridge_forecast(df_hourly, forecast_origin, with_weather=(args.with_weather.lower()=='true'))

    # 4) compute metrics comparing yhat and naive to actuals if available for backtest
    # (utils functions return MAE / WAPE / sMAPE)

    from src.utils import mae, wape, smape

    # We can only compute metrics if we have actuals for the last 24 hours before forecast_origin
    actuals = df_hourly['hourly_kwh'].iloc[-24:]
    preds = yhat['yhat'].values

    # Since actuals and predictions may differ in length, align safely
    actuals = actuals[-len(preds):]

    metrics = {
        'model': 'ridge_regression',
        'MAE': round(mae(actuals, preds), 2),
        'WAPE': round(wape(actuals, preds)*100, 2),
        'sMAPE': round(smape(actuals, preds), 2)
    }

    print("Metrics:", metrics)
    save_forecast_csv(yhat, f"{out_dir}/forecast_T_plus_24.csv")
    save_metrics_csv(metrics, f"{out_dir}/metrics.csv")


    if args.make_plots.lower()=='true':
        make_plots(df_hourly, yhat, naive_forecast, out_dir)

    print("Done. Artifacts in", out_dir)

if __name__ == "__main__":
    main()
