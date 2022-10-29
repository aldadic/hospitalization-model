from src.arima_model import ArimaModel
from src.data_loader import OccupancyData, CaseData
from src.utils import load_config, benchmark, NoIndent
import pandas as pd


INTERVALS = [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 10000]

# -----------------------------------------------------------------------------

CONFIG = load_config("config/general.yaml")
DATES = load_config("config/24_dates.yaml")["dates"]
CASE_DATA = CaseData("data/age_groups.csv")
OCCUPANCY_DATA = OccupancyData("data/hospitalization.csv")
FORECAST_DAYS = CONFIG["general"]["forecast_days"]
STATE = CONFIG["general"]["state"]
BED_TYPE = CONFIG["general"]["type"]

model = ArimaModel(
    state=STATE,
    bed_type=BED_TYPE,
    age_groups=CONFIG["general"]["age_groups"],
    from_date=CONFIG["general"]["date"],
    to_date=CONFIG["general"]["date"],
    case_data=CASE_DATA,
    occupancy_data=OCCUPANCY_DATA
)


def predict(date, days, interval):
    model.from_date = max(pd.to_datetime(date) - pd.to_timedelta(interval-1, unit="d"), OCCUPANCY_DATA.min_date())
    model.to_date = pd.to_datetime(date)
    print(f"Calibrating interval {model.from_date.strftime('%d.%m.%Y')} - {model.to_date.strftime('%d.%m.%Y')} ...")
    result = model.calibrate()
    calibration_stats = {
        "aic": result["aic"],
        "order": NoIndent(result["order"]),
        "seasonal_order": NoIndent(result["seasonal_order"])
    }
    print((f"AIC for {model.from_date.strftime('%d.%m.%Y')} - {model.to_date.strftime('%d.%m.%Y')}: {result['aic']}"))
    pred, _ = model.predict(days)
    return pred, calibration_stats


if __name__ == "__main__":
    for interval in INTERVALS:
        def forecast(date, days): return predict(date, days, interval)
        print(f"\nBenchmarking interval: {interval} days")
        filename = f"benchmarks/arima_model_{interval}day_interval.json"
        if interval in [7, 14]:  # interval of 7 and 14 days is to short for seasonality
            model.m = 1
        else:
            model.m = 7
        benchmark(forecast, DATES, FORECAST_DAYS, STATE, BED_TYPE, OCCUPANCY_DATA, filename)
