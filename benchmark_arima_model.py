from src.arima_model import ArimaModel
from src.data_loader import OccupancyData, CaseData
from src.utils import load_config, benchmark, NoIndent
import pandas as pd


CONFIG = load_config("config/general.yaml")
DATES = load_config("config/150_dates.yaml")["dates"]
CASE_DATA = CaseData("data/age_groups.csv")
OCCUPANCY_DATA = OccupancyData("data/hospitalization.csv")
FORECAST_DAYS = CONFIG["general"]["forecast_days"]
STATE = CONFIG["general"]["state"]
BED_TYPE = CONFIG["general"]["type"]

model = ArimaModel(
    state=STATE,
    bed_type=BED_TYPE,
    age_groups=CONFIG["general"]["age_groups"],
    from_date=OCCUPANCY_DATA.min_date(),
    to_date=CONFIG["general"]["date"],
    case_data=CASE_DATA,
    occupancy_data=OCCUPANCY_DATA
)


def forecast(date, days):
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
    filename = "benchmarks/arima_model.json"
    benchmark(forecast, DATES, FORECAST_DAYS, STATE, BED_TYPE, OCCUPANCY_DATA, filename)
