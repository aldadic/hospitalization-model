import pandas as pd
import numpy as np
import json
from ..data_loader import OccupancyData
from .json_no_indent import NoIndent, MyEncoder
from .error_functions import mase, mape, mae


def to_list(a):
    if isinstance(a, np.ndarray):
        return NoIndent(a.tolist())
    else:
        return NoIndent(list(a))


def benchmark(forecast: callable, dates: list, forecast_days: int, state: str, bed_type: str,
              occupancy_data: OccupancyData, path="benchmark.json"):
    successful = dict()
    failed = dict()
    for date in dates:
        try:
            prediction, calibration = forecast(date, forecast_days)
            from_date = pd.to_datetime(date) + pd.to_timedelta(1, unit="d")
            to_date = from_date + pd.to_timedelta(forecast_days-1, unit="d")
            reference = occupancy_data.get_array(from_date, to_date, state, bed_type)
            successful[date] = dict()
            successful[date]["actual"] = to_list(reference)
            successful[date]["prediction"] = to_list(prediction)
            successful[date]["diff"] = to_list(reference - prediction)
            successful[date]["mase"] = mase(reference, prediction)
            successful[date]["mape"] = mape(reference, prediction)
            successful[date]["mae"] = mae(reference, prediction)
            successful[date]["calibration_stats"] = calibration
            print(f"Forecast MASE for {date}: {mase(reference, prediction)}")
        except Exception as e:
            print(f"Forecast for {date} failed: {e}")
            failed[date] = str(e)
    result = {
        "forecast_days": forecast_days,
        "state": state,
        "bed_type": bed_type,
        "successful_forecasts": successful,
        "failed_forecasts": failed,
        "dates": dates
    }
    with open(path, "w") as f:
        json.dump(result, f, cls=MyEncoder, indent=4, ensure_ascii=False)
