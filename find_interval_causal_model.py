from src.causal_model import CausalModel
from src.data_loader import OccupancyData, CaseData
from src.utils import load_config, benchmark
import pandas as pd


INTERVALS = [7, 14, 21, 28, 35, 42, 49, 56] + [3, 4, 5, 6, 8, 9, 10, 11]

# -----------------------------------------------------------------------------

CONFIG = load_config("config/general.yaml", "config/causal_model.yaml")
DATES = load_config("config/24_dates.yaml")["dates"]
CASE_DATA = CaseData("data/age_groups.csv")
OCCUPANCY_DATA = OccupancyData("data/hospitalization.csv")
N = CONFIG["causal_model"]["monte_carlo_iterations"]
MAX_ITER = CONFIG["causal_model"]["max_iter"]
BOUNDS = [
    CONFIG["causal_model"]["param_ranges"]["hospitalization_p"],
    CONFIG["causal_model"]["param_ranges"]["poisson_lambda"],
    CONFIG["causal_model"]["param_ranges"]["truncnorm_loc"],
    CONFIG["causal_model"]["param_ranges"]["truncnorm_scale"]
]
FORECAST_DAYS = CONFIG["general"]["forecast_days"]
STATE = CONFIG["general"]["state"]
BED_TYPE = CONFIG["general"]["type"]

model = CausalModel(
    hospitalization_p=CONFIG["causal_model"]["params"]["hospitalization_p"],
    poisson_lambda=CONFIG["causal_model"]["params"]["poisson_lambda"],
    truncnorm_loc=CONFIG["causal_model"]["params"]["truncnorm_loc"],
    truncnorm_scale=CONFIG["causal_model"]["params"]["truncnorm_scale"],
    state=STATE,
    bed_type=BED_TYPE,
    age_groups=CONFIG["general"]["age_groups"],
    from_date=CONFIG["general"]["date"],
    to_date=CONFIG["general"]["date"],
    buffer=CONFIG["causal_model"]["buffer"],
    case_data=CASE_DATA,
    occupancy_data=OCCUPANCY_DATA,
    simulation_seed=CONFIG["causal_model"]["simulation_seed"],
    calibration_seed=CONFIG["causal_model"]["calibration_seed"],
)


def forecast(date, days, interval):
    calibration_from = pd.to_datetime(date) - pd.to_timedelta(interval-1, unit="d")
    calibration_to = pd.to_datetime(date)
    model.update(from_date=calibration_from, to_date=calibration_to)
    print(f"Calibrating interval {calibration_from.strftime('%d.%m.%Y')} - {calibration_to.strftime('%d.%m.%Y')} ...")
    result = model.calibrate(N, BOUNDS, MAX_ITER)
    print((f"Calibration error for {calibration_from.strftime('%d.%m.%Y')} "
           f"- {calibration_to.strftime('%d.%m.%Y')}: {result.fun}"))
    pred = model.predict(days, N)
    return pred, {"mape": result.fun}


if __name__ == "__main__":
    for interval in INTERVALS:
        def forecast_fct(date, days): return forecast(date, days, interval)
        print(f"\nBenchmarking interval: {interval} days")
        filename = f"benchmarks/causal_model_{interval}day_interval.json"
        benchmark(forecast_fct, DATES, FORECAST_DAYS, STATE, BED_TYPE, OCCUPANCY_DATA, filename)
