import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import truncnorm
from concurrent.futures import ProcessPoolExecutor
from ..data_loader import OccupancyData, CaseData
from ..utils import mape
from timeit import default_timer as timer


class CausalModel:
    """
    Wrapper class for the causal model.

    Attributes
    ----------
    hospitalization_p : float
        Probability of hospitalization given a positive test result.
    poisson_lambda : float
        Parameter of the Poisson distribution for the delay between a positive test result and hospitalization.
    truncnorm_loc : float
        Location parameter of the truncated normal distribution for the length of stay.
    truncnorm_scale : float
        Scale parameter of the truncated normal distribution for the length of stay.
    simulation_seed : int
        Seed for the random number generator used for the simulation.
    calibration_seed : int
        Seed for the random number generator used for the calibration.
    case_data : CaseData
        Case data to use for the model.
    occupancy_data : OccupancyData
        Occupancy data to use for the model.

    Methods
    -------
    update(state=None, bed_type=None, age_groups=None, from_date=None, to_date=None, buffer=None)
        Update the model parameters state, bed_type, age_groups, from_date, to_date and/or buffer.
    simulate(worker_id=0)
        Simulate the model.
    monte_carlo(n, method='parallel')
        Runs a Monte Carlo simulation of the model.
    calibrate()
        Calibrate the model parameters to match the reference data.
    predict(n, method='parallel')
        Predict the occupancy of the next n days.
    """

    def __init__(self, hospitalization_p: float, poisson_lambda: float, truncnorm_loc: float, truncnorm_scale: float,
                 state: str, bed_type: str, age_groups: list[str], from_date, to_date, buffer: int,
                 case_data: CaseData, occupancy_data: OccupancyData, simulation_seed=0x883540db8384824af9e5a57913286d0,
                 calibration_seed=0x2d06d0f2):
        """
        Constructor for the CausalModel class.

        Parameters
        ----------
        hospitalization_p : float
            Probability of hospitalization given a positive test result.
        poisson_lambda : float
            Parameter of the Poisson distribution for the delay between a positive test result and hospitalization.
        truncnorm_loc : float
            Location parameter of the truncated normal distribution for the length of stay.
        truncnorm_scale : float
            Scale parameter of the truncated normal distribution for the length of stay.
        params : list[float]
            List of parameters for the model in the order:
            [hospitalization_p, poisson_lambda, truncnorm_loc, truncnorm_scale].
        state : str
            State to use for the model. Possible values are 'Österreich', 'Burgenland', 'Kärnten', 'Niederösterreich',
            'Oberösterreich', 'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg' and 'Wien'.
        bed_type : str
            Type of bed to use for the model. Possible values are 'ICU' and 'normal'.
        age_groups : list[str]
            Age groups to use for the model. Possible values are '<5', '5-14', '15-24', '25-34', '35-44', '45-54',
            '55-64', '65-74', '75-84', '>84'.
        from_date : str
            Start date of the model in the format 'YYYY-MM-DD'.
        to_date : str
            End date of the model in the format 'YYYY-MM-DD'.
        buffer : int
            Number of days to ignore at the beginning of the model.
        case_data : CaseData
            Case data to use for the model.
        occupancy_data : OccupancyData
            Occupancy data to use for the model.
        simulation_seed : int, optional
            Seed for the random number generator used for the simulation.
            Default: 0x883540db8384824af9e5a57913286d0
        calibration_seed : int, optional
            Seed for the random number generator used for the calibration.
            Default: 0x2d06d0f2
        """
        self.hospitalization_p = hospitalization_p
        self.poisson_lambda = poisson_lambda
        self.truncnorm_loc = truncnorm_loc
        self.truncnorm_scale = truncnorm_scale
        self.simulation_seed = simulation_seed
        self.calibration_seed = calibration_seed
        self.case_data = case_data
        self.occupancy_data = occupancy_data

        self._state = state
        self._bed_type = bed_type
        self._age_groups = age_groups
        self._from_date = from_date
        self._to_date = to_date
        self._buffer = buffer
        self.update()

    def update(self, state=None, bed_type=None, age_groups=None, from_date=None, to_date=None, buffer=None):
        """
        Update the model parameters state, bed_type, age_groups, from_date, to_date and/or buffer.

        Parameters
        ----------
        state : str, optional
            State to use for the model. Possible values are 'Österreich', 'Burgenland', 'Kärnten', 'Niederösterreich',
            'Oberösterreich', 'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg' and 'Wien'.
            Default: None
        bed_type : str, optional
            Type of bed to use for the model. Possible values are 'ICU' and 'normal'.
            Default: None
        age_groups : list[str], optional
            Age groups to use for the model. Possible values are '<5', '5-14', '15-24', '25-34', '35-44', '45-54',
            '55-64', '65-74', '75-84', '>84'.
            Default: None
        from_date : str, optional
            Start date of the model in the format 'YYYY-MM-DD'.
            Default: None
        to_date : str, optional
            End date of the model in the format 'YYYY-MM-DD'.
            Default: None
        buffer : int, optional
            Number of days to ignore at the beginning of the model.
            Default: None

        Returns
        -------
        None
        """
        if state is not None:
            self._state = state
        if bed_type is not None:
            self._bed_type = bed_type
        if age_groups is not None:
            self._age_groups = age_groups
        if from_date is not None:
            self._from_date = from_date
        if to_date is not None:
            self._to_date = to_date
        if buffer is not None:
            self._buffer = buffer
        self.occupancy = self.occupancy_data.get_array(self._from_date, self._to_date, self._state, self._bed_type)
        self.cases = self.case_data.get_array(self._from_date, self._to_date, self._state, self._age_groups,
                                              buffer=self._buffer)

    @property
    def params(self):
        return [self.hospitalization_p, self.poisson_lambda, self.truncnorm_loc, self.truncnorm_scale]

    @params.setter
    def params(self, value):
        self.hospitalization_p = value[0]
        self.poisson_lambda = value[1]
        self.truncnorm_loc = value[2]
        self.truncnorm_scale = value[3]

    def simulate(self, worker_id=0):
        """
        Simulate the model.

        Parameters
        ----------
        worker_id : int, optional
            The worker_id is used in conjunction with the simulation_seed to generate a unique seed for each worker.
            Default: 0

        Returns
        -------
        np.ndarray
            List of simulated occupancy values.
        """
        rng = np.random.default_rng([worker_id, self.simulation_seed])
        n = len(self.cases)
        occupancy = np.zeros(n, dtype=int)
        left, right = 0, np.inf
        mean, std = self.truncnorm_loc, self.truncnorm_scale
        a, b = (left - mean) / std, (right - mean) / std
        hospitalizations = list(map(lambda x: int(np.floor(x * self.hospitalization_p)), self.cases))
        delays = rng.poisson(self.params[1], size=sum(hospitalizations))
        stays = np.round(truncnorm.rvs(a, b, loc=mean, scale=std, size=sum(hospitalizations), random_state=rng))

        pos = 0
        for t in range(n):
            for i in range(hospitalizations[t]):
                begin_stay = t + int(delays[pos+i])
                end_stay = begin_stay + int(stays[pos+i]) + 1
                occupancy[begin_stay:end_stay] += 1
            pos += hospitalizations[t]

        return occupancy[self._buffer:]

    def monte_carlo(self, n, method='parallel'):
        """
        Runs the model n times and returns the mean results.

        Parameters
        ----------
        n : int
            Number of simulations to run.
        method : str, optional
            Method to use for the simulation. Possible values are 'parallel' and 'sequential'.
            Default: 'parallel'

        Returns
        -------
        np.ndarray
            List of mean occupancy values.
        """
        if method == 'parallel':
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.simulate, worker_id)
                    for worker_id in range(n)
                ]
                result: np.ndarray = np.around(np.mean(np.array([future.result() for future in futures]), axis=0))
        elif method == 'sequential':
            simulations = [self.simulate(worker_id=i) for i in range(n)]
            result: np.ndarray = np.around(np.mean(np.array(simulations), axis=0))
        else:
            raise ValueError(f"Invalid method: {method}")
        return result.astype(int)

    def _get_mape(self, params, n, method='parallel'):
        """
        Internal function to calculate the mean absolute percentage error. Used for calibration.
        """
        self.params = params
        simulation = self.monte_carlo(n, method)
        return mape(self.occupancy, simulation)

    def calibrate(self, n, bounds, maxiter, use_x0=False, disp=False):
        """
        Calibrate the model parameters.

        Parameters
        ----------
        n : int
            Number of simulations to run in the monte carlo simulation.
        bounds : list[tuple[float, float]]
            List of tuples containing the lower and upper bounds of the parameters.
        maxiter : int
            Maximum number of iterations.
        use_x0 : bool, optional
            Use the current parameter values as initial value.
            Default: False
        disp : bool, optional
            Print the convergence message.
            Default: False

        Returns
        -------
        scipy.optimize.OptimizeResult
            The optimization result represented as a OptimizeResult object.
        """
        if use_x0:
            x0 = self.params
        else:
            x0 = None
        start = timer()
        result = differential_evolution(self._get_mape, args=(n, 'sequential'), bounds=bounds, updating='deferred',
                                        workers=-1, disp=disp, init='sobol', strategy='randtobest1bin',
                                        maxiter=maxiter, x0=x0, seed=self.calibration_seed)
        end = timer()
        if disp:
            print(f'Calibration took {end - start:.2f} seconds with final error {result.fun}')
        self.params = result.x
        return result

    def predict(self, days, n, method='parallel'):
        """
        Predict the occupancy of the next n days.

        Parameters
        ----------
        days : int
            Number of days to predict.
        n : int
            Number of simulations to run in the monte carlo simulation.
        method : str, optional
            Method to use for the monte carlo simulation. Possible values are 'parallel' and 'sequential'.
            Default: 'parallel'

        Returns
        -------
        np.ndarray
            List of predicted occupancy values.
        """
        to_date = pd.to_datetime(self._to_date)
        self.update(to_date=to_date + pd.Timedelta(days=days))
        simulation = self.monte_carlo(n, method)
        self.update(to_date=to_date)
        return simulation[-days:]
