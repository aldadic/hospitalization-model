import pmdarima as pm
import pandas as pd
import numpy as np
from src.data_loader import OccupancyData, CaseData


class ArimaModel:
    """
    Wrapper class for the ARIMA model.

    Attributes
    ----------
    model : pmdarima.arima.ARIMA
        The ARIMA model.
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
    case_data : CaseData
        Case data to use for the model.
    occupancy_data : OccupancyData
        Occupancy data to use for the model.
    m : int
        The number of periods in each season.
    max_p : int
        The maximum value of p.
    max_q : int
        The maximum value of q.
    max_d : int
        The maximum value of d.
    max_P : int
        The maximum value of P.
    max_Q : int
        The maximum value of Q.
    max_D : int
        The maximum value of D.

    Methods
    -------
    calibrate(disp=False)
        Calibrates the model.
    predict(days)
        Predicts the occupancy for the next days.
    """

    def __init__(self, state: str, bed_type: str, age_groups: list[str], from_date, to_date, case_data: CaseData,
                 occupancy_data: OccupancyData, m=7, max_p=7, max_q=7, max_d=1, max_P=2, max_Q=2, max_D=1):
        """
        Constructor for the ArimaModel class.

        Parameters
        ----------
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
        case_data : CaseData
            Case data to use for the model.
        occupancy_data : OccupancyData
            Occupancy data to use for the model.
        m : int, optional
            The number of periods in each season.
            Default: 7
        max_p : int, optional
            The maximum value of p.
            Default: 7
        max_q : int, optional
            The maximum value of q.
            Default: 7
        max_d : int, optional
            The maximum value of d.
            Default: 1
        max_P : int, optional
            The maximum value of P.
            Default: 2
        max_Q : int, optional
            The maximum value of Q.
            Default: 2
        max_D : int, optional
            The maximum value of D.
            Default: 1

        Returns
        -------
        None
        """
        self.model = None
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_D = max_D
        self.state = state
        self.bed_type = bed_type
        self.age_groups = age_groups
        self.from_date = from_date
        self.to_date = to_date
        self.case_data = case_data
        self.occupancy_data = occupancy_data

    def calibrate(self, disp=False):
        """
        Calibrate the model with pmdarima.auto_arima.

        Parameters
        ----------
        disp : bool, optional
            Whether to print the calibration results.
            Default: False

        Returns
        -------
        dict
            Dictionary containing the calibration results (i.e. model parameters, AIC, etc.).
        """
        occupancy = self.occupancy_data.get_array(self.from_date, self.to_date, self.state, self.bed_type)
        cases = (self.case_data.get_df(self.from_date, self.to_date, self.state, self.age_groups)
                               .rolling(7, min_periods=1).mean().values.reshape(-1, 1))
        self.model: pm.arima.ARIMA = pm.auto_arima(y=occupancy, X=cases, m=self.m, max_p=self.max_p, max_q=self.max_q,
                                                   max_d=self.max_d, max_P=self.max_P, max_Q=self.max_Q,
                                                   max_D=self.max_D, trace=disp, error_action="ignore")
        result = self.model.get_params()
        result["aic"] = self.model.aic()
        return result

    def predict(self, days):
        """
        Predict the occupancy for the next days.

        Parameters
        ----------
        days : int
            Number of days to predict.

        Returns
        -------
        np.ndarray
            Array containing the predicted occupancy.
        """
        cases = (self.case_data.get_df(from_date=pd.to_datetime(self.to_date) + pd.to_timedelta(1, unit="d"),
                                       to_date=pd.to_datetime(self.to_date) + pd.to_timedelta(days, unit="d"),
                                       state=self.state, age_groups=self.age_groups)
                               .rolling(7, min_periods=1).mean().values.reshape(-1, 1))
        pred, conf_int = self.model.predict(n_periods=days, X=cases, return_conf_int=True)
        return np.around(pred).astype(int), conf_int
