import pandas as pd


class CaseData:
    """
    Class for the case data from https://covid19-dashboard.ages.at/data/CovidFaelle_Altersgruppe.csv

    Methods
    -------
    get_df(from_date, to_date, state, age_groups, buffer=0)
        Get the case data as a pandas DataFrame.
    get_array(from_date, to_date, state, age_groups, buffer=0)
        Get the case data as a numpy array.
    min_date()
        Get the minimum date of the case data.
    max_date()
        Get the maximum date of the case data.
    states()
        Get the states of the case data.
    age_groups()
        Get the age groups of the case data.
    """
    def __init__(self, path):
        """
        Constructor for CaseData.

        Parameters
        ----------
        path : str
            Path to the raw case data.

        Returns
        -------
        None
        """
        self.data = pd.read_csv(path, sep=";", dayfirst=True, parse_dates=["Time"])

    def get_df(self, from_date, to_date, state, age_groups: tuple, buffer=0):
        """
        Get the case data as a pandas DataFrame.

        Parameters
        ----------
        from_date : str
            Start date in the format YYYY-MM-DD.
        to_date : str
            End date in the format YYYY-MM-DD.
        state : str
            Which state to get the data for. Possible values are: 'Österreich', 'Burgenland', 'Kärnten',
            'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg' and 'Wien'.
        age_groups : list[str]
            Which age groups should be included in the data. Possible values are: '<5', '5-14', '15-24', '25-34',
            '35-44', '45-54', '55-64', '65-74', '75-84', '>84'.
        buffer : int, optional
            How many days to include before from_date.
            Default: 0

        Returns
        -------
        pandas.DataFrame
            The case data as a pandas DataFrame.
        """
        data = self.data.copy()
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        start_date = from_date - pd.to_timedelta(buffer, unit="d")
        data = data.loc[data["Bundesland"] == state]
        data = data[data["Altersgruppe"].isin(age_groups)]
        data = data.groupby(['Time']).sum().reset_index()
        data["Anzahl"] = data["Anzahl"].diff()
        data = data[["Time", "Anzahl"]]
        data.columns = ["date", "cases"]
        data = data.loc[(start_date <= data["date"]) & (data["date"] <= to_date)]
        data = data.astype({"cases": int})
        return data.set_index("date", drop=True)

    def get_array(self, from_date, to_date, state, age_groups: tuple, buffer=0):
        """
        Get the case data as a numpy array.

        Parameters
        ----------
        from_date : str
            Start date in the format YYYY-MM-DD.
        to_date : str
            End date in the format YYYY-MM-DD.
        state : str
            Which state to get the data for. Possible values are: 'Österreich', 'Burgenland', 'Kärnten',
            'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg' and 'Wien'.
        age_groups : list[str]
            Which age groups should be included in the data. Possible values are: '<5', '5-14', '15-24', '25-34',
            '35-44', '45-54', '55-64', '65-74', '75-84', '>84'.
        buffer : int, optional
            How many days to include before from_date.
            Default: 0

        Returns
        -------
        numpy.ndarray
            The case data as a numpy array.
        """
        return self.get_df(from_date, to_date, state, age_groups, buffer).values.flatten()

    def min_date(self):
        """
        Returns the minimum date of the case data.

        Returns
        -------
        str
            The minimum date of the case data.
        """
        return self.data["Time"].min()

    def max_date(self):
        """
        Returns the maximum date of the case data.

        Returns
        -------
        str
            The maximum date of the case data.
        """
        return self.data["Time"].max()

    def states(self):
        """
        Returns the states in the case data.

        Returns
        -------
        list[str]
            The states in the case data.
        """
        return self.data["Bundesland"].unique()

    def age_groups(self):
        """
        Returns the age groups in the case data.

        Returns
        -------
        list[str]
            The age groups in the case data.
        """
        return self.data["Altersgruppe"].unique()
