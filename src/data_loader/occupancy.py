import pandas as pd


class OccupancyData:
    """
    Class for the occupancy data from https://covid19-dashboard.ages.at/data/Hospitalisierung.csv

    Methods
    -------
    get_df(from_date, to_date, state, type)
        Get the occupancy data as a pandas DataFrame.
    get_array(from_date, to_date, state, type)
        Get the occupancy data as a numpy array.
    min_date()
        Get the minimum date of the occupancy data.
    max_date()
        Get the maximum date of the occupancy data.
    states()
        Get the states in the occupancy data.
    """

    def __init__(self, path):
        """
        Constructor for OccupancyData.

        Parameters
        ----------
        path : str
            Path to the raw occupancy data.

        Returns
        -------
        None
        """
        self.data = pd.read_csv(path, sep=";", dayfirst=True, parse_dates=["Meldedatum"])

    def get_df(self, from_date, to_date, state, type):
        """
        Get the occupancy data as a pandas DataFrame.

        Parameters
        ----------
        from_date : str
            Start date in the format YYYY-MM-DD.
        to_date : str
            End date in the format YYYY-MM-DD.
        state : str
            Which state to get the data for. Possible values are: 'Österreich', 'Burgenland', 'Kärnten',
            'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg' and 'Wien'.
        type : str
            Which type of beds to get the data for. Possible values are 'ICU' and 'normal'.

        Returns
        -------
        pandas.DataFrame
            Occupancy data as a pandas DataFrame.
        """
        data = self.data.copy()
        if type == "ICU":
            data_col = "IntensivBettenBelCovid19"
        elif type == "normal":
            data_col = "NormalBettenBelCovid19"
        else:
            raise ValueError("type must be either 'ICU' or 'normal'")
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        data = data.loc[data["Bundesland"] == state]
        data = data[["Meldedatum", data_col]]
        data.columns = ["date", "occupancy"]
        data = data.loc[(from_date <= data["date"]) & (data["date"] <= to_date)]
        return data.set_index("date", drop=True)

    def get_array(self, from_date, to_date, state, type):
        """
        Get the occupancy data as a numpy array.

        Parameters
        ----------
        from_date : str
            Start date in the format YYYY-MM-DD.
        to_date : str
            End date in the format YYYY-MM-DD.
        state : str
            Which state to get the data for. Possible values are: 'Österreich', 'Burgenland', 'Kärnten',
            'Niederösterreich', 'Oberösterreich', 'Salzburg', 'Steiermark', 'Tirol', 'Vorarlberg' and 'Wien'.
        type : str
            Which type of beds to get the data for. Possible values are 'ICU' and 'normal'.

        Returns
        -------
        numpy.ndarray
            Occupancy data as a numpy array.
        """
        return self.get_df(from_date, to_date, state, type).values.flatten()

    def min_date(self):
        """
        Returns the minimum date of the occupancy data.

        Returns
        -------
        str
            The minimum date of the occupancy data.
        """
        return self.data["Meldedatum"].min()

    def max_date(self):
        """
        Returns the maximum date of the occupancy data.

        Returns
        -------
        str
            The maximum date of the occupancy data.
        """
        return self.data["Meldedatum"].max()

    def states(self):
        """
        Returns the states in the occupancy data.

        Returns
        -------
        list[str]
            The states in the occupancy data.
        """
        return self.data["Bundesland"].unique()
