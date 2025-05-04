import numpy as np


def time_related_id(dataframe) -> None:
    """
    Adds a new column 'ydhm_id' to the dataframe with a special format to
    support personalized output when building plots with plotly.

    Parameters
    ----------
    dataframe : pandas.core.frame.DataFrame
        The original dataframe. Should contain columns 'Year', 'Date',
        'Hour', 'Minute' for merging them into one - 'ydhm_id'.

    Returns
    -------
    None

    Notes
    -----
    The new column 'ydhm_id' is added to the dataframe and the original columns
    'Year', 'Date', 'Hour', 'Minute' are dropped.
    """
    from pandas import to_datetime

    # Функция для преобразования порядкового номера дня в году в дату
    def day_to_date(day):
        date = to_datetime(day, format="%j")
        return date.strftime("%m-%d")

    # Применение функции к столбцу 'Day' для создания нового столбца 'Date'
    dataframe["Date"] = dataframe["Day"].apply(day_to_date)

    # Функция для преобразования значения к нужному виду
    def format_value(value):
        return f"{value:02}"

    # Создание столбца 'ydhm_id'
    def make_cell(row):
        return str(
            f"{row['Year']}"
            + f"-{row['Date']}"
            + f"T{format_value(row['Hour'])}"
            + f":{format_value(row['Minute'])}"
        )

    dataframe["ydhm_id"] = dataframe.apply(make_cell, axis=1)

    # Rid of useless columns
    dataframe.drop(["Year", "Day", "Hour", "Minute", "Date"], axis=1, inplace=True)


def sep_gaps(gaps_ind: list, space=3):
    """
    Separating spaces into classes by spacing between them
    """
    acceptable_gaps = []
    bad_gaps = []
    for i in range(len(gaps_ind) - 1):
        if gaps_ind[i + 1] - gaps_ind[i] <= space:
            bad_gaps.append(gaps_ind[i + 1])
            if i == 0:
                bad_gaps.append(gaps_ind[i])
        else:
            acceptable_gaps.append(gaps_ind[i + 1])
            if i == 0:
                acceptable_gaps.append(gaps_ind[i])
    return acceptable_gaps, bad_gaps


def increm(arr):
    """
    Calculate increments of given array.

    First value is NaN by default

    Parameters
    ----------
    arr : array_like
        Input data.

    Returns
    -------
    new_ar : array_like
        Increments of given array.
    """
    new_ar = [None]
    for i in range(1, len(arr)):
        inc = arr[i] - arr[i - 1]
        new_ar.append(inc)
    return new_ar


def fill_gaps(data, fill_by=2):
    """
    Fill gaps in given data by mean of nearest neighbors.

    Parameters
    ----------
    data : array_like
        Input data.
    fill_by : int, optional
        Number of neighbors to be taken from both sides for mean calculation.
        The default is 2.

    Returns
    -------
    filled_data : array_like
        Data with filled gaps.

    Notes
    -----
    Gaps are supposed to be NaNs.
    """
    from pandas import isna
    from numpy import isnan, mean

    filled_data = data.copy()
    for i, val in enumerate(data):
        if isna(val):
            if i - fill_by < 0:
                vicinity = data[0 : i + 2 * fill_by]
            elif i + fill_by > len(data):
                vicinity = data[i - 2 * fill_by : i]
            else:
                vicinity = data[i - fill_by : i + fill_by]
            filled_data[i] = mean(vicinity[~isnan(vicinity)])
    return filled_data


def smooth(data, N=20):
    """
    Smooth given data by replacing each value with mean of N neighboring values

    Parameters
    ----------
    data : array_like
        Array of values to be smoothed
    N : int, optional
        Number of neighboring values to be used for mean calculation. Default is 20.

    Returns
    -------
    smoothed : array_like
        Smoothed array of values
    """
    return np.convolve(data, np.ones(N) / N, mode="valid")
