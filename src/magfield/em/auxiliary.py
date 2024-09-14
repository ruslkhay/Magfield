def time_related_id(dataframe) -> None:
    """
    Процедура добавляет к исходной таблице дополнительную колонку "ydhm_id"
    (Year Date Hour Minute id) с временным форматом, поддерживающим
    персонализированный вывод при построении графиков plotly.

    Параметры
    ----------
    dataframe : pandas.core.frame.DataFrame
        Исходная таблица. Должна содержать колонки 'Year', 'Date',
        'Hour', 'Minute', для обхединения их в одну - "ydhm_id".

    Возвращает: None
    ----------
    dataframe: pandas.core.frame.DataFrame
        Исходная таблица с новым столбцом "ydhm_id" и без исходных столбцов
        'Year', 'Date', 'Hour', 'Minute'
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
    """
    new_ar = [None]
    for i in range(1, len(arr)):
        inc = arr[i] - arr[i - 1]
        new_ar.append(inc)
    return new_ar


def fill_gaps(data, fill_by=2):
    """
    Filling gaps with mean value of 2*fill_by neighboring counts
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


def smooth(data, wind_size=20):
    from numpy.lib.stride_tricks import sliding_window_view
    from numpy import mean

    windows = sliding_window_view(data, wind_size)
    smoothed = []
    for wind in windows:
        smoothed.append(mean(wind))
    return smoothed
