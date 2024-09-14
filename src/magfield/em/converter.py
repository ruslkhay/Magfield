"""
This class contains all methods that are to be impied on row data.
What do i want from this class:
    1. Load data (in .csv in partucal). May be convert from and to different
    data tpe is reasonable.
    2. In future impliment interpolation methods.
    3. Convert columns to differnet representation
    4.
"""


class converter:
    dtype = {
        "Year": int,  # str на самом деле, просто с int удобнее работать
        "Day": int,  # str
        "Hour": int,  # str
        "Minute": int,  # str
        "BX": float,
        "BY": float,
        "BZ": float,
        "Vx_Velocity": float,
        "Vy_Velocity": float,
        "Vz_Velocity": float,
        "SYM/D": int,
        "SYM/H": int,
        "ASY/D": int,
        "ASY/H": int,
    }

    # !Write an exeption for case if dir arg is not a directory. Use os module

    def __init__(
        self,
        *args,
        dir: str,  # relevant directory path
    ) -> None:
        from os import getcwd
        from os.path import join
        from pandas import read_csv

        dir = join(getcwd(), dir)
        [
            self.__setattr__(
                arg.rstrip(".csv"), read_csv(join(dir, arg), dtype=self.dtype)
            )
            for arg in args
        ]
        self.dir = dir

    @staticmethod
    # Создает колонку с временным форматов, поддерживаемым html
    def time_related_id(dataframe):
        """
        Процедура добавляет к исходной таблице дополнительную колонку "ydhm_id"
        (Year Date Hour Minute id) с временным форматом, поддерживающим
        персонализированный вывод при построении графиков plotly.

        Параметры
        ----------
        dataframe : pandas.core.frame.DataFrame
            Исходная таблица. Должна содержать колонки 'Year', 'Date',
            'Hour', 'Minute', для обхединения их в одну - "ydhm_id".

        Возвращает:
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

    def change_time_format(self, *specific_attrs):
        from os.path import isfile, join

        attrs = specific_attrs or self.__dict__.keys()

        def check_isfile(attr_name):
            return isfile(join(self.dir, attr_name + ".csv"))

        dataframes_attrs = list(filter(check_isfile, attrs))
        [self.time_related_id(self.__getattribute__(df)) for df in dataframes_attrs]

    # def apply_to_all_csv_attr(self, func):
    #     attrs = specific_attrs or self.__dict__.keys()
    #     check_isfile = lambda attr_name: isfile(
    #         join(self.dir, attr_name +'.csv')
    #     )
    #     dataframes_attrs = list(filter(check_isfile, attrs))
    #     [func(
    #         self.__getattribute__(df)
    #         )
    #         for df in dataframes_attrs]

    def save(self):
        pass
