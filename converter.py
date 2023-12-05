def convert_asc_to_csv(input_file, output_file, mode_='w', header=False):
    from csv import writer
    from tqdm.notebook import tqdm 
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    data = [line.split() for line in (lines)] # tdqm можно использовать тут

    with open(output_file, mode=mode_, newline='') as f:
        writer = writer(f)
        if header:
            column_names = [
                'Year', 'Month', 'Day', 'Hour', 'Min', 'Sec',
                'Bx_GSE', 'By_GSE', 'Bz_GSE', 'By_GSM', 'Bz_GSM', 
                'Bt_', 'RMS_Bx_GSE', 'RMS_By_GSM', 'RMS_Bz_GSM', 'RMS_Bt', 
                'number_fts_pts', 'Rx_GSE', 'Ry_GSE', 'Rz_GSE', 'Ry_GSM',
                'Rz_GSM'
            ]
            writer.writerow(column_names)
        writer.writerows(data)





def convert_lst_to_csv(lst_file_path, csv_file_path, mode_='w', header=False):
    '''
    Конвертер файла из типа *.lst в *.csv.
    mode: 'w' - запись в файл (write), 'a' - добавление в файл (append)
    '''
    with open(lst_file_path, 'r') as lst_file:
        ''' Выполните здесь необходимую обработку данных из файла .lst
        и сохраните результаты в виде списка списков.'''
        
        row_data = lst_file.readlines()
        
        # Если нужно убрать дефектные значения из данных
        cond_9999 = lambda word: word != 99999.9 and word != 9999.99 and word != 99999
        replace_9999 = lambda word: word if cond_9999(word) else np.nan
        filt_9999 = lambda _list: list(map(replace_9999, _list))
        
        def define_types(word):
            if(word.isnumeric()): return int(word)
            elif(not word.isnumeric() and word.isalnum()): return str(word)
            else: return float(word)
        filt_types = lambda _list: list(map(define_types, _list))
        
        results = [filt_9999(filt_types(line.split())) for line in row_data]
        
    with open(csv_file_path, mode=mode_, encoding='UTF8', newline='') as csv_file:
        writer = writer(csv_file)
        if header:
            writer.writerow(list(TYPES.keys()))
        writer.writerows(results)
    
    if mode_=='w':
        print(f"Файл успешно конвертирован в формат CSV и сохранен по пути: {csv_file_path}")
    elif mode_=='a':
        print(f"Файл успешно конвертирован и дописан: {csv_file_path}")
 

"""
# Пример использования

# Указание путей к конвентируемым файлам
january_march, april_jun, july_september, october = ['omni_min_def_20230101_20230331.lst',
                                                     'omni_min_def_20230401_20230630.lst',
                                                     'omni_min_def_20230701_20230930.lst',
                                                     'omni_min_def_20231001_20231010.lst']

path_1 = PATH_DATA + january_march
path_2 = PATH_DATA + april_jun
path_3 = PATH_DATA + july_september
path_4 = PATH_DATA + october

# Указание пути для сохранения файла .csv
csv_file_path = PATH_DATA + "FULL_DATA.csv"

# Вызов функции для выполнения конвертации 
convert_lst_to_csv(path_2, csv_file_path, mode_='w', header=True)
convert_lst_to_csv(path_3, csv_file_path, mode_='a')
convert_lst_to_csv(path_4, csv_file_path, mode_='a')


# Сначала в файл "FULL_DATA" записывается конвентированная информация path_2. Затем в файл дозаписываются данные, лежащие
# находящиеся по путям path_3, path_4.
"""

from pandas import isna

def fill_nan(data_frame):
    '''
    Заменяет все значения NaN в таблице, кроме первой и последних строк,
    на среднее между значениями ближайших соседних числовых отсчетов
    '''
    for col in data_frame.columns:
        for i in range(1, len(data_frame.loc[:,col])-1):
            if isna(data_frame.loc[i, col]):
                previous_val = data_frame.loc[i-1, col]
                next_val = data_frame.loc[i+1, col]
                if isna(previous_val) and pd.isna(next_val):  # если оба значения NaN
                    data_frame.loc[i, col] = 0  # заменяем на 0 или другое значение по умолчанию
                elif isna(previous_val):  # если предыдущее значение NaN
                    data_frame.loc[i, col] = next_val
                elif isna(next_val):  # если последующее значение NaN
                    data_frame.loc[i, col] = previous_val
                else:
                    data_frame.loc[i, col] = (previous_val + next_val) / 2

# Создает колонку с временным форматов, поддерживаемым html
def time_related_id(dataframe):
    '''
    Процедура добавляет к исходной таблице дополнительную колонку "ydhm_id" 
    (Year Date Hour Minute id) с временным форматом, поддерживающим персонализированный 
    вывод при построении графиков plotly.
    
    Параметры
    ----------
    dataframe : pandas.core.frame.DataFrame
        Исходная таблица. Должна содержать колонки 'Year', 'Date', 'Hour', 'Minute',
        для обхединения их в одну - "ydhm_id".

    Возвращает:
    ----------
    dataframe: pandas.core.frame.DataFrame
        Исходная таблица с новым столбцом "ydhm_id" и без исходных столбцов 'Year', 
        'Date', 'Hour', 'Minute'
    '''
    from pandas import to_datetime
    
    # Функция для преобразования порядкового номера дня в году в дату
    def day_to_date(day):
        date = to_datetime(day, format='%j')
        return date.strftime('%m-%d')

    # Применение функции к столбцу 'Day' для создания нового столбца 'Date'
    dataframe['Date'] = dataframe['Day'].apply(day_to_date)

    # Функция для преобразования значения к нужному виду
    def format_value(value):
        return f'{value:02}'

    # Создание столбца 'ydhm_id'
    make_cell = lambda row: str(f"{row['Year']}"+
                                f"-{row['Date']}"+
                                f"T{format_value(row['Hour'])}"+
                                f":{format_value(row['Minute'])}")
    dataframe['ydhm_id'] = dataframe.apply(make_cell, axis=1)
    dataframe.drop(['Year', 'Date', 'Hour', 'Minute'], axis=1, inplace=True)
                    
