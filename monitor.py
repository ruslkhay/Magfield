
# Строит объемный 3D график

def hist3D(data, window_size=1000, bins=20, step=1):

    def visualize_3D_hist(hist3D):
        '''
        Строит объемный график, представляющий динамику изменения гистограмм в 
        зависимости от положения окна. Сечение, перпендикулярное оси y "№ Окна" - 
        это гистограмма в соответсвующем окне.
        '''
        import plotly.graph_objects as go

        # Выделение данных
        x = hist3D['bins'].values
        y = hist3D['wind_numb'].values
        z = hist3D['hist_freq'].values

        # Построение 3D поверхности
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

        # Персонализация изолиний и проекции
        custom_contours_z = dict(
                                show=True,
                                usecolormap=True,
                                highlightcolor="limegreen",
                                project_z=True)
        fig.update_traces(contours_z = custom_contours_z)

        # Персонализация осей
        custom_scene = dict(
                    xaxis = dict(
                        title='Интервалы гист-ы',
                        color='grey'),
                    yaxis = dict(
                        title='№ Окна',
                        color='grey'),
                    zaxis = dict(
                        title = 'Приращения '+hist3D.attrs.get('data_name')+", нТ",
                        color = 'grey'))

        # Название графика
        custom_title = f"Компонента: {hist3D.attrs.get('data_name')}, " \
                f"кол-во данных: {hist3D.attrs.get('data_length')}, " \
                f"размер окна: {hist3D.attrs.get('window_size')}, "\
                f"кол-во интервалов {hist3D.attrs.get('bin_size')}, " \
                f"длина шага: {hist3D.attrs.get('step_size')}."
        
        # Персонализация графика
        fig.update_layout(title=custom_title,
                        scene=custom_scene,
                        autosize=True,
                        width=1200, height=600,
                        margin=dict(l=65, r=50, b=65, t=90))
        return fig
    
    def construct_hist3D(data, window_size, bins, step):
        """
        Функция составляет 3D гистограммы.

        Параметры
        ----------
        data : pandas.core.series.Series
            Колонка таблицы pandas.core.frame.DataFrame.
        window_size : int
            Длина поднабора (окна) data, который будет использоваться при анализе.
        bins : int
            Ширина ячейки гистограммы.
        step : int
            Величина смещения окна, относительно предыдущего.

        Возвращает:
        ----------
        df: pandas.core.frame.DataFrame
            Таблица с координатами точек привязок: bins, wind_numb, -
            и высотами столбцов - hist_freq.
            df.attrs: dict
                Содержит вспомогательную информацию, используемую при кастомизации.
        """
        from pandas import DataFrame
        from numpy import histogram, meshgrid
        
        num_windows = len(data) - window_size + 1
        #file_save_name = f"{data.name}-l{len(data)}-ws{window_size}-s{step}-b{bins}"
        df = DataFrame({'bins':[], 'wind_numb':[], 'hist_freq':[]})
        df.attrs = {"data_name": data.name,
                    "data_length": len(data),
                    "window_size": window_size,
                    "step_size": step,
                    "bin_size": bins}
        
        for i in range(0, num_windows, step):
            window = data[i:i+window_size]
            hist, _bins = histogram(window, bins=bins)
            xpos, ypos = meshgrid(_bins[:-1], i)
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            dz = hist.flatten()
            df.loc[len(df.index)-1] = [xpos, ypos, dz]
        return df
    
    return visualize_3D_hist(construct_hist3D(data, window_size, bins, step))
    

# Basic data visualizations. 
TIME_RELATED_XTICK= [
    dict(dtickrange=[None, 1000], value="%H:%M:%S.%L, ms"),
    dict(dtickrange=[1000, 60000], value="%H:%M:%S, sec"),
    dict(dtickrange=[60000, 3600000], value="%H:%M, min"),
    dict(dtickrange=[3600000, 86400000], value="%H:%M, hours"),
    dict(dtickrange=[86400000, 604800000], value="%e, %b days"),
    dict(dtickrange=[604800000, "M1"], value="%e. %b, weeks"),
    dict(dtickrange=["M1", "M12"], value="%b '%y, months"),
    dict(dtickrange=["M12", None], value="%Y, year")
]
def show_genral_info(series, bins=200, add_title='', add_xaxis=None):
    '''
    Иллюстрирует ключевые характеристики данных: вид данных и гистограмму для 
    них.
    
    Параметры
    ----------
    series : pd.core.series.Series
        данные, которые будут визуализированы
    '''
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    X = series.index if add_xaxis is None else add_xaxis 
    
    fig = make_subplots(
        rows=2, 
        cols=1, 
        column_widths=[1],
        row_heights=[0.6,0.4],
        subplot_titles=['Данные',
                        'Гистограмма данных'],
        row_titles=["B, нТ", "Частота"]
        )

    fig.add_trace(
        go.Scatter(
            x=X, 
            y=series, 
            name='график данных')
            )
    fig.add_trace(
        go.Histogram(
            x = series, 
            bingroup=bins,
            histnorm='probability density',
            name='гистограмма'),
            row = 2, col = 1
            )
    
    fig.update_layout(
        autosize=False,
        xaxis_tickformatstops=TIME_RELATED_XTICK,
        width=1000,
        height=800,
        title_text=f"Визуализация данных {series.name} и их гистограммы " +
                   f"с {bins} интервалами. " + add_title
    )

    fig.show()
