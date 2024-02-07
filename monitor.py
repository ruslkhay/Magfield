
from plotly.graph_objects import Figure
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

class Monitor:
    # Basic data visualizations. 
    TIME_RELATED_TICKS= [
        dict(dtickrange=[None, 1_000], value="%H:%M:%S.%L, ms"),
        dict(dtickrange=[1_000, 60_000], value="%H:%M:%S, sec"),
        dict(dtickrange=[60_000, 3_600_000], value="%H:%M, min"),
        dict(dtickrange=[3_600_000, 86_400_000], value="%H:%M, hours"),
        dict(dtickrange=[86_400_000, 604_800_000], value="%e, %b days"),
        dict(dtickrange=[604_800_000, "M1"], value="%e. %b, weeks"),
        dict(dtickrange=["M1", "M12"], value="%b '%y, months"),
        dict(dtickrange=["M12", None], value="%Y, year")
    ]

    # Colors for mixtures with number of components between 1 and 8
    COLORS = [
        "#84C318", "#C45AB3", "#EDD892", "#C44536",
        "#4BB3FD", "#FC944A", "#4AFC94", "#00A9A5"
        ]

    @staticmethod
    def construct_hist_plot(
        orig_data, 
        data_plot, 
        probs,
        bins, 
        part_i=None
        ):
        '''
        Используется как вложенная функция в static2D_mixture 
        для совместного вывода всех смесей.
        '''
        from numpy import linspace
        from plotly.graph_objects import Scatter # Для потсроения кривых
        from plotly.express import histogram # Для построения гистограммы
        
        x_axis = linspace(min(orig_data), max(orig_data), bins)
        y_axis = probs

        fig = histogram(
            orig_data, 
            nbins=bins, 
            histnorm='Probability density'
            )

        span = range(len(data_plot)) if part_i is None else part_i

        for i in span:
            fig.add_trace(
                Scatter(
                    x=x_axis,
                    y=mixture(data_plot[i]).prob(x),
                    name=f"Смесь {i+1} законов"
                    )
                )

        return fig
    
    def construct_hist_subplots(
            data_hist, 
            data_plot, 
            bins
            ):
        '''
        Используется как вложенная функция в static2D_mixture
        для раздельного вывода всех смесей.
        '''
        from numpy import linspace
        from plotly.subplots import make_subplots
        from plotly.graph_objects import Histogram, Scatter
        
        x = linspace(min(data_hist), max(data_hist), bins)
        
        plots_names = ["Нормальное распределение (смесь из 1 закона)"]
        plots_names += ([f"Смесь из {k} законов" for k
                            in range(2, len(data_plot)+1)])

        fig = make_subplots(rows=len(data_plot), cols=1,
                            subplot_titles=plots_names,
                            row_titles=None)
        #, title='Гистограмма всего набора исходных данных BX'
        for i in range(len(data_plot)):
            fig.add_trace(Histogram(x = data_hist,
                                    bingroup=bins,
                                    histnorm='probability density',
                                    name='гистограмма'),
                            row = i+1, col = 1)
            fig.add_trace(Scatter(x=x, 
                                y=mixture(data_plot[i]).prob(x),
                                name=f"Смесь {i+1} законов"),
                        row = i+1, col = 1)

        return fig
    
    def static2D_mixture(
            mix_dicts, 
            series, 
            bins=200, 
            mode='one plot'
            ):
        '''
        Функция для визуализации получившихся смесей совместно с гистограммой данных
        '''

        if mode=='one plot':
            fig = construct_hist_plot(series, mix_dicts, bins=bins)
        
            # Персонализация холста
            fig.update_layout(template='plotly_dark',
                            legend = dict(font=dict(size=12,
                                                    color="#000066"),
                                            bgcolor="#FFFFFF",
                                            bordercolor="#FF0000",
                                            borderwidth=2)
                            )
            
        elif mode=='subplots':
            fig = construct_hist_subplots(series, mix_dicts, bins=bins)
            
            # Персонализация холста
            fig.update_layout(autosize=False,
                            template='plotly_dark',
                            legend = dict(font=dict(size=12,
                                                    color="#000066"),
                                            bgcolor="#FFFFFF",
                                            bordercolor="#FF0000",
                                            borderwidth=2),
                            width=1300,
                            height=4000
                            )
            
        # Персонализация осей
        fig.update_xaxes(mirror=True, ticks='outside', showline=True,
            linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showline=True,
            linecolor='black', gridcolor='lightgrey')
        fig.show()

    # @staticmethod
    def construct_mixture_2Dplot(
        num_comps: int,
        parameters: dict,
        x_ticks = None
        ):
        '''
        График для вывода весов, мат.ож-ий и ср.кв.откл-ий для 
        смесей распределений
        '''
        from plotly.graph_objects import Scatter, Figure # Для потсроения кривых   
        #from essentials import mixture
        from plotly.subplots import make_subplots
        
        # Создает соответсвующие названия для построение графиков (колонок)
        def custom_param_names(params):
            names = []
            for param in params:
                if param == 'probs':
                    names.append(
                        "Изменение весов компонент смеси"
                        )
                elif param == 'mus':
                    names.append(
                        "Изменение математических ожиданий компонент смеси"
                        )
                elif param == 'sigmas':
                    names.append(
                        "Изменение сред.-кв. отклонений компонент смеси"
                        )
                elif param == 'llh':
                    names.append(
                        "Маргинальная log функция правдоподобия"
                    )
                elif param == 'entr':
                    names.append(
                        "Энтропия"
                    )
            return names
        #=============
        # To Delete
        lows_colors = ["#84C318", "#C45AB3", "#EDD892",
                    "#C44536", "#4BB3FD", "#FC944A",
                    "#4AFC94", "#00A9A5"]

        #=============

        # индексы измерений (всего их сколько и окон)
        X = x_ticks
    
        lows = num_comps # число законов в смеси

        params = parameters # число параметров для каждого закона

        params_names = custom_param_names(params.keys())
        params_vals = tuple(params.values())
        # params_names.append('Энтропия')
        # params_names.append('Маргинальная log функция правдоподобия')
        # num_rows = len(params)+1 + 1 # +1 для энтропии +1 для ф-ии лог-маргинального правдоподобия
        fig = make_subplots(
            rows=len(params), #num_rows, 
            cols=1,
            subplot_titles=params_names,
            row_titles=None,
            vertical_spacing=0.24/len(params) #num_rows
            )

        def insert_plot(
                fig: Figure,
                cell_coord: (int, int),
                graph: (tuple, tuple),
                customs: dict = None
        ):
            X, Y = graph
            row_ind, col_ind = cell_coord
            fig.add_trace(
                Scatter(
                    x=X, 
                    y=Y,
                    name=f"Component №{i+1}",
                    legendgroup=f"Component №{i+1}",
                    showlegend=False, #legend,
                    mode='lines', 
                    line=dict(
                        color=__class__.COLORS[i]
                    ),
                    hoverlabel=dict(
                        font_color='blue'
                    )
                ),
                row = row_ind,
                col = col_ind,
                )
        
        # Графики характеристик смесей
        for row_index, param in enumerate(list(parameters.items())):
            
            if param[0] in ('probs', 'mus', 'sigmas'):
                
                for i in range(num_comps):
                    Y = param[1][i]
                    insert_plot(
                        fig,
                        (row_index+1, 1),
                        (X, Y)
                    )
            else:
                Y = param[1]
                insert_plot(
                    fig,
                    (row_index+1, 1),
                    (X, Y)
                )
        # for row_ind, parameter in enumerate(params):
        #     legend = True if row_ind==len(params)-1 else False
        #     for i, comp in enumerate(num_comps):
        #         # Y = data_multicol.loc[:,(parameter, comp)].values
        #         Y = params_vals[i]
        #         fig.add_trace(
        #             Scatter(
        #                 x=X, 
        #                 y=Y,
        #                 name=f"Закон №{i+1}",
        #                 legendgroup=f"Закон №{i+1}",
        #                 showlegend=legend,
        #                 mode='lines', 
        #                 line=dict(
        #                     color=Monitor.COLORS[i]
        #                 ),
        #                 hoverlabel=dict(
        #                     font_color='blue'
        #                 )
        #             ),
        #         row = row_ind + 1,
        #         col = 1,
        #         )

        # # График энтропии
        # from essentials import entrophy
        # entr = entrophy(data_multicol)
        # fig.add_trace(Scatter(x=X, 
        #                     y=entr,
        #                     name=f"Энтропия",
        #                     showlegend=legend,
        #                     mode='lines', line=dict(
        #                         color=lows_colors[-1])),
        #             row = num_rows - 1,
        #             col = 1,
        #             )
        
        # # График функции правдоподобия
        # Y = data_multicol.loc[:, 'LL_hist'].values
        # fig.add_trace(Scatter(x=X, 
        #                     y=Y,
        #                     name=f"Фун-ия правдоподобия",
        #                     showlegend=legend,
        #                     mode='lines', line=dict(
        #                         color=lows_colors[-2])),
        #             row = num_rows,
        #             col = 1,
        #             )
        
        # # Название графика
        # if data_multicol.attrs.get('num_of_iter') is not None:
        #     em_cond_description = f"Итераций {data_multicol.attrs.get('num_of_iter')}. "
        # elif data_multicol.attrs.get('conv_prime') is not None:
        #     em_cond_description = f"Точность весов: {data_multicol.attrs.get('conv_prime')}. "
            
        # custom_title = str(f"Смесь из {len(lows)} законов, "+
        #         f"{data_multicol.attrs.get('data_length')} отсчётов "+
        #         f"{data_multicol.attrs.get('data_name')}. "+
        #         f"Окно: {data_multicol.attrs.get('window_size')}. "+
        #         em_cond_description+
        #         f"Шаг: {data_multicol.attrs.get('step_size')}.")
        
        # Персонализация холста
        fig.update_layout(autosize=False,
                        xaxis_tickformatstops=TIME_RELATED_XTICK,
                        title=dict(text='custom_title',
                                    font=dict(size=22)),
                        legend = dict(font=dict(size=12,
                                                color="#000066"),
                                        bgcolor="#FFFFFF",
                                        bordercolor="#FF0000",
                                        borderwidth=2),
                        width=1000,
                        height=400*len(params)
                        )
        return fig