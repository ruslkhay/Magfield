#from mpl_toolkits.mplot3d import Axes3D

# Макропеременные 

'''
Шкала, зависящая от приближения. Используется в методах `show_genral_info`
    и `construct_mixture_plot`.
'''
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



# Строит 2D гистограмму. НЕ АКТУАЛЬНЫЙ
def static2D_hist(data, window_size=1000, bins=20, step=0, figure=None):
    '''
    Предназначена для анимированного вывода гистограмм.
    При анимации на вход подаются полные данные, задается размер плавующего
    окна, итерации (кадры) задаются параметром `step`
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    
    num_windows = len(data) - window_size + 1
    fig = plt.figure() if figure is None else figure

    data_val = data.loc[step:window_size+step,'BX']
    hist, bins = np.histogram(data_val, bins=bins)
    plt.stairs(hist, bins, fill=True)
    plt.vlines(bins, 0, hist.max(), colors='w')


    time_start = (data.loc[step,'Hour'], data.loc[step,'Minute'])
    time_end = (data.loc[window_size+step,'Hour'], 
                data.loc[window_size+step,'Minute'])
    year_day_end = str(f"Год {data.loc[window_size+step,'Year']},"+
                       f"день {data.loc[window_size+step,'Day']}.")
    plt.title(year_day_end + "\nВремя (час:минута)  %s:%s - " % (time_start)
             + "%s:%s." % (time_end))
    plt.show()


# Метод выдает блеклую картинку по стравнению с movable3D_hist. НЕ АКТУАЛЬНЫЙ.
def static3D_hist(data, window_size=1000, bins=20, step=1):
    """
    Функция строит статичный 3D график гистограмм.
    data : pandas.core.series.Series
        колонка таблицы pd.DataFrame
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Выделение окон данных и построение гистограмм
    bx_data = data
    num_windows = len(bx_data) - window_size + 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, num_windows, step):
        window = bx_data[i:i+window_size]
        hist, _bins = np.histogram(window, bins=bins)

        # Координаты точек привязок для столбцов.
        xpos, ypos = np.meshgrid(_bins[:-1], i)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # The ширина, глубина и выысота колонок соответсвенно.
        dx = np.ones_like(zpos)
        dy = np.ones_like(zpos)
        dz = hist.flatten()

        # Пределы цветовой гаммы и построение гистограммы на сечении (i-ом шаге)
        cmap=plt.cm.magma(plt.Normalize(-50,300)(dz))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cmap, zsort='average')

    ax.set_xlabel('Интервалы гист-ы')
    ax.set_ylabel('№ Окна')
    ax.set_zlabel('Частота гист-ы')
    ax.set_title(f" Данные: {data.name}, кол-во данных: {len(bx_data)},\
                    размер окна: {window_size},\n длина шага: {step}, \
                    кол-во интервалов {bins}.")
    plt.show()

#--------------------------------------------------------------------------------

# Строит объемный 3D график
def movable3D_hist(hist3D):
    '''
    Строит объемный график, представляющий динамику изменения гистограмм в 
    зависимости от положения окна. Сечение, перпендикулярное оси у "№ Окна" - 
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
    fig.show()
    '''
    !pip install dash
    # Вывод графика в отдельном окне (для Jupyter в теории)
    from dash import Dash, dcc, html
    app = Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    # Turn off reloader if inside Jupyter
    app.run_server(debug=True, use_reloader=False)
    '''
#--------------------------------------------------------------------------------
    
def show_genral_info(series, bins=200, add_title='', add_xaxis=None):
    '''
    Иллюстрирует ключевые характеристики данных: вид данных и гистограмму для 
    них.
    
    Параметры
    ----------
    series : pd.core.series.Series
        данные, которые будут визуализированы
    '''
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    X = series.index if add_xaxis is None else add_xaxis 
    
    fig = make_subplots(rows=2, cols=1, column_widths=[1],
                    row_heights=[0.6,0.4],
                    subplot_titles=['Данные',
                                    'Гистограмма данных'],
                    row_titles=["B, нТ", "Частота"])

    fig.add_trace(go.Scatter(x=X, y=series, name='график данных'))
    fig.add_trace(go.Histogram(x = series, bingroup=bins,
                            histnorm='probability density',
                            name='гистограмма'),
                    row = 2, col = 1)
    
    fig.update_layout(
        autosize=False,
        xaxis_tickformatstops=TIME_RELATED_XTICK,
        width=1000,
        height=800,
        title_text=f"Визуализация данных {series.name} и их гистограммы " +
                   f"с {bins} интервалами. " + add_title
    )

    fig.show()

#--------------------------------------------------------------------------------

# 2D Графики для вывода гистограммы и её приближения смесью
def construct_hist_plot(data_hist, data_plot, bins, part_i=None):
    '''
    Используется как вложенная функция в static2D_mixture 
    для совместного вывода всех смесей.
    '''
    from numpy import linspace
    from plotly.graph_objects import Scatter # Для потсроения кривых
    from plotly.express import histogram # Для построения гистограммы
    
    from essentials import mixture
    
    x = linspace(min(data_hist), max(data_hist), bins)

    fig = histogram(data_hist, 
                   nbins=bins, 
                   histnorm='probability density')

    span = range(len(data_plot)) if part_i is None else part_i
    for i in span:
        fig.add_trace(Scatter(x=x,
                                 y=mixture(data_plot[i]).prob(x),
                                 name=f"Смесь {i+1} законов"))

    return fig

def construct_hist_subplots(data_hist, data_plot, bins):
    '''
    Используется как вложенная функция в static2D_mixture
    для раздельного вывода всех смесей.
    '''
    from numpy import linspace
    from plotly.subplots import make_subplots
    from plotly.graph_objects import Histogram, Scatter
    
    from essentials import mixture
    
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

def static2D_mixture(mix_dicts, series, bins=200, mode='one plot'):
    '''
    Основная функция для визуализации получившихся смесей
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
    
#--------------------------------------------------------------------------------

# График для вывода весов, мат.ож-ий и ср.кв.откл-ий для смесей распределений
def construct_mixture_2Dplot(data_multicol):
    '''
    
    '''
    from plotly.graph_objects import Scatter # Для потсроения кривых   
    #from essentials import mixture
    from plotly.subplots import make_subplots
    
    # Создает соответсвующие названия для построение графиков (колонок)
    def custom_param_names(params):
        names = []
        for param in params:
            if param == 'cl_prob':
                names.append("Изменение весов компонент смеси")
            elif param == 'math_exp':
                names.append("Изменение математических ожиданий компонент смеси")
            elif param == 'st_dev':
                names.append("Изменение среднеквадратичных отклонений компонент смеси")
        return names
    
    # индексы измерений (всего их сколько и окон)
    X = data_multicol.attrs.get('custom_xaxis')
    
    lows = list(data_multicol.columns.levels[1]) # число законов в смеси
    lows.remove('')
    lows_colors = ["#84C318", "#C45AB3", "#EDD892",
                   "#C44536", "#4BB3FD", "#FC944A",
                   "#4AFC94", "#00A9A5"]
    params = list(data_multicol.columns.levels[0]) # число параметров для каждого закона
    params.remove('LL_hist')
    params_names = custom_param_names(params)
    params_names.append('Энтропия')
    params_names.append('Маргинальная log функция правдоподобия')
    
    num_rows =len(params)+1 + 1 # +1 для энтропии +1 для ф-ии лог-маргинального правдоподобия
    fig = make_subplots(rows=num_rows, cols=1,
                        subplot_titles=params_names,
                        row_titles=None,
                        vertical_spacing=0.24/num_rows)
    # Графики характеристик смесей
    for row_ind, parameter in enumerate(params):
        legend = True if row_ind==len(params)-1 else False
        for i, low in enumerate(lows):
            Y = data_multicol.loc[:,(parameter, low)].values
            fig.add_trace(Scatter(x=X, 
                                  y=Y,
                                  name=f"Закон №{i+1}",
                                  legendgroup=f"Закон №{i+1}",
                                  showlegend=legend,
                                  mode='lines', line=dict(
                                      color=lows_colors[i]),
                          hoverlabel=dict(font_color='blue')),
                          row = row_ind + 1,
                          col = 1,
                          )
    # График энтропии
    from essentials import entrophy
    entr = entrophy(data_multicol)
    fig.add_trace(Scatter(x=X, 
                          y=entr,
                          name=f"Энтропия",
                          showlegend=legend,
                          mode='lines', line=dict(
                              color=lows_colors[-1])),
                  row = num_rows - 1,
                  col = 1,
                  )
    
    # График функции правдоподобия
    Y = data_multicol.loc[:, 'LL_hist'].values
    fig.add_trace(Scatter(x=X, 
                          y=Y,
                          name=f"Фун-ия правдоподобия",
                          showlegend=legend,
                          mode='lines', line=dict(
                              color=lows_colors[-2])),
                  row = num_rows,
                  col = 1,
                  )
    
    # Название графика
    if data_multicol.attrs.get('num_of_iter') is not None:
        em_cond_description = f"Итераций {data_multicol.attrs.get('num_of_iter')}. "
    elif data_multicol.attrs.get('conv_prime') is not None:
        em_cond_description = f"Точность весов: {data_multicol.attrs.get('conv_prime')}. "
        
    custom_title = str(f"Смесь из {len(lows)} законов, "+
            f"{data_multicol.attrs.get('data_length')} отсчётов "+
            f"{data_multicol.attrs.get('data_name')}. "+
            f"Окно: {data_multicol.attrs.get('window_size')}. "+
            em_cond_description+
            f"Шаг: {data_multicol.attrs.get('step_size')}.")
    
    # Персонализация холста
    fig.update_layout(autosize=False,
                      xaxis_tickformatstops=TIME_RELATED_XTICK,
                      title=dict(text=custom_title,
                                 font=dict(size=22)),
                      template='plotly_dark',
                      legend = dict(font=dict(size=12,
                                              color="#000066"),
                                    bgcolor="#FFFFFF",
                                    bordercolor="#FF0000",
                                    borderwidth=2),
                      width=1000,
                      height=400*len(params)
                     )
    return fig