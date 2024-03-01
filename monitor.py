# custom_title = f"Компонента: {hist3D.attrs.get('data_name')}, " \
#         f"кол-во данных: {hist3D.attrs.get('data_length')}, " \
#         f"размер окна: {hist3D.attrs.get('window_size')}, "\
#         f"кол-во интервалов {hist3D.attrs.get('bin_size')}, " \
#         f"длина шага: {hist3D.attrs.get('step_size')}."

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

# Colors for mixtures with number of components between 1 and 8
COLORS = [
    "#84C318", "#C45AB3", "#EDD892", "#C44536",
    "#4BB3FD", "#FC944A", "#4AFC94", "#00A9A5"
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

    # индексы измерений (всего их сколько и окон)
    X = x_ticks

    params = parameters # число параметров для каждого закона

    params_names = custom_param_names(params.keys())

    fig = make_subplots(
        rows=len(params), #num_rows, 
        cols=1,
        subplot_titles=params_names,
        row_titles=None,
        vertical_spacing=0.24/len(params) #num_rows
        )

    def insert_plot(
            fig: Figure,
            cell_coord,
            graph,
            customs = None
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
                    color=COLORS[i]
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

def visualize_3D_hist(
        hist3D: dict,
        custom_title = 'Histogram'
        ):
        '''
        Constructs a three-dimensional graph representing the dynamics of 
        changes in histograms in depending on the position of the window. 
        Section perpendicular to the y-axis "Window №." - is a histogram in the 
        corresponding window.
        '''
        import plotly.graph_objects as go

        # Выделение данных
        x = hist3D['bins']
        y = hist3D['wind_numb']
        z = hist3D['hist_freq']

        # Построение 3D поверхности
        fig = go.Figure(
            data=[
                go.Surface(
                    z=z, 
                    x=x, 
                    y=y)
                ]
            )

        # Персонализация изолиний и проекции
        custom_contours_z = dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True
        )
        fig.update_traces(contours_z = custom_contours_z)

        # Персонализация осей
        custom_scene = dict(
            xaxis = dict(
                title='Bins',
                color='grey'
                ),
            yaxis = dict(
                title='№ Wind',
                color='grey'
                ),
            zaxis = dict(
                title = 'Increments',
                color = 'grey'
                )
            )
        
        # Персонализация графика
        fig.update_layout(title=custom_title,
                        scene=custom_scene,
                        autosize=True,
                        width=1200, height=600,
                        margin=dict(l=65, r=50, b=65, t=90))
        return fig
