import numpy as np

def diffusion_component(mixture):
    mus = mixture['math_exp'].values
    class_probs = mixture['cl_prob'].values
    total_mu = np.sum(mus * class_probs, axis=1, keepdims=True)
    return np.sum((mus-total_mu)**2 * class_probs, axis=1, keepdims=True)

def dynamic_component(mixture):
    sigmas = mixture['st_dev'].values
    class_probs = mixture['cl_prob'].values
    return np.sum(sigmas**2 * class_probs, axis=1, keepdims=True)

def construct_mixture_2Dplot(data_multicol):
    '''
    График для вывода весов, мат.ож-ий и ср.кв.откл-ий для смесей распределений
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
    params_names.append('Валатильность')
    
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
    
    # График валатильности
    dicom = diffusion_component(data_multicol).reshape(-1)
    dycom = dynamic_component(data_multicol).reshape(-1)

    Y = data_multicol.loc[:, 'LL_hist'].values
    fig.add_trace(Scatter(x=X, 
                          y=dicom,
                          name=f"Диффузионная комп-а",
                          showlegend=legend,
                          mode='lines', line=dict(
                              color='#FF0000')),
                  row = num_rows,
                  col = 1,
                  )
    fig.add_trace(Scatter(x=X, 
                          y=dycom,
                          name=f"Динамическая комп-а",
                          showlegend=legend,
                          mode='lines', line=dict(
                              color='#FFFF22')),
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
    from histplot import TIME_RELATED_XTICK
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