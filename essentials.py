
def construct_hist3D(data, window_size=1000, bins=20, step=1):
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

#--------------------------------------------------------------------------------

import silence_tensorflow.auto

def EM_with_iterations(dataset, n_classes, n_iterations, random_seed, prog_bar=True):
    from tqdm.notebook import tqdm # шкала прогресса для ноутбука
    import tensorflow_probability as tfp
    import numpy as np
    n_samples = dataset.shape[0]

    # Задаем начальные значения характеристик случайным образом
    np.random.seed(random_seed)
    mus = np.random.rand(n_classes)
    sigmas = np.random.rand(n_classes)
    # задает веса так, что их сумма равна 1
    class_probs = np.random.dirichlet(np.ones(n_classes))
    
    span = tqdm(range(n_iterations)) if prog_bar else range(n_iterations)
    for em_iter in span:
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas,allow_nan_stats=False).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        delete_r = responsibilities
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)
        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        for c in range(n_classes):
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:, c] * (dataset - mus[c])**2) / class_responsibilities[c]
            )
        if False: # ОТЛАДОЧНАЯ ПЕЧАТЬ
            print(f'ITERATION: {em_iter}','responsibilities1: ',np.isnan(delete_r).any()) # PRINT
            print('responsibilities2 : ',np.isnan(responsibilities).any()) # PRINT
            print('class_responsibilities : ',np.isnan(class_responsibilities).any(),'\n') # PRINT
            if em_iter==15:
                print('tfp.distributions.Normal: ',tfp.distributions.Normal(loc=mus, scale=sigmas).prob(dataset.reshape(-1, 1)).numpy(),'\n')
                print(f'ITERATION: {em_iter}','responsibilities1: ',delete_r) # PRINT            
                print('responsibilities2 : ',responsibilities) # PRINT
                print('class_responsibilities : ',class_responsibilities,'\n') # PRINT
                print('class_probs : ',class_probs,'\n') # PRINT
                print('mus : ',mus,'\n') # PRINT
                print('sigmas : ',sigmas,'\n') # PRINT
                return None
    return class_probs, mus, sigmas

#--------------------------------------------------------------------------------

def EM(dataset, n_classes, eps, random_seed, prog_bar=True):
    #from tqdm.notebook import tqdm
    import tensorflow_probability as tfp
    import numpy as np
    
    n_samples = dataset.shape[0]
    np.random.seed(random_seed)
    
    # Инициализация начальных значений характеристик случайным образом
    mus = np.random.rand(n_classes)
    sigmas = np.random.rand(n_classes)
    class_probs = np.random.dirichlet(np.ones(n_classes))

    converged = False
    iteration = 0
    prev_class_responsibilities = np.zeros((n_samples, n_classes))
    
    while not converged:
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas, allow_nan_stats=False).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)
        
        class_responsibilities = np.sum(responsibilities, axis=0)
 
        # Проверка условия остановки
        cond_nan = np.any(np.isnan(class_responsibilities)) # Если вдруг ср.кв. отлк-е = 0 у одной из комп-т
        cond_eps = np.all(np.abs((class_responsibilities - prev_class_responsibilities)/ n_samples) <= eps)
        if cond_nan or cond_eps:
            converged = True
            break

        prev_class_responsibilities = class_responsibilities.copy()
        
        # M-шаг
        for c in range(n_classes):
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:, c] * (dataset - mus[c])**2) / class_responsibilities[c]
            )

        iteration += 1
        #span.set_description(f"Iteration {iteration}")

    return class_probs, mus, sigmas

'''
НЕ ПОМНЮ, ЧТОБЫ ИСПОЛЬЗОВАЛ ЭТОТ КОД
def mixture(mix_param):
    import tensorflow as tf
    import tensorflow_probability as tfp

    pi_cluster = tf.constant(mix_param.get("class_probs"))
    mus = tf.constant(mix_param.get("mus"))
    sigmas = tf.constant(mix_param.get("sigmas"))

    cluster_distribution = tfp.distributions.Categorical(probs=pi_cluster,
                                                        name="cluster")
    factor_distribution = tfp.distributions.Normal(loc=mus ,scale=sigmas,
                                                name="factors")
    normal_mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution = cluster_distribution,
        components_distribution = factor_distribution)
    return normal_mixture
'''

#--------------------------------------------------------------------------------

def mixture_param(series, window_size=1000, step=10, n_iter=None, eps=0.1, n_class=6,
                  rseed=42, EM_prog_bar=False, xaxis_id=None):
    
    from pandas import MultiIndex, DataFrame
    from tqdm.notebook import tqdm # шкала прогресса для ноутбука
    
    num_windows = len(series) - window_size + 1
    
    # Параметры для ЕМ-алгоритма
    random_seed = rseed
#     n_iterations = n_iter # Используется, если EM_with_iterations
    epsilon=eps
    n_classes = n_class
    
    # Создаем DataFrame с мультииндексом
    sub_col = [f"low{i}" for i in range(1, n_classes+1)]
    main_col = ['math_exp', 'st_dev', 'cl_prob']
    col = MultiIndex.from_product([main_col, sub_col])
    param_df = DataFrame(columns=col,dtype=float)
    
    param_df.attrs = {"data_name": series.name,
            "data_length": len(series),
            "window_size": window_size,
            "step_size": step,
            "num_of_iter": n_iter,
            "eps": epsilon,
            "num_of_lows": n_class,
            "custom_xaxis": []}

    # Перемещение окна
    for i in tqdm(range(0, num_windows, step)):
        window = series[i:i+window_size]
        class_probs, mus, sigmas = EM(window.values,
                                      n_classes, 
                                      epsilon,
                                      random_seed,
                                      prog_bar=EM_prog_bar)
        
        ind = len(param_df.index) # Индекс строки в формируемой таблице
        if xaxis_id is not None: # Добавляет временные привзяки
            rel_i = series.index[i] # Индекс, привязанный к исходным данным
            (param_df.attrs.get('custom_xaxis')).append(xaxis_id[rel_i])
        param_df.loc[ind, 'math_exp'] = mus
        param_df.loc[ind, 'st_dev'] = sigmas
        param_df.loc[ind, 'cl_prob'] = class_probs


    return param_df

#--------------------------------------------------------------------------------

def entrophy(mixtures):
    columns = list(mixtures.columns)
    # Проверяем принадлежность ячеек к колонке с весами
    cond_cl_p = lambda col: col[0] == 'cl_prob'
    # Выписываем кол-во законов
    lows = [col[1] for col in columns if cond_cl_p(col)]
    class_prob = mixtures.loc[:,'cl_prob'].values
    
    class_weights = []
    for i in lows:
        class_weights.append(mixtures.loc[:,('cl_prob', i)].values)
        
    # Считаемэнтропию        
    from scipy.stats import entropy
    H = entropy(class_weights, axis=0)
    return H