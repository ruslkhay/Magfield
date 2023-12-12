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
#-------------------------------------------------------------------------------

def mixture_predictions(mix_param_indexed, xaxis):
    # Используется при визуализации смесей в histplot.py
    import tensorflow as tf
    import tensorflow_probability as tfp
    from numpy import array

    class_probs = tf.constant(mix_param_indexed['cl_prob'])
    mus = tf.constant(mix_param_indexed['math_exp'])
    sigmas = tf.constant(mix_param_indexed['st_dev'])

    cluster_distribution = tfp.distributions.Categorical(probs=class_probs,
                                                        name="cluster")
    factor_distribution = tfp.distributions.Normal(loc=mus ,scale=sigmas,
                                                name="factors")
    
    normal_mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution = cluster_distribution,
        components_distribution = factor_distribution)
    
    return array(list(map(normal_mixture.prob, xaxis)))

def construct_mixture3D(mixture, series, window_size=1000, bins=20, step=1):
    """
    Функция формирует данные о смеси для 3D визуализации.

    Параметры
    ----------
    mixture : pandas.core.frame.DataFrame
        Таблица со значениями парамаетров компонент смеси. Каждая строка
        содержит параметры, полученные ЕМ-алгоритмом на конкретном окне
    window_size : int
        Длина поднабора (окна) data, который будет использоваться при анализе.
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
    from numpy import meshgrid
    import numpy as np
    from tqdm.notebook import tqdm

    num_windows = len(series) - window_size
    
    df = DataFrame({'bins':[], 'step':[], 'density':[]})
    df.attrs = {"series_name": series.name,
                "data_length": len(mixture),
                "window_size": window_size,
                "step_size": step,
                "comp_num": (mixture.shape[1]-1)//3}
    
    for i in tqdm(range(0, num_windows, step)):
        window = series[i:i+window_size]
        _bins = np.histogram_bin_edges(window, bins=bins)
        
        dens = mixture_predictions(mixture.loc[[i//step]], _bins[:-1])
        xpos, ypos = meshgrid(_bins[:-1], i)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        dz = dens.flatten()
        df.loc[len(df.index)] = [xpos, ypos, dz]
    return df
import silence_tensorflow.auto
#-------------------------------------------------------------------------------

def log_likelihood(class_probs, mus, sigmas, data): 
    # Calculate the marginal log likelihood
    import numpy as np
    from scipy.special import logsumexp
    import tensorflow_probability as tfp

    result = np.sum(
        logsumexp(
            np.log(class_probs)
            +
            tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
                data.reshape(-1, 1)
            ).numpy()
            ,
            axis=1
        )
        ,
        axis=0
    )
    return result

def EM_iterative(
    dataset,
    n_iterations,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    import numpy as np
    from tqdm import tqdm
    import tensorflow_probability as tfp

    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    for _ in tqdm(range(n_iterations)):
        # E-Step
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1,
                                            keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        class_probs = class_responsibilities / n_samples
        mus = np.sum(responsibilities * dataset.reshape(-1, 1),
                      axis=0) / class_responsibilities
        sigmas = np.sqrt(
            np.sum(responsibilities * (dataset.reshape(-1, 1) - mus.reshape(1, -1))**2, axis=0)
            / class_responsibilities
        )
    # Возвращаю значение ф-ии правд-ия для лучего результата
    log_lh = log_likelihood(class_probs, mus, sigmas, dataset)
    return class_probs, mus, sigmas, log_lh

def EM_adap(
    dataset,
    convergence_accuracy,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    import numpy as np
    import tensorflow_probability as tfp
    
    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]
    epsilon = convergence_accuracy

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    prev_class_probs = np.zeros(n_classes) # intial valurs for condition
    converged = False

    while not converged:
        # E-Step
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        # Если попалась неудачная генерация
        if np.any(np.isnan(responsibilities)): #or np.any(responsibilities == 0)
            class_probs = np.random.dirichlet(np.ones(n_classes))
            mus = np.random.rand(n_classes)
            sigmas = np.random.rand(n_classes)
            continue
        # break

        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, 
                                           keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        class_probs = class_responsibilities / n_samples
        mus = np.sum(responsibilities * 
                     dataset.reshape(-1, 1), axis=0) / class_responsibilities
        sigmas = np.sqrt(
            np.sum(responsibilities * (dataset.reshape(-1, 1) - 
                                       mus.reshape(1, -1))**2, axis=0)
            / class_responsibilities
        )

        # if(np.any(np.abs(class_probs-class_probs_initial) > 0.3 ) or
        #    np.any(np.abs(mus-mus_initial) > 0.3 ) or
        #    np.any(np.abs(sigmas-sigmas_initial) > 0.3 )):
        #     class_probs = np.random.dirichlet(np.ones(n_classes))
        #     mus = np.random.rand(n_classes)
        #     sigmas = np.random.rand(n_classes)
        #     continue

        if(np.all(class_probs - prev_class_probs <= epsilon)):
            break

        prev_class_probs = class_probs.copy()

    # Возвращаю значение ф-ии правд-ия для лучего результата
    log_lh = log_likelihood(class_probs, mus, sigmas, dataset)

    if(np.any(np.abs(class_probs-class_probs_initial) > 0.2 )):
        #Сортируем в правильном порядке
        sorted_indices = sorted(range(len(class_probs_initial)),
                                key=lambda k: class_probs_initial[k])
        class_probs = np.array([class_probs[i] for i in sorted_indices])
        mus = np.array([mus[i] for i in sorted_indices])
        sigmas = np.array([sigmas[i] for i in sorted_indices])

    return class_probs, mus, sigmas, log_lh

def EM_sieved(
    dataset,
    n_classes: int,
    n_iter_initial: int,
    convergence_accuracy_prime: float,
    n_candidates=100, # кол-во наборов смесей
    n_chosen_ones=1, # кол-во (лучших) наборов, которые мы хотим получить в результате
    random_seed=42,
    prog_bar=False,
    prev_params=None
):
    import numpy as np
    from tqdm.notebook import tqdm
    
    # (1) Генерирование первичных наборов параметров смесей

    Mus = []
    Sigmas = []
    Class_probs = []
    LL_histories = []
    
    # Возвращает шкалу прогресса либо область итерирования
    def pbar(span, title): 
        return tqdm(span).set_description(title) if prog_bar else span

    # Просеивание кандитатов    
    for candidate_id in pbar(range(n_candidates), "Генерация параметров"):
        
        # Задает новое состояние случайного генератора при смене кандидата
        np.random.seed(random_seed + candidate_id)
        
        # Инициализация случайным образом начальных значений
        mus = np.random.rand(n_classes)
        sigmas = np.random.rand(n_classes)
        class_probs = np.random.dirichlet(np.ones(n_classes))
        
        # Рассчитываем параметры ЕМ-алгоритмом
        class_probs, mus, sigmas, ll = EM_iterative(
            dataset,
            n_iter_initial,
            class_probs,
            mus,
            sigmas,
        )
        
        # Сохраняем параметры
        Mus.append(mus)
        Sigmas.append(sigmas)
        Class_probs.append(class_probs)
        LL_histories.append(ll)

    # Добавляем предыдущие парам-ы, если нужно
    if prev_params is not None:
        Mus.append(prev_params[0])
        Sigmas.append(prev_params[1])
        Class_probs.append(prev_params[2])
        LL_histories.append(prev_params[3])

    # (2) Отбор результатов.
    log_likelihood_history_array = np.array(LL_histories)
    
    # Выбор лучших параметров наборов и отсеивание лишних результатов
    ordered_candidate_ids = np.argsort( - log_likelihood_history_array)
    chosen_ones_ids = ordered_candidate_ids[:n_chosen_ones]

    # (3) Запуск ЕМ-алгоритма для лучших параметров смесей
    Mus_chosen = []
    Sigmas_chosen = []
    Class_probs_chosen = []
    LL_histories_chosen = []

    for chosen_one_id in pbar(chosen_ones_ids+[-1], "ЕМ для лучших парам-ов"):
        class_probs, mus, sigmas, log_likelihood_history = EM_adap(
            dataset,
            convergence_accuracy_prime,
            Class_probs[chosen_one_id],
            Mus[chosen_one_id],
            Sigmas[chosen_one_id],
        )

        Mus_chosen.append(mus)
        Sigmas_chosen.append(sigmas)
        Class_probs_chosen.append(class_probs)
        LL_histories_chosen.append(log_likelihood_history)
    
    # (4) Выбор лучшего кандидата
    
    # Подгоняю под один размер. Сохраняю историю лучших. Нахожу номер наилучшего

    log_likelihood_history_chosen_ones_array = np.array(LL_histories_chosen)
    # log_likelihood_history_chosen_ones_array = np.array(log_likelihood_history_chosen_ones_list)

    ordered_chosen_ones_ids = np.argsort( - log_likelihood_history_chosen_ones_array)
    
    # Выделяю лучщие параметры
    best_chosen_one_id = ordered_chosen_ones_ids[0]
    best_mus = Mus_chosen[best_chosen_one_id]
    best_sigmas = Sigmas_chosen[best_chosen_one_id]
    best_class_probs = Class_probs_chosen[best_chosen_one_id]

    # Сортировка первого списка и получение индексов перестановки
    # sorted_indices = sorted(range(len(best_class_probs)),
    #                          key=lambda k: best_class_probs[k])
    # best_class_probs = np.array([best_class_probs[i] for i in sorted_indices])
    # best_mus = np.array([best_mus[i] for i in sorted_indices])
    # best_sigmas = np.array([best_sigmas[i] for i in sorted_indices])


    return best_class_probs, best_mus, best_sigmas, log_likelihood_history_chosen_ones_array
#-------------------------------------------------------------------------------

def mixture_extraction(series, 
                       window_size=1000, 
                       step=10, 
                       n_iter=None, 
                       n_iter_initial=100, 
                       conv_prime=0.0001,
                       n_class=6,
                       n_candidates = 11,
                       n_choose=1,
                       rseed=42, 
                       EM_prog_bar=False, 
                       xaxis_id=None):
    
    from pandas import MultiIndex, DataFrame
    from tqdm.notebook import tqdm # шкала прогресса для ноутбука
    
    num_windows = len(series) - window_size + 1
    
    # Параметры для ЕМ-алгоритма
    n_classes  = n_class
    n_iter_initial = n_iter_initial
    n_candidates = n_candidates
    convergence_accuracy_prime = conv_prime
    n_chosen_ones = n_choose # кол-во (лучших) наборов, которые мы хотим получить в результате
    random_seed = rseed   
    
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
            "conv_prime": conv_prime,
            "num_of_lows": n_class,
            "custom_xaxis": []}



    # Initial
    intial_param = EM_sieved(series[:window_size].values,
                                n_classes,
                                n_iter_initial,
                                convergence_accuracy_prime,
                                n_candidates,
                                n_chosen_ones,       
                                random_seed,
                                prog_bar=EM_prog_bar)
    
    class_probs = intial_param[0]
    mus = intial_param[1]
    sigmas = intial_param[2]
    log_lh = intial_param[3]

    # Перемещение окна
    progress_bar = tqdm(range(step, num_windows, step))
    
    for i in progress_bar:
        progress_bar.set_description(f"Обработка в окнах")
        window = series[i:i+window_size]
        class_probs, mus, sigmas, log_lh = EM_adap(window.values,
                                                conv_prime,
                                                class_probs,
                                                mus,
                                                sigmas)

        ind = len(param_df.index) # Индекс строки в формируемой таблице
        if xaxis_id is not None: # Добавляет временные привзяки
            rel_i = series.index[i] # Индекс, привязанный к исходным данным
            (param_df.attrs.get('custom_xaxis')).append(xaxis_id[rel_i])
        param_df.loc[ind, 'math_exp'] = mus
        param_df.loc[ind, 'st_dev'] = sigmas
        param_df.loc[ind, 'cl_prob'] = class_probs
        param_df.loc[ind, 'LL_hist'] = log_lh # best performence
        # prev_params = (mus, sigmas, class_probs, log_lh) 

    return param_df
#-------------------------------------------------------------------------------

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

