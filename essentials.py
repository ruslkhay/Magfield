
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

#--------------------------------------------------------------------------------
def EM_iterative(
    dataset,
    n_iterations,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    import numpy as np
    from scipy.special import logsumexp
    from tqdm import tqdm
    import tensorflow_probability as tfp

    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    log_likelihood_history = []

    for em_iter in tqdm(range(n_iterations)):
        # E-Step
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1,
                                            keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        class_probs = class_responsibilities / n_samples
        mus = np.sum(responsibilities * dataset.reshape(-1, 1), axis=0) / class_responsibilities
        sigmas = np.sqrt(
            np.sum(responsibilities * (dataset.reshape(-1, 1) - mus.reshape(1, -1))**2, axis=0)
            / class_responsibilities
        )

        # Calculate the marginal log likelihood
        log_likelihood = np.sum(
            logsumexp(
                np.log(class_probs)
                +
                tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
                    dataset.reshape(-1, 1)
                ).numpy()
                ,
                axis=1
            )
            ,
            axis=0
        )
        
        log_likelihood_history.append(log_likelihood)
    print(class_probs)
    return class_probs, mus, sigmas, log_likelihood_history

def EM_adaptive(
    dataset,
    convergence_accurcy,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    import numpy as np
    from scipy.special import logsumexp
    from tqdm import tqdm
    import tensorflow_probability as tfp
    
    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()
    log_likelihood_history = []
    # prev_class_responsibilities = np.full((n_samples, n_classes), n_samples)
    prev_class_responsibilities = np.zeros((n_samples, n_classes))
    converged = False
    i=0
    while not converged:
        
        # E-шаг
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)
        
        # Проверка условия игнорирования хотя бы одного копмонента смеси
        if np.any(np.isnan(class_responsibilities)):
            # Запускаем ЕМ завново, но с новыми значениями
            mus = np.random.rand(n_classes)
            sigmas = np.random.rand(n_classes)
            class_probs = np.random.dirichlet(np.ones(n_classes))
            continue
            
        # M-шаг
        class_probs = class_responsibilities / n_samples
        mus = np.sum(responsibilities * dataset.reshape(-1, 1), axis=0) / class_responsibilities
        sigmas = np.sqrt(
            np.sum(responsibilities * (dataset.reshape(-1, 1) - mus.reshape(1, -1))**2, axis=0)
            / class_responsibilities
        )
        
        # Упорядычиваем значения, чтобы "пронумировать" компоненты смеси
        class_probs.sort()
        mus.sort()
        sigmas.sort()
        
        # Вычисление маргинальной функции правдоподобия по сгенерированным параметрам
        log_likelihood = np.sum(
            logsumexp(
                np.log(class_probs)
                +
                tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
                    dataset.reshape(-1, 1)
                ).numpy()
                ,
                axis=1
            )
            ,
            axis=0
        )
        log_likelihood_history.append(log_likelihood)
        

        cond_eps = np.all(np.abs((class_responsibilities - prev_class_responsibilities) / n_samples <= convergence_accurcy))
        if cond_eps:
            print(i, '\t',class_responsibilities/n_samples,'\n\t',prev_class_responsibilities/ n_samples,'\n\n')
            converged = True
            break           
        i+=1
        prev_class_responsibilities = class_responsibilities.copy()
    
    return class_probs, mus, sigmas, log_likelihood_history


def em_adap(
    dataset,
    convergence_accuracy,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    import numpy as np
    from scipy.special import logsumexp
    import tensorflow_probability as tfp
    
    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]
    epsilon = convergence_accuracy

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    log_likelihood_history = []

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

        if(np.all(class_probs - prev_class_probs <= epsilon)):
            break

        prev_class_probs = class_probs.copy()

        # Calculate the marginal log likelihood
        log_likelihood = np.sum(
            logsumexp(
                np.log(class_probs)
                +
                tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
                    dataset.reshape(-1, 1)
                ).numpy()
                ,
                axis=1
            )
            ,
            axis=0
        )
        
        log_likelihood_history.append(log_likelihood)
    return class_probs, mus, sigmas, log_likelihood_history

def EM_sieved(
    dataset,
    n_classes,
    convergence_accuracy_initial,
    n_candidates, # кол-во наборов смесей
    convergence_accuracy_prime,
    n_chosen_ones, # кол-во (лучших) наборов, которые мы хотим получить в результате
    random_seed,
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
        class_probs, mus, sigmas, log_likelihood_history = em_adap(
            dataset,
            convergence_accuracy_initial,
            class_probs,
            mus,
            sigmas,
        )
        
        # Сохраняем параметры
        Mus.append(mus)
        Sigmas.append(sigmas)
        Class_probs.append(class_probs)
        LL_histories.append(log_likelihood_history)

    # Добавляем предыдущие парам-ы, если нужно
    if prev_params is not None:
        Mus.append(prev_params[0])
        Sigmas.append(prev_params[1])
        Class_probs.append(prev_params[2])
        LL_histories.append(prev_params[3])

    # (2) Отбор результатов.
    from itertools import zip_longest
    
    # Подгоняю списики с историей функций правдоподобия под один размер.
    LL_hist_same_size = [list(tpl) for tpl in 
                            zip(*zip_longest(*LL_histories,
                                              fillvalue=np.NaN))]
    log_likelihood_history_array = np.array(LL_hist_same_size)
    

    # Выбор лучших параметров наборов и отсеивание лишних результатов
    ordered_candidate_ids = np.argsort( - log_likelihood_history_array[:, -1])
    chosen_ones_ids = ordered_candidate_ids[:n_chosen_ones]

    # (3) Запуск ЕМ-алгоритма для лучших параметров смесей
    Mus_chosen = []
    Sigmas_chosen = []
    Class_probs_chosen = []
    LL_histories_chosen = []

    for chosen_one_id in pbar(chosen_ones_ids+[-1], "ЕМ для лучших парам-ов"):
        class_probs, mus, sigmas, log_likelihood_history = em_adap(
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
    LL_histories_chosen_same_size = [list(tpl) for tpl in 
                                        zip(*zip_longest(*LL_histories_chosen,
                                                        fillvalue=np.NaN))]
    log_likelihood_history_chosen_ones_array = np.array(LL_histories_chosen_same_size)
    # log_likelihood_history_chosen_ones_array = np.array(log_likelihood_history_chosen_ones_list)

    ordered_chosen_ones_ids = np.argsort( - log_likelihood_history_chosen_ones_array[:, -1])
    
    # Выделяю лучщие параметры
    best_chosen_one_id = ordered_chosen_ones_ids[0]
    best_mus = Mus_chosen[best_chosen_one_id]
    best_sigmas = Sigmas_chosen[best_chosen_one_id]
    best_class_probs = Class_probs_chosen[best_chosen_one_id]

    # Сортировка первого списка и получение индексов перестановки
    sorted_indices = sorted(range(len(best_sigmas)),
                             key=lambda k: best_sigmas[k])
    best_class_probs = np.array([best_class_probs[i] for i in sorted_indices])
    best_mus = np.array([best_mus[i] for i in sorted_indices])
    best_sigmas = np.array([best_sigmas[i] for i in sorted_indices])


    return best_class_probs, best_mus, best_sigmas, log_likelihood_history_chosen_ones_array


def em_sieved(
    dataset,
    n_classes,
    n_iterations_pre_sieving,
    n_candidates,
    n_iterations_post_sieving,
    n_chosen_ones,
    random_seed,
):
    import numpy as np
    from scipy.special import logsumexp
    from tqdm import tqdm
    import tensorflow_probability as tfp
    # (1) Pre-Sieving

    mus_list = []
    sigmas_list = []
    class_probs_list = []
    log_likelihood_history_list = []

    for candidate_id in range(n_candidates):
        np.random.seed(random_seed + candidate_id)

        mus = np.random.rand(n_classes)
        sigmas = np.random.rand(n_classes)
        class_probs = np.random.dirichlet(np.ones(n_classes))

        class_probs, mus, sigmas, log_likelihood_history = EM_iterative(
            dataset,
            n_iterations_pre_sieving,
            class_probs,
            mus,
            sigmas,
        )
        mus_list.append(mus)
        sigmas_list.append(sigmas)
        class_probs_list.append(class_probs)
        log_likelihood_history_list.append(log_likelihood_history)
    
    # (2) Sieving, select the best candidates
    log_likelihood_history_array = np.array(log_likelihood_history_list)

    # Sort in descending order
    ordered_candidate_ids = np.argsort( - log_likelihood_history_array[:, -1])
    chosen_ones_ids = ordered_candidate_ids[:n_chosen_ones]

    # (3) Post-Sieving
    mus_chosen_ones_list = []
    sigmas_chosen_ones_list = []
    class_probs_chosen_ones_list = []
    log_likelihood_history_chosen_ones_list = []
    for chosen_one_id in chosen_ones_ids:
        class_probs, mus, sigmas, log_likelihood_history = EM_iterative(
            dataset,
            n_iterations_post_sieving,
            class_probs_list[chosen_one_id],
            mus_list[chosen_one_id],
            sigmas_list[chosen_one_id],
        )

        mus_chosen_ones_list.append(mus)
        sigmas_chosen_ones_list.append(sigmas)
        class_probs_chosen_ones_list.append(class_probs)
        log_likelihood_history_chosen_ones_list.append(log_likelihood_history)
    
    # (4) Select the very best candidate
    log_likelihood_history_chosen_ones_array = np.array(log_likelihood_history_chosen_ones_list)

    # Sort in descending order
    ordered_chosen_ones_ids = np.argsort( - log_likelihood_history_chosen_ones_array[:, -1])

    best_chosen_one_id = ordered_chosen_ones_ids[0]
    best_mus = mus_chosen_ones_list[best_chosen_one_id]
    best_sigmas = sigmas_chosen_ones_list[best_chosen_one_id]
    best_class_probs = class_probs_chosen_ones_list[best_chosen_one_id]

    return best_class_probs, best_mus, best_sigmas, log_likelihood_history_chosen_ones_array
#--------------------------------------------------------------------------------

def mixture_extraction(series, 
                       window_size=1000, 
                       step=10, 
                       n_iter=None, 
                       conv_initial=0.001, 
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
    convergence_accuracy_initial = conv_initial
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

    # Перемещение окна
    progress_bar = tqdm(range(0, num_windows, step))
    prev_params = None
    for i in progress_bar:
        progress_bar.set_description(f"Обработка в окнах")
        window = series[i:i+window_size]
        class_probs, mus, sigmas, LL_history = EM_sieved(window.values,
                                      n_classes, 
                                      convergence_accuracy_initial,
                                      n_candidates,
                                      convergence_accuracy_prime,
                                      n_chosen_ones,       
                                      random_seed,
                                      prog_bar=EM_prog_bar,
                                      prev_params=prev_params)

        ind = len(param_df.index) # Индекс строки в формируемой таблице
        if xaxis_id is not None: # Добавляет временные привзяки
            rel_i = series.index[i] # Индекс, привязанный к исходным данным
            (param_df.attrs.get('custom_xaxis')).append(xaxis_id[rel_i])
        param_df.loc[ind, 'math_exp'] = mus
        param_df.loc[ind, 'st_dev'] = sigmas
        param_df.loc[ind, 'cl_prob'] = class_probs
        param_df.loc[ind, 'LL_hist'] = LL_history[0][0] # best performence
        prev_params = (mus, sigmas, class_probs, LL_history[0]) 

    return param_df

#--------------------------------------------------------------------------------

def mixture_param(series, window_size=1000, step=10, n_iter=None, eps=0.001, n_class=6,
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

def log_likelihood(mixture, series):
    from numpy import sum as npsum, log as nplog
    import numpy as np
    from scipy.special import logsumexp
    from tqdm import tqdm
    import tensorflow_probability as tpf
    log_likelihood_history = []
    
    class_probs = mixture.loc[:,'cl_prob'].values
    mus = mixture.loc[:,'math_exp'].values
    sigmas = mixture.loc[:,'st_dev'].values
    for row in range(mixture.shape[0]):
        # Вычисление маргинальной функции правдоподобия по сгенерированным параметрам
        log_likelihood = npsum(
            logsumexp(
                nplog(class_probs[row])
                +
                tpf.distributions.Normal(loc=mus[row], scale=sigmas[row]).log_prob(
                    series.reshape(-1, 1)
                ).numpy()
                ,
                axis=1
            )
            ,
            axis=0
        )
        log_likelihood_history.append(log_likelihood)
    return log_likelihood_history