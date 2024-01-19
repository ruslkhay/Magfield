'''
This class will contain all enitial functionality for data processing
What do i want:
    1. '''
import tensorflow_probability as tfp

def log_likelihood(
        class_probs, 
        mus, 
        sigmas, 
        data
        ): 
    # Calculate the marginal log likelihood
    import numpy as np
    from scipy.special import logsumexp

    result = np.sum(
        logsumexp(
            np.log(class_probs) + 
            tfp.distributions.Normal(
                loc=mus, 
                scale=sigmas).log_prob(
                    data.reshape(-1, 1)
                    ).numpy(),
            axis=1
        )
        ,
        axis=0
        )
    
    return result

def check_validaty(respons):
    pass

def E_step(
        dataset, 
        distribution: tfp.distributions, 
        class_probs,
        mus, 
        sigmas
        ):
        from numpy.linalg import norm as numpy_linalg_norm

        responsibilities = distribution(
            loc=mus, 
            scale=sigmas
            ).prob(
                dataset.reshape(-1, 1)
                ).numpy() * class_probs

        responsibilities /= numpy_linalg_norm(
            responsibilities, 
            axis=1, 
            ord=1,
            keepdims=True
            )
        
        return responsibilities

def M_step(
        dataset, 
        responsibilities
        ):
    from numpy import sum as numpy_sum, sqrt as numpy_sqrt

    n_samples = dataset.shape[0]
    reshaped_dataset = dataset.reshape(-1, 1)

    class_responsibilities = numpy_sum(
        responsibilities, 
        axis=0
        )
    
    class_probs = class_responsibilities / n_samples

    mus = numpy_sum(
        responsibilities * reshaped_dataset,
        axis=0
        ) / class_responsibilities
    
    sigmas = numpy_sqrt(
        numpy_sum(
            responsibilities * (reshaped_dataset - mus.reshape(1, -1))**2, 
            axis=0
            ) / class_responsibilities
    )

    return class_probs, mus, sigmas

def reset_params(class_probs, mus, sigmas):
    from numpy.random import dirichlet, rand
    from numpy import ones
    n_classes = len(class_probs)
    class_probs = dirichlet(ones(n_classes))
    mus, sigmas = rand(n_classes), rand(n_classes)
    return class_probs, mus, sigmas

def EM_iterative(
    dataset,
    n_iterations,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
    ):
    from tqdm import tqdm
    from numpy import isnan, any

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    for i in tqdm(range(n_iterations)):

        responsibilities = E_step(
            dataset, 
            tfp.distributions.Normal, 
            class_probs,
            mus, 
            sigmas
            )
        
        # Если попалась неудачная генерация
        if any(isnan(responsibilities)):
            class_probs, mus, sigmas = reset_params(class_probs, mus, sigmas)
            print('Bad selection. Restarting the iteration ', i)
            i = i - 1
            continue

        class_probs, mus, sigmas = M_step(
            dataset, 
            responsibilities)
        
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
    from numpy import isnan, any
    
    n_classes = class_probs_initial.shape[0]
    epsilon = convergence_accuracy

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    prev_class_probs = np.zeros(n_classes) # intial valurs for condition
    i=0
    while True:
        responsibilities = E_step(
            dataset, 
            tfp.distributions.Normal, 
            class_probs,
            mus, 
            sigmas
            )

        # Если попалась неудачная генерация
        if any(isnan(responsibilities)):
            class_probs, mus, sigmas = reset_params(class_probs, mus, sigmas)
            print('Bad selection. Restarting the iteration ', i)
            i = i - 1
            continue

        class_probs, mus, sigmas = M_step(
            dataset, 
            responsibilities)
        i+=1
        
        # Stop-condition
        if(np.all(class_probs - prev_class_probs <= epsilon)):
            print(i," iteration's taken before convergance")
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


class Algorithm:
    def __init__(self, series, adj_var, *params):
        self.series = series
        self.params = params

class IterAlg(Algorithm):
    def __init__(self, series, iterations):
        super().__init__(series)
        self.iterations = iterations

class AdaptAlg(Algorithm):
    def __init__(self, series, accuracy):
        super().__init__(series)
        self.accuracy = accuracy


