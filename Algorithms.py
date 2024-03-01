'''
This class will contain all enitial functionality for data processing
'''
import numpy as np
# Debug prints
round_sort = lambda param, n: [round(elem, n) for elem in np.sort(param)]

def candid_print(
        sertian_param, 
        loglikelihood=None, 
        text='INER print probs best'
    ):
    if loglikelihood:
        print(text, 
            *[f"{round(llh,4)} | {round_sort(param, 4)}" for 
                param, llh in zip(sertian_param, loglikelihood)], 
                '\n', 
                sep='\n\t'
                )
    else:
        print(text, 
            *[f"{round_sort(param, 4)}" for param in sertian_param], 
            '\n', 
            sep='\n\t'
            )

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

def log_likelihood_from_book(
        class_probs, 
        mus, 
        sigmas, 
        data
        ): 
    # Calculate the marginal log likelihood
    import numpy as np
    from scipy.special import logsumexp

    densities = tfp.distributions.Normal(
                    loc=mus, 
                    scale=sigmas
                    ).prob(data.reshape(-1, 1)).numpy()
    
    result = np.sum(
        np.log(
            np.sum(
                class_probs * densities,
                axis=1
            )
        )
    )
    
    return result

def E_step(
        dataset, 
        distribution, 
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

def initialize_params(num_comps, random_seed=42):
    from numpy.random import dirichlet, rand, uniform, seed
    from numpy import ones

    seed(random_seed)

    class_probs = dirichlet(ones(num_comps))
    mus = rand(num_comps)*20
    sigmas = uniform(0.5, num_comps*2, num_comps)

    return class_probs, mus, sigmas

def EM_iter(
    dataset,
    n_iterations,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
    ):
    from numpy import isnan, any
    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()
    i = 0
    count = 0
    while i <= n_iterations:

        responsibilities = E_step(
            dataset, 
            tfp.distributions.Normal, 
            class_probs,
            mus, 
            sigmas
            )
        
        # Если попалась неудачная генерация
        if any(isnan(responsibilities)):
            count += 1
            class_probs, mus, sigmas = initialize_params(len(class_probs), count)
            print('Bad selection in ITER. Restarting the iteration ', i)
            i = 0
            continue

        class_probs, mus, sigmas = M_step(
            dataset, 
            responsibilities)
        
        i+=1
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
    from numpy import isnan, any, all, zeros, abs as numpy_abs
    
    n_classes = class_probs_initial.shape[0]
    epsilon = convergence_accuracy

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    prev_class_probs = zeros(n_classes) # intial valurs for condition
    prev_mus, prev_sigmas = zeros(n_classes), zeros(n_classes) 
    i=0
    count = 0
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
            count += 1
            class_probs, mus, sigmas = initialize_params(len(class_probs),count)
            print('Bad selection in ADAP. Restarting the iteration ', i)
            i = 0
            continue

        class_probs, mus, sigmas = M_step(
            dataset, 
            responsibilities)
        i+=1
        
        # Stop-condition
        if(
            all(numpy_abs(class_probs - prev_class_probs) <= epsilon) and
            all(numpy_abs(mus - prev_mus <= epsilon)) and
            all(numpy_abs(sigmas - prev_sigmas <= epsilon))
           ):
            # print(i," iteration's taken before convergance")
            break
        prev_class_probs = class_probs.copy()
        prev_mus = mus.copy()
        prev_sigmas = sigmas.copy()

    # Возвращаю значение ф-ии правд-ия для лучего результата
    log_lh = log_likelihood(class_probs, mus, sigmas, dataset)

    #Сортируем в правильном порядке
    if(any(numpy_abs(class_probs-class_probs_initial) > 0.1 )):
        sorted_indices = sorted(range(len(class_probs_initial)),
                                key=lambda k: class_probs_initial[k])
        class_probs = np.array([class_probs[i] for i in sorted_indices])
        mus = np.array([mus[i] for i in sorted_indices])
        sigmas = np.array([sigmas[i] for i in sorted_indices])

    return class_probs, mus, sigmas, log_lh

def EM_sieved(
    dataset,
    num_params: int,
    num_iter_candid_initial: int = 100,
    n_candid: int = 20, # кол-во наборов смесей
    n_best_candid: int = 4, # кол-во (лучших) наборов, которые мы хотим получить в результате
    accur_best_candid: float = 0.01,
    random_seed: int = 42,
    prog_bar=False,
    prev_pmsl=None
):
    import numpy as np
    from tqdm.notebook import tqdm

    # (1) Генерирование первичных наборов параметров смесей
    all_candid_params = ([], [], [], [])
    add_params = lambda param_list, predic: [param.append(val) for param, val 
                                             in zip(param_list, predic)]
            
    # Возвращает шкалу прогресса либо область итерирования
    pbar = lambda span, title: tqdm(span, title) if prog_bar else span

    # Просеивание кандитатов    
    for candidate_id in pbar(range(n_candid), "Candidates generation"):
        # Задает новое состояние случайного генератора при смене кандидата
        rseed = random_seed + candidate_id
        # Рассчитываем параметры ЕМ-алгоритмом
        predictions = EM_iter(
            dataset,
            num_iter_candid_initial,
            *initialize_params(num_params, random_seed=rseed)
        )
        # Сохраняем параметры
        add_params(all_candid_params, predictions)

    # Добавляем предыдущие парам-ы, если нужно
    if prev_pmsl is not None:
        add_params(all_candid_params, prev_pmsl)

    # (2) Отбор результатов.
    probs, mus, sigmas, loglike = all_candid_params

    # Выбор лучших параметров наборов и отсеивание лишних результатов
    ids_best = np.argsort(-np.array(loglike))[:n_best_candid]
    best_candid_params = ([], [], [], [])
    
    # candid_print(probs, loglike)
    # print(np.mean(loglike))
    # candid_print([probs[bid] for bid in ids_best], 
    #              [loglike[bid] for bid in ids_best],
    #              'best, based on loglikelihood')
    
    # candid_print([mus[bid] for bid in ids_best], text="Mus for best ones")
    # candid_print([sigmas[bid] for bid in ids_best], text="Sigmas for best ones")


    for i in pbar(ids_best, "ЕМ для лучших парам-ов"):
        predictions = EM_adap(
            dataset,
            accur_best_candid,
            probs[i],
            mus[i],
            sigmas[i],
        )
        add_params(best_candid_params, predictions)
    
    # (3) Выбор лучшего кандидата
    probs, mus, sigmas, loglike = best_candid_params

    # ids_best = np.argsort(-np.array(loglike))
    # candid_print([probs[bid] for bid in ids_best], 
    #              [loglike[bid] for bid in ids_best],
    #              'best, based on loglikelihood')
    
    loglike_history = np.sort(np.array(loglike))[::-1]
    id_prime = np.argsort(-np.array(loglike))[0]
    prob = probs[id_prime]
    mu = mus[id_prime]
    sigma = sigmas[id_prime]

    return prob, mu, sigma, loglike_history[0]

