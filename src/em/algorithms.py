"""Contain implementations of EM-algorithms and connected functions."""

from typing import List
from scipy.stats._stats_py import KstestResult
import tensorflow_probability as tfp
import numpy as np


def sort_params(by_id: int = 0, *params: List[int] | List[float]):
    """Sort params by a specific parameter and other correspondingly.

    :param by_id: index of a parameter by which sorting will go
    :param params: collection of parameters to sort
    """
    sort_var = params[by_id]
    sorted_var_id = sorted(range(len(sort_var)), key=lambda k: sort_var[k])

    def _sort(param):
        return [param[i] for i in sorted_var_id]

    return tuple(map(_sort, params))


def candid_print(certain_param, loglikelihood=None, text="INNER print probs best"):
    r"""Debug print for ... idk for what ¯\_(ツ)_/¯."""

    def round_sort(param, n):
        return [round(el, n) for el in np.sort(param)]

    msg = [f"{round_sort(param, 4)}" for param in certain_param]
    if loglikelihood:
        msg = [
            f"{round(llh, 4)} | {round_sort(param, 4)}"
            for param, llh in zip(certain_param, loglikelihood)
        ]
    print(text, *msg, "\n", sep="\n\t")


def initialize_params(n: int, dataset=[], random_seed: int = 42):
    """Generate starting values for all mixture parameters.

    :param n: Number of components of mixture
    :param dataset: If given then it slightly adjusts initial parameters
    :param random_seed: Set state for random generator
    """
    from numpy.random import dirichlet, rand, uniform, seed
    from numpy import ones

    seed(random_seed)
    class_probs = dirichlet(ones(n))
    mus = rand(n) * (np.mean(dataset) if any(dataset) else 1)
    sigmas = uniform(0.5, 1.5, n) * (np.std(dataset) if any(dataset) else 1)
    return class_probs, mus, sigmas


def log_likelihood(class_probs, mus, sigmas, data):
    """Calculate the mixture's marginal log-likelihood.

    :param class_probs: Probability parameter values
    :param mus: Mathematical expectation values
    :param sigmas: Standard deviation values
    :param data: Data on which likelihood is measured
    """
    import numpy as np
    from scipy.special import logsumexp

    result = np.sum(
        logsumexp(
            np.log(class_probs)
            + tfp.distributions.Normal(loc=mus, scale=sigmas)
            .log_prob(data.reshape(-1, 1))
            .numpy(),
            axis=1,
        ),
        axis=0,
    )
    return result


def log_likelihood_from_book(class_probs, mus, sigmas, data):
    """Marginal log-likelihood as is written in Korolev's grey book."""
    import numpy as np

    densities = (
        tfp.distributions.Normal(loc=mus, scale=sigmas)
        .prob(data.reshape(-1, 1))
        .numpy()
    )

    result = np.sum(np.log(np.sum(class_probs * densities, axis=1)))
    return result


def E_step(dataset, class_probs, mus, sigmas, distribution=tfp.distributions.Normal):
    """Expectation step.

    Define the expected value of the log-likelihood function.

    :param dataset: Input data for processing
    :param class_probs: Values of mixture's components probabilities
    :param mus: Values of mixture's components mathematical expectations
    :param sigmas: Values of mixture's components standard deviations
    :param distribution: Distribution of mixture's components, defaults to
        tensorflow_probability.distributions.Normal
    """
    from numpy.linalg import norm as numpy_linalg_norm

    response = (
        distribution(loc=mus, scale=sigmas).prob(dataset.reshape(-1, 1)).numpy()
        * class_probs
    )
    response /= numpy_linalg_norm(response, axis=1, ord=1, keepdims=True)
    return response


def M_step(dataset, response):
    """Maximization step.

    Find the parameters that maximize value of the log-likelihood function,
    calculated on previous E-step.

    :param dataset: Domain of definition of the maximization parameter
    :param response: Log-likelihood value from E-step (current conditional
        distribution), that will be maximized
    """
    from numpy import sum as numpy_sum, sqrt as numpy_sqrt

    n_samples = dataset.shape[0]
    reshaped_dataset = dataset.reshape(-1, 1)

    class_response = numpy_sum(response, axis=0)
    class_probs = class_response / n_samples
    mus = numpy_sum(response * reshaped_dataset, axis=0) / class_response
    _terms = response * (reshaped_dataset - mus.reshape(1, -1)) ** 2
    sigmas = numpy_sqrt(numpy_sum(_terms, axis=0) / class_response)
    return class_probs, mus, sigmas


def EM_iter(dataset, n_iterations, class_probs_init, mus_init, sigmas_init):
    """Iterative EM-algorithm.

    :param dataset: Input data for processing
    :param n_iterations: Number of iterations for calculating mixture
        parameters values
    :param class_probs_init: Starting values for probability parameter
    :param mus_init: Starting values for mathematical expectation
        parameter
    :param sigmas_init: Starting vales for standard deviation parameter
    """
    from numpy import isnan, any

    class_probs = class_probs_init.copy()
    mus = mus_init.copy()
    sigmas = sigmas_init.copy()
    n = len(class_probs)
    i, count = 0, 0
    while i <= n_iterations:
        response = E_step(dataset, class_probs, mus, sigmas)
        # If bad parameter generation occurred
        if any(isnan(response)):
            count += 1
            new_params = initialize_params(n, dataset, count)
            class_probs, mus, sigmas = new_params
            print("Bad selection in ITER. Restarting the iteration ", i)
            i = 0
            continue
        class_probs, mus, sigmas = M_step(dataset, response)
        i += 1
    # log-likelihood value of best result
    log_lh = log_likelihood(class_probs, mus, sigmas, dataset)
    return class_probs, mus, sigmas, log_lh


def EM_adap(dataset, conv_acc, class_probs_init, mus_init, sigmas_init):
    """Adaptive EM-algorithm.

    :param dataset: Input data for processing
    :param conv_acc: Convergence accuracy - difference between all neighboring
        parameters upon reaching which will stop parameter's calculational
        process
    :param class_probs_init: Starting values for probability parameter
    :param mus_init: Starting values for mathematical expectation
        parameter
    :param sigmas_init: Starting vales for standard deviation parameter
    """
    from numpy import isnan, any, all, zeros, abs as numpy_abs

    n = class_probs_init.shape[0]
    epsilon = conv_acc

    class_probs = class_probs_init.copy()
    mus = mus_init.copy()
    sigmas = sigmas_init.copy()

    prev_class_probs, prev_mus, prev_sigmas = zeros(n), zeros(n), zeros(n)
    i, count = 0, 0
    while True:
        response = E_step(dataset, class_probs, mus, sigmas)
        if any(isnan(response)):  # Case of bad generation
            count += 1
            new_params = initialize_params(n, dataset, count)
            class_probs, mus, sigmas = new_params
            print("Bad selection in ADAP. Restarting the iteration ", i)
            i = 0
            continue
        class_probs, mus, sigmas = M_step(dataset, response)
        i += 1
        # Stop-condition
        if (
            all(numpy_abs(class_probs - prev_class_probs) <= epsilon)
            and all(numpy_abs(mus - prev_mus <= epsilon))
            and all(numpy_abs(sigmas - prev_sigmas <= epsilon))
        ):
            # print(i," iteration's taken before convergence")
            break

        prev_class_probs = class_probs.copy()
        prev_mus = mus.copy()
        prev_sigmas = sigmas.copy()

    # Log-likelihood for best result
    log_lh = log_likelihood(class_probs, mus, sigmas, dataset)
    return class_probs, mus, sigmas, log_lh


def EM_sieved(
    dataset,
    num_params: int,
    num_iter_candid_init: int = 100,
    n_candid: int = 20,  # кол-во наборов смесей
    # кол-во (лучших) наборов, которые мы хотим получить в результате
    n_best_candid: int = 4,
    accur_best_candid: float = 0.01,
    random_seed: int = 42,
    prog_bar=False,
    prev_pmsl=None,
    sort=True,
):
    """Sieving EM-algorithm.

    This is a combination of iterative and adaptive EM-algorithms.
    Primely generates multiple parameters sets and runs on them iterative
    EM. Secondarily choses subset of the best parameters based on
    log-likelihood values. Finally runs adaptive EM on this subset.

    :param dataset:
    :param num_params
    :param num_iter_candid_init
    :param n_candid  # кол-во наборов смесей
    :param n_best_candid  # кол-во (лучших) наборов, которые в результате
    :param accur_best_candid
    :param random_seed
    :param prog_bar
    :param prev_pmsl
    :param sort
    """
    import numpy as np
    from tqdm.notebook import tqdm

    # (1) Генерирование первичных наборов параметров смесей
    all_candid_params = ([], [], [], [])

    def add_params(param_list, predic):
        return [param.append(val) for param, val in zip(param_list, predic)]

    # Возвращает шкалу прогресса либо область итерирования
    def pbar(span, title):
        return tqdm(span, title) if prog_bar else span

    # Просеивание кандитатов
    for candidate_id in pbar(range(n_candid), "Candidates generation"):
        # Задает новое состояние случайного генератора при смене кандидата
        rseed = random_seed + candidate_id
        # Рассчитываем параметры ЕМ-алгоритмом
        predictions = EM_iter(
            dataset,
            num_iter_candid_init,
            *initialize_params(
                num_params,
                dataset=dataset,
                random_seed=rseed,
            ),
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
    # candid_print([sigmas[bid] for bid in ids_best],
    #              text="Sigmas for best ones")

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
    res = tuple([prob, mu, sigma, loglike_history[0]])
    if sort:
        # best way is to sort by 'sigma'
        res = sort_params(2, *res)
    return res


def KS_test(data, **mix_param) -> KstestResult:
    from scipy.stats import kstest
    import tensorflow_probability as tfp

    # Data generation
    norm_mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=mix_param["probs"]),
        components_distribution=tfp.distributions.Normal(
            loc=mix_param["mus"],
            scale=mix_param["sigmas"],
        ),
    )
    return kstest(data, lambda x: norm_mixture.cdf(x).numpy())


def EM_KS(
    dataset,
    train_perc,  # percentage of validational dataset size
    class_probs_init,
    mus_init,
    sigmas_init,
    relprev_pos,  # relative position of previous p-value count to compare with
    random_seed,
    epsilon=4,
):
    """Kolmogorov-Smirnov EM-algorithm.

    EM algorithm that prevent deviation fading using Kolmogorov-Smirnov test
    for detecting p-value's decreasing (worsen). P-value is evaluated by
    Kolmogorov-Smirnov statistic for fitting given data with mixture model
    on current step.
    """
    from numpy import isnan, abs, any
    from numpy.random import shuffle, seed

    num_comp = class_probs_init.shape[0]

    seed(random_seed)
    # Shuffling th e dataset
    data = dataset.copy()
    shuffle(data)

    # Separation of validating and training data
    train_size = int(train_perc * len(data))
    data_train = data[:train_size]
    data_valid = data[train_size:]

    # Saving components to process
    probs = class_probs_init.copy()
    mus = mus_init.copy()
    sigmas = sigmas_init.copy()

    # Initialize components for previous processed window
    pvalue_prev = 0

    # Counters for prints and rseed change
    iter_counter = 0
    count = 0
    while True:
        # (I) EM step for train data
        response = E_step(data_train, tfp.distributions.Normal, probs, mus, sigmas)
        # Если попалась неудачная генерация
        if any(isnan(response)):
            count += 1
            probs, mus, sigmas = initialize_params(
                num_comp,
                dataset=data_train,
                random_seed=count,
            )
            print("Bad selection in EM_KS. Restarting the iteration ", iter_counter)
            iter_counter = 0
            continue

        probs, mus, sigmas = M_step(data_train, response)
        iter_counter += 1

        # (II) Calculating p-value for data_valid
        pvalue = KS_test(data_valid, probs=probs, mus=mus, sigmas=sigmas)[1]
        # print('running \t', pvalue, pvalue_prev, sep=' | ')
        # Stop-condition
        slow_speed_cond = round(abs(pvalue - pvalue_prev), epsilon) == 0
        better_pval_cond = pvalue <= pvalue_prev
        if better_pval_cond or slow_speed_cond:
            # print('break at', pvalue, pvalue_prev, sep=' | ')
            break

        # Saving previous p-value
        if iter_counter % relprev_pos == 0:
            pvalue_prev = pvalue

    # Возвращаю значение ф-ии правд-ия для лучего результата
    log_lh = log_likelihood(probs, mus, sigmas, dataset)
    # Сортируем в правильном порядке
    sorted_indices = sorted(range(len(class_probs_init)), key=lambda k: sigmas[k])
    probs = np.array([probs[i] for i in sorted_indices])
    mus = np.array([mus[i] for i in sorted_indices])
    sigmas = np.array([sigmas[i] for i in sorted_indices])

    return probs, mus, sigmas, log_lh
