"""Expectation-Maximization algorithms for **static** Gaussian mixture models.

This module contains implementations of Expectation-Maximization algorithms
that are used for segregation data on clusters. Each cluster correspond to
a gaussian distribution.
"""
import numpy as np
import numpy.linalg as nlg
from scipy.special import logsumexp
from scipy.stats import kstest
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture

import warnings
from tqdm.notebook import tqdm


class __EM:
    def __init__(
        self,
        num_comp=1,
        variances=None,
        means=None,
        probs=None,
        distrib="norm",
        warm_start=False,
        rseed=42,
    ):
        if num_comp >= 1 and isinstance(num_comp, int):
            self._num_comp = num_comp
        else:
            raise ValueError(
                f"Number components should be a positive integer number. Not {num_comp}"
            )

        self._variances = variances
        self._means = means
        self._probs = probs
        self._warm_start = warm_start
        self._rseed = rseed  # Need for initialization if no warm start
        if not self._warm_start:
            self._initialize_params()
        self._llh = None  # log-likelihood

        match distrib:
            case "norm":
                self._distrib = tfp.distributions.Normal
            case _:
                raise ValueError("No such distribution is supported.")

    def __repr__(self):
        def Round(arr, dig=4):
            return list(map(lambda numb: round(numb, dig), arr))

        p = Round(self._probs)
        m = Round(self._means)
        v = Round(self._variances)

        res = f"{self.__class__.__name__}\n"
        res += f"probs={p},\n"
        res += f"means={m},\n"
        res += f"variances={v}\n"
        return res

    def _e_step(self, dataset):
        """Expectation step.

        Define the expected value of the log-likelihood function.

        :param dataset: Input data for processing
        :param class_probs: Values of mixture's components probabilities
        :param mus: Values of mixture's components mathematical expectations
        :param sigmas: Values of mixture's components standard deviations
        :param distribution: Distribution of mixture's components, defaults to
            tensorflow_probability.distributions.Normal
        """

        response = (
            self._distrib(loc=self._means, scale=self._variances)
            .prob(dataset.reshape(-1, 1))
            .numpy()
            * self._probs
        )
        response /= nlg.norm(response, axis=1, ord=1, keepdims=True)
        return response

    def _m_step(self, data, response):
        """Maximization step.

        Find the parameters that maximize value of the log-likelihood function,
        calculated on previous E-step.

        :param data: Domain of definition of the maximization parameter
        :param response: Log-likelihood value from E-step (current conditional
            distribution), that will be maximized
        """

        n_samples = data.shape[0]
        reshaped_dataset = data.reshape(-1, 1)

        class_response = np.sum(response, axis=0)
        probs = class_response / n_samples
        means = np.sum(response * reshaped_dataset, axis=0) / class_response
        _terms = response * (reshaped_dataset - means.reshape(1, -1)) ** 2
        vars = np.sqrt(np.sum(_terms, axis=0) / class_response)
        self._probs, self._means, self._variances = probs, means, vars

    def _initialize_params(self, data=[]) -> None:
        """Generate starting values for empty mixture parameters.

        :param dataset: If given tRound(self._variances)hen it slightly adjusts initial parameters
        :param random_seed: Set state for random generator
        """
        np.random.seed(self._rseed)
        if not np.any(self._probs):
            self._probs = np.random.dirichlet(np.ones(self._num_comp))
        if not np.any(self._means):
            self._means = np.random.rand(self._num_comp) * (
                np.mean(data) if any(data) else 1
            )
        if not np.any(self._variances):
            self._variances = np.random.uniform(0.5, 1.5, self._num_comp) * (
                np.std(data) if any(data) else 1
            )

    def _log_likelihood(self, data):
        """Calculate the mixture's marginal log-likelihood.

        :param class_probs: Probability parameter values
        :param mus: Mathematical expectation values
        :param sigmas: Standard deviation values
        :param data: Data on which likelihood is measured
        """
        result = np.sum(
            logsumexp(
                np.log(self._probs)
                + tfp.distributions.Normal(loc=self._means, scale=self._variances)
                .log_prob(data.reshape(-1, 1))
                .numpy(),
                axis=1,
            ),
            axis=0,
        )
        return result

    def fit(self, data):
        self._initialize_params(data)
        self._llh = self._log_likelihood(data)

    @property
    def parameters(self):
        return dict(probs=self._probs, means=self._means, variances=self._variances)

    @parameters.setter
    def parameters(self, vals: dict):
        self._probs = vals["probs"]
        self._means = vals["means"]
        self._variances = vals["variances"]

    @property
    def aic(self):
        k = self._num_comp * 3 - 1 + 2  # 2 is a default value for AIC
        return 2 * k - 2 * self._llh


class EMscklearn(GaussianMixture, __EM):
    pass
    # def fit(self, data):
    #     __start = gmm["window"]["step"]
    #     __stop = len(gmm["series"]) - gmm["window"]["size"]
    #     # __stop = gmm["window"]["step"] + 100
    #     __step = gmm["window"]["step"]

    #     for i in tqdm(range(__start, __stop, __step)):
    #         # Take current window and fit GMM on it
    #         values = gmm["series"][i : gmm["window"]["size"] + i].reshape(-1, 1)
    #         vals = values[~np.isnan(values)].reshape(-1, 1)
    #         mixture = model.fit(vals)
    #         # If one of the variances is lower then threshold then create new model
    #         j = 0
    #         while min(mixture.covariances_.reshape(-1, 1)) < GMMkwargs["tol"] and j < 20:
    #             model = GaussianMixture(**GMMkwargs)
    #             mixture = model.fit(vals)
    #             j += 1

    #         # Initialize container for parameters
    #         gmm["weights"] = np.append(gmm["weights"], mixture.weights_.reshape(-1, 1), axis=1)
    #         gmm["means"] = np.append(gmm["means"], mixture.means_, axis=1)
    #         gmm["variances"] = np.append(
    #             gmm["variances"], mixture.covariances_.reshape(-1, 1), axis=1
    #         )
    #         if any(np.isnan(values)):
    #             gaps_number = np.isnan(values).sum()
    #             fillings, _ = model.sample(gaps_number)
    #             values[np.isnan(values)] = fillings.reshape(-1)


class EMiter(__EM):
    """Iterative __EM-algorithm."""

    def __init__(
        self,
        num_comp=1,
        num_iter=10,
        variances=None,
        means=None,
        probs=None,
        distrib="norm",
        warm_start=False,
        rseed=42,
    ):
        super().__init__(num_comp, variances, means, probs, distrib, warm_start, rseed)
        self._num_iter = num_iter

    @property
    def aic(self):
        k = self._num_comp * 3 - 1 + 2  # 2 is a default value for AIC
        k += 1
        return 2 * k - 2 * self._llh

    def fit(self, data, pbar=False):
        i, count = 0, 0
        if pbar:
            pbar = tqdm(range(self._num_iter), "Iterating __EM")
        else:
            pbar = range(self._num_iter)
        for i in pbar:
            response = self._e_step(data)
            # If bad parameter generation occurred
            if np.any(np.isnan(response)):
                count += 1
                self._initialize_params(data, count)
                warnings.warn(
                    f"Bad selection in {self.__class__.__name__} at {i} count. Restarting the iteration."
                )
                pbar.reset()
                continue
            self._m_step(data, response)
        self._llh = self._log_likelihood(data)


class EMadap(__EM):
    def __init__(
        self,
        num_comp=1,
        epsilon=0.001,
        variances=None,
        means=None,
        probs=None,
        distrib="norm",
        warm_start=False,
        rseed=42,
    ):
        super().__init__(num_comp, variances, means, probs, distrib, warm_start, rseed)
        self._eps = epsilon
        # Previous parameters to detect convergence
        self._pprobs = np.zeros(num_comp)
        self._pmeans = np.zeros(num_comp)
        self._pvariances = np.zeros(num_comp)

    @property
    def aic(self):
        k = self._num_comp * 3 - 1 + 2  # 2 is a default value for AIC
        k += 1
        return 2 * k - 2 * self._llh

    def stop_condition(self):
        """
        Check the stop condition for the adaptive __EM algorithm.

        This function evaluates whether the change in mixture parameters is
        within a s or self._probs is Nonepecified convergence accuracy (epsilon) for all components.
        """

        p = np.all(np.abs(self._probs - self._pprobs) <= self._eps)
        m = np.all(np.abs(self._means - self._pmeans) <= self._eps)
        v = np.all(np.abs(self._variances - self._pvariances) <= self._eps)
        return p and m and v

    def fit(self, data):
        count = 0
        while True:
            response = self._e_step(data)
            if np.any(np.isnan(response)):
                count += 1
                self._initialize_params(data, count)
                warnings.warn(f"Bad selection in {self.__class__.__name__}.")
                continue
            self._m_step(data, response)
            if self.stop_condition():
                break

            self._pprobs = self._probs
            self._pmeans = self._means
            self._pvariances = self._variances
        self._llh = self._log_likelihood(data)


class EMsiev(__EM):
    def __init__(
        self,
        num_comp=1,
        num_init=10,
        num_iter=100,
        num_best=1,
        epsilon=0.001,
        variances=None,
        means=None,
        probs=None,
        distrib="norm",
        warm_start=False,
        rseed=42,
    ):
        """
        Initialize parameters for __EM-sieving algorithm.

        Parameters
        ----------
        num_comp: int
            Number of components in the mixture.
        num_init: int
            Number of candidates for finding best initial parameters sets.
        num_iter: int
            Number of iterations for iterative __EM-algorithm for calculating
            parameters for each of the candidates.
        num_best: int
            Number of best candidates out of the initial candidates to process.
        epsilon: float
            Convergence accuracy for adaptive __EM-algorithm that applies on best
            initial parameters.
        """
        self._num_init = num_init
        self._num_iter = num_iter
        self._num_best = num_best
        self._epsilon = epsilon
        super().__init__(num_comp, variances, means, probs, distrib, warm_start, rseed)

    @property
    def aic(self):
        k = self._num_comp * 3 - 1 + 2  # 2 is a default value for AIC
        k += 4
        return 2 * k - 2 * self._llh

    def fit(
        self,
        data,
        prog_bar=False,
    ):
        # (1) Генерирование первичных наборов параметров смесей
        all_candid_params = ([], [], [], [])

        def add_params(param_list, predic):
            return [param.append(val) for param, val in zip(param_list, predic)]

        def pbar(span, title):
            return tqdm(span, title) if prog_bar else span

        # Sieve through initial candidates
        for candidate_id in pbar(
            range(self._num_init), "Initial parameters. Iterative __EM"
        ):
            # Задает новое состояние случайного генератора при смене кандидата
            rseed = self._rseed + candidate_id
            iterative = EMiter(self._num_comp, self._num_iter, rseed=rseed)
            iterative.fit(data)
            add_params(
                all_candid_params, (*iterative.parameters.values(), iterative._llh)
            )
        # Add parameters from previous frame if segregating dynamic mixture model
        # if self._warm_start is not None:
        #     add_params(all_candid_params, prev_pmsl)

        probs, mus, sigmas, loglike = all_candid_params

        # Best parameters, based on loglikelihood
        ids_best = np.argsort(-np.array(loglike))[: self._num_best]
        best_candid_params = ([], [], [], [])

        # __EM for best initial parameters
        for i in pbar(ids_best, "Adaptive __EM"):
            adaptive = EMadap(
                self._num_comp,
                self._epsilon,
                probs=probs[i],
                means=mus[i],
                variances=sigmas[i],
                warm_start=True,
            )
            adaptive.fit(data)
            add_params(
                best_candid_params, (*adaptive.parameters.values(), adaptive._llh)
            )

        # Best set of parameters, based on loglikelihood
        probs, mus, sigmas, loglike = best_candid_params
        loglike_history = np.sort(np.array(loglike))[::-1]
        id_prime = np.argsort(-np.array(loglike))[0]
        self.parameters = {
            "probs": probs[id_prime],
            "means": mus[id_prime],
            "variances": sigmas[id_prime],
        }
        self._llh = loglike_history[0]


class EMKS(__EM):
    from scipy.stats._stats_py import KstestResult

    def __init__(
        self,
        num_comp=1,
        variances=None,
        means=None,
        probs=None,
        distrib="norm",
        warm_start=False,
        rseed=42,
    ):
        super().__init__(num_comp, variances, means, probs, distrib, warm_start, rseed)

    @property
    def aic(self):
        k = self._num_comp * 3 - 1 + 2  # 2 is a default value for AIC
        k += 3
        return 2 * k - 2 * self._llh

    def __ks_test(self, data) -> KstestResult:
        norm_mixture = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=self._probs),
            components_distribution=tfp.distributions.Normal(
                loc=self._means,
                scale=self._variances,
            ),
        )
        return kstest(data, lambda x: norm_mixture.cdf(x).numpy())

    def fit(
        self,
        data,
        train_perc,  # percentage of validational dataset size
        relprev_pos=2,  # relative position of previous p-value count to compare with
        conv_speed=0.0001,  # convergence speed between p-value changes
    ):
        """Kolmogorov-Smirnov EM-algorithm.

        EM algorithm that prevent deviation fading using Kolmogorov-Smirnov test
        for detecting p-value's decreasing (worsen). P-value is evaluated by
        Kolmogorov-Smirnov statistic for fitting given data with mixture model
        on current step.
        """
        np.random.seed(self._rseed)
        np.random.shuffle(data)

        # Separation of validating and training data
        train_size = int(train_perc * len(data))
        data_train = data[:train_size]
        data_valid = data[train_size:]

        # Saving components to process
        # Initialize components for previous processed window
        pvalue_prev = 0
        # Counters for prints and rseed change
        iter_counter = 0
        count = 0
        while True:
            # (I) EM step for train data
            response = self._e_step(data_train)
            if np.any(np.isnan(response)):
                count += 1
                self._initialize_params(data, count)
                warnings.warn(
                    f"Bad selection in {self.__class__.__name__}. Restarting the iteration {iter_counter}"
                )
                iter_counter = 0
                continue
            self._m_step(data_train, response)
            iter_counter += 1

            # (II) Calculating p-value for data_valid
            pvalue = self.__ks_test(data_valid)[1]
            # Stop-condition
            slow_speed_cond = np.abs(pvalue - pvalue_prev) < conv_speed
            better_pval_cond = pvalue <= pvalue_prev
            if better_pval_cond or slow_speed_cond:
                break

            # Saving previous p-value
            if iter_counter % relprev_pos == 0:
                pvalue_prev = pvalue

        self._llh = self._log_likelihood(data)
