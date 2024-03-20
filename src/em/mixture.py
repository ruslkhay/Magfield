class Mixture:
    """
    Create static, randomly generated mixtures of distributions.
    Predict mixture values with EM-algorithms.
    Calculate volatility.

    No interaction with 'window', i.e. no dynamic mixtures.
    """

    import tensorflow_probability as tfp
    import numpy as np

    def __init__(
        self,
        num_comps: int,  # amount of different r.v. in mixture
        distrib,  # type of r.v. distribution
        random_seed: int = 42,
        rand_initialize=False,
        comp_probs=None,  # the 'weights' of corresponding r.v.
        math_expects=None,  # the 'means' of corr-ding r.v.
        stand_devs=None,  # the 'dispersion' of corr-ding r.v.
    ):
        self.num_comps = num_comps
        self.distrib = distrib
        self._rseed = random_seed
        if rand_initialize:
            # Random initialization of main mixture parameters
            self.initialize_probs_mus_sigmas(random_seed)

        self.probs = comp_probs
        self.mus = math_expects
        self.sigmas = stand_devs

    def __str__(self) -> str:
        return str(self.__dict__).replace(" '", "\n '")

    def initialize_probs_mus_sigmas(self, random_seed: int, **kwargs) -> None:
        """
        Generate initial values for mixture parameters.
        Originally number of components and their's distribution type should
        be defined.
        """
        from numpy.random import seed

        from src.em.algorithms import initialize_params

        seed(random_seed)

        probs, mus, sigmas = initialize_params(
            self.num_comps, random_seed=random_seed, **kwargs
        )
        setattr(self, "probs", probs)
        setattr(self, "mus", mus)
        setattr(self, "sigmas", sigmas)

    def generate_samples(self, n_samples: int, random_seed: int = 42) -> None:
        """
        Generate values and their probabilities for mixture.
        This data can be used in visual or testing parts of this project.
        """
        import tensorflow_probability as tfp
        import tensorflow as tf

        # Data generation
        univariate_gmm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=self.probs,
            ),
            components_distribution=tfp.distributions.Normal(
                loc=self.mus,
                scale=self.sigmas,
            ),
        )

        tf.random.set_seed(  # set's random seed for sampling (for tfp)
            random_seed
        )

        self.samples = univariate_gmm.sample(n_samples).numpy()

        self.samples_probs = univariate_gmm.prob(self.samples).numpy()

    @staticmethod
    def construct_tpf_mixture(probs, mus, sigmas):
        """
        Generally irrelevant, but could be useful for future testing or
        visualization. Because returns tensorflow_probability.distribution
        object, that certainly has much more methods and options to test for.
        """
        import tensorflow_probability as tfp

        # Data generation
        univariate_gmm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=probs),
            components_distribution=tfp.distributions.Normal(
                loc=mus,
                scale=sigmas,
            ),
        )
        return univariate_gmm

    def log_likelihood(self, data) -> float:
        """
        Calculate value, representing the 'validity' measure (or 'correctness')
        for how good mixture emplanes given data behavior.
        """
        from src.em.algorithms import log_likelihood

        res = log_likelihood(self.probs, self.mus, self.sigmas, data)
        return res

    @property
    def volatility(self):
        """
        Calculate volatility.

        If mixture parameters would ever be represented other then 1D matrix,
        then it's needed to implement next change:
            axis = 1 if self.class_probability.ndim > 1 else 0
        """
        if not hasattr(self, "volat_comps"):
            from numpy import sum as numpy_sum

            def diffusion_component(self):
                total_mu = numpy_sum(self.mus * self.probs, axis=0, keepdims=True)
                res = numpy_sum(
                    (self.mus - total_mu) ** 2 * self.probs, axis=0, keepdims=True
                )
                return res

            def dynamic_component(self):
                res = numpy_sum(self.sigmas**2 * self.probs, axis=0, keepdims=True)
                return res

            self.volat_comps = {
                "diffusion_comp": diffusion_component(self),
                "dynamic_comp": dynamic_component(self),
            }
        return self.volat_comps

    @property
    def parameters(self):
        res = dict(probs=self.probs, mus=self.mus, sigmas=self.sigmas)
        return res

    @parameters.setter
    def parameters(self, vals: dict):
        self.probs = vals.get("probs")
        self.mus = vals.get("mus")
        self.sigmas = vals.get("sigmas")

    def _update_params(self, functions_list, point):
        for param, funcs in zip(("probs", "mus", "sigmas"), functions_list):
            # new_val = self.__getattribute__(param) + func(point)
            new_val = [func(point) for func in funcs]
            self.__setattr__(param, new_val)

    # ---------------------------------------------------------------------------

    def EM_iterative(self, dataset, n_iterations: int) -> tuple:
        """
        Predicts mixture parameters such variables weights, means and
        dispersion. Automatically returns the log-likelihood for parameters
        predictions.

        Estimate parameters based on amount of iterations.
        """
        from src.em.algorithms import EM_iter

        # class_probs, mus, sigmas, log_lh,
        predictions = EM_iter(dataset, n_iterations, self.probs, self.mus, self.sigmas)
        return predictions

    def EM_adaptive(self, dataset, accuracy: float):
        """
        Predicts mixture parameters such variables weights, means and
        dispersion. Automatically returns the log-likelihood for parameters
        predictions.

        Estimate parameters based on limiting converges distance, i.e.
        accuracy.
        """
        from src.em.algorithms import EM_adap

        predictions = EM_adap(dataset, accuracy, self.probs, self.mus, self.sigmas)
        return predictions

    def EM_sieving(
        self,
        dataset,
        iter_initial,
        num_candid,
        num_best_candid,
        accur_final,
        random_seed,
        **kwargs,
    ):
        """Combination of iterative and adaptive EMs. Primarily"""
        from src.em.algorithms import EM_sieved

        candidates = EM_sieved(
            dataset,
            num_params=self.num_comps,
            num_iter_candid_init=iter_initial,
            n_candid=num_candid,
            n_best_candid=num_best_candid,
            accur_best_candid=accur_final,
            random_seed=random_seed,
            **kwargs,
        )
        from numpy import array

        # Сортировка первого списка и получение индексов перестановки
        p, m, s, h = candidates
        sorted_indices = sorted(range(len(m)), key=lambda k: s[k])
        p = array([p[i] for i in sorted_indices])
        m = array([m[i] for i in sorted_indices])
        s = array([s[i] for i in sorted_indices])
        return p, m, s, h

    def Kolmogorov_EM(self, dataset, train_perc=0.5, relprev_pos=1, random_seed=42):
        from src.em.algorithms import EM_KS
        from numpy import array

        p, m, s, h = EM_KS(
            dataset=dataset,
            train_perc=train_perc,
            class_probs_init=self.probs,
            mus_init=self.mus,
            sigmas_init=self.sigmas,
            relprev_pos=relprev_pos,
            random_seed=random_seed,
        )
        # Sorting mixture parameters
        sorted_indices = sorted(range(len(m)), key=lambda k: s[k])
        p = array([p[i] for i in sorted_indices])
        m = array([m[i] for i in sorted_indices])
        s = array([s[i] for i in sorted_indices])
        return p, m, s, h

    # ---------------------------------------------------------------------------

    def show_samples(self):
        from plotly.express import scatter

        fig = scatter(x=self.samples, y=self.samples_probs)
        return fig


# ---------------------------------------------------------------------------


class DynamicMixture(Mixture):
    def __init__(
        self,
        num_comps,
        distrib,
        window_shape=(4300, 60),
        time_span=None,
        records=None,
        random_seed=42,
    ):
        super().__init__(num_comps, distrib, random_seed)

        self.time_span = time_span  # values to mark time axis
        self.records = records  # signal to process

        # Container for window shape info
        self.frame = dict(length=window_shape[0], step=window_shape[1])

        # Container for each window
        self.windows = []

        # Container for mixtures parameters and time relevant indexes info
        self.__params = dict(probs=[], mus=[], sigmas=[], llh=[])

        # Process X coefficients (dX=a(t)dt+b(t)dW)
        self.__proc_coefs = dict(a=[], b=[])

        # Container for synthetic data (used for testing EMs)
        self.samples = dict(records=[], probs=[])

    @staticmethod
    def _update_dict(dic, new_vals) -> None:
        """
        Extend dict with given values
        """
        for name, value in zip(dic.keys(), new_vals):
            dic[name].append(value)

    # ROW/ INCORRECT!!!
    def generate_samples(
        self, n_samples: int = 1, params_behave=None, random_seed: int = 42
    ) -> None:
        """
        Generates multitude of mixture objects. Where each next mixture is
        made out of previous one, but corrected by parameters time behavior
        """
        # Initial mixture
        curr_mix = Mixture(
            self.num_comps, self.distrib, self._rseed, rand_initialize=False
        )

        # Initialize time axis if needed
        if self.time_span is None:
            self.time_span = tuple(range(n_samples))

        for t in self.time_span:
            # Updating initial mixture parameters
            curr_mix._update_params(functions_list=params_behave, point=t)

            # Generating data
            curr_mix.generate_samples(n_samples, random_seed)

            # Saving data
            self.samples["records"].append(*curr_mix.samples)
            self.samples["probs"].append(*curr_mix.samples_probs)

            # Saving parameters
            self.__class__._update_dict(
                self.samples["params"],
                new_vals=(
                    t,  # stands for time_id
                    curr_mix.probs,
                    curr_mix.mus,
                    curr_mix.sigmas,
                    curr_mix.log_likelihood(curr_mix.samples),  # stands for llh
                ),
            )

    def _window_segregation(self, data):
        """
        Segregate given data into data chunks, corresponding to window size
        """
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(data, self.frame["length"])[:: self.frame["step"]]

        # Don't know for what reasons, but method above returns read-only array
        self.windows = windows.copy()

        return self.windows

    def rewrite_as_normal_human_this_initialization(
        self, random_seed: int, **kwargs
    ) -> None:
        """
        Generate initial values for mixture parameters.
        Originally number of components and their's distribution type should
        be defined.
        """
        del self.parameters
        from numpy.random import seed

        from src.em.algorithms import initialize_params

        seed(random_seed)

        probs, mus, sigmas = initialize_params(self.num_comps, random_seed, **kwargs)
        self._update_dict(self.__params, (probs, mus, sigmas, None))

    def predict(
        self,
        data,
        EM_params=dict(
            iter_initial=12, num_candid=20, num_best_candid=8, accur_final=0.001
        ),
        random_seed=42,
    ):
        """
        On each windows applies sieving EM algorithm on freshly generated
        mixture parameters 'candidates'
        """
        from tqdm.notebook import tqdm

        record_wind = self._window_segregation(data)

        first_wind = record_wind[0]
        record_wind = record_wind[1:]

        initial_guess = super().EM_sieving(
            dataset=first_wind, **EM_params, random_seed=random_seed
        )

        for frame in tqdm(record_wind):
            temporal_guess = super().EM_sieving(
                dataset=frame,
                **EM_params,
                random_seed=random_seed,
                prev_pmsl=initial_guess,
            )
            self._update_dict(self.__params, temporal_guess)
            initial_guess = temporal_guess
        return "finished"

    def predict_light(
        self,
        data,
        EM_params=dict(
            iter_initial=20,
            num_candid=40,
            num_best_candid=8,
            accur_final=0.005,
            prog_bar=True,
        ),
        random_seed=42,
    ):
        """
        Apply sieving EM only on starting window. Further previously
        calculated mixture parameters are fed to adaptive EM algorithm
        and no new parameters a.k.a. 'candidates' are considered
        """
        del self.parameters  # clear results from previous launches

        from tqdm.notebook import tqdm

        record_wind = self._window_segregation(data)

        first_wind = record_wind[0]
        record_wind = record_wind[1:]

        initial_guess = super().EM_sieving(
            dataset=first_wind, **EM_params, random_seed=random_seed
        )
        self._update_dict(self.__params, initial_guess)

        for frame in tqdm(record_wind):
            temporal_guess = super().EM_sieving(
                dataset=frame,
                iter_initial=0,
                num_candid=0,
                num_best_candid=1,
                prev_pmsl=initial_guess,
                accur_final=0.001,
                random_seed=random_seed,
            )

            self._update_dict(self.__params, temporal_guess)
            initial_guess = temporal_guess
        return "finished"

    def predict_ks(
        self,
        data,
        train_perc,  # percentage of validation dataset size
        relprev_pos,
        random_seed=42,
    ):
        """
        Apply sieving EM only on starting window. Further previously
        calculated mixture parameters are fed to adaptive EM algorithm
        and no new parameters a.k.a. 'candidates' are considered
        """
        # del self.parameters # clear results from previous launches
        from tqdm.notebook import tqdm
        from src.em.algorithms import EM_KS

        # Shuffling windows content
        record_wind = self._window_segregation(data)

        # first_wind = record_wind[0]
        # record_wind = record_wind[1:]

        # initial_guess = super().Kolmogorov_EM(
        #     dataset=first_wind,
        #     train_perc=train_perc,
        #     random_seed=random_seed
        # )
        # self._update_dict(
        #     self.__params,
        #     initial_guess
        # )

        for ind, frame in enumerate(tqdm(record_wind)):
            temporal_guess = EM_KS(
                dataset=frame,
                train_perc=train_perc,
                class_probs_init=self.parameters["probs"][ind],
                mus_init=self.parameters["mus"][ind],
                sigmas_init=self.parameters["sigmas"][ind],
                relprev_pos=relprev_pos,
                random_seed=random_seed,
            )

            self._update_dict(self.__params, temporal_guess)
            # initial_guess = temporal_guess
        return

    def reshape_params(self, params: tuple = ("probs", "mus", "sigmas")) -> dict:
        """
        Segregate parameters values by specific mixture component and
        unite it's values on each frame into one array.
        Previously params attribute contained values of all components on
        specific window (frame)
        """
        import numpy as np

        reshaped = dict()
        for key, value in self.__params.items():
            if key in params:
                specif_comp_vals = []
                for i in range(len(value[0])):
                    specif_comp_vals.append(np.array([arr[i] for arr in value]))
                reshaped[key] = specif_comp_vals
        return reshaped

    def reconstruct_process_coef(self):
        """
        Calculate stochastic process coefficients by known mixture parameters
        """
        import numpy as np

        del self.process_coefs  # clear results from previous launches

        def procpar(pk, ak, bk):
            "for a specific window"
            a = np.sum(pk * ak)
            b = np.sum(pk * (bk**2 + ak**2)) - a**2
            return a, b

        p = self.parameters["probs"]
        a = self.parameters["mus"]
        b = self.parameters["sigmas"]
        coef_a, coef_b = [], []

        for i in range(len(p)):
            a_t, b_t = procpar(p[i], a[i], b[i])
            coef_a.append(a_t)
            coef_b.append(b_t)
            self._update_dict(self.__proc_coefs, (a_t, b_t))
        return np.array(coef_a), np.array(coef_b)

    @property
    def parameters(self):
        return self.__params

    @parameters.deleter
    def parameters(self):
        print("Deleted")
        for key in self.__params.keys():
            self.__params[key] = []

    @property
    def process_coefs(self):
        return self.__proc_coefs

    @process_coefs.deleter
    def process_coefs(self):
        print("Deleted proc params")
        for key in self.__proc_coefs.keys():
            self.__proc_coefs[key] = []

    def process_volatility(self):
        """
        Calculate stochastic process coefficients by known mixture parameters
        """
        import numpy as np

        def procvolat(pk, ak, bk):
            "for a specific window"
            a = np.sum(pk * ak)
            dynamic_term = np.sum(pk * (ak - a) ** 2)
            diffusion_term = np.sum(pk * bk**2)
            return dynamic_term, diffusion_term

        p = self.parameters["probs"]
        a = self.parameters["mus"]
        b = self.parameters["sigmas"]
        dynamic_comp, diffusion_comp = [], []

        for i in range(len(p)):
            dynamic, diffus = procvolat(p[i], a[i], b[i])
            dynamic_comp.append(dynamic)
            diffusion_comp.append(diffus)

        volatility = dict(
            dynamic=np.array(dynamic_comp), diffusional=np.array(diffusion_comp)
        )
        return volatility

    from plotly.graph_objects import Figure

    def show_parameters(self) -> Figure:
        from monitor import construct_mixture_2Dplot

        probs_mus_sigmas = self.reshape_params()
        return construct_mixture_2Dplot(
            num_comps=self.num_comps,
            parameters=probs_mus_sigmas,
            x_ticks=self.time_span,
        )

    def __construct_hist_3d(self, bins):
        hist_attr = {
            "bins": [],
            "wind_numb": [],  # relative window number
            "hist_freq": [],  # 2D histogram values
        }
        from numpy import histogram, meshgrid

        hist_attr = {
            "bins": [],
            "wind_numb": [],  # relative window number
            "hist_freq": [],  # 2D histogram values
        }
        for i, window in enumerate(self.windows):
            hist, _bins = histogram(window, bins=bins)
            xpos, ypos = meshgrid(_bins[:-1], i)

            hist_attr["bins"].append(xpos.flatten())
            hist_attr["wind_numb"].append(ypos.flatten())
            hist_attr["hist_freq"].append(hist.flatten())

        return hist_attr

    def show_hist_3d(self, bins):
        from monitor import visualize_3D_hist

        hist_attr = self.__construct_hist_3d(bins)
        return visualize_3D_hist(hist_attr)

    def save(self, file_name, path=""):
        """Save class instance into file."""
        import pickle
        from os.path import join

        with open(join(path, f"{file_name}.pickle"), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """Load saved mixture file."""
        import pickle

        with open(file_path, "rb") as file:
            print(file_path)
            mixture = pickle.load(file)
        return mixture
