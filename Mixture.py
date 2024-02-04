class frame:
    def __init__(self, *, size, step, bins) -> None:
        self.size = size
        self.step = step
        self.BINS = bins


class mixture:
    '''
    Create static, randomly generated mixtures of distributions.
    Predict mixture values with EM-algorithms.
    Calculate volatilty.
    
    No interaction with 'window', i.e. no dynamic mixtures.
    '''
    import tensorflow_probability as tfp
    import numpy as np

    def __init__(
            self,
            num_comps: int, # amount of different r.v. in mixture
            distrib: tfp.distributions, # type of r.v. distribution
            random_seed: int = 42,
            comp_probs: np.array = None, # the 'weights' of corresponding r.v.
            math_expects: np.array = None, # the 'means' of corr-ding r.v.
            stand_devs: np.array = None # the 'dispersion' of corr-ding r.v.
    ):
        self.num_comps = num_comps
        self.distrib = distrib
        
        # Random initialization of main mixture parameters
        self.initialize_probs_mus_sigmas(random_seed)

        # self.probs = comp_probs
        # self.mus = math_expects
        # self.sigmas = stand_devs


    def __str__(self) -> str:
        return str(self.__dict__).replace(' \'','\n \'')

    def initialize_probs_mus_sigmas(
            self,
            random_seed: int 
            ) -> None: 
        """
        Generate initial values for mixture parameters.
        Originaly number of components and their's distribution type should
        be defined.
        """
        from numpy.random import seed

        from Algorithms import initialize_params

        seed(random_seed)
        
        probs, mus, sigmas = initialize_params(self.num_comps, random_seed)
        setattr(self, 'probs', probs)
        setattr(self, 'mus', mus)
        setattr(self, 'sigmas', sigmas)

    def generate_samples(
            self,
            n_samples: int,
            random_seed: int = 42
            ) -> None:
        '''
        Generate values and their probabilities for mixture.
        This data can be used in vizual or testing parts of this project.
        '''
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
            )
        )

        tf.random.set_seed( # set's random seed for sampling (for tfp)
            random_seed
        )

        self.samples = univariate_gmm.sample(
            n_samples
            ).numpy()   
        
        self.samples_probs = univariate_gmm.prob(
            self.samples).numpy()   

    def construct_tpf_mixture(
        self,
        ) -> tfp.distributions.MixtureSameFamily:
        '''
        Generaly irrelevant, but could be useful for future testing or
        visualization. Because returns tenserflow_probability.distribution
        object, that certainly has much more methods and options to test for.
        '''
        import tensorflow_probability as tfp
        # Data generation
        univariate_gmm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=self.probs
            ),
            components_distribution=tfp.distributions.Normal(
                loc=self.mus,
                scale=self.sigmas,
            )
        )
        return univariate_gmm

    def log_likelihood(
        self,
        data: np.array
        ) -> float: 
        '''
        Calcualte value, representing the 'validaty' measure (or 'correctness')
        for how good mixture explanes given data behaviour.
        '''
        from Algorithms import log_likelihood
        res = log_likelihood(
            self.probs, 
            self.mus, 
            self.sigmas, 
            data
            )
        return res
    
    @property
    def volatility(
            self
            ):
        """
        Calculate volatility.
        
        If mixture parameters would ever be represented other then 1D matrix, 
        then it's needed to implement next change:
            axis = 1 if self.class_probability.ndim > 1 else 0
        """
        if not hasattr(self, 'volat_comps'):
            from numpy import sum as numpy_sum
            def diffusion_component(self):
                total_mu = numpy_sum(
                    self.mus * self.probs,
                    axis=0, 
                    keepdims=True
                    )
                res = numpy_sum(
                    (self.mus-total_mu)**2 * self.probs, 
                    axis=0, 
                    keepdims=True
                    )
                return res

            def dynamic_component(self):
                res = numpy_sum(
                    self.sigmas**2 * self.probs, 
                    axis=0, 
                    keepdims=True
                    )
                return res
            self.volat_comps = {
                'diffusion_comp': diffusion_component(self),
                'dynamic_comp': dynamic_component(self)
            }
        return self.volat_comps
    
    #---------------------------------------------------------------------------

    def EM_iterative(
            self,
            dataset: np.array,
            n_iterations: int
            ) -> tuple:
        '''
        Predicts mixture parameters such variables weights, means and 
        dispersion. Automaticly returnes the log-likelihood for parameters
        predictions.
        
        Estimate parameters based on amount of iterations.
        '''
        from Algorithms import EM_iter

        # class_probs, mus, sigmas, log_lh,
        predictions = EM_iter(
            dataset, 
            n_iterations, 
            self.probs, 
            self.mus, 
            self.sigmas
            )
        return predictions
    
    def EM_adaptive(
            self,
            dataset: np.array,
            accuracy: float
            ):
        '''
        Predicts mixture parameters such variables weights, means and 
        dispersion. Automaticly returnes the log-likelihood for parameters
        predictions.

        Estimate parameters based on limiting convergens distance, i.e. 
        accuracy.
        '''
        from Algorithms import EM_adap
        predictions = EM_adap(
            dataset, 
            accuracy, 
            self.probs, 
            self.mus, 
            self.sigmas
            )
        return predictions
    
    def EM_sieving(
            self,
            dataset,
            iter_initial,
            num_candid,
            num_best_candid,
            accur_final,
            random_seed
    ):
        from Algorithms import EM_sieved
        candidates = EM_sieved(
            dataset,
            num_params=self.num_comps,
            num_iter_candid_initial=iter_initial,
            n_candid=num_candid,
            n_best_candid=num_best_candid,
            accur_best_candid=accur_final,
            random_seed=random_seed
        )
        return candidates
    #---------------------------------------------------------------------------
    def plot(self):
        from plotly.express import scatter
        fig = scatter(x=self.samples,
                      y=self.samples_probs)
        return fig