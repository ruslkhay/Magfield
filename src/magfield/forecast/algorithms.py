import numpy as np


class FloatingWindow:
    def __init__(self, size, step, sort=None) -> None:
        self.size = size
        self.step = step
        self.sort_method = sort


class EDF:
    """Make forecast based on estimated distribution function."""

    from typing import List

    def __init__(self, window: List[int], M: int = 0, ord_quant: int = 0) -> None:
        """Generate class object.

        :param window: length and step of floating window.
        :param K: Number of grid points
        """
        self.quants = []
        self.probs = []
        self.M = M
        self.quant_probs = np.linspace(0, 1, ord_quant, endpoint=False)[1:]
        self.window = {"length": window[0], "step": window[1]}

    def fit(self, input):
        from tqdm.notebook import tqdm
        from scipy.stats import ecdf
        import numpy as np
        from scipy.stats.mstats import mquantiles

        i = 0
        # Main window shifting part
        for i in tqdm(range(len(input) - self.window["length"] + 1)):
            counts = input[i : self.window["length"] + i + 1]
            if self.quant_probs.size > 0:
                self.quants.append(list(mquantiles(counts, prob=self.quant_probs)))
            if self.M:
                distrib_func = ecdf(counts).cdf
                prob = np.linspace(0, 1, self.M, endpoint=False)[1:]
                quants = mquantiles(counts, prob=prob)
                self.probs.append(list(distrib_func.evaluate(quants)))


# class Quant:
#     """Make forecast based on quantiles."""

#     from typing import List

#     def __init__(self, window: List[int], order: int) -> None:
#         """Generate class object.

#         :param window: length and step of floating window.
#         :param order: Order of quantiles. For example if `order=10` then
#         we have business with deciles.
#         """
#         self.order = order
#         self.window = {"length": window[0], "step": window[1]}

#     def fit(self, input):
#         from tqdm.notebook import tqdm
#         from scipy.stats.mstats import hdquantiles
#         from scipy.stats import ecdf

#         probs = np.linspace(0, 1, self.order)
#         quants = []
#         i = 0
#         # Main window shifting part
#         for i in tqdm(range(len(input) - self.window["length"] + 1)):
#             counts = input[i : self.window["length"] + i]
#             order_stats = window.cdf.quantiles
#             quants.append(hdquantiles(order_stats, prob=probs))
#             i += self.window["step"]
#         self.T = np.array(quants)
