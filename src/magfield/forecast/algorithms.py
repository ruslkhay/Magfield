import numpy as np


class FloatingWindow:
    def __init__(self, size, step, sort=None) -> None:
        self.size = size
        self.step = step
        self.sort_method = sort


class EDF:
    """Make forecast based on estimated distribution function."""

    from typing import List

    def __init__(self, window: List[int], K: int) -> None:
        """Generate class object.

        :param window: length and step of floating window.
        :param K: Number of grid points
        """
        self.K = K
        self.window = {"length": window[0], "step": window[1]}

    def fit(self, input):
        from numpy.typing import NDArray
        from tqdm.notebook import tqdm
        from scipy.stats import ecdf
        import numpy as np

        def make_grid(series: NDArray, h) -> NDArray:
            """Construct grid to evaluating empirical distribution function."""
            x_K = series[-1]
            grid = series[::h]
            if len(grid) < self.K:
                grid = np.append(grid, x_K)
            return grid

        h = (self.window["length"] - 1) // (self.K - 1)
        i = 0
        quants = []
        # Main window shifting part
        for i in tqdm(range(len(input) - self.window["length"] + 1)):
            counts = input[i : self.window["length"] + i + 1]
            window = ecdf(counts)
            # order_stats = window.cdf.quantiles
            T = np.apply_along_axis(window.cdf.evaluate, 0, counts[::h]).tolist()
            # T = np.apply_along_axis(window.cdf.evaluate, 0, grid)
            quants.append(T)
            i += self.window["step"]
        self.T = np.array(quants)


class Quant:
    """Make forecast based on quantiles."""

    from typing import List

    def __init__(self, window: List[int], order: int) -> None:
        """Generate class object.

        :param window: length and step of floating window.
        :param order: Order of quantiles. For example if `order=10` then
        we have business with deciles.
        """
        self.order = order
        self.window = {"length": window[0], "step": window[1]}

    def fit(self, input):
        from tqdm.notebook import tqdm
        from scipy.stats.mstats import hdquantiles
        from scipy.stats import ecdf

        probs = np.linspace(0, 1, self.order)
        quants = []
        i = 0
        # Main window shifting part
        for i in tqdm(range(len(input) - self.window["length"] + 1)):
            counts = input[i : self.window["length"] + i]
            window = ecdf(counts)
            order_stats = window.cdf.quantiles
            quants.append(hdquantiles(order_stats, prob=probs))
            i += self.window["step"]
        self.T = np.array(quants)
