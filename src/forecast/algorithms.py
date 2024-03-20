class FloatingWindow:
    def __init__(self, size, step, sort=None) -> None:
        self.size = size
        self.step = step
        self.sort_method = sort
        pass


class Quant:
    from typing import List
    from scipy.stats import ecdf

    """Make forecast based on quantiles."""

    def __init__(self, window: List[int], K: int) -> None:
        """Generate class object.

        :param window: length and step of floating window.
        :param K: Number of grid points
        """
        self.K = K
        self.window = {"length": window[0], "step": window[1]}

    def fit(self, input):
        from numpy.typing import NDArray
        from scipy.stats import ecdf
        import numpy as np

        # def make_var_series(series):
        #     """Construct variational series of given series."""
        #     increments = []
        #     for i in range(1, len(series)):
        #         increments.append(series[i] - series[i-1])
        #     return increments

        def make_grid(series: NDArray) -> NDArray:
            """Construct grid to evaluating empirical distribution function."""
            x_K = series[-1]
            grid = series[::h]
            if len(grid) < self.K:
                grid = np.append(grid, x_K)
            return grid

        h = (self.window["length"] - 1) // (self.K - 1)
        i = 0
        # Main window shifting part
        for counts in input[i : self.window["length"] + i : self.window["step"]]:
            window = ecdf(counts)
            order_stats = window.cdf.quantiles
            grid = make_grid(order_stats)
            T = np.apply_along_axis(window.cdf.evaluate, 0, grid)
            print(T)

        pass
