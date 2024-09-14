# from algorithms import *

import scipy.stats as stat
import pandas as pd
import plotly.express as ple

if __name__ == "__main__":
    data = pd.read_csv("Data/nice_jan_march.csv", nrows=10_000)
    bx = data["BX"].values
    ple.line(bx).show()
    print(type(bx))
    distr = stat.ecdf(bx)  # type: ignore
    # print(distr.cdf.quantiles)
    ple.line(distr.cdf.quantiles).show()
