import numpy as np
from scipy.optimize import leastsq
import plotly.graph_objects as go
from scipy.stats import norm
from plotly.subplots import make_subplots


def find_frequency(data, sampling_rate):
    """"""
    n = len(data)
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    spectrum = abs(fft_result)
    idx = np.argmax(spectrum[1:]) + 1  # Set initial frequency
    freq = freqs[idx]
    return abs(freq)


def harmonic_approximation(data, time, harmonics_num=4, smooth=0, title="", limit=-1):
    if smooth:

        def smoothing(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode="same")
            return y_smooth

        data = smoothing(data, smooth)

    fig = make_subplots(rows=3, cols=1)
    # fig = make_subplots(rows=2, cols=1)
    data_orig = data
    t = np.array(range(len(data)))
    params = []
    harmonical_signal = np.zeros(len(data))

    for _ in range(harmonics_num):
        # guess_mean = np.mean(data)
        guess_phase = 0
        guess_freq = find_frequency(data, len(data)) * np.pi * 2 / len(data)
        guess_amp = max(data) - min(data)

        def optimize_func(x):
            # return x[0] * np.sin(x[1] * t + x[2]) + x[3] - data
            return x[0] * np.sin(x[1] * t + x[2]) - data

        params_sin = leastsq(
            # optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean]
            optimize_func,
            [guess_amp, guess_freq, guess_phase],
        )[0]

        params.append(params_sin)
        # est_amp, est_freq, est_phase, est_mean = params_sin
        est_amp, est_freq, est_phase = params_sin

        # data_fit = est_amp * np.sin(est_freq * t + est_phase) + est_mean
        data_fit = est_amp * np.sin(est_freq * t + est_phase)
        data = data - data_fit
        harmonical_signal += data_fit

    # Approximation visualization
    fig.add_trace(
        go.Scatter(x=time[:limit], y=data_orig[:limit], marker=dict(color="#3058B0")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time[:limit], y=harmonical_signal[:limit], marker=dict(color="#FF7F50")
        ),
        row=1,
        col=1,
    )
    # fig.add_annotation(
    #     xref="x domain",
    #     yref="y domain",
    #     x=0.5,
    #     y=1.2,
    #     showarrow=False,
    #     font=dict(size=22),
    #     text=f"<b>Approximation ({harmonics_num} harmonics)<b>",
    #     row=1,
    #     col=1,
    # )
    #
    # Residuals
    fig.add_trace(
        go.Scatter(x=time[:limit], y=data[:limit], marker=dict(color="purple")),
        row=2,
        col=1,
    )
    # fig.add_annotation(
    #     xref="x domain",
    #     yref="y domain",
    #     x=0.5,
    #     y=1.2,
    #     showarrow=False,
    #     font=dict(size=22),
    #     text="<b>Residuals<b>",
    #     row=2,
    #     col=1,
    # )
    # Histogramm
    fig.add_trace(
        go.Histogram(
            x=data[:limit], histnorm="probability density", marker=dict(color="#3058B0")
        ),
        row=3,
        col=1,
    )
    x = np.linspace(min(data) * 1.2, max(data) * 1.2, 100)
    fig.add_trace(
        go.Scatter(
            x=x[:limit],
            y=norm.pdf(x[:limit], *norm.fit(data[:limit])),
            marker=dict(color="#FF7F50"),
        ),
        row=3,
        col=1,
    )

    def cdf(x):
        return norm.cdf(x, loc=norm.fit(data)[0], scale=norm.fit(data)[1])

    # pval = kstest(data, cdf=cdf).pvalue
    # fig.add_annotation(
    #     xref="x domain",
    #     yref="y domain",
    #     x=0.5,
    #     y=1.3,
    #     showarrow=False,
    #     font=dict(size=22),
    #     text="<b>Residuals histogram<b>",
    #     row=3,
    #     col=1,
    # )
    # fig.add_annotation(
    #     xref="x domain",
    #     yref="y domain",
    #     x=0.5,
    #     y=1.165,
    #     showarrow=False,
    #     font=dict(size=20),
    #     text=f"Normal test p-value = {pval:.3f}",
    #     row=3,
    #     col=1,
    # )

    fig.update_layout(
        width=1200,
        height=500,
        # title=dict(font=dict(size=20), text=f"<b>{title}<b>"),
        showlegend=False,
    )
    return fig, data, params
