from plotly.subplots import make_subplots
import plotly.graph_objects as go


def long_plot(series, title, n, dates, showlegend=True):
    """Represent long time series.
    Slice it down into `n` sections and put one under each other.
    :param series: 1D data to visualize.
    :param dates: Custom units for X-axis.
    :param title: Custom name of the figure.
    :param n: Number of subplots that would be one figure. Each of them
    represent one n'th of original series."""

    fig = make_subplots(rows=n, cols=1)

    # Calculate the length of each partition
    total_length = len(series)
    partition_length = total_length // n

    # Create each part and add to the figure
    for i in range(n):
        start_index = i * partition_length
        # To handle the last segment which may include extra elements
        end_index = (i + 1) * partition_length if (i + 1) < n else total_length

        # Slice the series data for this part
        part = series[start_index:end_index]

        # Add the plot for the current part
        fig.append_trace(
            go.Scatter(
                x=dates[start_index:end_index],  # Use the corresponding dates
                y=part,
                name=f"Part {i + 1}",
            ),
            row=i + 1,
            col=1,
        )

    # Update the layout of the figure
    fig.update_layout(
        height=300 * n, width=1200, title_text=title, showlegend=showlegend
    )

    return fig
