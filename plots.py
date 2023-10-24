import numpy as np
import plotly.express as px
import plotly.graph_objects as go

REGRESSION_POINTS = 500
REGRESSION_OVERHEAD = 0.5


def plot_regression_points(data_input, expected_output):
    fig = px.scatter(x=data_input.reshape((-1)), y=expected_output.reshape((-1)), opacity=0.65)
    fig.show()


# expected_output can be replaced with output to see mappings of test points
def plot_regression_line(data_input, expected_output, model):
    x_range = np.linspace(data_input.min() - REGRESSION_OVERHEAD,
                          data_input.max() + REGRESSION_OVERHEAD, REGRESSION_POINTS)
    y_range = np.array([model.operation([x_range[i]]) for i in range(REGRESSION_POINTS)]).reshape((-1))

    fig = px.scatter(x=data_input.reshape((-1)), y=expected_output.reshape((-1)), opacity=0.65)
    fig.add_traces(
        go.Scatter(
            x=x_range,
            y=y_range,
            name='Regression Fit')
    )
    fig.show()


def plot_classification_points(data_input, expected_output):
    output_length = expected_output.shape[0]
    output_label = np.argmax(expected_output, axis=1)

    plots = []
    class_number = np.max(output_label) + 1
    for color in range(class_number):
        color_input = np.array([data_input[i] for i in range(output_length) if output_label[i] == color])
        # color_score = np.array([expected_output[i][color] for i in range(output_length) if output_label[i] == color])

        plot = go.Scatter(
            x=color_input[:, 0], y=color_input[:, 1],
            mode='markers',
            hoverinfo='none',
            marker=dict(
                color='red' if color == 0 else 'green' if color == 1 else 'blue',
                symbol='circle' if color == 0 else 'diamond' if color == 1 else 'square',
                size=15,
                line_width=1)
            )
        plots.append(plot)

    fig = go.Figure(data=plots)
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    fig.show()


# data_input and expected_output can be replaced with test data to see values of test points
def plot_classification_score(data_input, expected_output, model):
    output_length = expected_output.shape[0]
    output_label = np.argmax(expected_output, axis=1)
    score = np.array([model.operation(data_input[i])[output_label[i]] for i in range(output_length)])

    fig = px.scatter(
        x=data_input[:, 0], y=data_input[:, 1],
        color=score, color_continuous_scale='Greys',
        symbol=output_label, symbol_map={0: 'circle', 1: 'diamond', 2: 'square'},
        labels={'symbol': 'label', 'color': 'score of <br>correct class'}
    )
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    fig.show()


# data_input and expected_output can be replaced with test data to see values of test points
def plot_classification_score_by_class(data_input, expected_output, model, class_number):
    output_length = expected_output.shape[0]
    output_label = np.argmax(expected_output, axis=1)
    score = np.array([model.operation(data_input[i])[class_number] for i in range(output_length)])

    fig = px.scatter(
        x=data_input[:, 0], y=data_input[:, 1],
        color=score, color_continuous_scale='Greys',
        symbol=output_label, symbol_map={0: 'circle', 1: 'diamond', 2: 'square'},
        labels={'symbol': 'label', 'color': f'score of <br>class {class_number}'}
    )
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    fig.show()
