import numpy as np
import plotly.graph_objects as go

REGRESSION_POINTS = 2000
REGRESSION_OVERHEAD = 0.2


def plot_regression(train_input, train_output, test_input, test_output, model):
    x_range = np.linspace(min(train_input.min(), test_input.min()) - REGRESSION_OVERHEAD,
                          max(train_input.max(), test_input.max()) + REGRESSION_OVERHEAD, REGRESSION_POINTS)
    y_range = np.array([model.operation([x_range[i]]) for i in range(REGRESSION_POINTS)]).reshape((-1))

    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=test_input.reshape((-1)),
            y=test_output.reshape((-1)),
            mode='markers',
            marker=dict(color='gray',
                        opacity=0.5),
            name='Testing set',
            showlegend=True
        ))
    fig.add_traces(
        go.Scatter(
            x=train_input.reshape((-1)),
            y=train_output.reshape((-1)),
            mode='markers',
            marker=dict(color='blue',
                        opacity=1),
            name='Training set',
            showlegend=True
        ))
    fig.add_traces(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='markers',
            marker=dict(color='red',
                        size=3,
                        opacity=1),
            name='Regression fit',
            showlegend=True
        ))
    fig.show()


def plot_classification(train_input, train_output, test_input, test_output, model):
    plot_classification_points(train_input, train_output, test_input, test_output)
    plot_classification_score(train_input, train_output, test_input, test_output, model)
    for i in range(train_output.shape[1]):
        plot_classification_score_by_class(train_input, train_output, test_input, test_output, model, i + 1)


def plot_classification_points(train_input, train_output, test_input, test_output):
    output_length = train_output.shape[0]
    train_label = np.argmax(train_output, axis=1)
    test_label = np.argmax(test_output, axis=1)

    plots = []
    colors = np.max(train_label) + 1
    for color in range(colors):
        plot_train = np.array([train_input[i] for i in range(output_length) if train_label[i] == color])
        plot_test = np.array([test_input[i] for i in range(output_length) if test_label[i] == color])
        # train_score = np.array([train_output[i][color] for i in range(output_length) if train_label[i] == color])

        plots.append(
            go.Scatter(
                x=plot_train[:, 0], y=plot_train[:, 1],
                marker=dict(
                    color='red' if color == 0 else 'green' if color == 1 else 'blue',
                    symbol='circle' if color == 0 else 'diamond' if color == 1 else 'square',
                    size=15,
                    line_width=1),
                name='Training set',
                mode='markers',
                hovertext=f'Class {color + 1} (training set)'
            ))
        plots.append(
            go.Scatter(
                x=plot_test[:, 0], y=plot_test[:, 1],
                marker=dict(
                    color='red' if color == 0 else 'green' if color == 1 else 'blue',
                    symbol='circle-dot' if color == 0 else 'diamond-dot' if color == 1 else 'square-dot',
                    size=15,
                    line_width=1),
                name='Testing set',
                mode='markers',
                hovertext=f'Class {color + 1} (testing set)'
            ))

    fig = go.Figure(data=plots,
                    layout=go.Layout(
                        title='Correct division of points among classes',
                        titlefont_size=16
                    ))
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    fig.show()


def plot_classification_score(train_input, train_output, test_input, test_output, model):
    output_length = train_output.shape[0]
    train_label = np.argmax(train_output, axis=1)
    test_label = np.argmax(test_output, axis=1)

    plots = []
    colors = np.max(train_label) + 1
    for color in range(colors):
        plot_train = np.array([train_input[i] for i in range(output_length) if train_label[i] == color])
        plot_test = np.array([test_input[i] for i in range(output_length) if test_label[i] == color])
        train_score = np.array([model.operation(train_input[i])[train_label[i]] for i in range(output_length)
                                if train_label[i] == color])
        test_score = np.array([model.operation(test_input[i])[test_label[i]] for i in range(output_length)
                               if test_label[i] == color])

        plots.append(go.Scatter(
            x=plot_train[:, 0],
            y=plot_train[:, 1],
            marker=dict(
                cmax=1,
                cmin=0,
                color=train_score,
                colorscale='Greys',
                colorbar=dict(
                    thickness=20),
                symbol='circle' if color == 0 else 'diamond' if color == 1 else 'square',
            ),
            name=f'Class {color + 1} (training set)',
            mode='markers',
            hovertext=train_score
        ))
        plots.append(go.Scatter(
            x=plot_test[:, 0],
            y=plot_test[:, 1],
            marker=dict(
                cmax=1,
                cmin=0,
                color=test_score,
                colorscale='Greys',
                colorbar=dict(
                    thickness=20),
                symbol='circle-dot' if color == 0 else 'diamond-dot' if color == 1 else 'square-dot',
            ),
            name=f'Class {color + 1} (testing set)',
            mode='markers',
            hovertext=test_score
        ))

    fig = go.Figure(data=plots,
                    layout=go.Layout(
                        title='Score assigned to the correct class of each point',
                        titlefont_size=16
                    ))
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    fig.show()


def plot_classification_score_by_class(train_input, train_output, test_input, test_output, model, class_number):
    output_length = train_output.shape[0]
    train_label = np.argmax(train_output, axis=1)
    test_label = np.argmax(test_output, axis=1)

    plots = []
    colors = np.max(train_label) + 1
    for color in range(colors):
        plot_train = np.array([train_input[i] for i in range(output_length) if train_label[i] == color])
        plot_test = np.array([test_input[i] for i in range(output_length) if test_label[i] == color])
        train_score = np.array([model.operation(train_input[i])[class_number - 1] for i in range(output_length)
                                if train_label[i] == color])
        test_score = np.array([model.operation(test_input[i])[class_number - 1] for i in range(output_length)
                               if test_label[i] == color])

        plots.append(go.Scatter(
            x=plot_train[:, 0],
            y=plot_train[:, 1],
            marker=dict(
                cmax=1,
                cmin=0,
                color=train_score,
                colorscale='Greys',
                colorbar=dict(
                    thickness=20),
                symbol='circle' if color == 0 else 'diamond' if color == 1 else 'square',
            ),
            name=f'Class {color + 1} (training set)',
            mode='markers',
            hovertext=train_score
        ))
        plots.append(go.Scatter(
            x=plot_test[:, 0],
            y=plot_test[:, 1],
            marker=dict(
                cmax=1,
                cmin=0,
                color=test_score,
                colorscale='Greys',
                colorbar=dict(
                    thickness=20),
                symbol='circle-dot' if color == 0 else 'diamond-dot' if color == 1 else 'square-dot',
            ),
            name=f'Class {color + 1} (testing set)',
            mode='markers',
            hovertext=test_score
        ))

    fig = go.Figure(data=plots,
                    layout=go.Layout(
                        title=f'Score assigned to class no. {class_number}',
                        titlefont_size=16
                    ))
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    fig.show()
