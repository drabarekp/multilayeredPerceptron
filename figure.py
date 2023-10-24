import plotly.graph_objects as go


def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]


def get_figure(data, layers):

    layer_count = len(layers)
    placements_x = [0.1 + 0.8 * x / (layer_count - 1) for x in range(layer_count)]
    print(placements_x)

    max_layer = max(layers)
    placements_y = [0.15 + 0.8 * y / (max_layer - 1) for y in range(max_layer)]
    print(placements_y)

    node_x = []
    node_y = []
    for lx in range(layer_count):
        for ly in range(layers[lx]):
            node_x.append(placements_x[lx])
            node_y.append(placements_y[ly])

    index = 0
    edge_x = []
    edge_y = []
    for i in range(layer_count - 1):
        for start in range(index, index + layers[i]):
            for end in range(index + layers[i], index + layers[i] + layers[i + 1]):
                edge_x.append(node_x[start])
                edge_x.append((2 * node_x[start] + node_x[end]) / 3)
                edge_x.append(node_x[end])

                edge_y.append(node_y[start])
                edge_y.append((2 * node_y[start] + node_y[end]) / 3)
                edge_y.append(node_y[end])

        index += layers[i]

    edge_width = flatten_comprehension(flatten_comprehension(data[0]))
    edge_change = flatten_comprehension(flatten_comprehension(data[2]))

    node_value = flatten_comprehension(data[1])
    node_change = flatten_comprehension(data[3])

    edge_trace = [go.Scatter(
            x=edge_x[3 * i:3 * i + 3], y=edge_y[3 * i:3 * i + 3],
            mode='lines+text',
            text=['', '<br>{:4.2f}<br>(Δ{:+4.2f})'.format(edge_width[i], edge_change[i]), ''],
            hoverinfo='none',
            textposition='bottom center',
            line=dict(width=abs(edge_width[i]), color='red' if edge_width[i] < 0 else 'green'))
        for i in range(len(edge_width))]

    node_trace = [go.Scatter(
        x=[node_x[i]], y=[node_y[i]],
        mode='markers+text',
        hoverinfo='none',
        text='bias:<br>{:4.2f} (Δ{:+4.2f})'.format(node_value[i], node_change[i])
        if i >= layers[0] else '<br>IN {}'.format(i + 1),
        textposition='bottom center',
        marker=dict(
            color='red' if node_value[i] < 0 else 'green' if node_value[i] > 0 else 'azure',
            size=15,
            line_width=1)
    ) for i in range(len(node_change))]

    edge_trace.extend(node_trace)
    fig = go.Figure(data=edge_trace,
                    layout=go.Layout(
                        title='Bias and weights distribution against iterations',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="S. Górski, P. Drabarek",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    # fig.show()
    return fig
