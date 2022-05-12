import numpy as np
import pandas as pd
import networkx as nx

import plotly
import plotly.graph_objects as go
import plotly.express as px


def __trace_nodes(g, sep_nodes_by_color, colors, size):
    """
    Trace nodes to draw them on 3D plot

    Parameters
    ----------
    g: nx.Graph object
        graph object with N nodes
    sep_nodes_by_color: bool or np.ndarray
        contains list of length N where each number
    colors: str or list
        contains list of colors
    size: int or list
        size of node

    Returns
    -------
    traced_nodes: np.array
        traced_nodes

    === Example ===
    Let's consider graph with 3 nodes, and we need to colorize first two by 'fuchsia' color
    and last one by 'lightgreen' color, therefore:
    sep_nodes_by_color = [0, 0, 1];
    colors = ['fuchsia', 'lightgreen']
    If you want to control color opacity - use rgba format: 'rgba(255, 255, 255, 0.)'
    """
    traced_nodes = []
    node_text = [f'{n}' for n in g.nodes()]
    coords = nx.get_node_attributes(g, 'coords')

    x = np.array([coords[i][0] for i in coords.keys()])
    y = np.array([coords[i][1] for i in coords.keys()])
    z = np.array([coords[i][2] for i in coords.keys()])

    if sep_nodes_by_color is not False:
        if not isinstance(sep_nodes_by_color, np.ndarray):
            raise TypeError('sep_nodes_by_color should have numpy.ndarray data type')

        for i, s in zip(np.unique(sep_nodes_by_color), size):
            traced_nodes.append(
                go.Scatter3d(x=x[sep_nodes_by_color == i], y=y[sep_nodes_by_color == i], z=z[sep_nodes_by_color == i],
                             mode='markers',
                             marker=dict(symbol='circle', size=s, color=colors[i]),
                             opacity=0.9,
                             text=node_text, hoverinfo='text')
            )
    else:
        traced_nodes.append(
            go.Scatter3d(x=x, y=y, z=z,
                         mode='markers',
                         marker=dict(symbol='circle', size=size, color=colors),
                         opacity=0.9,
                         text=node_text, hoverinfo='text')
        )

    return traced_nodes


def __edges_coords(coords, g_edges):
    """
    Create coordinates of edges

    Parameters
    ----------
    coords: dict
        dict of coordinates of nodes: {node: (x,y,z), ...}
    g_edges: np.array
        Array containing edges

    Returns
    -------
    x_edges, y_edges, z_edges: np.array
        For edges [(n1, n2), ...] 'x_edges' has form [n1_x, n2_x, None, ...]
    """
    x_edges, y_edges, z_edges = [], [], []

    for edge in g_edges:
        x_coords = [coords[edge[0]][0], coords[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [coords[edge[0]][1], coords[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [coords[edge[0]][2], coords[edge[1]][2], None]
        z_edges += z_coords

    return x_edges, y_edges, z_edges


def __trace_edges(g, sep_edges_by_color, colors, width):
    """
    Trace edges to draw them on 3D plot

    Parameters
    ----------
    g: nx.Graph object
        graph object with N nodes
    sep_edges_by_color: bool or np.ndarray
        contains list of ...
    colors: str or list
        contains list of colors
    width: int or list
        width of edge

    === Example ===
    if sep_edges_by_color = False;
    colors = 'rgba(74, 54, 54, 0.7)'; width = 2
    """
    trace_edges = []

    coords = nx.get_node_attributes(g, 'coords')
    g_edges = np.array(list(g.edges()))

    if sep_edges_by_color is not False:
        if not isinstance(sep_edges_by_color, np.ndarray):
            raise TypeError('sep_edges_by_color should have numpy.ndarray data type')

        for i, w in zip(np.unique(sep_edges_by_color), width):
            x_edges, y_edges, z_edges = __edges_coords(coords, g_edges[sep_edges_by_color == i])
            trace_edges.append(
                go.Scatter3d(x=x_edges, y=y_edges, z=z_edges,
                             mode='lines',
                             line=dict(color=colors[i], width=w),
                             hoverinfo='none')
            )

    else:
        x_edges, y_edges, z_edges = __edges_coords(coords, g_edges)
        trace_edges.append(
            go.Scatter3d(x=x_edges, y=y_edges, z=z_edges,
                         mode='lines',
                         line=dict(color=colors, width=width),
                         hoverinfo='none')
        )

    return trace_edges


def __draw_cube(cube_dim, e_color='rgba(20, 20, 20, 0.9)', line_width=3):
    """
    Draw cube which has unit dimensions [d x d x d] where d = cube_dim
    """
    d = cube_dim
    trace = []

    x = [0, d, d, 0, 0]
    y = [0, 0, d, d, 0]
    z1 = [0, 0, 0, 0, 0]
    z2 = [d, d, d, d, d]

    # trace top and bottom
    trace.append(go.Scatter3d(x=x, y=y, z=z1, mode='lines',
                              line=dict(color=e_color, width=line_width), hoverinfo='none'))
    trace.append(go.Scatter3d(x=x, y=y, z=z2, mode='lines',
                              line=dict(color=e_color, width=line_width), hoverinfo='none'))

    # trace 4 vertical lines
    for i in range(4):
        trace.append(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[z1[0], z2[0]], mode='lines',
                                  line=dict(color=e_color, width=line_width), hoverinfo='none'))

    return trace


def __setup_layout(text_title):
    # set plot properties
    axis = dict(showbackground=False, showline=False, zeroline=False,
                showgrid=False, showticklabels=False, title='')
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1, y=2, z=1.2)
    )
    layout = go.Layout(title=text_title,
                       scene_camera=camera,
                       width=1080, height=1080,
                       showlegend=False,
                       scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                       margin=dict(t=100),
                       hovermode='closest')

    return layout


def draw_cnt_graph(g, cube_dim, full_graph, **kwargs):
    """
    Draw main graph
    """
    data = []

    # draw cube
    data.extend(__draw_cube(cube_dim))

    if full_graph:
        # add nodes and edges
        appearance = kwargs['appearance']
        data.extend(__trace_nodes(g, sep_nodes_by_color=False, colors=appearance['nodes']['colors'],
                                  size=appearance['nodes']['size']))
        data.extend(__trace_edges(g, sep_edges_by_color=False, colors=appearance['edges']['colors'],
                                  width=appearance['edges']['width']))
    else:
        nodes_to_go = kwargs['nodes_to_go']
        appearance = kwargs['appearance']

        # define colorscale - to separate nodes involved in voltage measurements from other nodes
        node_colorscale = []
        for n in g.nodes():
            if g.nodes[n]['on_surface'] == 1:
                if n in nodes_to_go: node_colorscale.append(1)
                else: node_colorscale.append(2)
            else: node_colorscale.append(0)

        # add nodes and edges
        data.extend(__trace_nodes(g, sep_nodes_by_color=np.array(node_colorscale),
                                  colors=appearance['nodes']['colors'],
                                  size=appearance['nodes']['size']))
        data.extend(__trace_edges(g, sep_edges_by_color=False, colors=appearance['edges']['colors'],
                                  width=appearance['edges']['width']))

    fig = go.Figure(data=data, layout=__setup_layout(text_title=appearance['text']))
    fig.show()


def __draw_cylinder(r, h, p0, angles=False):
    """
    Draw cylinder (which represent form of damage)

    Parameters
    ----------
    r: float
        radius of the cylinder
    h: float
        half of height of the cylinder
    p0: tuple
        base point (center point of the cylinder)
    nt, nv: int
        define quality of the mesh grid
    """
    # define cylinder coordinates
    nt = 90
    nv = 50
    theta = np.linspace(0, 2 * np.pi, nt)
    v = np.linspace(0 - h, 0 + h, nv)
    theta, v = np.meshgrid(theta, v)
    x = r * np.cos(theta)  # + p0[0]
    y = r * np.sin(theta)  # + p0[1]
    z = v

    if angles:
        a, b, g = angles
        euler_matrix = np.array([
            [np.cos(b)*np.cos(g), np.sin(a)*np.sin(b)*np.cos(g) - np.cos(a)*np.sin(g), np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.sin(g)],
            [np.cos(b)*np.sin(g), np.sin(a)*np.sin(b)*np.sin(g) + np.cos(a)*np.cos(g), np.cos(a)*np.sin(b)*np.sin(g) - np.sin(a)*np.cos(g)],
            [-np.sin(b), np.sin(a)*np.cos(b), np.cos(a)*np.cos(b)]
        ])

        rot = np.dot(euler_matrix,
                     np.array([x.ravel(), y.ravel(), z.ravel()]))

        x = rot[0, :].reshape(x.shape)
        y = rot[1, :].reshape(y.shape)
        z = rot[2, :].reshape(z.shape)

    colorscale = [[0, 'blue'], [1, 'blue']]
    cyl_trace = go.Surface(x=x+p0[0], y=y+p0[1], z=z+p0[2], colorscale=colorscale, opacity=0.1,
                           showlegend=False, showscale=False, hoverinfo='none')

    return cyl_trace


def draw_fracture_form(r, h, p0, angles):
    data = []
    data.extend(__draw_cube(cube_dim=0.5))
    data.append(__draw_cylinder(r, h, p0, angles))

    fig = go.Figure(data=data, layout=__setup_layout('Fracture form: cylinder'))
    fig.show()



