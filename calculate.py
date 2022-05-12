import numpy as np
import pandas as pd

import networkx as nx
import scipy.sparse.linalg as lg

import os
import random
import itertools
from tqdm import tqdm
from copy import deepcopy


# ==================================
# === (1) CNT graph construction ===
# ==================================

def get_node_inside_cell(df, split):
    """
    Find center node in the cells into which the surface is divided
    (each plane of RVE is supposed to be divided on [n x n] cells
    and this func. finds nodes in the center of cells for given plane of RVE)

    Parameters
    ----------
    df: pd.DataFrame
        dataframe with nodes and coordinates lying on curtain plane
    split: list
        interval [0, RVE_length] divided on 'n' parts
        for interval [0, 0.5] and n=2 ---> [0, 0.25, 0.5]

    Returns
    -------
    nodes: list
        nodes lying in the center of cells
    """
    nodes = []
    coord1, coord2 = df.columns[1:]
    step = split[1] - split[0]

    points_coords = np.meshgrid(split[:-1], split[:-1])
    for x1, x2 in zip(points_coords[0].flatten(), points_coords[1].flatten()):
        id_nodes_in_cell = list(
            df[(df[coord1] > x1) & (df[coord1] < x1 + step) & (df[coord2] > x2) & (df[coord2] < x2 + step)].index)

        idx = df.loc[id_nodes_in_cell, :].sort_values(by=[coord1, coord2]).index[len(id_nodes_in_cell) // 2]
        nodes.append(df.loc[idx, :].n.item())

    return nodes


def get_cnt_graph(path_nodes, path_edges, nodes_to_go_mode, del_b_connections=True):
    """
    Create CNT graph from raw data

    Parameters
    ----------
    path_nodes: str
        path to csv file which contains information about nodes (coordinates)
    path_edges: str
        path to csv file which contains information about edges (conductivity and type)
    nodes_to_go_mode: str
        the way how to consider nodes for measurements
        Options:
        'all' - consider all nodes on the surface and take each pair of them
        'grid n' - split surfaces [n x n] parts and take one node in each cell
    del_b_connections: bool
        boundary ('b') connections can be removed from edges (the default is True)

    Returns
    -------
    g: nx.Graph
        output CNT graph
    nodes_to_go: list
        contains list of nodes on surface for measurements
    pairs_to_go: list
        contains list of pairs from 'nodes_to_go'
    """
    print('CNT-graph -> building ...')

    # get input data
    df_nodes = pd.read_csv(path_nodes, names=["x", "y", "z"])
    df_nodes['on_surface'] = [0] * df_nodes.shape[0]  # add column 'on_surface'
    df_edges = pd.read_csv(path_edges, names=["node1", "node2", "G", "type"])

    # data comes from Matlab where indexing starts from 1
    df_edges.loc[:, ['node1', 'node2']] = df_edges.loc[:, ['node1', 'node2']] - 1

    # check if graph has multiple connections
    df_edges_shape = df_edges.shape
    df_edges.drop_duplicates(subset=['node1', 'node2'], inplace=True)
    assert df_edges_shape == df_edges.shape, 'GRAPH_ERROR: multiple connections between nodes -> take new graph'

    cnt_nodes = []
    cnt_edges = []

    print('CNT-graph -> nodes processing')
    for i in tqdm(df_nodes.index):
        cnt_nodes.append((i, {'coords': (df_nodes.iloc[i].x, df_nodes.iloc[i].y, df_nodes.iloc[i].z),
                              'on_surface': df_nodes.iloc[i].on_surface}))

    print('CNT_graph -> edges processing')
    for i in tqdm(df_edges.index):
        cnt_edges.append((df_edges.iloc[i].node1, df_edges.iloc[i].node2,
                          {'G': df_edges.iloc[i].G, 'type': df_edges.iloc[i].type}))

    # create CNT graph object
    g = nx.Graph()
    g.add_nodes_from(cnt_nodes)
    g.add_edges_from(cnt_edges)

    # delete boundary ('b') connections
    if del_b_connections:
        print('CNT_graph -> removing b-connections')
        b_edges = [e for e in g.edges() if (g.edges[e]['type'] == 'b')]
        g.remove_edges_from(b_edges)  # remove edges which have type 'b'
        # after deleting 'b'-edges, free nodes can appear (nodes without neighbors, not connected with other nodes)
        free_nodes = [n for n in g.nodes() if len(list(g.neighbors(n))) == 0]
        g.remove_nodes_from(free_nodes)  # remove free nodes

    # after conducting operations above, sub-graphs (or tails) can appear
    # we should take the largest graph and drop other entities
    connected_coms = list(nx.connected_components(g))
    if len(connected_coms) > 1:
        print(f'   graph has {len(connected_coms)} sub-graphs')
        subgraph_nodes_num = [len(subgraph_nodes) for subgraph_nodes in connected_coms]
        idx = np.argmax(subgraph_nodes_num)
        other_sub_graphs = [ns for ns in subgraph_nodes_num if ns != subgraph_nodes_num[idx]]
        sg_sorted = sorted(other_sub_graphs, reverse=True)
        mean_length = np.mean(other_sub_graphs).item()
        print(f'   main graph with {subgraph_nodes_num[idx]} nodes; other: [{sg_sorted[0]}, {sg_sorted[1]}, ...]',
              f'with mean: {round(mean_length, 2)} nodes (total nodes lost: {np.sum(other_sub_graphs)})')
        # remove unuseful sub-graphs
        [g.remove_nodes_from(connected_coms[i]) for i in range(len(connected_coms)) if i != idx]
        assert len(list(nx.connected_components(g))) == 1, 'GRAPH_ERROR: Sub-graphs were not removed'
        print('   tails of tubes are removed')

    # determine the nodes lying on the planes of the cube
    # first, take max and min values of each coordinate of the cube
    n_max = max(df_nodes.x)
    n_min = min(df_nodes.x)
    dx = 1e-15  # deviation which comes from raw data (some nodes can go out of the cube by some tiny value < dx)
    for n in g.nodes():
        x, y, z = g.nodes[n]['coords']
        on_surf = (x > n_max - dx) or (x < n_min + dx) or (y > n_max - dx) or (y < n_min + dx) or \
                  (z > n_max - dx) or (z < n_min + dx)
        if on_surf: g.nodes[n]['on_surface'] = 1

    # define nodes and pairs of them to conduct voltage measurements
    if nodes_to_go_mode.split(sep=' ')[0] == 'all':
        # here we just take all nodes on the surface and for pairs - all combinations of them without repetitions
        # (!) ATTENTION: this case will consume significant computational time -> better to choose 'grid n' method
        nodes_to_go = [n for n in g.nodes() if g.nodes[n]['on_surface'] == 1]
        pairs_to_go = list(itertools.combinations(nodes_to_go, 2))

    elif nodes_to_go_mode.split(sep=' ')[0] == 'grid':
        d = int(nodes_to_go_mode.split(sep=' ')[1]) + 1
        split = np.linspace(n_min, n_max, d)

        nodes_on_surface = [n for n in g.nodes() if g.nodes[n]['on_surface'] == 1]
        df_on_surf = pd.DataFrame(data=[(n, g.nodes[n]['coords'][0],
                                         g.nodes[n]['coords'][1],
                                         g.nodes[n]['coords'][2]) for n in nodes_on_surface],
                                  columns=['n', 'x', 'y', 'z'])

        nodes_to_go = []
        # go by each (6) planes of cube
        for coord in ['x', 'y', 'z']:
            nodes_to_go.extend(
                get_node_inside_cell(df_on_surf[df_on_surf[coord] < n_min + dx].drop(columns=[coord]), split))

            nodes_to_go.extend(
                get_node_inside_cell(df_on_surf[df_on_surf[coord] > n_max - dx].drop(columns=[coord]), split))

        nodes_to_go = list(map(int, nodes_to_go))
        pairs_to_go = list(itertools.combinations(nodes_to_go, 2))

    print('CNT_graph -> done!\n')

    return g, nodes_to_go, pairs_to_go


# ================================
# === (2) Graph check function ===
# ================================


def graph_check(g, nodes_to_go, strict=True):
    """
    Graph check function go through following steps
    1) Drop internal sub-graphs ('nodes_to_go' are not touched or only one included)
    2) (strict) if there are other sub-graphs (> 1) - skip this case: flag = False
    3) (strict) check if all 'nodes_to_go' included into graph
    4) drop tails (they should not start with node from 'nodes_to_go')
    """

    # (1) drop internal sub-graphs
    sg_nodes_list = list(nx.connected_components(g))
    for sg_nodes in sg_nodes_list:
        nodes_inside = [n for n in sg_nodes if n not in nodes_to_go]
        if len(sg_nodes) - len(nodes_inside) < 2: g.remove_nodes_from(nodes_inside)

    # (2)-(3) here we finally end up with one graph or skip this case
    if strict:
        sg_nodes_list = list(nx.connected_components(g))
        if len(sg_nodes_list) != 1: return False
        included = [1 for n in nodes_to_go if n in list(sg_nodes_list)[0]]
        if sum(included) != len(nodes_to_go): return False

    # (4) drop tails (num of neighbors <= 1 and this node doesn't belong to 'nodes_to_go')
    while True:
        free_nodes = [n for n in g.nodes() if (len(list(g.neighbors(n))) <= 1) and (n not in nodes_to_go)]
        if len(free_nodes) == 0: break
        g.remove_nodes_from(free_nodes)

    return True


# ========================================
# === (3) Fracture (cracks) generation ===
# ========================================


def get_nodes_to_delete(g, nodes_to_go, r, h):
    """
    Define set of nodes to delete and simulate fracture in nanocomposite
    p0 - center point of cylinder

    Parameters
    ----------
    g: nx.Graph
        initial circuit graph
    nodes_to_go: list
        nodes on the cube surface
    r: float
        radius of cylinder
    h: float
        half of height of cylinder

    Returns
    -------
    nodes_to_delete: dict
        Dict {p0 of cylinder : list of nodes to delete}
    """
    p0_coords = []
    nodes_to_delete = []
    cube_dim = 0.5
    # permutations with repetitions
    xy_range = list(itertools.product(np.arange(r, cube_dim, r, dtype=np.float32), repeat=2))
    z_range = np.arange(h, cube_dim, h)

    # combinations with repetitions from two lists
    for p in itertools.product(xy_range, z_range):
        p0_coords.append((p[0][0], p[0][1], p[1]))

    print('Generating fracture cases ...')
    for p0 in tqdm(p0_coords):
        nodes_p0 = []
        for n in g.nodes():
            x, y, z = g.nodes[n]['coords']
            inside_cylinder = ((x - p0[0]) ** 2 + (y - p0[1]) ** 2) <= r ** 2 and (z >= p0[2] - h) and (z <= p0[2] + h)
            # exclude nodes on the surface
            if inside_cylinder and (n not in nodes_to_go): nodes_p0.append(n)
        nodes_to_delete.append(nodes_p0)

    p0_nodes_to_delete = dict(zip(p0_coords, nodes_to_delete))

    # add initial state
    p0_nodes_to_delete[('nan', 'nan', 'nan')] = [-1]

    return p0_nodes_to_delete


# ===========================================
# === (4) MNA method applied to CNT graph ===
# ===========================================


def get_conductivity_matrix(g, base_node):
    """
    Function creates conductivity matrix of circuit graph

    Parameters
    ----------
    g: nx.Graph
        current circuit graph
    base_node: int
        base node where voltage is equal to zero

    Return
    ------
    scipy.sparse.csr_matrix
    """
    # create attribute matrix and add sum by row/column on diagonal
    art_matrix = nx.attr_sparse_matrix(g, edge_attr='G')[0] * (-1)
    diag_sum = np.squeeze(np.asarray(art_matrix.sum(axis=1)))
    art_matrix.setdiag(np.abs(diag_sum))

    # lil_matrix -> csr_matrix
    art_matrix = art_matrix.tocsr()

    # delete row and column corresponding to the base node (here it's last row and column)
    idx = sorted(g.nodes()).index(base_node)
    mask = np.ones(art_matrix.shape[0], dtype=bool)
    mask[idx] = False
    return art_matrix[mask][:, mask], idx


def calculate_graph(g, pair, conductivity_matrix, idx_del):
    """
    Function applies MNA to given circuit (graph)
    and finds the solution for linear system $G \times V = I$

    Parameters
    ----------
    g: nx.Graph
        current circuit graph
    pair: tuple
        pair of nodes connected to current source
    conductivity_matrix: scipy.sparse.csr_matrix
        conductivity matrix of circuit graph
    idx_del: int
        idx_del

    Returns
    -------
    voltage: int
        voltage between two nodes connected to the current source
    """
    nodes_list = sorted(g.nodes())
    # idx_del = nodes_list.index(base_node)

    # supply current: 400 microA
    i_supply = 400 * 1e-6

    # building current vector
    i_matrix = np.zeros(g.number_of_nodes()).reshape(-1, 1)
    i_matrix[nodes_list.index(pair[0])] = -i_supply
    i_matrix[nodes_list.index(pair[1])] = i_supply

    i_matrix = np.delete(i_matrix, idx_del, 0)
    del nodes_list[idx_del]

    v_matrix = lg.spsolve(conductivity_matrix, i_matrix)

    n_voltage = v_matrix[nodes_list.index(pair[1])] - v_matrix[nodes_list.index(pair[0])]
    return np.abs(n_voltage)


# ================================
# === (5) Calculation function ===
# ================================


def get_data(data_path, params, how_many='anyone'):
    """
    Calculation engine

    Parameters
    ----------
    data_path: str
        path to folder where data is stored
    params: dict
        parameters: 'r' (radius of cylinder), 'h' (half of height), way to select 'nodes_to_go'
    how_many: str
        how many graphs to take
    """
    raw_data_path = os.path.join(data_path, 'raw')
    csv_list = [csv for csv in os.listdir(raw_data_path)
                if (not csv.startswith('.')) and (csv.endswith(".csv"))]

    cnt_ids = np.unique([int(csv.split(' ')[0]) for csv in csv_list])

    if how_many == 'all': graph_ids = cnt_ids
    elif how_many == 'anyone': graph_ids = [random.choice(cnt_ids)]
    else: graph_ids = [cnt_ids[0]]

    df_labels = pd.DataFrame(columns=['g_name', 'x', 'y', 'z'])

    for k, idx in enumerate(graph_ids):
        print(f'CNT graph #{idx} - processing')
        path_nodes = os.path.join(raw_data_path, f'{idx} graphNodes.csv')
        path_edges = os.path.join(raw_data_path, f'{idx} graphEdges.csv')
        gr, nodes_to_go, pairs_to_go = get_cnt_graph(path_nodes, path_edges,
                                                     nodes_to_go_mode=params['nodes_to_go_mode'])
        graph_check(gr, nodes_to_go, strict=params['strict_check_func'])
        print(f'CNT graph #{idx} - done!\n')

        # create graph of measurements
        g_output = nx.Graph()
        gout_nodes = [(n, {'coords': gr.nodes[n]['coords']}) for n in nodes_to_go]
        gout_edges = [(int(p[0]), int(p[1]), {'U': np.NaN}) for p in pairs_to_go]
        g_output.add_nodes_from(gout_nodes)
        g_output.add_edges_from(gout_edges)

        # get nodes combinations to drop from main graph and hence simulate break
        nodes_to_delete = get_nodes_to_delete(gr, nodes_to_go, params['damage_r'], params['damage_h'])

        print(f'CNT graph #{idx} - damage cases calculation')
        thrown_cases = []
        for case_idx, p0 in tqdm(enumerate(nodes_to_delete.keys())):
            # create subgraph of main circuit graph
            nodes_del = nodes_to_delete[p0]
            nodes_to_stay = list(set(gr.nodes()) - set(nodes_del))
            g = gr.subgraph(nodes_to_stay).copy()
            if not graph_check(g, nodes_to_go, strict=params['strict_check_func']):
                thrown_cases.append(case_idx)
                continue

            gout = deepcopy(g_output)

            # choose base node randomly
            base_node = random.choice([n for n in g.nodes() if g.nodes[n]['on_surface'] == 0])

            # get conductivity matrix and index of base node
            conductivity_matrix, idx_del = get_conductivity_matrix(g, base_node)

            sg_nodes_list = list(nx.connected_components(g))
            for i, pair in enumerate(pairs_to_go):
                if params['strict_check_func']:
                    gout.edges[pair]['U'] = calculate_graph(g, pair, conductivity_matrix, idx_del)
                else:
                    connected = False
                    for sg_nodes in sg_nodes_list:
                        if (pair[0] in sg_nodes) and (pair[1] in sg_nodes): connected = True

                    if connected: gout.edges[pair]['U'] = calculate_graph(g, pair, conductivity_matrix, idx_del)
                    else: gout.edges[pair]['U'] = 0.

            df_labels.loc[case_idx + k*len(nodes_to_delete), :] = [f'cnt_{idx}_case_{case_idx}.gml', p0[0], p0[1], p0[2]]

            # SAVE output graph
            nx.write_gml(gout, f'data/graphs/cnt_{idx}_case_{case_idx}.gml')
            del g, gout
        print(f'for graph #{idx} {len(thrown_cases)}/{len(nodes_to_delete)} have been removed')

    # SAVE labels
    df_labels.to_csv(f'data/df_labels.csv')
