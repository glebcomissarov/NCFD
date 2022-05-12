import numpy as np
import pandas as pd
import networkx as nx

import os
from tqdm import tqdm


class CNTdata(object):
    def __init__(self, data_path, labels_path, save_to, update, transform):
        # labels
        self.__labels = pd.read_csv(labels_path, index_col=0)

        # initial graph object
        self.graph = self.__get_initial_graph(data_path)
        # dataframe with cracks + initial case
        self.df = self.__graphs2df(data_path, save_to, update, transform)
        self.transform = transform

    def __make_df(self, data_path, transform):
        graphs_ids = list(self.__labels[~self.__labels.isnull().any(axis=1)].index)
        # to make initial graph last
        graphs_ids.append(self.__labels[self.__labels.isnull().any(axis=1)].index.item())

        feature_names = [f'U_{e[0]}_{e[1]}' for e in self.graph.edges()]
        feature_names.extend(['g_name', 'x', 'y', 'z'])

        df = pd.DataFrame(columns=feature_names)

        for idx in tqdm(graphs_ids):
            g_name = self.__labels.loc[idx, 'g_name']

            g = nx.read_gml(os.path.join(data_path, 'graphs', g_name))

            df.loc[idx, feature_names[:-4]] = [g.edges[e]['U'] for e in g.edges()]

            df.loc[idx, ['g_name', 'x', 'y', 'z']] = [g_name,
                                                      self.__labels.loc[idx, 'x'],
                                                      self.__labels.loc[idx, 'y'],
                                                      self.__labels.loc[idx, 'z']]

            del g

        if transform == 'divide-to-calm-graph':
            df.iloc[:-1, :-4] = df.iloc[:-1, :-4] / df.iloc[-1, :-4]
        return df

    def __graphs2df(self, data_path, save_to, update, transform):
        if save_to in os.listdir(data_path) and not update:
            return pd.read_csv(os.path.join(data_path, save_to), index_col=0)
        else:
            df = self.__make_df(data_path, transform)
            df.to_csv(os.path.join(data_path, save_to))
            return df

    def __get_initial_graph(self, data_path):
        # find index of initial graph in table of labels
        idx = self.__labels[self.__labels.isnull().any(axis=1)].index.item()

        return nx.read_gml(
            os.path.join(data_path, 'graphs', self.__labels.loc[idx, 'g_name']))
