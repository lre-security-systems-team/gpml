"""
This module provides a base class for Graph Anomaly Detection algorithms.

All GAD algorithms should derive this class to avoid code duplication
for certains methods

Class:
    Detector - Detector class

Public Methods:
    get_score - returns the anomalies.
    l21_norm - computes l2,1 norm.
    dump_results - dumps a new dataframe with GAD metrics.

Contributors:
    Lyes BOURENNANI

Project started on:
    11/10/2022
"""

from datetime import timedelta
from abc import ABC, abstractmethod

import pandas as pd
import networkx as nx
import numpy as np
import tensorflow as tf

class Detector(ABC):
    """Abstract class for detectors."""

    def __init__(self, graph: nx.Graph, attributes: np.ndarray = None):
        """
        Detector Constructor.

        Parameters
        ----------
        :param graph: NetworkX graph
        :param attributes: Numpy array of attributes
        """
        self.graph = graph
        self.attributes = attributes
        self.score = None
        if attributes is None:
            attributes = np.eye(len(graph.nodes()))

    @abstractmethod
    def fit(self):
        """
        Abstract class for fit method.

        It feeds data for the implemented algorithms.
        """

    def l21_norm(self, m: tf.Variable):
        """
        Return the l2,1 norm.

        Compute the l2,1 norm for matrix operations from a tf.Variable.
        
        Returns
        -------
        output tf.Variable holding the l2,1 norm of the given matrix.

        Parameters
        ----------
        :param m: The given tf.Variable containing the input matrix
        """
        return tf.reduce_sum(tf.norm(m, ord='euclidean', axis=1))

    def get_score(self):
        """
        Return the score obtained by implemented algorithms.

        It returns the score fed by the fit method.

        Returns
        -------
        output Vertices that are anomalies (set)

        """
        return self.score

    @staticmethod
    def dump_results(df : pd.DataFrame, algorithm, time_attr: str, src_attr: str, dst_attr: str, delta: int=5, **args):
        """
        Return a new pandas dataframe containing graph anomaly detection metrics.

        It returns a deepcopy of the given dataframe with a new column with the graph anomaly detection algorithm used.
        
        Returns
        -------
        output Vertices that are anomalies (pd.Dataframe)

        Parameters
        ----------
        :param df: The given pd.Dataframe
        :param algorithm: The chosen algorithm (SCAN, RADAR, ANOMALOUS)
        :param time_attr: Name of Datetime column in Dataframe
        :param src_attr: Name of source nodes column in Dataframe
        :param dst_attr: Name of destinatop, nodes column in Dataframe
        :param delta: Number of interval in minutes (Defaults to 5)
        :param **args: Mapping for GAD Algorithms' fit methods
        """
        new_df = df.copy()

        if time_attr is not None:
            new_df[time_attr] = pd.to_datetime(new_df[time_attr])
            new_df.sort_values(by=time_attr)

        start_time, end_time = new_df[time_attr].min(), new_df[time_attr].max()

        new_df[algorithm.__name__] = None

        current_time = start_time
        while current_time <= end_time:
            next_time = current_time + timedelta(minutes=delta)
            interval_df = new_df[(new_df[time_attr] >= current_time) & (new_df[time_attr] <= next_time)]

            G = nx.DiGraph()

            for _, row in interval_df.iterrows():
                G.add_edge(row[src_attr], row[dst_attr])

            #TODO: Need to construct attributes from dataframe
            model = algorithm(G, np.eye(len(G.nodes())))
            model.fit(**args)
            score = model.score

            for idx, row in interval_df.iterrows():
                new_df.at[idx, algorithm.__name__] = score[row[src_attr]]

            current_time = next_time

        return new_df
