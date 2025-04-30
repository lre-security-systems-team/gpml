"""
This module provides an implementation of SCAN: A Structural Clustering Algorithm for Networks.

Xu, Xiaowei, et al. "Scan: a structural clustering algorithm for networks."
Proceedings of the 13th ACM SIGKDD international conference on Knowledge
discovery and data mining. 2007.

It includes a class to process NetworkX graphs read to retrieve anomalies.

Class:
    SCAN - SCAN Algorithm class

Public Methods:
    fit - feed data to the SCAN object and compute anomalies.
    get_score - returns the anomalies

Contributors:
    Lyes BOURENNANI

Project started on:
    11/10/2022
"""

import math
import networkx as nx
from gpml.anomaly_detection.detector import Detector

class SCAN(Detector):
    """
    SCAN Class.

    Can be instantiated with the studied graph 
    It directly stores the result of the fit() method.
    (clusters, hubs and outliers).
    """

    def __init__(self, graph: nx.Graph):
        """
        SCAN Constructor.

        Parameters
        ----------
        :param graph: NetworkX graph
        """
        super().__init__(graph)
        self.clusters = {}

        self.hubs = set()
        self.score = None

    def __neighborhood(self, v):
        """
        Compute the set of the adjacency list of a particular node and itself.

        It consists of retrieving adjacent vertex and the current vertex.

        Returns
        -------
        output Set of vertices

        Parameters
        ----------
        :param v: vertex
        """
        return set(list(self.graph[v]) + [v])

    def __struct_sim(self, v, w):
        """
        Compute the structural similarity between two vertices.

        It consists of computing the geometric mean of the neighborhood of the vertices.

        Returns
        -------
        output Geometric mean (Double)

        Parameters
        ----------
        :param v: first vertex
        :param w: second vertex
        """
        n_v = self.__neighborhood(v)
        n_w = self.__neighborhood(w)

        #n_v and n_w are never zero since it contains the vertex itself

        inter = n_v.intersection(n_w)

        return len(inter) / math.sqrt(len(n_v) * len(n_w))

    def __e_neighborhood(self, v, epsilon):
        """
        Neighborhood but using espilon as a threshold to filter vertices.

        It consists of retrieving each vertex of the neighborhood where
        their structure similarity is greater or equal to epsilon.

        Returns
        -------
        output Set of vertices

        Parameters
        ----------
        :param v: vertex
        :param epsilon: epsilon
        """
        n = self.__neighborhood(v)
        return {w for w in n if self.__struct_sim(v, w) >= epsilon}

    def __is_core(self, v, epsilon, mu):
        """
        Check if a vertex is a core.

        A vertex is a core if its epsilon-neighborhood contains an equal or
        superior amount of vertices than the mu threshold.

        Returns
        -------
        output boolean

        Parameters
        ----------
        :param v: vertex
        :param epsilon: epsilon
        :param mu: mu
        """
        return len(self.__e_neighborhood(v, epsilon)) >= mu

    def __direct_reach(self, v, w, epsilon, mu):
        """
        Check if the direct structure reachability between two vertices.

        It means that for two vertices v and w, v has to be a core and
        w is in the epsilon neighborhood of v.

        Returns
        -------
        output boolean

        Parameters
        ----------
        :param G: graph
        :param v: first vertex
        :param w: second vertex
        :param epsilon: epsilon
        :param mu: mu
        """
        return self.__is_core(v, epsilon, mu) and w in self.__e_neighborhood(v, epsilon)

    def fit(self, epsilon, mu):
        """
        Compute the SCAN algorithm on the graph G (NetworkX graph).

        It fills the instantiated SCAN object with the found clusters, hubs
        and outliers.

        Returns
        -------
        output void

        Parameters
        ----------
        :param epsilon: epsilon
        :param mu: mu
        """
        cluster_id = -1

        self.clusters = {}
        for v in self.graph:
            self.clusters[v] = []

        node_id = {}
        for v in self.graph:
            node_id[v] = -1

        self.hubs.clear()
        self.score = {node: False for node in self.graph.nodes()}

        C = {}
        for v in self.graph:
            C[v] = False

        non_member = []

        for v in self.graph:
            if self.__is_core(v, epsilon, mu):
                cluster_id += 1
                q = list(self.__e_neighborhood(v, epsilon))

                while len(q) != 0:
                    y = q.pop(0)
                    R = [x for x in self.graph if self.__direct_reach(y, x, epsilon, mu)]

                    for x in R:
                        if not C[x] or x in non_member:
                            node_id[x] = cluster_id
                            self.clusters[list(self.graph.nodes)[cluster_id]].append(x)
                        if not C[x]:
                            q.append(x)
                            C[x] = True
            else:
                non_member.append(v)

        for v in non_member:
            n = self.__neighborhood(v)
            t = False
            for x in n:
                for y in n:
                    if x != y and node_id[x] != node_id[y] and (node_id[x] != -1 and node_id[y] != -1):
                        t = True

            if t:
                self.hubs.add(v)
            else:
                self.score[v] = True
