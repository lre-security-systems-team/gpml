"""
This module provides tools for constructing and analyzing network graphs from CSV.

Contributors:
    Julien MICHEL

Project started on:
    11/10/2022
"""

import pandas as pd
import networkx as nx


def create_static_graph_from_csv(file_name, edge_source, edge_dest, label, sampling=1):
    """
    Take a csv and represent rows as edges of a graph to create the graph representation.

    Returns
    -------
    return a multigraph representing the chosen csv file

    Parameters
    ----------
    :param file_name: Input csv file
    :param edge_source: Name of field in csv for edge source as list
    :param edge_dest: Name of field in csv for edge destination as list
    :param label: Name of the field for label in the csv
    :param sampling: Percentage of sample taken from csv data
    """
    if not isinstance(edge_source, list) or not isinstance(edge_dest, list):
        print('Edge vertice have to be given as list')
        return 0

    if sampling > 1:
        return "Can't oversample"

    MG = nx.MultiGraph()

    dataframe = pd.read_csv(file_name, skiprows=lambda x: x % int(1 / sampling) != 0)
    limit = dataframe.shape[0]
    count = 0
    while count < limit:

        if len(edge_source) > 1:  # Multiple column as edge source

            source = "_".join([str(dataframe[item][count]) for item in edge_source])
        else:
            source = dataframe[edge_source[0]][count]
        if len(edge_dest) > 1:  # Multiple column as edge dest
            destination = "_".join([str(dataframe[item][count]) for item in edge_dest])
        else:
            destination = dataframe[edge_dest[0]][count]
        MG.add_edge(source, destination, label=dataframe[label][count], index=count)
        count = count + 1

    return MG
