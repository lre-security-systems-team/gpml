"""
This module provides functionalities for analyzing and computing network community metrics over dynamic graphs.

It includes functions to process graphs based on data read from CSV files, compute various
1rst and 2nd order community metrics, and incorporate those metrics back into the data
for further analysis.
The module supports multiplecommunity detection strategies and offers tools
for dynamic graph analysis.

Functions:
    chunk_read_and_insert_gc_metrics - Processes chunks of CSV data to compute community metrics
    and outputs to a new CSV.
    insert_metrics_to_dataframe - Computes community metrics for a given dataframe
    and updates it with new metrics columns.

Contributors:
    Julien MICHEL

Project started on:
    11/10/2022
"""

import sys
import time
from datetime import datetime
import itertools
import networkx as nx
import networkx.algorithms.community as nx_comm
import igraph as ig
import pandas as pd
from library.community import community_louvain
import gpml.metrics.graph_community as gc

sys.path.append('.')


def nested_insert_metrics_to_dataframe(dataframe, time_interval, date_time, edge_source, edge_dest, label, name,
                                       date_timestamp, community_strategy='louvain', time_0=0, buffer_graph=None):
    """
    Take a dataframe, represent it on dynamic graph and compute community metrics.

    Returns
    -------
    return dataframe with community metrics

    Parameters
    ----------
    :param dataframe: pd.DataFrame()
    :param time_interval: timedelta of graph window
    :param date_time: Name of the date field in dataframe
    :param edge_source: Name of field in dataframe for edge source id as list
    :param edge_dest: Name of field in dataframe for edge destination id as list
    :param label: Name of the target column in dataframe as str
    :param name: name to attach to new column
    :param date_timestamp: True or False if date column type is already timestamp
    :param buffer_graph: None if first chunk, last graph otherwise
    """
    if (not isinstance(edge_source, list) or not isinstance(edge_dest, list)):
        print('edge vertice have to be given as list')
        return dataframe

    if date_timestamp is False:
        dataframe[date_time] = dataframe[date_time].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    # sort dataframe by date to make the time interval selection
    dataframe = dataframe.sort_values(date_time)
    # get the first date
    first = dataframe[date_time][0]
    count = 0
    stop = 0
    size = 0
    t = 1
    current_stop = time_interval

    community = [0] * dataframe.shape[0]
    # 1rs Order metrics
    anchor = [0] * dataframe.shape[0]
    average_degree_c = [0] * dataframe.shape[0]
    average_degree_g = [0] * dataframe.shape[0]
    nb_of_edges_c = [0] * dataframe.shape[0]
    nb_of_nodes_c = [0] * dataframe.shape[0]
    nb_of_edges_g = [0] * dataframe.shape[0]
    nb_of_nodes_g = [0] * dataframe.shape[0]

    # 2nd Order metrics
    density_c = [0] * dataframe.shape[0]
    density_g = [0] * dataframe.shape[0]
    edges_dist = [0] * dataframe.shape[0]
    externality = [0] * dataframe.shape[0]
    conductance = [0] * dataframe.shape[0]
    expansion = [0] * dataframe.shape[0]
    NED = [0] * dataframe.shape[0]
    NED_index = [0] * dataframe.shape[0]
    mean_community_size = [0] * dataframe.shape[0]
    # Dynamic metrics
    last_stability = [0] * dataframe.shape[0]
    stability = [0] * dataframe.shape[0]
    delta_node = [0] * dataframe.shape[0]
    delta_density = [0] * dataframe.shape[0]
    delta_connectivity = [0] * dataframe.shape[0]
    delta_degree = [0] * dataframe.shape[0]

    ##########################################
    # Dynamic metric holder
    dynamic_metrics = {'delta_node': [], 'delta_connectivity': [], 'delta_density': [], 'delta_degree': []}
    ##########################################

    # Initial graph
    MG = nx.MultiGraph()

    while dataframe[date_time][count] < first + current_stop:

        if len(edge_source) > 1:  # Multiple column as edge source

            src = "_".join([str(dataframe[item][count]) for item in edge_source])
        else:
            src = dataframe[edge_source[0]][count]
        if len(edge_dest) > 1:  # Multiple column as edge dest
            dst = "_".join([str(dataframe[item][count]) for item in edge_dest])
        else:
            dst = dataframe[edge_dest[0]][count]
        MG.add_edge(src, dst, label=dataframe[label][count], index=count)
        count = count + 1

    if community_strategy == 'lpa':  # Label propagation algorithm
        pre_partition = list(nx_comm.label_propagation.label_propagation_communities(MG))
        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    elif community_strategy == 'k_clique':  # Label propagation algorithm
        pre_partition = list(nx_comm.k_clique_communities(MG, 2))
        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    elif community_strategy == 'girvan_newman':
        gm = nx_comm.centrality.girvan_newman(MG)
        slice_count = 0
        for communities in itertools.islice(gm, 2):
            slice_count = slice_count + 1
            if slice_count == 2:
                pre_partition = list(c for c in communities)

        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    elif community_strategy == 'walktrap':
        mg = ig.Graph.from_networkx(MG)
        wtrap = mg.community_walktrap()
        part_count = 0
        partition = {}
        for graph in wtrap.as_clustering().subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    elif community_strategy == 'eigenvector':
        mg = ig.Graph.from_networkx(MG)
        pre_part = mg.community_leading_eigenvector()
        part_count = 0
        partition = {}
        for graph in pre_part.subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    elif community_strategy == 'leiden':
        mg = ig.Graph.from_networkx(MG)
        pre_part = mg.community_leiden()
        part_count = 0
        partition = {}
        for graph in pre_part.subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    elif community_strategy == 'infomap':
        mg = ig.Graph.from_networkx(MG)
        pre_part = mg.community_infomap()
        part_count = 0
        partition = {}
        for graph in pre_part.subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    else:  # (default or louvain) # Louvain algorithm
        partition = community_louvain.best_partition(MG)
    nx.set_node_attributes(MG, partition, "community")

    current_stop = current_stop + time_interval

    forder_metrics_c, forder_metrics_g = gc.gc_metrics_first_order(MG)
    center = forder_metrics_c['center']
    so_metrics_c, so_metrics_g = gc.gc_metrics_second_order(forder_metrics_c, forder_metrics_g)
    ########## Assign metrics value to dataframe ##########

    # Dynamic update #

    dynamic_metrics['delta_density'] = so_metrics_c['density']
    dynamic_metrics['delta_connectivity'] = so_metrics_c['expansion']
    dynamic_metrics['delta_degree'] = so_metrics_c['average_degree']
    dynamic_metrics['delta_node'] = forder_metrics_c['nb_of_nodes']
    ##################
    for elm in MG.edges.data():
        community[elm[2]['index']] = MG.nodes[elm[0]]['community']
        # 1rs Order metrics
        anchor[elm[2]['index']] = forder_metrics_c['anchor'][MG.nodes[elm[0]]['community']]
        average_degree_c[elm[2]['index']] = so_metrics_c['average_degree'][MG.nodes[elm[0]]['community']]
        average_degree_g[elm[2]['index']] = so_metrics_g['average_degree']
        nb_of_edges_c[elm[2]['index']] = forder_metrics_c['nb_of_edges'][MG.nodes[elm[0]]['community']]
        nb_of_nodes_c[elm[2]['index']] = forder_metrics_c['nb_of_nodes'][MG.nodes[elm[0]]['community']]
        nb_of_edges_g[elm[2]['index']] = forder_metrics_g['nb_of_edges']
        nb_of_nodes_g[elm[2]['index']] = forder_metrics_g['nb_of_nodes']

        # 2nd Order metrics
        density_c[elm[2]['index']] = so_metrics_c['density'][MG.nodes[elm[0]]['community']]
        density_g[elm[2]['index']] = so_metrics_g['density']
        edges_dist[elm[2]['index']] = so_metrics_g['edges_dist']
        externality[elm[2]['index']] = so_metrics_c['externality'][MG.nodes[elm[0]]['community']]
        conductance[elm[2]['index']] = so_metrics_c['conductance'][MG.nodes[elm[0]]['community']]
        expansion[elm[2]['index']] = so_metrics_c['expansion'][MG.nodes[elm[0]]['community']]
        NED[elm[2]['index']] = so_metrics_c['NED'][MG.nodes[elm[0]]['community']]
        NED_index[elm[2]['index']] = so_metrics_g['NED_index']
        mean_community_size[elm[2]['index']] = so_metrics_g['mean_community_size']
    old_stabilities = [0] * dataframe.shape[0]
    if time_0 != 0:
        size += (max(gc.propagate_communities(buffer_graph[0], MG, buffer_graph[1], center)) + 1)
    print('First_graph generated and metrics computed')

    while stop == 0:
        t = t + 1
        # Graph t+1
        MG2 = nx.MultiGraph()
        try:
            # print( dataframe[date_time][count], first,current_stop)
            while dataframe[date_time][count] < first + current_stop:

                if len(edge_source) > 1:  # Multiple column as edge source

                    s = "_".join([str(dataframe[item][count]) for item in edge_source])
                else:
                    s = dataframe[edge_source[0]][count]
                if len(edge_dest) > 1:  # Multiple column as edge dest
                    d = "_".join([str(dataframe[item][count]) for item in edge_dest])
                else:
                    d = dataframe[edge_dest[0]][count]

                MG2.add_edge(s, d, label=dataframe[label][count], index=count)
                count = count + 1

            if community_strategy == 'lpa':  # Label propagation algorithm
                pre_partition = list(nx_comm.label_propagation.label_propagation_communities(MG2))
                part_count = 0
                partition2 = {}
                for elm in pre_partition:
                    for key in elm:
                        partition2[key] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'k_clique':  # Label propagation algorithm
                pre_partition = list(nx_comm.k_clique_communities(MG2, 2))
                part_count = 0
                partition2 = {}
                for elm in pre_partition:
                    for key in elm:
                        partition2[key] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'girvan_newman':
                gm = nx_comm.centrality.girvan_newman(MG2)
                slice_count = 0
                for communities in itertools.islice(gm, 2):
                    slice_count = slice_count + 1
                    if slice_count == 2:
                        pre_partition = list(c for c in communities)
                part_count = 0
                partition2 = {}
                for elm in pre_partition:
                    for key in elm:
                        partition2[key] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'walktrap':
                mg = ig.Graph.from_networkx(MG2)
                wtrap = mg.community_walktrap()
                part_count = 0
                partition2 = {}
                for graph in wtrap.as_clustering().subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'eigenvector':
                mg = ig.Graph.from_networkx(MG2)
                pre_part = mg.community_leading_eigenvector()
                part_count = 0
                partition2 = {}
                for graph in pre_part.subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'leiden':
                mg = ig.Graph.from_networkx(MG2)
                pre_part = mg.community_leiden()
                part_count = 0
                partition2 = {}
                for graph in pre_part.subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1
            elif community_strategy == 'infomap':
                mg = ig.Graph.from_networkx(MG2)
                pre_part = mg.community_infomap()
                part_count = 0
                partition2 = {}
                for graph in pre_part.subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1

            else:  # (default or louvain) # Louvain algorithm
                partition2 = community_louvain.best_partition(MG2)

            current_stop = current_stop + time_interval
            nx.set_node_attributes(MG2, partition2, "community")
            # print(MG2.number_of_nodes())
            forder_metrics_c, forder_metrics_g = gc.gc_metrics_first_order(MG2)
            center2 = forder_metrics_c['center']
            so_metrics_c, so_metrics_g = gc.gc_metrics_second_order(forder_metrics_c, forder_metrics_g)

            for elm in MG2.edges.data():
                # 1rs Order metrics
                anchor[elm[2]['index']] = forder_metrics_c['anchor'][MG2.nodes[elm[0]]['community']]
                average_degree_c[elm[2]['index']] = so_metrics_c['average_degree'][MG2.nodes[elm[0]]['community']]
                average_degree_g[elm[2]['index']] = so_metrics_g['average_degree']
                nb_of_edges_c[elm[2]['index']] = forder_metrics_c['nb_of_edges'][MG2.nodes[elm[0]]['community']]
                nb_of_nodes_c[elm[2]['index']] = forder_metrics_c['nb_of_nodes'][MG2.nodes[elm[0]]['community']]
                nb_of_edges_g[elm[2]['index']] = forder_metrics_g['nb_of_edges']
                nb_of_nodes_g[elm[2]['index']] = forder_metrics_g['nb_of_nodes']

                # 2nd Order metrics
                density_c[elm[2]['index']] = so_metrics_c['density'][MG2.nodes[elm[0]]['community']]
                density_g[elm[2]['index']] = so_metrics_g['density']
                edges_dist[elm[2]['index']] = so_metrics_g['edges_dist']
                externality[elm[2]['index']] = so_metrics_c['externality'][MG2.nodes[elm[0]]['community']]
                conductance[elm[2]['index']] = so_metrics_c['conductance'][MG2.nodes[elm[0]]['community']]
                expansion[elm[2]['index']] = so_metrics_c['expansion'][MG2.nodes[elm[0]]['community']]
                NED[elm[2]['index']] = so_metrics_c['NED'][MG2.nodes[elm[0]]['community']]
                NED_index[elm[2]['index']] = so_metrics_g['NED_index']
                mean_community_size[elm[2]['index']] = so_metrics_g['mean_community_size']
                # Dynamic metrics

            # Dynamic update #
            size += (max(gc.propagate_communities(MG, MG2, center, center2)) + 1)
            dynamic_metrics['delta_density'] += [0] * (size - len(dynamic_metrics['delta_density']))
            dynamic_metrics['delta_connectivity'] += [0] * (size - len(dynamic_metrics['delta_connectivity']))
            dynamic_metrics['delta_degree'] += [0] * (size - len(dynamic_metrics['delta_degree']))
            dynamic_metrics['delta_node'] += [0] * (size - len(dynamic_metrics['delta_node']))
            old_stabilities += [0] * (size - len(old_stabilities))

            ##################
            stabilities, new_stabilities = gc.compute_stabilities(MG, MG2, size, old_stabilities, t)

            for elm in MG2.edges.data():
                community[elm[2]['index']] = MG2.nodes[elm[0]]['community']
                last_stability[elm[2]['index']] = stabilities[MG2.nodes[elm[0]]['community']]
                stability[elm[2]['index']] = new_stabilities[MG2.nodes[elm[0]]['community']]
                delta_node[elm[2]['index']] = forder_metrics_c['nb_of_nodes'][MG2.nodes[elm[0]]['old_community']] - \
                                              dynamic_metrics['delta_node'][MG2.nodes[elm[0]]['community']]
                try:
                    delta_density[elm[2]['index']] = abs((so_metrics_c['density'][MG2.nodes[elm[0]]['old_community']] -
                                                          dynamic_metrics['delta_density'][
                                                              MG2.nodes[elm[0]]['community']]) /
                                                         ((so_metrics_c['density'][MG2.nodes[elm[0]]['old_community']] +
                                                           dynamic_metrics['delta_density'][
                                                               MG2.nodes[elm[0]]['community']]) / 2))
                except Exception:
                    delta_density[elm[2]['index']] = 0
                try:
                    delta_connectivity[elm[2]['index']] = abs((so_metrics_c['expansion'][
                                                                   MG2.nodes[elm[0]]['old_community']] -
                                                               dynamic_metrics['delta_connectivity'][
                                                                   MG2.nodes[elm[0]]['community']]) /
                                                              ((so_metrics_c['expansion'][
                                                                    MG2.nodes[elm[0]]['old_community']] -
                                                                dynamic_metrics['delta_connectivity'][
                                                                    MG2.nodes[elm[0]]['community']]) / 2))
                except Exception:
                    delta_connectivity[elm[2]['index']] = 0

                delta_degree[elm[2]['index']] = so_metrics_c['average_degree'][MG2.nodes[elm[0]]['old_community']] - \
                                                dynamic_metrics['delta_degree'][MG2.nodes[elm[0]]['community']]

            old_stabilities = new_stabilities.copy()

            dynamic_metrics['delta_density'] = so_metrics_c['density']
            dynamic_metrics['delta_connectivity'] = so_metrics_c['expansion']
            dynamic_metrics['delta_degree'] = so_metrics_c['average_degree']
            dynamic_metrics['delta_node'] = forder_metrics_c['nb_of_nodes']
            MG = MG2.copy()

            ####################
            ### store_graphe to implement
            ####################

        except Exception as err:
            print(err)
            stop = 1

    # initialize new columns to 0
    dataframe['community_' + name] = community
    dataframe['Nb_of_nodes_c_' + name] = nb_of_nodes_c
    dataframe['Nb_of_nodes_g_' + name] = nb_of_nodes_g
    dataframe['Nb_of_edges_c_' + name] = nb_of_edges_c
    dataframe['Nb_of_edges_g_' + name] = nb_of_edges_g
    dataframe['average_degree_c_' + name] = average_degree_c
    dataframe['average_degree_g_' + name] = average_degree_g
    dataframe['anchor_' + name] = anchor
    dataframe['density_c_' + name] = density_c
    dataframe['density_g_' + name] = density_g
    dataframe['edges_dist_' + name] = edges_dist
    dataframe['externality_' + name] = externality
    dataframe['conductance_' + name] = conductance
    dataframe['expansion_' + name] = expansion
    dataframe['mean_community_size_' + name] = mean_community_size
    dataframe['NED_' + name] = NED
    dataframe['NED_index_' + name] = NED_index
    dataframe['last_stability_' + name] = last_stability
    dataframe['stability_' + name] = stability
    dataframe['delta_density_' + name] = delta_density
    dataframe['delta_Node_' + name] = delta_node
    dataframe['delta_Degree_' + name] = delta_degree
    dataframe['delta_expansion_' + name] = delta_connectivity

    return (dataframe, (MG2, center2))


def chunk_read_and_insert_gc_metrics(file_name, output_csv, time_interval, header=None, community_strategy='louvain',
                                     chunksize=5000000, date_time="Date time", edge_source=None, edge_dest=None,
                                     label='Label', name='default'):
    """
    Use pandas to read a csv by chunk and write another csv with the same field + graph communities metrics.

    Returns
    -------
    output csv file with community metrics

    Parameters
    ----------
    :param file_name: input csv file
    :param output_csv: output csv file
    :param time_interval: timedelta of graph window
    :param community_strategy: Name of community strategy
    :param chunksize: nb of csv line to load in memory as int
    :param date_time: Name of the date field in csv
    :param edge_source: Name of field in csv for edge source id as list
    :param edge_dest: Name of field in csv for edge destination id as list
    :param label: Name of the label filed in csv
    :param name: naming suffix for metrics
    """
    timelist = []
    start = time.time()
    ## First Initialisation of dataframe ######
    if header is None:
        df = pd.read_csv(file_name, nrows=chunksize)
    else:
        df = pd.read_csv(file_name, names=header, nrows=chunksize)

    if edge_source is None:
        edge_source = ['Source IP']
    if edge_dest is None:
        edge_dest = ['Destination IP']

    df[date_time] = df[date_time].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df = df.sort_values(date_time)

    skip = chunksize + 1
    current_stop = time_interval

    df, (last_multigraph, last_centers) = nested_insert_metrics_to_dataframe(df, time_interval, date_time, edge_source,
                                                                             edge_dest, label, name=name,
                                                                             date_timestamp=True,
                                                                             community_strategy=community_strategy)

    df.to_csv(output_csv, index=False, mode='w')

    timelist.append(time.time() - start)

    current_stop = current_stop + time_interval
    stop = 0
    while stop == 0:
        if header is None:
            df = pd.read_csv(file_name, nrows=chunksize, skiprows=range(1, skip))
        else:
            df = pd.read_csv(file_name, names=header, nrows=chunksize, skiprows=range(0, skip))

        df[date_time] = df[date_time].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df = df.sort_values(date_time)
        skip = skip + chunksize
        try:
            # print(df.head(),df.shape[0])
            df, (last_multigraph, last_centers) = nested_insert_metrics_to_dataframe(df, time_interval, date_time,
                                                                                     edge_source, edge_dest,
                                                                                     label, name=name,
                                                                                     date_timestamp=True,
                                                                                     community_strategy=community_strategy,
                                                                                     time_0=1, buffer_graph=(
                last_multigraph, last_centers))
        except Exception:
            stop = 1
        df.to_csv(output_csv, index=False, header=False, mode='a')

        timelist.append(time.time() - start)

        if df.shape[0] < chunksize:
            stop = 1
    timedf = pd.DataFrame()
    timedf['Time'] = timelist
    timedf.to_csv("scalability_" + output_csv, index=False)


def insert_metrics_to_dataframe(dataframe, time_interval, date_time, edge_source, edge_dest, label, name,
                                community_strategy='louvain',continuity=True):
    """
    Take a dataframe, represent it on dynamic graph and compute community metrics.

    Returns
    -------
    return dataframe with community metrics

    Parameters
    ----------
    :param dataframe: pd.DataFrame()
    :param time_interval: timedelta of graph window
    :param date_time: Name of the date field in dataframe
    :param edge_source: Name of field in dataframe for edge source id as list
    :param edge_dest: Name of field in dataframe for edge destination id as list
    :param label: Name of the target column in dataframe as str
    :param name: name to attach to new column
    :param continuity: True of False if dataset has continuity in his timestamp
    """
    if not isinstance(edge_source, list) or not isinstance(edge_dest, list):
        print('edge vertice have to be given as list')
        return dataframe

    try:
        dataframe[date_time] = dataframe[date_time].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    except Exception:
        pass
    # sort dataframe by date to make the time interval selection
    dataframe = dataframe.sort_values(date_time)
    # get the first date
    first = dataframe[date_time][0]
    count = 0
    stop = 0
    size = 0
    t = 1
    current_stop = time_interval

    community = [0] * dataframe.shape[0]
    # 1rs Order metrics
    anchor = [0] * dataframe.shape[0]
    average_degree_c = [0] * dataframe.shape[0]
    average_degree_g = [0] * dataframe.shape[0]
    nb_of_edges_c = [0] * dataframe.shape[0]
    nb_of_nodes_c = [0] * dataframe.shape[0]
    nb_of_edges_g = [0] * dataframe.shape[0]
    nb_of_nodes_g = [0] * dataframe.shape[0]

    # 2nd Order metrics
    density_c = [0] * dataframe.shape[0]
    density_g = [0] * dataframe.shape[0]
    edges_dist = [0] * dataframe.shape[0]
    externality = [0] * dataframe.shape[0]
    conductance = [0] * dataframe.shape[0]
    expansion = [0] * dataframe.shape[0]
    NED = [0] * dataframe.shape[0]
    NED_index = [0] * dataframe.shape[0]
    mean_community_size = [0] * dataframe.shape[0]
    # Dynamic metrics
    last_stability = [0] * dataframe.shape[0]
    stability = [0] * dataframe.shape[0]
    delta_node = [0] * dataframe.shape[0]
    delta_density = [0] * dataframe.shape[0]
    delta_connectivity = [0] * dataframe.shape[0]
    delta_degree = [0] * dataframe.shape[0]

    # Dynamic metric holder
    dynamic_metrics = {'delta_node': [], 'delta_connectivity': [], 'delta_density': [], 'delta_degree': []}

    # Initial graph
    MG = nx.MultiGraph()

    while dataframe[date_time][count] < first + current_stop:

        if len(edge_dest) > 1:  # Multiple edge dest
            destination = "_".join([str(dataframe[item][count]) for item in edge_dest])
        else:
            destination = dataframe[edge_dest[0]][count]
        if len(edge_source) > 1:  # Multiple source
            source = "_".join([str(dataframe[item][count]) for item in edge_source])
        else:
            source = dataframe[edge_source[0]][count]
        MG.add_edge(source, destination, label=dataframe[label][count], index=count)
        count = count + 1

    if community_strategy == 'lpa':  # Label propagation algorithm
        pre_partition = list(nx_comm.label_propagation.label_propagation_communities(MG))
        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    elif community_strategy == 'k_clique':  # Label propagation algorithm
        pre_partition = list(nx_comm.k_clique_communities(MG, 2))
        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    elif community_strategy == 'girvan_newman':
        gm = nx_comm.centrality.girvan_newman(MG)
        slice_count = 0
        for communities in itertools.islice(gm, 2):
            slice_count = slice_count + 1
            if slice_count == 2:
                pre_partition = list(c for c in communities)

        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    elif community_strategy == 'walktrap':
        mg = ig.Graph.from_networkx(MG)
        wtrap = mg.community_walktrap()
        part_count = 0
        partition = {}
        for graph in wtrap.as_clustering().subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    elif community_strategy == 'eigenvector':
        mg = ig.Graph.from_networkx(MG)
        pre_part = mg.community_leading_eigenvector()
        part_count = 0
        partition = {}
        for graph in pre_part.subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    elif community_strategy == 'leiden':
        mg = ig.Graph.from_networkx(MG)
        pre_part = mg.community_leiden()
        part_count = 0
        partition = {}
        for graph in pre_part.subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    elif community_strategy == 'infomap':
        mg = ig.Graph.from_networkx(MG)
        pre_part = mg.community_infomap()
        part_count = 0
        partition = {}
        for graph in pre_part.subgraphs():
            for elm in graph.vs:
                partition[elm['_nx_name']] = part_count
            part_count = part_count + 1

    else:  # (default or louvain) # Louvain algorithm
        partition = community_louvain.best_partition(MG)
    nx.set_node_attributes(MG, partition, "community")

    current_stop = current_stop + time_interval
    if continuity is False: # check if a hole is encountered in the data
        if dataframe[date_time][count] > first + current_stop:
            current_stop = time_interval
            first = dataframe[date_time][count]
    else: # if a hole is encountered but continuity is supposed to be true
        if dataframe[date_time][count] > first + current_stop:
            sys.exit("Continuity is set to True, but dataset doesn't have time continuity")
    forder_metrics_c, forder_metrics_g = gc.gc_metrics_first_order(MG)
    center = forder_metrics_c['center']
    so_metrics_c, so_metrics_g = gc.gc_metrics_second_order(forder_metrics_c, forder_metrics_g)
    # Assign metrics value to dataframe

    # Dynamic update #
    dynamic_metrics['delta_density'] = so_metrics_c['density']
    dynamic_metrics['delta_connectivity'] = so_metrics_c['expansion']
    dynamic_metrics['delta_degree'] = so_metrics_c['average_degree']
    dynamic_metrics['delta_node'] = forder_metrics_c['nb_of_nodes']

    for elm in MG.edges.data():
        community[elm[2]['index']] = MG.nodes[elm[0]]['community']
        # 1st Order metrics
        anchor[elm[2]['index']] = forder_metrics_c['anchor'][MG.nodes[elm[0]]['community']]
        average_degree_c[elm[2]['index']] = so_metrics_c['average_degree'][MG.nodes[elm[0]]['community']]
        average_degree_g[elm[2]['index']] = so_metrics_g['average_degree']
        nb_of_edges_c[elm[2]['index']] = forder_metrics_c['nb_of_edges'][MG.nodes[elm[0]]['community']]
        nb_of_nodes_c[elm[2]['index']] = forder_metrics_c['nb_of_nodes'][MG.nodes[elm[0]]['community']]
        nb_of_edges_g[elm[2]['index']] = forder_metrics_g['nb_of_edges']
        nb_of_nodes_g[elm[2]['index']] = forder_metrics_g['nb_of_nodes']

        # 2nd Order metrics
        density_c[elm[2]['index']] = so_metrics_c['density'][MG.nodes[elm[0]]['community']]
        density_g[elm[2]['index']] = so_metrics_g['density']
        edges_dist[elm[2]['index']] = so_metrics_g['edges_dist']
        externality[elm[2]['index']] = so_metrics_c['externality'][MG.nodes[elm[0]]['community']]
        conductance[elm[2]['index']] = so_metrics_c['conductance'][MG.nodes[elm[0]]['community']]
        expansion[elm[2]['index']] = so_metrics_c['expansion'][MG.nodes[elm[0]]['community']]
        NED[elm[2]['index']] = so_metrics_c['NED'][MG.nodes[elm[0]]['community']]
        NED_index[elm[2]['index']] = so_metrics_g['NED_index']
        mean_community_size[elm[2]['index']] = so_metrics_g['mean_community_size']
    old_stabilities = [0] * dataframe.shape[0]

    print('First_graph generated and metrics computed')

    while stop == 0:
        t = t + 1
        # Graph t+1
        MG2 = nx.MultiGraph()
        try:
            while dataframe[date_time][count] < first + current_stop:

                if len(edge_dest) > 1:  # Multiple column as edge dest
                    destination = "_".join([str(dataframe[item][count]) for item in edge_dest])
                else:
                    destination = dataframe[edge_dest[0]][count]
                if len(edge_source) > 1:  # Multiple column as edge source

                    source = "_".join([str(dataframe[item][count]) for item in edge_source])
                else:
                    source = dataframe[edge_source[0]][count]
                MG2.add_edge(source, destination, label=dataframe[label][count], index=count)
                count = count + 1

            if community_strategy == 'lpa':  # Label propagation algorithm
                pre_partition = list(nx_comm.label_propagation.label_propagation_communities(MG2))
                part_count = 0
                partition2 = {}
                for elm in pre_partition:
                    for key in elm:
                        partition2[key] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'k_clique':  # Label propagation algorithm
                pre_partition = list(nx_comm.k_clique_communities(MG2, 2))
                part_count = 0
                partition2 = {}
                for elm in pre_partition:
                    for key in elm:
                        partition2[key] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'girvan_newman':
                gm = nx_comm.centrality.girvan_newman(MG2)
                slice_count = 0
                for communities in itertools.islice(gm, 2):
                    slice_count = slice_count + 1
                    if slice_count == 2:
                        pre_partition = list(c for c in communities)
                part_count = 0
                partition2 = {}
                for elm in pre_partition:
                    for key in elm:
                        partition2[key] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'walktrap':
                mg = ig.Graph.from_networkx(MG2)
                wtrap = mg.community_walktrap()
                part_count = 0
                partition2 = {}
                for graph in wtrap.as_clustering().subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'eigenvector':
                mg = ig.Graph.from_networkx(MG2)
                pre_part = mg.community_leading_eigenvector()
                part_count = 0
                partition2 = {}
                for graph in pre_part.subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1

            elif community_strategy == 'leiden':
                mg = ig.Graph.from_networkx(MG2)
                pre_part = mg.community_leiden()
                part_count = 0
                partition2 = {}
                for graph in pre_part.subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1
            elif community_strategy == 'infomap':
                mg = ig.Graph.from_networkx(MG2)
                pre_part = mg.community_infomap()
                part_count = 0
                partition2 = {}
                for graph in pre_part.subgraphs():
                    for elm in graph.vs:
                        partition2[elm['_nx_name']] = part_count
                    part_count = part_count + 1

            else:  # (default or louvain) # Louvain algorithm
                partition2 = community_louvain.best_partition(MG2)

            current_stop = current_stop + time_interval
            if continuity is False: # check if a hole is encountered in the data
                if dataframe[date_time][count] > first + current_stop:
                    current_stop = time_interval
                    first = dataframe[date_time][count]
            else: # if a hole is encountered but continuity is supposed to be true
                if dataframe[date_time][count] > first + current_stop:
                    sys.exit("Continuity is set to True, but dataset doesn't have time continuity")

            nx.set_node_attributes(MG2, partition2, "community")

            forder_metrics_c, forder_metrics_g = gc.gc_metrics_first_order(MG2)
            center2 = forder_metrics_c['center']
            so_metrics_c, so_metrics_g = gc.gc_metrics_second_order(forder_metrics_c, forder_metrics_g)

            for elm in MG2.edges.data():
                # 1rs Order metrics
                anchor[elm[2]['index']] = forder_metrics_c['anchor'][MG2.nodes[elm[0]]['community']]
                average_degree_c[elm[2]['index']] = so_metrics_c['average_degree'][MG2.nodes[elm[0]]['community']]
                average_degree_g[elm[2]['index']] = so_metrics_g['average_degree']
                nb_of_edges_c[elm[2]['index']] = forder_metrics_c['nb_of_edges'][MG2.nodes[elm[0]]['community']]
                nb_of_nodes_c[elm[2]['index']] = forder_metrics_c['nb_of_nodes'][MG2.nodes[elm[0]]['community']]
                nb_of_edges_g[elm[2]['index']] = forder_metrics_g['nb_of_edges']
                nb_of_nodes_g[elm[2]['index']] = forder_metrics_g['nb_of_nodes']

                # 2nd Order metrics
                density_c[elm[2]['index']] = so_metrics_c['density'][MG2.nodes[elm[0]]['community']]
                density_g[elm[2]['index']] = so_metrics_g['density']
                edges_dist[elm[2]['index']] = so_metrics_g['edges_dist']
                externality[elm[2]['index']] = so_metrics_c['externality'][MG2.nodes[elm[0]]['community']]
                conductance[elm[2]['index']] = so_metrics_c['conductance'][MG2.nodes[elm[0]]['community']]
                expansion[elm[2]['index']] = so_metrics_c['expansion'][MG2.nodes[elm[0]]['community']]
                NED[elm[2]['index']] = so_metrics_c['NED'][MG2.nodes[elm[0]]['community']]
                NED_index[elm[2]['index']] = so_metrics_g['NED_index']
                mean_community_size[elm[2]['index']] = so_metrics_g['mean_community_size']
                # Dynamic metrics

            # Dynamic update #
            size += (max(gc.propagate_communities(MG, MG2, center, center2)) + 1)
            dynamic_metrics['delta_density'] += [0] * (size - len(dynamic_metrics['delta_density']))
            dynamic_metrics['delta_connectivity'] += [0] * (size - len(dynamic_metrics['delta_connectivity']))
            dynamic_metrics['delta_degree'] += [0] * (size - len(dynamic_metrics['delta_degree']))
            dynamic_metrics['delta_node'] += [0] * (size - len(dynamic_metrics['delta_node']))
            old_stabilities += [0] * (size - len(old_stabilities))

            stabilities, new_stabilities = gc.compute_stabilities(MG, MG2, size, old_stabilities, t)

            for elm in MG2.edges.data():
                community[elm[2]['index']] = MG2.nodes[elm[0]]['community']
                last_stability[elm[2]['index']] = stabilities[MG2.nodes[elm[0]]['community']]
                stability[elm[2]['index']] = new_stabilities[MG2.nodes[elm[0]]['community']]
                delta_node[elm[2]['index']] = forder_metrics_c['nb_of_nodes'][MG2.nodes[elm[0]]['old_community']] - \
                                              dynamic_metrics['delta_node'][MG2.nodes[elm[0]]['community']]
                try:
                    delta_density[elm[2]['index']] = abs((so_metrics_c['density'][MG2.nodes[elm[0]]['old_community']] -
                                                          dynamic_metrics['delta_density'][
                                                              MG2.nodes[elm[0]]['community']]) /
                                                         ((so_metrics_c['density'][MG2.nodes[elm[0]]['old_community']] +
                                                           dynamic_metrics['delta_density'][
                                                               MG2.nodes[elm[0]]['community']]) / 2))
                except Exception:
                    delta_density[elm[2]['index']] = 0
                try:
                    delta_connectivity[elm[2]['index']] = abs((so_metrics_c['expansion'][
                                                                   MG2.nodes[elm[0]]['old_community']] -
                                                               dynamic_metrics['delta_connectivity'][
                                                                   MG2.nodes[elm[0]]['community']]) /
                                                              ((so_metrics_c['expansion'][
                                                                    MG2.nodes[elm[0]]['old_community']] -
                                                                dynamic_metrics['delta_connectivity'][
                                                                    MG2.nodes[elm[0]]['community']]) / 2))
                except Exception:
                    delta_connectivity[elm[2]['index']] = 0

                delta_degree[elm[2]['index']] = so_metrics_c['average_degree'][MG2.nodes[elm[0]]['old_community']] - \
                                                dynamic_metrics['delta_degree'][MG2.nodes[elm[0]]['community']]

            old_stabilities = new_stabilities.copy()

            dynamic_metrics['delta_density'] = so_metrics_c['density']
            dynamic_metrics['delta_connectivity'] = so_metrics_c['expansion']
            dynamic_metrics['delta_degree'] = so_metrics_c['average_degree']
            dynamic_metrics['delta_node'] = forder_metrics_c['nb_of_nodes']
            MG = MG2.copy()

        except Exception as err:
            print(err)
            stop = 1

    # initialize new columns to 0
    dataframe['community_' + name] = community
    dataframe['Nb_of_nodes_c_' + name] = nb_of_nodes_c
    dataframe['Nb_of_nodes_g_' + name] = nb_of_nodes_g
    dataframe['Nb_of_edges_c_' + name] = nb_of_edges_c
    dataframe['Nb_of_edges_g_' + name] = nb_of_edges_g
    dataframe['average_degree_c_' + name] = average_degree_c
    dataframe['average_degree_g_' + name] = average_degree_g
    dataframe['anchor_' + name] = anchor
    dataframe['density_c_' + name] = density_c
    dataframe['density_g_' + name] = density_g
    dataframe['edges_dist_' + name] = edges_dist
    dataframe['externality_' + name] = externality
    dataframe['conductance_' + name] = conductance
    dataframe['expansion_' + name] = expansion
    dataframe['mean_community_size_' + name] = mean_community_size
    dataframe['NED_' + name] = NED
    dataframe['NED_index_' + name] = NED_index
    dataframe['last_stability_' + name] = last_stability
    dataframe['stability_' + name] = stability
    dataframe['delta_density_' + name] = delta_density
    dataframe['delta_Node_' + name] = delta_node
    dataframe['delta_Degree_' + name] = delta_degree
    dataframe['delta_expansion_' + name] = delta_connectivity

    return dataframe
