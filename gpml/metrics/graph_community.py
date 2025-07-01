"""
This module provides all functions related to community graphs.

Contributors:
    Julien MICHEL

Project started on:
    11/10/2022
"""

# contributors: Julien MICHEL
# project started on 11/10/2022

import sys
from statistics import mean

import networkx as nx

sys.path.append('.')


def gc_metrics(G, partition):
    """
    Compute different graph community metrics.

    Return`s
    -------
    List of the different calculated graph community metrics

    Parameters
    ----------
    :param G: networkx  multigraph
    :param partition: a community partition of G
    """
    # pre-conditions
    if G.number_of_nodes() < 2:
        print('The graph should have at least 3 nodes')
        return 0

    # Initialize all necessary variables
    # for coverage
    indegree = 0  # indegree of G
    indegree_c = 0  # indegree of connexion =/= edges
    total_c = 0  # total number of connexion in G

    # Get the number of community
    n_community = max(G.nodes[elm]['community'] for elm in G.nodes) + 1  # complexity N
    nb_node_com = [0] * n_community
    center = [0] * n_community

    # Initialise indegree ;
    nx.set_node_attributes(G, 0, "indegree")
    nx.set_node_attributes(G, 0, "outdegree")

    density = [0] * n_community
    fail_density = [0] * n_community
    externality = [0] * n_community
    in_community = [(0, 0)] * n_community  # different edges count, total edge count
    ex_community = [0] * n_community  # total edge count

    # for modularity
    partition_size = [0] * (max(partition.values()) + 1)

    # for unifiability
    uni_in = [0] * n_community
    uni_out = [0] * n_community
    uni = []
    # for isolability
    out = [0] * n_community
    size = [0] * n_community
    iso = [1] * n_community

    # Loop over each edge in the graph
    for edge in G.edges:

        # if the edge has both node in the community
        if G.nodes[edge[0]]['community'] == G.nodes[edge[1]]['community']:

            indegree = indegree + 1

            # if the edge is the first repertoried between the two nodes
            if edge[2] == 0:
                indegree_c = indegree_c + 1

                in_community[G.nodes[edge[0]]['community']] = (in_community[G.nodes[edge[0]]['community']][0] + 1,
                                                               in_community[G.nodes[edge[0]]['community']][1] + 1)
                # Update node indegree
                G.nodes[edge[0]]['indegree'] = G.nodes[edge[0]]['indegree'] + 1
                G.nodes[edge[1]]['indegree'] = G.nodes[edge[1]]['indegree'] + 1

                # if old center has less indegree, it's not the center anymore
                if center[G.nodes[edge[0]]['community']] == 0:
                    center[G.nodes[edge[0]]['community']] = edge[0]
                elif G.nodes[center[G.nodes[edge[0]]['community']]]['indegree'] < G.nodes[edge[0]]['indegree']:
                    if G.nodes[edge[0]]['indegree'] >= G.nodes[edge[1]]['indegree']:
                        center[G.nodes[edge[0]]['community']] = edge[0]
                    else:
                        center[G.nodes[edge[0]]['community']] = edge[1]

            # Any other edge in the community
            else:
                in_community[G.nodes[edge[0]]['community']] = (
                    in_community[G.nodes[edge[0]]['community']][0],
                    in_community[G.nodes[edge[0]]['community']][1] + 1)

        # if the edge is between two nodes of different community
        else:
            ex_community[G.nodes[edge[0]]['community']] = ex_community[G.nodes[edge[0]]['community']] + 1
            ex_community[G.nodes[edge[1]]['community']] = ex_community[G.nodes[edge[1]]['community']] + 1
            G.nodes[edge[0]]['outdegree'] = G.nodes[edge[0]]['outdegree'] + 1
            G.nodes[edge[1]]['outdegree'] = G.nodes[edge[1]]['outdegree'] + 1

        # if the edge is the first between two nodes
        if edge[2] == 0:
            total_c = total_c + 1

    # Loop on graph nodes
    for node in G.nodes:
        nb_node_com[G.nodes[node]['community']] = nb_node_com[G.nodes[node]['community']] + 1  # Complexity N

        # Unifiability
        uni_in[G.nodes[node]['community']] = uni_in[G.nodes[node]['community']] + G.nodes[node]['indegree']
        uni_out[G.nodes[node]['community']] = uni_out[G.nodes[node]['community']] + G.nodes[node]['outdegree']

        # Isolability
        # If there is at least 1 edge going out of the community by this node
        if G.nodes[node]['outdegree'] > 0:
            out[G.nodes[node]['community']] = out[G.nodes[node]['community']] + 1
        size[G.nodes[node]['community']] = size[G.nodes[node]['community']] + 1

    # Loop on partition
    for elm in partition:
        partition_size[partition[elm]] = partition_size[partition[elm]] + 1
    nb_edges = G.number_of_edges()
    nb_nodes = G.number_of_nodes()
    total = 0
    for elm in partition_size:
        total = total + (nb_edges / nb_nodes) * (elm / nb_nodes) * elm
    for incr in range(0, n_community):
        uni.append(uni_out[incr] / uni_in[incr])
        iso[incr] = 1 - (out[incr] / size[incr])

    # Assign return values

    # For each community
    for i in range(0, n_community):  # Complexity P
        density[i] = 1
        density[i] = in_community[i][0] / (((nb_node_com[i] - 1) * nb_node_com[i]) / 2)
        fail_density[i] = 1
        fail_density[i] = in_community[i][0] / (((nb_node_com[i] + 1) * nb_node_com[i]) / 2)
        externality[i] = ex_community[i] / (ex_community[i] + in_community[i][1])

    mean_partition_size = mean(partition_size)
    coverage = indegree / nb_edges
    total = total / nb_edges

    modularity = coverage - total
    number_community_per_data = n_community / nb_edges

    metrics = {'coverage': coverage, 'modularity': modularity, 'density': density, 'er_density': fail_density,
               'externality': externality,
               'mean_partition_size': mean_partition_size, 'number_community_per_data': number_community_per_data,
               'isolability': iso, 'unifiability': uni, 'center': center}

    return metrics


def compute_stabilities(g1, g2, nb_of_communities, old_stabilities, t):
    """
    Compute stabilities between 2 graph and update global stabilities if there is a need to.

    Returns
    -------
    Stabilies between communities of g1 and g2, and global stabilities for corresponding communities.

    Parameters
    ----------
    :param g1: networkx multigraph
    :param g2: networkx multigraph
    :param nb_of_communities: int
    :param old_stabilities: float List
    :t: int
    """
    same = [0] * nb_of_communities
    not_same = [0] * nb_of_communities
    nb_node_comm2 = [0] * nb_of_communities
    nb_node_comm1 = [0] * nb_of_communities
    # For each nodes in the graph at t+1
    for elm in g2.nodes:
        nb_node_comm2[g2.nodes[elm]['community']] = nb_node_comm2[g2.nodes[elm]['community']] + 1
        try:
            if g2.nodes[elm]['community'] == g1.nodes[elm]['community']:
                same[g2.nodes[elm]['community']] = same[g2.nodes[elm]['community']] + 1
            else:
                not_same[g2.nodes[elm]['community']] = not_same[g2.nodes[elm]['community']] + 1
        # Ip not in g1 at time t
        except Exception:
            not_same[g2.nodes[elm]['community']] = not_same[g2.nodes[elm]['community']] + 1

    for elm in g1.nodes:
        nb_node_comm1[g1.nodes[elm]['community']] = nb_node_comm1[g1.nodes[elm]['community']] + 1

    stabilities = [0] * nb_of_communities
    new_stabilities = old_stabilities.copy()
    for i in range(0, nb_of_communities):
        if nb_node_comm2[i] != 0 and nb_node_comm1[i] != 0:
            stabilities[i] = same[i] / nb_node_comm2[i] - (
                    not_same[i] / nb_node_comm2[i] + (nb_node_comm1[i] - same[i]) / nb_node_comm1[i]) / 2
        new_stabilities[i] = (old_stabilities[i] * (t - 1) + stabilities[i]) / t
    return stabilities, new_stabilities


def propagate_communities(g1, g2, center, center_t):
    """
    Propagate stabilities from g1 to g2.

    Returns
    -------
    New centers of communities in g2 and update of communities index in g2

    Parameters
    ----------
    :param g1: networkx multigraph
    :param g2: networkx multigraph
    :param center: Nodes id list
    :param center_t: Nodes id list
    """
    # Get old center new community
    center_where = []
    not_in = len(center)
    for elm in center_t:
        try:
            center_where.append(g1.nodes[elm]['community'])
        except Exception:  # not_in
            center_where.append(not_in)
            not_in = not_in + 1
    # For each node in the graph at time t+1
    for elm in g2.nodes:
        g2.nodes[elm]['old_community'] = g2.nodes[elm]['community']
        g2.nodes[elm]['community'] = center_where[g2.nodes[elm]['community']]
    return center_where


def gc_metrics_first_order(G):
    """
    Compute different graph and graph community metrics of first order.

    This function calculates metrics by one run over the edges and nodes of the graph
    and does not require the values of other metrics for calculation.

    Returns
    -------
    List of the different calculated graph community metrics
    List of the different calculated graph metrics
    Parameters
    ----------
    :param G: networkx  multigraph
    """
    # pre-conditions
    nb_node = G.number_of_nodes()
    if nb_node < 2:
        print('The graph should at least have 3 nodes')
        return 0

    indegree = 0  # indegree of G
    outdegree = 0  # outdegree of G
    n_community = max(G.nodes[elm]['community'] for elm in G.nodes) + 1  # complexity N
    indegree_c = [0] * n_community
    outdegree_c = [0] * n_community
    nb_edge = 0
    nb_node_com = [0] * n_community
    nb_edge_com = [0] * n_community
    anchor = [0] * n_community
    n_connexion = 0
    n_connexion_c = [0] * n_community
    center = [0] * n_community

    # Initialise indegree
    nx.set_node_attributes(G, 0, "indegree")
    nx.set_node_attributes(G, 0, "outdegree")
    for edge in G.edges:
        node1_community = G.nodes[edge[0]]['community']
        node2_community = G.nodes[edge[1]]['community']
        nb_edge = nb_edge + 1
        # if the edge has both node in the community
        if node1_community == node2_community:
            # Increment the number of edges within the same community
            nb_edge_com[node1_community] += 1
            # Increment the overall indegree count
            indegree += 2
            # Update node indegree
            G.nodes[edge[0]]['indegree'] += 1
            G.nodes[edge[1]]['indegree'] += 1
            # Update the indegree count for the community
            indegree_c[node1_community] += 2
            # if the edge is the first repertoried between the two nodes
            if edge[2] == 0:
                n_connexion += 1
                n_connexion_c[node1_community] += 1
                # if old center has less indegree, it's not the center anymore
                if center[node1_community] == 0:
                    center[node1_community] = edge[0]
                elif G.nodes[center[node1_community]]['indegree'] < G.nodes[edge[0]]['indegree']:
                    if G.nodes[edge[0]]['indegree'] >= G.nodes[edge[1]]['indegree']:
                        center[node1_community] = edge[0]
                    else:
                        center[node1_community] = edge[1]
        # if the edge is between two node of different community
        else:
            nb_edge_com[node1_community] += 1
            nb_edge_com[node2_community] += 1
            outdegree_c[node1_community] += 1
            outdegree_c[node2_community] += 1
            outdegree = outdegree + 2
            G.nodes[edge[0]]['outdegree'] = G.nodes[edge[0]]['outdegree'] + 1
            G.nodes[edge[1]]['outdegree'] = G.nodes[edge[1]]['outdegree'] + 1
            if edge[2] == 0:
                n_connexion = n_connexion + 1
                anchor[node1_community] = anchor[node1_community] + 1
                anchor[node2_community] = anchor[node1_community] + 1
    for node in G.nodes:
        nb_node_com[G.nodes[node]['community']] = nb_node_com[G.nodes[node]['community']] + 1  # Complexity N

    # metrics
    metrics_c = {'nb_of_nodes': nb_node_com,
                 'in_degree': indegree_c,
                 'out_degree': outdegree_c,
                 'nb_of_edges': nb_edge_com,
                 'anchor': anchor,
                 'n_connection': n_connexion_c,
                 'center': center}
    metrics_g = {'nb_of_nodes': nb_node,
                 'in_degree': indegree,
                 'out_degree': outdegree,
                 'nb_of_edges': nb_edge,
                 'n_connection': n_connexion}
    return metrics_c, metrics_g


def gc_metrics_second_order(forder_metrics_c, forder_metrics_g):
    """
     Compute different graph and graph community metrics of second order.

    This function calculates metrics by one run over all the first-order metrics
    for a given set of community and a graph.
    Returns
    -------
    List of the different calculated graph community metrics
    List of the different calculated graph metrics
    Parameters
    ----------
    :param forder_metrics_c: community metrics of first order
    :param forder_metrics_g: graph metrics of first order
    """
    n_community = len(forder_metrics_c['in_degree'])
    average_degree = (forder_metrics_g['in_degree'] + forder_metrics_g['out_degree']) / forder_metrics_g['nb_of_nodes']
    average_degree_c = [0] * n_community
    density = forder_metrics_g['n_connection'] / ((forder_metrics_g['nb_of_nodes'] * (forder_metrics_g['nb_of_nodes'] - 1)) / 2)
    density_c = [0] * n_community
    edges_dist = 0
    externality = [0] * n_community
    conductance = [0] * n_community
    expansion = [0] * n_community
    NED = [0] * n_community
    NED_index = 0

    for i in range(0, n_community):

        in_degree_i = forder_metrics_c['in_degree'][i]
        out_degree_i = forder_metrics_c['out_degree'][i]
        nb_of_edges_i = forder_metrics_c['nb_of_edges'][i]
        nb_of_nodes_i = forder_metrics_c['nb_of_nodes'][i]
        n_connection_i = forder_metrics_c['n_connection'][i]
        denominator_density = (nb_of_nodes_i * (nb_of_nodes_i - 1)) / 2

        # Calculate the average degree for node i, handling potential division by zero
        if nb_of_nodes_i > 0:
            average_degree_c[i] = (in_degree_i + out_degree_i) / nb_of_nodes_i
        else:
            average_degree_c[i] = 0
        try:
            if denominator_density > 0:
                density_c[i] = n_connection_i / denominator_density
            else:
                density_c[i] = 1
        except (TypeError, ValueError) :
            density_c[i] = 1

        edges_dist = edges_dist + nb_of_edges_i
        externality[i] = out_degree_i / (out_degree_i + (in_degree_i / 2))
        conductance[i] = out_degree_i / (out_degree_i + in_degree_i)
        expansion[i] = out_degree_i / nb_of_nodes_i
        NED[i] = (nb_of_nodes_i + n_connection_i + in_degree_i) / (
                nb_of_nodes_i + ((nb_of_nodes_i * (nb_of_nodes_i - 1)) / 2) + in_degree_i + out_degree_i)
        NED_index = NED_index + NED[i] * in_degree_i

    edges_dist = edges_dist / forder_metrics_g['nb_of_edges']
    NED_index = NED_index / (forder_metrics_g['in_degree'] + forder_metrics_g['out_degree'])
    mean_community_size = mean(forder_metrics_c['nb_of_nodes'])
    # metrics
    so_metrics_c = {'average_degree': average_degree_c,
                    'density': density_c,
                    'externality': externality,
                    'conductance': conductance,
                    'expansion': expansion,
                    'NED': NED}
    so_metrics_g = {'average_degree': average_degree,
                    'density': density,
                    'edges_dist': edges_dist,
                    'NED_index': NED_index,
                    'mean_community_size': mean_community_size}
    return so_metrics_c, so_metrics_g
