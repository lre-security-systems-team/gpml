# contributors: Pierre Parrend, Amani Abou-Rida
# project started on 11/10/2022

import networkx as nx
import matplotlib.pyplot as plt
from data.graphviz_config_bot_iot import *


def extract_graph(dataset, graph_type, label, src_addr, dst_addr, src_port=0, dst_port=0, src_mac=0, dst_mac=0):

    """ Extract the graph attack according to the type we need to show

        Returns
        -------
        graph of nodes and edges , attack_list for the type of attack found on the edge between two nodes (ip sources) ,
        attack_labels for the label of the attacks in the dataset

        Parameters
        ----------
        :param dataset: dataset
        :param dst_mac: destination mac
        :param src_mac: source mac
        :param dst_port: destination port
        :param src_port: source port
        :param dst_addr: destination ip
        :param src_addr: source ip
        :param label: label of the attack
        :param graph_type: type of graph we need to show (ip, ip + proto, or ip + mac)
        """

    format_supported = True
    if graph_type == 'ip':
        dataset['source_node'] = dataset[src_addr]
        dataset['dst_node'] = dataset[dst_addr]
    else:
        if graph_type == 'ip_proto':
            comp_field1 = 'proto'
            comp_field2 = 'proto'
        elif graph_type == 'mac':
            comp_field1 = src_mac
            comp_field2 = dst_mac
        else:
            format_supported = False
        if format_supported:
            dataset['source_node'] = dataset[src_addr] + ':' + dataset[comp_field1].astype(str)
            dataset['dst_node'] = dataset[dst_addr] + ':' + dataset[comp_field2].astype(str)
        else:
            print('Graph type not supported :' + graph_type)

    dataset['edge'] = dataset['source_node'] + ':' + dataset['dst_node']
    dataset['edge_label'] = dataset[src_port].astype(str) + ':' + dataset[dst_port].astype(str) + ':' + dataset[label].astype(str)
    graph = nx.DiGraph()
    pos = nx.spring_layout(graph)
    sip_list = dataset['source_node']
    dip_list = dataset['dst_node']
    ip_list = [*sip_list, *dip_list]

    graph.add_nodes_from(ip_list)
    attack_labels = dataset[label]
    edge_label_list = dataset['edge_label']

    attack_list = []

    for u, v, d in zip(sip_list, dip_list, edge_label_list):

        label = d.split(':')[2]
        colors = {
            0: 'blue',
            1: 'red',
        }
        color = colors[int(label[0])]
        graph.add_edge(u, v, label=d, color=color)

    for k, v, d in graph.edges(data=True):
        attack_list.append(d['label'].split(':')[2])

    return graph, attack_list, attack_labels

def show_graph(graph):

    """ Show the graph we extracted as nodes and edges in networkx

        Parameters
        ----------
        graph : graph with nodes and edges

        Returns
        -------
        display a graph with nodes and edges using networkx
        """

    plt.figure(figsize=(20, 8))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph)
    plt.show()


def show_graph_as_html(graph, attack_name, url, title):

    """ Show the graph we extracted as nodes and edges in html file

        Returns
        -------
        display a graph with nodes and edges using pyvis and save it as html file

        Parameters
        ----------
        :param graph: graph with nodes and edges
        :param title: title of the html page
        :param url: URL for saving html file
        :param attack_name: name of the attack
        """

    from pyvis.network import Network
    net = Network(notebook=True, directed=True, height='600px', width='90%',
                  bgcolor='#222222', font_color="purple",
                  heading=attack_name)
    net.from_nx(graph)
    net.show(url + '/' + title + ".html")
    net.show_buttons(filter_=['physics'])

def print_graph(dataset, graph_type, label, src_addr, dst_addr, url=url,title=title,attack_name=attack_name, src_port=0, dst_port=0, src_mac=0, dst_mac=0):

    """ print the graph we extracted as nodes and edges in html file and networkx

        Returns
        -------
        print a graph with nodes and edges


        Parameters
        ----------
        :param dataset: dataset
        :param dst_mac: destination mac
        :param src_mac: source mac
        :param dst_port: destination port
        :param src_port: source port
        :param dst_addr: destination ip
        :param src_addr: source ip
        :param label: label of the attack
        :param graph_type: type of graph we need to show (ip, ip + proto, or ip + mac)
        """
    connectivity_graph, attack_list, attack_labels = extract_graph(dataset, graph_type, label, src_addr,
                                                                   dst_addr, src_port, dst_port, src_mac, dst_mac)
    show_graph(connectivity_graph)
    show_graph_as_html(connectivity_graph, attack_name, url, title)
