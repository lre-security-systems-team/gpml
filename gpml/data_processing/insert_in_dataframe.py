# contributors: Julien MICHEL
# project started on 11/10/2022

import unittest
import networkx as nx
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import scipy
import matplotlib.cm as cm
import time
from statistics import mean
import timeit
import sys
sys.path.append('.')
from library.community import community_louvain
import gpml.metrics.graph_community as gc

def chunk_read_and_insert_gc_metrics(file_name,output_csv, time_interval, community_strategy='louvain',
            chunksize = 5000000,date_time = "Date time",edge_source = "Source IP",edge_dest = "Destination IP"):
    """  Use pandas to read a csv by chunk and write another csv with the same field + Ip graph communities metrics

        Returns
        -------
        output csv file with community metrics

        Parameters
        ----------
        :param file_name: input csv file
        :param time_interval: timedelta of graph window
        :param community_strategy: Nodes id list
        :param chunksize: nb of csv line to load in memory as int
        :param date_time: Name of the date field in csv
        :param edge_source: Name of field in csv for edge source id
        :parma edge_dest: Name of field in csv for edge destination id
        """

###### First Initialisation of dataframe ######
    df = pd.read_csv(file_name,nrows = chunksize)
    df.head()
    df[date_time] = df[date_time].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df = df.sort_values(date_time)

######
    Density = [0] * df.shape[0]
    Externality = [0] * df.shape[0]
    last_stability = [0] * df.shape[0]
    Stability = [0] * df.shape[0]
    ##########################################
    # Communities relevance metrics
    Coverage = [0] * df.shape[0]
    Coverage_co = [0] * df.shape[0]
    Modularity = [0] * df.shape[0]
    Mean_communities_size = [0] * df.shape[0]
    Unifiability = [0] * df.shape[0]
    Isolability = [0] * df.shape[0]
    Number_community_per_data = [0] * df.shape[0]
    ##########################################


 #get the first date
    first = df[date_time][0]
    count = 0
    old_count = 0
    stop = 0
    size = 0
    skip = chunksize +1
    t = 1
    current_stop = time_interval

    #create first graph
    MG = nx.MultiGraph()
    while df[date_time][count] <  first + current_stop:
        MG.add_edge(df[edge_source][count],df[edge_dest][count],label=df['Label'][count],index=count)
        count = count + 1

    ########### Community partition making ###########
    #

    if community_strategy == 'lpa' : # Label propagation algorithm
        pre_partition = list(nx_comm.label_propagation.label_propagation_communities(MG))
        part_count = 0
        partition = {}
        for elm in pre_partition:
            for key in elm:
                partition[key] = part_count
            part_count = part_count + 1

    else : #(default or louvain) # Louvain algorithm
        partition = community_louvain.best_partition(MG)
    nx.set_node_attributes(MG, partition, "community")

    metrics = gc.gc_metrics(MG, partition)
    center = metrics['center']
    ########## Assign metrics value to dataframe ##########


    for elm in MG.edges.data():
        Density[elm[2]['index']] = metrics['density'][MG.nodes[elm[0]]['community']]
        Externality[elm[2]['index']] = metrics['externality'][MG.nodes[elm[0]]['community']]
        Coverage[elm[2]['index']] = metrics['coverage']
        Modularity[elm[2]['index']] = metrics['modularity']
        Mean_communities_size[elm[2]['index']] = metrics['mean_partition_size']
        Unifiability[elm[2]['index']] = metrics['unifiability'][MG.nodes[elm[0]]['community']]
        Isolability[elm[2]['index']] =  metrics['isolability'][MG.nodes[elm[0]]['community']]
        Number_community_per_data[elm[2]['index']] = metrics['number_commmunity_per_data']



    ######### Write metrics to csv ##########

    # initialize new columns to 0
    df['Density'] = Density
    df['Externality'] = Externality
    df['Coverage'] = Coverage
    df['Modularity'] = Modularity
    df['Mean_communities_size'] =  Mean_communities_size
    df['Unifiability'] = Unifiability
    df['Isolability'] = Isolability
    df['Number_community_per_data'] = Number_community_per_data

    df['last_stability'] = 0 # Stay 0 for first graph
    df['Stability'] = 0 # Stay 0 for first graph
    old_stabilities = [0] * df.shape[0]


    df[0:count].to_csv(output_csv,index = False,mode='w') # mode 'w' since it's the start of the file

    # At this point the first graph is computed and written to csv.
    current_stop = current_stop + time_interval
    """ Tentative 1"""
    ######## Compute others graphs ##########
    create = 1
    offset = count
    lmt = count

    dataframe = pd.DataFrame()
    sep_chunk = 0
    while stop == 0:
        # Graph t+1
        if create == 1 :
            t = t + 1
            MG2 = nx.MultiGraph()
            create = 0
        try:
            #print('vu ' + str(df[date_time][count%chunksize]))

            if( sep_chunk == 0):
                while df[date_time][count%chunksize] <  first + current_stop :
                    MG2.add_edge(df[edge_source][count%chunksize],df[edge_dest][count%chunksize],label=df['Label'][count%chunksize],index=count)
                    if ((offset)%chunksize == 0 ): ### fin d'un chunk
                        print('fin chunk')
                        plant
                    count = count + 1
                    offset = count

            else :
                #print( 'test')
                for elm in MG2.edges.data():
                    elm[2]['index'] = elm[2]['index'] - lmt
                #print('test 2')
                while df[date_time][count%chunksize] <  first + current_stop :
                    MG2.add_edge(df[edge_source][count%chunksize],df[edge_dest][count%chunksize],label=df['Label'][count%chunksize],index=(chunksize + 1 -lmt) + count%chunksize)
                    if ((offset)%chunksize == 0 ): ### fin d'un chunk
                        print('fin chunk')
                        plant
                    count = count + 1
                    offset = count

            #print('pasvu')
            create = 1 # next time in the loop a new graph will be needed
            partition2 = community_louvain.best_partition(MG2)
            #(print('ok'))
            current_stop = current_stop + time_interval
            nx.set_node_attributes(MG2, partition2, "community")
            #print('?')
            metrics2 = gc.gc_metrics(MG2, partition2)
            #print(metrics2['isolability'])
            center2 = metrics2['center']
            ######## Assign metrics value to dataframe ########
            #print('ahahahahaha')
            size += (max(gc.propagate_community(MG,MG2,center,center2)) + 1)
            #print( "ici")
            old_stabilities += [0] * (size - len(old_stabilities))
            #print("la")
            stabilities,new_stabilities = gc.compute_stabilities(MG,MG2,size,old_stabilities,t)
            #(print('avant boucle'))
            for elm in MG2.edges.data():

                Density[elm[2]['index']%chunksize] = metrics2['density'][MG2.nodes[elm[0]]['old_community']]

                Externality[elm[2]['index']%chunksize] = metrics2['externality'][MG2.nodes[elm[0]]['old_community']]

                Coverage[elm[2]['index']%chunksize] = metrics2['coverage']

                Modularity[elm[2]['index']%chunksize] = metrics2['modularity']

                Mean_communities_size[elm[2]['index']%chunksize] = metrics2['mean_partition_size']
                Unifiability[elm[2]['index']%chunksize] = metrics2['unifiability'][MG2.nodes[elm[0]]['old_community']]
                Isolability[elm[2]['index']%chunksize] =  metrics2['isolability'][MG2.nodes[elm[0]]['old_community']]
                Number_community_per_data[elm[2]['index']%chunksize] = metrics2['number_commmunity_per_data']



                last_stability[elm[2]['index']%chunksize] = stabilities[MG2.nodes[elm[0]]['old_community']]
                Stability[elm[2]['index']%chunksize] = new_stabilities[MG2.nodes[elm[0]]['old_community']]




            #print('avant center')
            center = center2
            ##### Write community metrics to the csv #######
            #(print('apres boucle'))

            if sep_chunk == 0:

                df['Density'] = pd.Series(Density)
                df['Externality'] = pd.Series(Externality)
                df['Coverage'] = pd.Series(Coverage)
                df['Modularity'] = pd.Series(Modularity)
                df['Mean_communities_size'] =  pd.Series(Mean_communities_size)
                df['Unifiability'] = pd.Series(Unifiability)
                df['Isolability'] = pd.Series(Isolability)

                df['Number_community_per_data'] = pd.Series(Number_community_per_data)

                df['last_stability'] = pd.Series(last_stability)
                df['Stability'] = pd.Series(Stability)

                df[lmt:(count-1)%chunksize].to_csv(output_csv,index = False,header = False,mode='a') # mode 'a' since we want to append
                lmt = (count-1)%chunksize
                #(print('ecriture'))


            if sep_chunk == 1:
                #print('cas special')
                sep_chunk = 0

                dataframe = pd.concat([dataframe,df[0:(count-1)%chunksize]])

                dataframe['Density'] = Density[0:dataframe.shape[0]]
                dataframe['Externality'] = Externality[0:dataframe.shape[0]]
                dataframe['Coverage'] = Coverage[0:dataframe.shape[0]]
                dataframe['Modularity'] = Modularity[0:dataframe.shape[0]]
                dataframe['Mean_communities_size'] =  Mean_communities_size[0:dataframe.shape[0]]
                dataframe['Unifiability'] = Unifiability[0:dataframe.shape[0]]
                dataframe['Isolability'] = Isolability[0:dataframe.shape[0]]
                dataframe['Number_community_per_data'] = Number_community_per_data[0:dataframe.shape[0]]

                dataframe['last_stability'] = last_stability[0:dataframe.shape[0]]
                dataframe['Stability'] = Stability[0:dataframe.shape[0]]

                dataframe.to_csv(output_csv,index = False,header = False,mode='a') # mode 'a' since we want to append
                lmt = (count-1)%chunksize


            MG = MG2.copy()
            ####################
            ### store_graphe to implement
            ####################

        except Exception as err:
            try : # read the next chunk
                (print(count))
################################################################################################################################
                if(old_count) == count:

                    partition2 = community_louvain.best_partition(MG2)
                    #(print('ok'))
                    current_stop = current_stop + time_interval
                    nx.set_node_attributes(MG2, partition2, "community")
                    #print('?')
                    metrics2 = gc.gc_metrics(MG2, partition2)
                    #print('wtf?')
                    center2 = metrics2['center']
                    ######## Assign metrics value to dataframe ########
                    #print('ahahahahaha')
                    size += (max(gc.propagate_community(MG,MG2,center,center2)) + 1)
                    #print( "ici")
                    old_stabilities += [0] * (size - len(old_stabilities))
                    #print("la")
                    stabilities,new_stabilities = gc.compute_stabilities(MG,MG2,size,old_stabilities,t)
                    #(print('avant boucle'))
                    for elm in MG2.edges.data():

                        Density[elm[2]['index']%chunksize] = metrics2['density'][MG2.nodes[elm[0]]['old_community']]

                        Externality[elm[2]['index']%chunksize] = metrics2['externality'][MG2.nodes[elm[0]]['old_community']]

                        Coverage[elm[2]['index']%chunksize] = metrics2['coverage']

                        Modularity[elm[2]['index']%chunksize] = metrics2['modularity']

                        Mean_communities_size[elm[2]['index']%chunksize] = metrics2['mean_partition_size']
                        Unifiability[elm[2]['index']%chunksize] = metrics2['unifiability'][MG2.nodes[elm[0]]['old_community']]
                        Isolability[elm[2]['index']%chunksize] =  metrics2['isolability'][MG2.nodes[elm[0]]['old_community']]
                        Number_community_per_data[elm[2]['index']%chunksize] = metrics2['number_commmunity_per_data']



                        last_stability[elm[2]['index']%chunksize] = stabilities[MG2.nodes[elm[0]]['old_community']]
                        Stability[elm[2]['index']%chunksize] = new_stabilities[MG2.nodes[elm[0]]['old_community']]




                    #print('avant center')
                    center = center2
                    ##### Write community metrics to the csv #######
                    #(print('apres boucle'))

                    copy['Density'] = Density[0:copy.shape[0]]
                    copy['Externality'] = Externality[0:copy.shape[0]]
                    copy['Coverage'] = Coverage[0:copy.shape[0]]
                    copy['Modularity'] = Modularity[0:copy.shape[0]]
                    copy['Mean_communities_size'] =  Mean_communities_size[0:copy.shape[0]]
                    copy['Unifiability'] = Unifiability[0:copy.shape[0]]
                    copy['Isolability'] = Isolability[0:copy.shape[0]]
                    copy['Number_community_per_data'] = Number_community_per_data[0:copy.shape[0]]

                    copy['last_stability'] = last_stability[0:copy.shape[0]]
                    copy['Stability'] = Stability[0:copy.shape[0]]

                    copy.to_csv(output_csv,index = False,header = False,mode='a') # mode 'a' since we want to append
                    stop = 1
################################################################################################################################

                copy = df[lmt:(count)%chunksize].copy()
                dataframe = df[lmt:]

                df = pd.read_csv(file_name,nrows = chunksize, skiprows= range(1,skip))
                df[date_time] = df[date_time].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                df = df.sort_values(date_time)
                skip = skip + chunksize
                offset = count + 1
                sep_chunk = 1


                old_count = count

            except :
                stop = 1

    """ ........ """
    return

def insert_metrics_to_dataframe(dataframe,time_interval,date_time,edge_source,edge_dest,label,name,date_timestamp):
    """  Take a dataframe, represent it on dynamic graph and compute community metrics.

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
        """
    if(type(edge_source) != list or type(edge_dest) != list):
        print('edge vertice have to be given as list')
        return dataframe

    if(date_timestamp == False):
        dataframe[date_time] = dataframe[date_time].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
     # sort dataframe by date to make the time interval selection
    dataframe = dataframe.sort_values(date_time)
    #get the first date
    first = dataframe[date_time][0]
    count = 0
    stop = 0
    size = 0
    t = 1
    current_stop = time_interval

    Community = [0] * dataframe.shape[0]
    #1rs Order metrics
    Anchor = [0] * dataframe.shape[0]
    Average_degree_c = [0] * dataframe.shape[0]
    Average_degree_g = [0] * dataframe.shape[0]
    nb_of_edges_c = [0] * dataframe.shape[0]
    nb_of_nodes_c = [0] * dataframe.shape[0]


    #2nd Order metrics
    Density_c = [0] * dataframe.shape[0]
    Density_g = [0] * dataframe.shape[0]
    Edges_dist = [0] * dataframe.shape[0]
    Externality = [0] * dataframe.shape[0]
    Conductance = [0] * dataframe.shape[0]
    Expansion = [0] * dataframe.shape[0]
    NED = [0] * dataframe.shape[0]
    NEDIndex = [0] * dataframe.shape[0]
    Mean_community_size = [0] * dataframe.shape[0]
    #Dynamic metrics
    last_stability = [0] * dataframe.shape[0]
    Stability = [0] * dataframe.shape[0]



    ##########################################
    # Communities relevance metrics
    Mean_communities_size = [0] * dataframe.shape[0]
    Number_community_per_data = [0] * dataframe.shape[0]
    ##########################################
    # Initial graph
    MG = nx.MultiGraph()

    while dataframe[date_time][count] <  first + current_stop:

        if(len(edge_source) > 1): # Multiple column as edge source

            source ="_".join([str(dataframe[item][count]) for item in edge_source])
        else:
            source = dataframe[edge_source[0]][count]
        if(len(edge_dest) > 1): # Multiple column as edge dest
            destination ="_".join([str(dataframe[item][count]) for item in edge_dest])
        else:
            destination =  dataframe[edge_dest[0]][count]
        MG.add_edge(source,destination,label=dataframe[label][count],index=count)
        count = count + 1
    partition = community_louvain.best_partition(MG)
    nx.set_node_attributes(MG, partition, "community")

    current_stop = current_stop + time_interval

    fo_metrics_c,fo_metrics_g = gc.gc_metrics_first_order(MG)
    center = fo_metrics_c['center']
    so_metrics_c, so_metrics_g = gc.gc_metrics_second_order(fo_metrics_c,fo_metrics_g)
    ########## Assign metrics value to dataframe ##########


    for elm in MG.edges.data():

        Community[elm[2]['index']] = MG.nodes[elm[0]]['community']
        #1rs Order metrics
        Anchor[elm[2]['index']] = fo_metrics_c['anchor'][MG.nodes[elm[0]]['community']]
        Average_degree_c[elm[2]['index']] = so_metrics_c['average_degree'][MG.nodes[elm[0]]['community']]
        Average_degree_g[elm[2]['index']] = so_metrics_g['average_degree']
        nb_of_edges_c[elm[2]['index']] = fo_metrics_c['nb_of_edges'][MG.nodes[elm[0]]['community']]
        nb_of_nodes_c[elm[2]['index']] = fo_metrics_c['nb_of_nodes'][MG.nodes[elm[0]]['community']]


        #2nd Order metrics
        Density_c[elm[2]['index']] = so_metrics_c['density'][MG.nodes[elm[0]]['community']]
        Density_g[elm[2]['index']] = so_metrics_g['density']
        Edges_dist[elm[2]['index']] = so_metrics_g['edges_dist']
        Externality[elm[2]['index']] = so_metrics_c['externality'][MG.nodes[elm[0]]['community']]
        Conductance[elm[2]['index']] = so_metrics_c['conductance'][MG.nodes[elm[0]]['community']]
        Expansion[elm[2]['index']] = so_metrics_c['expansion'][MG.nodes[elm[0]]['community']]
        NED[elm[2]['index']] = so_metrics_c['NED'][MG.nodes[elm[0]]['community']]
        NEDIndex[elm[2]['index']] = so_metrics_g['NEDIndex']
        Mean_community_size[elm[2]['index']] = so_metrics_g['mean_community_size']
    old_stabilities = [0] * dataframe.shape[0]

    print('First_graph generated and metrics computed')

    while stop == 0:
        t = t + 1
        # Graph t+1
        MG2 = nx.MultiGraph()
        try:
            while dataframe[date_time][count] <  first + current_stop:

                if(len(edge_source) > 1): # Multiple column as edge source

                    source ="_".join([str(dataframe[item][count]) for item in edge_source])
                else:
                    source = dataframe[edge_source[0]][count]
                if(len(edge_dest) > 1): # Multiple column as edge dest
                    destination ="_".join([str(dataframe[item][count]) for item in edge_dest])
                else:
                    destination =  dataframe[edge_dest[0]][count]

                MG2.add_edge(source,destination,label=dataframe[label][count],index=count)
                count = count + 1
            partition2 = community_louvain.best_partition(MG2)

            current_stop = current_stop + time_interval
            nx.set_node_attributes(MG2, partition2, "community")

            fo_metrics_c,fo_metrics_g = gc.gc_metrics_first_order(MG2)
            center2 = fo_metrics_c['center']
            so_metrics_c, so_metrics_g = gc.gc_metrics_second_order(fo_metrics_c,fo_metrics_g)



            for elm in MG2.edges.data():
                Community[elm[2]['index']] = MG2.nodes[elm[0]]['community']
                #1rs Order metrics
                Anchor[elm[2]['index']] = fo_metrics_c['anchor'][MG2.nodes[elm[0]]['community']]
                Average_degree_c[elm[2]['index']] = so_metrics_c['average_degree'][MG2.nodes[elm[0]]['community']]
                Average_degree_g[elm[2]['index']] = so_metrics_g['average_degree']
                nb_of_edges_c[elm[2]['index']] = fo_metrics_c['nb_of_edges'][MG2.nodes[elm[0]]['community']]
                nb_of_nodes_c[elm[2]['index']] = fo_metrics_c['nb_of_nodes'][MG2.nodes[elm[0]]['community']]


                #2nd Order metrics
                Density_c[elm[2]['index']] = so_metrics_c['density'][MG2.nodes[elm[0]]['community']]
                Density_g[elm[2]['index']] = so_metrics_g['density']
                Edges_dist[elm[2]['index']] = so_metrics_g['edges_dist']
                Externality[elm[2]['index']] = so_metrics_c['externality'][MG2.nodes[elm[0]]['community']]
                Conductance[elm[2]['index']] = so_metrics_c['conductance'][MG2.nodes[elm[0]]['community']]
                Expansion[elm[2]['index']] = so_metrics_c['expansion'][MG2.nodes[elm[0]]['community']]
                NED[elm[2]['index']] = so_metrics_c['NED'][MG2.nodes[elm[0]]['community']]
                NEDIndex[elm[2]['index']] = so_metrics_g['NEDIndex']
                Mean_community_size[elm[2]['index']] = so_metrics_g['mean_community_size']

            size += (max(gc.propagate_community(MG,MG2,center,center2)) + 1)

            old_stabilities += [0] * (size - len(old_stabilities))

            stabilities,new_stabilities = gc.compute_stabilities(MG,MG2,size,old_stabilities,t)

            for elm in MG2.edges.data():

                last_stability[elm[2]['index']] = stabilities[MG2.nodes[elm[0]]['community']]
                Stability[elm[2]['index']] = new_stabilities[MG2.nodes[elm[0]]['community']]



            old_stabilities = new_stabilities.copy()
            MG = MG2.copy()

            ####################
            ### store_graphe to implement
            ####################

        except Exception as err:
            print(err)
            stop = 1

    # initialize new columns to 0
    dataframe['Community_'+name] = Community
    dataframe['Nb_of_nodes_c_'+name] = nb_of_nodes_c
    dataframe['Nb_of_edges_c_'+name] = nb_of_edges_c
    dataframe['Average_degree_c_'+name] = Average_degree_c
    dataframe['Average_degree_g_'+name] = Average_degree_g
    dataframe['Anchor_'+name] = Anchor
    dataframe['Density_c_'+name] = Density_c
    dataframe['Density_g_'+name] = Density_g
    dataframe['Edges_dist'+name] = Edges_dist
    dataframe['Externality_'+name] = Externality
    dataframe['Conductance_'+name] = Conductance
    dataframe['Expansion_'+name] = Expansion
    dataframe['Mean_community_size_'+name] =  Mean_community_size
    dataframe['NED_'+name] = NED
    dataframe['NEDIndex_'+name] = NEDIndex
    dataframe['last_stability_'+name] = last_stability
    dataframe['Stability_'+name] = Stability



    return(dataframe)
