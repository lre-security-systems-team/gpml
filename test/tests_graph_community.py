# contributors: Julien MICHEL
# project started on 11/10/2022
import sys
from datetime import timedelta, datetime
import unittest
import networkx as nx
import pandas as pd

sys.path.append('.')
from library.community import community_louvain
from gpml.metrics import graph_community as g_comm


class TestStability(unittest.TestCase):

    def setUp(self):
        root = '.'
        self.filename = 'data/ugr16/ugr_sample_100k.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_values(self):
        self.df['Date time'] = self.df['Date time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        self.df = self.df.sort_values("Date time")
        count = 0
        first = self.df["Date time"][0]

        MG = nx.MultiGraph()
        MG2 = nx.MultiGraph()
        while self.df["Date time"][count] < first + timedelta(minutes=5):
            MG.add_edge(self.df["Source IP"][count], self.df["Destination IP"][count], label=self.df['Label'][count],
                        index=count)
            count = count + 1
        while self.df["Date time"][count] < first + timedelta(minutes=10):
            MG2.add_edge(self.df["Source IP"][count], self.df["Destination IP"][count], label=self.df['Label'][count],
                         index=count)
            count = count + 1

            partition = community_louvain.best_partition(MG)
            nx.set_node_attributes(MG, partition, "community")
            metrics = g_comm.gc_metrics(MG, partition)

            partition2 = community_louvain.best_partition(MG2)
            nx.set_node_attributes(MG2, partition2, "community")
            metrics2 = g_comm.gc_metrics(MG2, partition)
            size = len(metrics['center']) + (
                    max(g_comm.propagate_communities(MG, MG2, metrics['center'], metrics2['center'])) + 1)
            old_stabilities = [0] * (size)

            stabilities, _ = g_comm.compute_stabilities(MG, MG2, size, old_stabilities, 1)

        self.assertEqual(min(stabilities) > -2, True)
        self.assertEqual(max(stabilities) <= 1, True)

    def test_all_metrics_minimal_graph(self):
        self.df['Date time'] = self.df['Date time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        self.df = self.df.sort_values("Date time")
        MG = nx.MultiGraph()
        MG.add_edge(self.df["Source IP"][0], self.df["Destination IP"][0], label=self.df['Label'][0], index=0)

        partition = community_louvain.best_partition(MG)
        nx.set_node_attributes(MG, partition, "community")

        metrics = g_comm.gc_metrics(MG, partition)

        self.assertTrue(metrics['coverage'] == 1)
        self.assertTrue(metrics['modularity'] == 0)
        self.assertTrue(metrics['mean_partition_size'] == 2)
        self.assertTrue(max(metrics['density']) == 1)
        self.assertTrue(min(metrics['externality']) == 0)
        self.assertTrue(min(metrics['unifiability']) == 0)
        self.assertTrue(max(metrics['isolability']) == 1)

    def test_all_metrics_void_graph(self):
        MG = nx.MultiGraph()

        partition = community_louvain.best_partition(MG)
        nx.set_node_attributes(MG, partition, "community")

        metrics = g_comm.gc_metrics(MG, partition)
        self.assertEqual(metrics, 0)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        root = '.'
        self.filename = 'data/ugr16/ugr_sample_100k.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_all_values(self):
        self.df['Date time'] = self.df['Date time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        self.df = self.df.sort_values("Date time")
        count = 0
        first = self.df["Date time"][0]
        MG = nx.MultiGraph()
        while self.df["Date time"][count] < first + timedelta(minutes=5):
            MG.add_edge(self.df["Source IP"][count], self.df["Destination IP"][count], label=self.df['Label'][count],
                        index=count)
            count = count + 1
        partition = community_louvain.best_partition(MG)
        nx.set_node_attributes(MG, partition, "community")

        forder_metrics_c, forder_metrics_g = g_comm.gc_metrics_first_order(MG)
        so_metrics_c, so_metrics_g = g_comm.gc_metrics_second_order(forder_metrics_c, forder_metrics_g)

        # Best partition is stochastic , so values may varie
        self.assertTrue(min(forder_metrics_c['nb_of_nodes']) >= 2)
        self.assertTrue(min(forder_metrics_c['nb_of_edges']) > 0)

        self.assertTrue(min(forder_metrics_c['in_degree']) > 0)
        self.assertTrue(min(forder_metrics_c['out_degree']) >= 0)

        self.assertTrue(min(forder_metrics_c['anchor']) <= min(forder_metrics_c['out_degree']))
        self.assertTrue(min(forder_metrics_c['n_connection']) <= min(forder_metrics_c['nb_of_edges']))

        self.assertTrue(forder_metrics_g['nb_of_nodes'] >= 2)
        self.assertTrue(forder_metrics_g['nb_of_edges'] > 0)

        self.assertTrue(forder_metrics_g['in_degree'] > 0)
        self.assertTrue(forder_metrics_g['out_degree'] >= 0)
        self.assertTrue(forder_metrics_g['n_connection'] <= forder_metrics_g['nb_of_edges'])

        self.assertTrue(min(so_metrics_c['average_degree']) > 0)

        self.assertTrue(max(so_metrics_c['density']) <= 1)
        self.assertTrue(min(so_metrics_c['density']) > 0)

        self.assertTrue(max(so_metrics_c['externality']) < 1)
        self.assertTrue(min(so_metrics_c['externality']) >= 0)

        self.assertTrue(max(so_metrics_c['conductance']) < 1)
        self.assertTrue(min(so_metrics_c['conductance']) >= 0)

        self.assertTrue(max(so_metrics_c['expansion']) <= 1)
        self.assertTrue(min(so_metrics_c['expansion']) >= 0)

        self.assertTrue(min(so_metrics_c['NED']) > 0)

        self.assertTrue(so_metrics_g['average_degree'] > 0)

        self.assertTrue(so_metrics_g['density'] <= 1)
        self.assertTrue(so_metrics_g['density'] > 0)

        self.assertTrue(so_metrics_g['edges_dist'] <= 1)
        self.assertTrue(so_metrics_g['edges_dist'] >= 0)

        self.assertTrue(so_metrics_g['NED_index'] < 1)
        self.assertTrue(so_metrics_g['NED_index'] >= 0)

        self.assertTrue(so_metrics_g['mean_community_size'] > 0)


if __name__ == '__main__':
    unittest.main()
