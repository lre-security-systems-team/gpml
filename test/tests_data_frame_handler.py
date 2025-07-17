# contributors: Julien MICHEL
# project started on 11/10/2022
import sys
from datetime import timedelta

import unittest
import pandas as pd

sys.path.append('.')
import gpml.data_preparation.data_frame_handler as itd


class TestChunk(unittest.TestCase):  # create csv with corresponding function and check csv

    def setUp(self):
        root = '.'
        self.filename = 'data/ugr16/ugr_sample_100k.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_chunk(self):  # test if different chunksize give equivalent result
        itd.chunk_read_and_insert_gc_metrics('./' + self.filename,
                                             'chunk_test1.csv', 360,edge_source = ['Source IP'],
                                             edge_dest = ['Destination IP'], chunksize=110000)
        itd.chunk_read_and_insert_gc_metrics('./' + self.filename,
                                             'chunk_test2.csv', 360,edge_source = ['Source IP'],
                                             edge_dest = ['Destination IP'], chunksize=15000)

        df1 = pd.read_csv('chunk_test1.csv')
        df2 = pd.read_csv('chunk_test2.csv')

        for i in range(0, 107894):
            self.assertEqual(df1["Date time"][i], df2["Date time"][i])

    def test_size(self):
        itd.chunk_read_and_insert_gc_metrics('./' + self.filename, 'chunk_test2.csv', 360,
                                             edge_source = ['Source IP'],edge_dest = ['Destination IP'],chunksize=15000)
        df2 = pd.read_csv('chunk_test2.csv')
        count = self.df.shape[0]
        count2 = df2.shape[0]

        self.assertEqual(count, count2)


class TestInsertMetrics(unittest.TestCase):  # Take any dataframe and compute community metrics

    def setUp(self):
        root = '.'
        self.filename = 'data/ugr16/ugr_sample_100k.csv'
        self.df = pd.read_csv(root + '/' + self.filename)

    def test_insert_ip(self):  # test computation and insert of graph community metric
        nb_col1 = len(self.df.columns)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)

    def test_insert_ipport(self):
        nb_col1 = len(self.df.columns)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP', 'Source Port'],
                                              ['Destination IP', 'Destination Port'], 'Label', 'ip5')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)

    def test_all_community_strategies(self):
        nb_col1 = len(self.df.columns)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5', community_strategy='lpa')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5',
                                              community_strategy='girvan_newman')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5', community_strategy='k_clique')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5', community_strategy='walktrap')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5',
                                              community_strategy='eigenvector')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5', community_strategy='leiden')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)
        df2 = itd.extract_community_metrics(self.df, 360, 'Date time', ['Source IP'],
                                              ['Destination IP'], 'Label', 'ip5', community_strategy='infomap')
        nb_col2 = len(df2.columns)
        self.assertTrue(nb_col2 > nb_col1)


if __name__ == '__main__':
    unittest.main()
